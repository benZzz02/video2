# 显式历史读取动作

`--history-read-mode action` 使用冻结的 OneStreamer 决定是否继续读取已存储的
历史视觉特征。当前实验关注已有视觉 memory 的读取量：近期证据可直接回答时
停止，问题依赖此前过程时继续扩展历史视图。

## 读取协议

每次查询从 recent 视图开始，历史视觉 token 预算依次为
`0 → 1024 → 4096 → 16384`。`0` 由程序自动添加，后面三档通过
`--state-history-budgets` 设置；该参数同时服务于 `state` 和 `action` 两种模式。
预算不包含 recent 帧或提示词，实际读取量受完整帧大小、存储容量和总上下文限制。

探测提示将问题、当前可见帧及真实的历史读取元数据一起交给模型：近期时间范围、
已经读取的历史、剩余记录的时间分布。元数据只描述存储和读取情况，不生成历史
caption，也不声称被压缩或丢弃的内容仍然可见。已读时间戳的最早和最晚值不等于
中间所有记录都已读。

模型输出一个动作：

- `A` 对应 `STOP`：当前证据能够支持问题要求的判断，停止扩展。
- `B` 对应 `READ_MORE`：需要读取更多已有历史，进入下一档可用预算。

提示重点区分当前画面的物体计数与一段时间内的事件累计次数。对于累计、首次、
末次或从未发生等判断，要求检查相关时间范围内尚未读取的记录；对于后续发展，
只请求已经观察并存储的过程证据。探测输入不包含任务类别或标准答案。

动作来自全词表的原始单 token argmax，探测前验证 `A`、`B` 均为单 token。
代码不把输出强制约束到这两个 token：其他输出被记录为协议错误，并在仍有可用
历史时继续扩展。动作探测与最终答案分别生成，最终答案沿用原回答提示和格式。

历史耗尽、预算耗尽或上下文放不下由执行器终止，并与模型主动 `STOP` 分别记录。
提示文本和历史元数据都参与上下文计数。历史选择沿用逐档嵌套方式，扩展时保留
已经选入的记录，避免预算增大反而丢失前一轮证据。

## 运行

双卡本地包装沿用已有机器路径、GPU 配置和 **131072（128K）总上下文**。
默认模式仍为 `fill`。先使用一个新的输出目录检查输入与参数：

```bash
/home/zhangnuohua/miniconda3/envs/vllm/bin/python -u tools/run_visual_ovo_local.py \
  --output results/ovo-action-memory-2gpu-20261013 \
  --history-read-mode action \
  --state-history-budgets 1024 4096 16384 \
  --dry-run
```

`--dry-run` 检查配置、数据和运行身份，不加载模型、不验证实际动作输出。
正式推理使用相同命令去掉 `--dry-run`。中断恢复时，保持全部参数、代码和输入一致，
追加 `--resume`。源码或提示词变化后使用新的结果目录，不能接续旧配置结果。

也可以使用通用入口并提供自己的模型、manifest 和视频路径：

```bash
EVAL_PYTHON=/path/to/python bash run_visual_memory.sh \
  --manifest /path/to/manifest.json \
  --model-path /path/to/OneStreamer \
  --video-root /path/to/videos \
  --output results/ovo-action-memory \
  --history-read-mode action \
  --context-tokens 131072 \
  --state-history-budgets 1024 4096 16384
```

通用脚本本身默认总上下文为 32768；上面的显式参数覆盖它。双卡包装使用的 128K
配置与此前这台机器上的完整 Fill / State 实验一致。

## 结果与审计

逐题预测记录 `action_rounds`、`action_probe_seconds`、`action_stop_reason`
和 `action_invalid_count`。每轮保留实际动作和读取视图，便于判断停止是否发生于
零历史，以及异常输出是否导致继续读取。

双卡包装的 `summary.json` 另外汇总：

- `action_probe_seconds`、`action_probe_count`：总探测耗时与次数。
- `action_counts`、`action_invalid_count`：STOP、READ_MORE 和异常输出数量。
- `action_stop_reasons`：模型停止及执行器边界的分布。

原有 `state_*` 字段继续独立汇总。动作探测耗时已包含在查询耗时中，不应重复相加。
模式、预算、两个读取控制器和模型适配器的源码哈希都进入运行身份校验，包括尚未
`git add` 的新控制器文件。

此模式不训练参数，保留现有 recent、历史写入与压缩方式，并沿用最终回答入口。
需要通过新的推理结果检验：累计计数中的零历史早停是否减少，以及 BT / RT / FT
的最终变化。当前代码与 CPU 测试不提供准确率结论。

## 本地验证（2026-10-13）

核心模块 114 项测试和本地运行器 19 项测试通过。核心测试使用
Transformers 4.57.6、PyTorch 2.9.1 与随机权重的小型 Qwen3-VL，检查真实
模型前向中的特征注入、动作 logits、上下文计数，以及插入探测前后的最终答案
tokens / logits 一致。控制器测试覆盖非法动作继续、非单调 metadata 长度、
预算与上下文耗尽；视频集成测试包含原采样器的分数起点量化行为。

AST 对比确认历史写入、pooling、recent 窗口、采样器、特征注入及最终回答函数
保持原实现。测试使用本地合成数据，没有执行正式 OneStreamer checkpoint 的
OVO / StreamingBench 推理。
