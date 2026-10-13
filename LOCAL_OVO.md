# 本机 OVO-Bench 视觉 memory 实验

新增 `action` 历史读取模式的协议、运行命令与输出字段见
[ACTION_READOUT.md](ACTION_READOUT.md)。该模式显式判断“停止 / 继续读取”，
不使用原生应答状态作为停止信号。以下已报告分数均属于对应的既有实验；
新模式需要使用独立结果目录评测，双卡包装默认仍为 `fill` 和 128K 总上下文。

## 来源与环境

- 仓库：`https://github.com/benZzz02/video2`
- 固定提交：`ebd5ef449e3510f38eb981a975c1d36372752f52`
- 模型：`/mnt/truenas_nfs/dataset/znh1302/OneStreamer`
- 数据：`/mnt/truenas_nfs/dataset/ovo-bench/chunked_videos`
- Python：`/home/zhangnuohua/miniconda3/envs/vllm/bin/python`
- 版本：PyTorch `2.7.1+cu128`，Transformers `4.57.6`，BF16，SDPA。
- tmux：`video2-ovo`；GPU 0 和 3，各一条独立推理进程。
- 额外依赖复用 `/home/zhangnuohua/medhorizon/OneStreamer/.runtime/ovo`，不升级共享 conda 环境。

输入来自前次完整 OneStreamer OVO 运行的 `manifest.json`，共 3035 个样本、
12 个子任务。原始标注和视频保持不变，旧 PHCM 结果不覆盖。

## 实验设置

采用仓库 `run_visual_memory.sh` 的 `progressive_pool + fill`，Recent16，
固定起点 1 fps，每次离开 recent 的特征以 stride=2 压缩，之后每 60 秒
视频时间再次压缩。保留原生视觉特征和全部 DeepStack 分支，不生成 caption。
CPU 历史库仍为 16384 视觉 token / 512 MiB。

显式使用 **131072（128K）总上下文**，覆盖仓库默认 32768。
原因是同一批数据中至少 172 个样本仅 Recent16 视觉 token 已超过 32K，
最大为 76800。保留全部样本、Recent16 和源分辨率，不能直接以默认 32K
完成全量评测。实际历史读入仍受库容量和剩余上下文限制。

本次是 **视觉 memory、Recent16、128K、SDPA** 设置。它与此前 PHCM 运行的
固定窗口采样相位和提示词也不同；分数差异不能只归因于记忆表示变化。
本次不额外运行 recent-only 消融。

## 执行与结果

独立外层包装 `tools/run_visual_ovo_local.py` 按视频时长将独立流分为两片，
调用原始入口，保持核心推理算法不变。正式结果目录：

```text
/home/zhangnuohua/medhorizon/video2/results/ovo-visual-memory-2gpu-20261011
```

运行命令（在 `vllm` 环境及仓库根目录）：

```bash
python -u tools/run_visual_ovo_local.py --output results/ovo-visual-memory-2gpu-20261011
```

中断后在同一命令末尾追加 `--resume`；检查参数和数据时追加 `--dry-run`。

正式任务于北京时间 **2026-10-11 11:05** 在 tmux `video2-ovo` 启动。
分片 0 在 GPU 0，共 1518 条；分片 1 在 GPU 3，共 1517 条。
两片视频总时长分别约 89.045 / 89.045 小时。`progress` 窗口跟踪实时日志。
输出根目录的 `status.json` 每 30 秒刷新完成数，`environment.json` 记录软件环境。

**已完成：** 北京时间 2026-10-11 21:55 正常结束，退出码 0，两片均一次完成，
3035 条预测全部成功，自动合并评分完成；全量墙钟耗时约 10 小时 50 分钟。
GPU 0 和 3 已释放。完整预测见输出目录 `predictions.jsonl`，评分见 `summary.json`。

| 指标 | 视觉 memory 本次结果 |
| --- | ---: |
| 三类任务等权综合分 | 73.20% |
| 历史回溯 | 78.01% |
| 实时视觉感知 | 79.47% |
| 前向主动响应 | 62.12% |
| 逐样本准确率 | 68.37%（2075/3035） |

此前 PHCM 综合分为 71.68%，此次高约 1.53 个百分点。由于采样相位、提示词和
上下文配置同时存在差异，这一结果不是只改变记忆形式的严格消融实验。

外层记录来源和配置，使用独立 attempt 目录保留逐条结果，恢复时跳过已成功
完成的样本。只有瞬时读取/解码错误会有限重试。不要在已有进程运行时重复启动。
全部完成后校验完整 ID 集合，再调用原始 `onestreamer.metrics.aggregate`
生成总体、三大类及 12 子任务分数。三大类内先对子任务取平均，最后对三类取平均。

## 启动前验证

2026-10-11：全量输入和双卡包装 dry-run 通过；仓库 61 项 CPU/微型模型测试、
包装 9 项 CPU 回归测试全部通过。包装测试覆盖部分写入后的恢复、身份校验、
读取重试预算、人工中断后的恢复、进程清理、运行中进度更新和完整结果合并。

真实 GPU smoke 的 5 条预测全部正常结束，不混入正式评分：

| 样本 | 视频时长 | 输入 token | 观察加回答耗时 | 验证 |
| --- | ---: | ---: | ---: | --- |
| ovo-1117 | 1 秒 | 1738 | 1.32 秒 | 无历史短视频 |
| ovo-0590 | 18 秒 | 3333 | 1.03 秒 | FIFO 移入历史 |
| ovo-0082 | 76 秒 | 59658 | 36.21 秒 | 1920×1440 源帧、历史容量丢弃 |
| ovo-0967 | 24 秒 | 86811 | 42.78 秒 | 2560×1920 源帧、完整 Recent16 |
| ovo-0836 | 120 秒 | 15829 | 8.17 秒 | 按年龄二次 pooling，104 个历史帧、149 次压缩 |

上述耗时不包含模型加载，只是 smoke 实测，不能代替全量性能或准确率。
前四条测试期间 1 秒采样的显存峰值分别为 GPU 0 28387 MiB、GPU 3 38053 MiB，
不是精确 allocator 峰值。结果保存在 `results/smoke-small-20261011`、
`results/smoke-high-20261011` 和 `results/smoke-age-20261011`。

## 状态驱动的历史读取（2026-10-12 新增）

当前为实验模式。GPU 0/3 上两轮各 12 条 smoke 均运行完成；v2 修复了输出格式，
但 25 次探测全部为 `</Silence>`，尚未实现有效的按需读取，不能据此宣称 RT 无损。
结果与限制见 `analysis/adaptive-visual-memory-20261012/IMPLEMENTATION.md`，
逐条记录在 `results/state-readout-smoke-v2-20261012/readout_audit.json`。

`--history-read-mode state` 保留同一套纯视觉缓存和 Recent16，使用冻结模型的
原生 `</Response>`、`</Standby>`、`</Silence>` 判断当前证据是否足以回答问题。
先只读 Recent16；输出 `</Response>` 时停止扩展历史，输出另两种状态时，
逐级增加历史视觉 token 预算。历史阶段采用嵌套选择，已读证据保留；达到预算、
上下文上限或没有更多可读历史时停止。若模型输出非预期状态，明确记录后在
当前视图回答。最终回答继续使用原有问题和格式要求，状态输出不进入评分文本。

默认历史预算为 `1024 4096 16384`，不包含 Recent16 或文本 token；实际读入量
受完整帧特征大小、存储容量和总上下文限制。路由器不接收 BT/RT/FT 标签或
标准答案，不调用 caption/summary 生成，也不训练新参数。它将已有状态语义
用于推理时的历史读取控制；是否能可靠保护 RT、保留 BT/FT 收益，需要实测。

在独立结果目录先检查完整输入与参数（此命令不启动推理）：

```bash
/home/zhangnuohua/miniconda3/envs/vllm/bin/python -u tools/run_visual_ovo_local.py \
  --output results/ovo-state-memory-v2-2gpu-20261012 \
  --history-read-mode state \
  --state-history-budgets 1024 4096 16384 \
  --dry-run
```

准备正式评测时去掉 `--dry-run`；恢复时保持全部参数一致并追加 `--resume`。
默认不指定读取模式仍为原有 `fill`。配置、历史预算和控制器源码哈希均进入
运行身份校验，不能把新配置续写进旧实验目录。

逐样本结果额外记录每轮状态、历史读取量、停止原因及状态探测耗时。
状态探测需要额外的模型前向计算；应将其计入查询耗时，同时对齐 Recent16、
采样、最终回答提示词和评分方式比较三类任务。评测重点是 RT 相对 recent-only
是否退化，以及 BT/FT 相对原视觉 memory 保留多少收益；实现完成不代表已证明
这些准确率目标。上述 2026-10-11 的 smoke 和分数属于原有 `fill` 实验。

随后已按用户要求在 tmux `onestreamer-state-full` 启动 v2 全量 3035 条实验，
使用相同 vllm 环境与 GPU 0、3，输出到上述 `ovo-state-memory-v2-2gpu-20261012`
目录。代码发布时实验仍在运行；最终状态分布、BT/RT/FT 得分以完整输出为准。
现有运行目录已包含 manifest，检查使用 `--dry-run`，需要恢复时使用 `--resume`；
不要重复启动已有运行。
