# V-CAST 风格的历史 token 保留（可选）

新增 `--history-compression vcast`。不训练模型，沿用现有采样、Recent-N、问题路由和回答流程，只替换历史帧的存储压缩。原有命令仍采用原来的模式和参数；`run_visual_memory.sh` 默认仍是 `progressive_pool`，本地运行器的 `full_context` 默认仍是 `budget_pool`。

## 保留什么

1. 帧离开 Recent-N 后，完整的原生视觉特征先进入 CPU 历史库。预算充足时不压缩。
2. 超出预算时，用各帧**原始特征均值**构成的轨迹计算曲率，以 softmax 分配各帧的 token 数。温度默认 `0.7`。
3. 帧内按照 `1 − cosine(token, frame_mean) + minmax(token_norm)` 排序，选择原始 token。选中的 token 不求平均，按原始空间顺序保存。
4. 最终视觉特征和每一个 DeepStack 分支使用相同索引。读入时移除未选 token 的占位符，保留原始网格坐标、时间戳、文本顺序以及生成阶段的位置偏移。

稀疏读出直接展开文本中的图像占位符，不为历史帧创建原尺寸空白图片。位置计算仍临时重建原始网格的整数索引，不重建完整视觉特征或 KV cache。

空间选择可能保留小物体、局部变化等原有池化会混合掉的证据，但分数不是语义正确性的保证，需要实际任务验证。

## 适配主方法（2026-10-20）

1. **延迟取出，结果不变。** 每帧的打分在进入记忆时就固定，平分时按原位置先后排序，所以任意一次保留的都是同一个排序的前 k 个，少保留的集合总是多保留的子集。存储只记录每帧保留数 `keep`；驻留的行数达到 `keep` 两倍时才真正取出，读取前也会取出。原来存储满后每进一帧都要把所有旧帧复制一遍。300 条随机流上与原实现逐项一致（保留位置、特征、统计）。237 帧 1080p 的 CPU 模拟中，建记忆从 60.7 s 降到 14.0 s。统计量按保留数计算；驻留量另记为 `resident_tokens`，最多为保留数的两倍。
2. **"刚才"路径覆盖的帧与主方法相同。** `budget_pool` 的最新历史帧都是 2×2 池化后的大小，"刚才"读取的是能放进 16K token 的最新若干帧。V-CAST 版先按池化后的大小选出同一批帧，若这批帧的保留 token 超过 16K，就按同样的曲率分配和帧内打分压进 16K（只在读取时选择，不改存储）。两种存储因此读取同样的时间范围、同样的 token 上限，只有帧内保留什么不同。逐题记录的 `route` 新增 `short_frames`、`short_pooled_span_tokens`、`short_retained_tokens`、`short_read_tokens`。

3. **最新历史帧使用真实曲率。** 帧离开最近窗口时，它的后一帧已经在最近窗口里（已观察到），用它计算这帧的曲率，后一帧本身不分配权重。原来最新历史帧按端点曲率 1 第一次分配，之后真实曲率更高也无法加回 token，结果越早的帧保留越多（ovo-0012 按时间四段平均每帧 463/457/369/301，ovo-0166 为 255/238/230/189）。400 帧模拟中，修改前逐帧与"按最终曲率一次分配"的相关为 0.37，修改后 399 帧全部一致。

## 与原版 V-CAST 的关系

参考[官方 Qwen3-VL 实现](https://github.com/xinyouu/V-CAST/blob/f15d2764e2381e0344fea9eb6228a4a58654f4f7/compressor/v_cast/modeling_qwen3_vl_v_cast.py)，固定参考提交 `f15d2764e2381e0344fea9eb6228a4a58654f4f7`。

本实现沿用帧内双分数和帧间曲率公式，是**有硬预算的流式适配**，不是论文离线固定保留比例的复现。分配只使用已观察且仍保留的历史帧；各帧可有不同网格大小。已丢弃 token 不恢复，后续压缩使用原始均值和存活 token 的原始分数，不重算一个压缩后的“新均值”。不保存完整特征的隐藏副本。

token、字节和帧数上限均强制执行。每帧的最少 token 数沿用 `--history-min-frame-tokens`（默认 `16`）；放不下这些最低配置时，沿用时间间隔抽稀规则删除帧。只剩一帧时可以降到该下限以下，仍不能突破预算。最低一个 token 都放不下时移除该帧。有限预算不保证永远保留所有历史帧。

## 使用

在已有 `run_visual_memory.sh` 命令后追加：

```bash
--history-compression vcast
```

建议与当前 `full_context` 同预算、同路由做对照。本地运行器的检查命令：

```bash
python tools/run_visual_ovo_local.py \
  --memory-layout full_context \
  --history-compression vcast \
  --history-read-mode routed \
  --route-threshold 0.8 --route-escalation SHORT \
  --output results/ovo-vcast-routed \
  --dry-run
```

这里显式写出当前 HEART 的 `.8/SHORT` 路由设置；V-CAST 不改变路由默认值。按实际机器传入原有 `--manifest`、`--model-path`、`--video-root`、`--python` 参数。运行器沿用现有 GPU 配置。检查通过后去掉 `--dry-run` 才会启动推理；本次开发没有启动正式实验。

若现有实验使用 `fill`，保留 `--history-read-mode fill`。温度可通过 `--vcast-temperature 0.7` 显式指定。推荐先使用默认值，无需增加学习模块。

使用新的输出目录，不能把不同压缩模式的结果续写到同一目录。代码指纹和配置会参与恢复校验；更新代码后的旧目录会因指纹不一致被拒绝，这是防止结果混用的既有行为。

## 检查记录

新运行版本标识为 `visual_memory_v10_vcast_streaming`。manifest 记录参考提交、适配方式及两个新模块的源码散列。逐题记录沿用 `history_timestamps`、`history_frame_tokens` 和 `history_visual_tokens`，新增 bank 统计：

- `selection_operations`、`selection_input_tokens`、`selection_output_tokens`：历史 token 选择次数及累计输入输出数。
- `stored_feature_bytes`、`metadata_bytes`、`stored_bytes`：特征与位置、选择所需元数据、两者总量。总量受字节上限约束，不代表 Python 对象或分配器的实际内存占用。
- `budget_cost`：历史 token 加逐帧提示开销，不能超过 `token_budget`。

稀疏回答还记录 `visual_position_mode=original_grid_sparse_mrope`、`image_retained_tokens` 和 `image_original_tokens`。这里的 `image_grid_thw` 始终是原始网格，不能用网格面积代替实际保留 token 数。原有 `history_pool_scales` 在本模式为 `1`；旧 pooling 累计量为 `0`，选择量见 bank 字段。

## 验证范围

支持范围沿用现有适配器：`transformers==4.57.6`、Qwen3-VL 架构、单样本图像帧输入、1 fps、单束生成。OneStreamer 使用同一架构接口。当前没有新增其他模型族支持。

本地测试覆盖原生全 token 输出一致性、稀疏原坐标与多步生成、DeepStack 对齐、不同网格、重复不可逆压缩、因果时间顺序、禁用历史、硬预算和旧模式回归。CPU 微型随机模型用于验证接口正确性；真实 checkpoint 的 GPU 运行、准确率和是否优于 `budget_pool` 仍待同设置实验确认。

2026-10-20 本地验证：Python 3.11、PyTorch 2.9.1、Transformers 4.57.6，下面的测试集 **181 项及 234 个子用例全部通过**。

```bash
python -m pytest -q Eval/Proactive_Eval/OneStreamer/tests tests_local/test_run_visual_ovo_local.py
```
