# 按问题选择是否读视觉历史（question router + visual check）

记忆写入时不知道会被问什么；问题到来后再决定读多少历史。不使用任务标签或题型规则。

1. **问题范围判断**（`router.py`，在线版为 `onestreamer/question_router.py`）：冻结的 OneStreamer
   只读问题文字（无画面、无选项、无任务标签），在 NOW / SHORT / LONG 三个单 token 标签上取全词表
   log-prob，三种定义顺序轮换取平均。问题放在 `<VIEWER_QUESTION>` 内，其中的答题格式指令被声明为
   留给之后作答；再减去空问题 "N/A" 的 log-prob 作校准（无需标注）。
2. **画面检查**（`visual_check.py`）：判为 NOW 的题用官方 Recent16 路径作答（1141/1141 与官方答案逐题一致），
   记录首个答案 token 的全词表概率；低于阈值则改读历史。
3. **读取**：NOW→官方 Recent16；SHORT→最新 16K 视觉 token 的历史；LONG→填满上下文的全部历史。

阈值规则事先固定：只在 half0（按视频哈希切分）上选，在 RT 不低于 Recent16−1 的前提下最大化 Overall，
得到 τ=0.80、升级到 SHORT。用已有三版预测离线拼接（`combine.py`、`combine_vcheck.py`、`curve.py`），
留出的 half1：

| half1 | BT | RT | FT | Overall |
|---|---:|---:|---:|---:|
| Recent16 | 75.82 | 83.03 | 51.34 | 70.06 |
| 旧 Fill | 79.14 | 78.57 | 60.10 | 72.61 |
| 只用问题判断 | 79.84 | 82.10 | 56.08 | 72.68 |
| 问题判断 + 画面检查 | 80.08 | 81.28 | 60.40 | 73.92 |

局限：CRR 题面写明"与最新帧有关"，问题判断全部判为 NOW，需画面检查补救；低置信度只说明 Recent16
可能错，对"此刻"类题读历史往往更差，因此 RT 仍比 Recent16 低约 1.75。

完整系统（`--history-read-mode routed --route-threshold 0.8 --route-escalation SHORT`，配合
`--memory-layout full_context`）的全量运行脚本为 `../run_full_routed_20261017.sh`，结果待补。
离线拼接中 SHORT 使用旧 Fill 的预测，完整系统使用新存储中最新 16K token 且近期帧已对齐，结果可能略有差异。
