# 按上下文填满的视觉历史（budget_pool）

2026-10-15。修复 `progressive_pool + fill` 的两个问题：

1. **长期历史实际没有存下。** 旧版存储上限固定 16,384 token；1080p 级视频每帧
   stride-2 后 690 token，只能存约 23 帧，帧在 60 秒的再压缩到期前就被丢弃。
   EPM 有 92% 样本丢掉了早期历史，中位回看仅 32 秒，128K 上下文约 70K 未使用。
2. **近期窗口比官方晚一帧。** 采样条件 `< end` 使最后一帧落在 `end−1s`；官方
   end-aligned 读法取 `ceil(end·fps)−1`。

## 新机制

- `--history-compression budget_pool`：帧离开 Recent16 时先做一次 stride-2 pooling 入库。
  超出预算时，对“token 最多的记录中最旧的一帧”再 pooling 一档，因此分辨率随时间
  单调不增。所有记录降到 `--history-min-frame-tokens`（默认 16，再压一次可能低于该值）后，
  才删除使相邻时间间隔最小的一帧，即均匀时间抽稀，并保留最早观察。
- `--history-store-budget context`：每条视频流在第一帧编码后计算历史预算：
  总上下文 − 16 帧近期 token − 每帧时间戳/包装文本 − 该流最长问题的文本 − 回答预留 − 64。
  每帧文本开销计入预算；fill 读取仍逐 token 精确校验，超出时退回原有选择逻辑。
- `--stream-include-end`：位于问题时刻的采样槽取截止前最后一个源帧，与官方近期
  最后一帧一致；仍严格早于问题时刻。
- 字节上限另设 4 GiB；每 token（含 DeepStack）约 20 KiB，86K token 约 1.7 GiB。

旧模式和默认参数行为不变；新版本号 `visual_memory_v7_context_budget_pool`。
历史无内容时的最终提示词与官方 Recent16 仍不同，本次未改。

## 运行

```bash
/home/zhangnuohua/miniconda3/envs/vllm/bin/python -u tools/run_visual_ovo_local.py \
  --output results/ovo-fill-full-context-2gpu-20261015 \
  --memory-layout full_context --dry-run
```

去掉 `--dry-run` 正式运行；中断后保持参数不变加 `--resume`。包装器固定使用 GPU 0/3。

逐题 `bank` 新字段：`token_budget`、`frame_overhead`、`budget_cost`、`earliest_timestamp`、
`frame_tokens_min/max`、`dropped_frames`。用 `earliest_timestamp` 和 `history_timestamps`
检查是否覆盖到视频开头。

## 验证

CPU 测试：核心 125 项（新增 11 项，覆盖预算不变量、分辨率单调、抽稀保留首帧、
1 小时 1080p 模拟、采样与官方最后一帧对齐、增量查询不丢槽、fill 端到端）和本地运行器
28 项均通过。

GPU 难题试验（`analysis/fullctx-hard-trial-20261015/`）：按固定哈希选取 Recent16 与旧 Fill
都答错的 90 题，另加 20 道 RT 对照（Recent16 对、旧 Fill 错）。新版修复：

| 组 | 修复 | 历史回看中位（旧 Fill → 新版） |
|---|---:|---:|
| EPM | 6/30 | 32 s → 258 s |
| SSR | 2/20 | 93 s → 145 s |
| CRR | 1/15 | 589 s（不变） |
| ASI | 0/10 | 93 s → 195 s |
| REC | 0/15 | 30 s（不变） |
| RT 对照恢复 | 3/20 | 93 s → 160 s |

EPM 修复均伴随历史回看显著延长；REC/CRR 历史本来完整，错误不在存储。RT 干扰基本仍在。
这是目的性抽样，不代表全量提升幅度。最长输入 129,814 token，RTX 6000D 正常，
单题问答中位 13 秒、最长 58 秒；无预算超限，所有题历史均从流起点开始。

全量（3,035 题）于 2026-10-15 14:37 以 `analysis/run_full_fullctx_20261015.sh` 启动。
