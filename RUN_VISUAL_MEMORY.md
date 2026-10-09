# OneStreamer 渐进视觉记忆：训练机运行说明

这是推理实验包，不需要训练。使用冻结的 OneStreamer-4B checkpoint。
包含官方仓库快照和新增视觉记忆模块、测试、使用文档；不包含模型权重、
完整评测视频数据或 Python 环境。官方仓库已有示例和标注仍保留。
克隆到独立目录即可，不需要覆盖训练机上已有的官方仓库。

## 1. 在训练机克隆

```bash
git clone git@github.com:benZzz02/video2.git
cd video2
```

后续更新使用 `git pull --ff-only`。下面所有命令都从仓库根目录执行。

## 2. 使用推理环境

优先激活训练机上已经能运行官方 OneStreamer 的环境。
适配器要求 **transformers==4.57.6**。不要直接升级到最新版。
官方环境为 Linux x86_64、Python 3.10.9、PyTorch 2.6.0 + CUDA 12.4。
如需新建环境，从仓库根目录执行官方安装步骤：

```bash
conda create -n onestreamer-visual python=3.10.9 -y
conda activate onestreamer-visual
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
python -m pip install packaging==25.0 psutil==7.2.2 ninja setuptools wheel
python -m pip install --no-build-isolation -r Eval/requirements.txt
python -m pip install --no-deps -r Eval/requirements-opencv.txt
```

FlashAttention 的 wheel/编译环境需要与训练机匹配。OpenCV 的 `--no-deps`
是官方安装要求，只用于该文件；环境版本说明见
[environment.md](Eval/docs/environment.md)。

## 3. 指定训练机上的三个路径

```bash
VM_MODEL=/data/models/OneStreamer-4B
VM_VIDEOS=/data/datasets/OVOBench
VM_MANIFEST=/data/evaluation/EPM/manifest.json
```

`VM_MANIFEST` 使用官方评测流程已经生成、含 `samples` 的 manifest。
不是任意原始 benchmark 标注 JSON。视频路径按 `VM_VIDEOS / sample.video`
解析，因此 `VM_VIDEOS` 应与 manifest 的 `video` 相对路径匹配。
也可将后续命令中的 `--manifest` 换成 `--samples`，读取官方统一 Sample
格式的 JSON 数组或 JSONL。单任务 manifest 不需要额外填写 `--subtasks`。

如果没有 manifest，先按官方
[Eval/README.md](Eval/README.md) 配置路径并运行评测 dry-run 生成。
本入口的 `--dry-run` 仅校验输入，不生成新的 manifest。

## 4. 先检查路径，再运行小样本

```bash
bash run_visual_memory.sh \
  --manifest "$VM_MANIFEST" --model-path "$VM_MODEL" \
  --video-root "$VM_VIDEOS" --output results/history20 \
  --limit 20 --dry-run
```

通过后运行历史特征版本：

```bash
CUDA_VISIBLE_DEVICES=0 bash run_visual_memory.sh \
  --manifest "$VM_MANIFEST" --model-path "$VM_MODEL" \
  --video-root "$VM_VIDEOS" --output results/history20 \
  --limit 20
```

运行同样样本、同样 recent 配置的无历史对照：

```bash
CUDA_VISIBLE_DEVICES=0 bash run_visual_memory.sh \
  --manifest "$VM_MANIFEST" --model-path "$VM_MODEL" \
  --video-root "$VM_VIDEOS" --output results/recent20 \
  --limit 20 --history-capacity 0
```

先后运行这两条命令。每次真实运行必须使用空的输出目录，不支持断点续跑。
`--limit 20` 是过滤后的前 20 个问题，不是每类 20 个，也不是随机样本。
先用来检查是否能跑通，再移除该参数运行相同的正式样本集。
如已具备其他依赖但缺少 FlashAttention，可临时追加
`--attn-implementation sdpa` 做跑通检查；耗时对比应固定后端。

## 5. 启动脚本采用的配置

- Recent16 保留完整特征；追加 `--recent-frames 8` 或其他正整数即可修改。
- 特征离开 recent 时做 stride=2 空间 pooling；此后每过 60 秒视频时间再 pooling。
- 历史帧压至单 token 后停止继续压缩；存储超限时丢最早的历史记录。
- `progressive_pool + fill` 已由启动脚本显式启用。
- 32,768 是总上下文上限，包含 recent、历史、提示词和回答预留。
- CPU 历史库默认另有 16,384 视觉 token / 512 MiB 存储上限；fill 不保证每次填满32K。
- 无 PHCM、无 caption 生成、无训练、无问题到来后回读已丢弃的视频。

两组实验都通过同一个启动脚本，保持相同 recent 数、采样、上下文、checkpoint
和问题集。这个 matched recent-only 与官方 question-end-aligned recent16
采样协议有区别，正式报告中要区分。完整机制说明见
[VISUAL_MEMORY.md](Eval/Proactive_Eval/OneStreamer/VISUAL_MEMORY.md)。

结果在各自目录的 `predictions.jsonl`、`summary.json` 和 `manifest.json`。
效率比较包括从流开始的观察编码时间与问答时间；不要只比较问答耗时。

## 6. 验证范围

本地 61 项合成视频/微型随机 Qwen3-VL CPU 测试已通过。
尚未用发布的 OneStreamer-4B 权重验证 CUDA/bfloat16 推理、BT 收益或实际速度。
训练机先执行上面的路径检查和小样本运行。
如需重新运行模块测试，从仓库根目录执行：

```bash
PYTHONDONTWRITEBYTECODE=1 python -m unittest discover \
  -s Eval/Proactive_Eval/OneStreamer/tests -p 'test_*.py' -q
```

代码来源和新增模块清单见 `SOURCE_INFO.json`。Git 提交标识可用于固定实验版本。
