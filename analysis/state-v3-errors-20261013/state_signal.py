#!/usr/bin/env python3
"""Read-only, descriptive audit of native-state signals and paired fill answers.

No models are loaded and no threshold is fitted. Thresholds/bins are declared
before reading results. Only recorded recent-only Response endpoints have an
observed recent-only answer; intermediate probe states do not supply answers.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics

HERE = Path(__file__).resolve().parent
FIX = HERE.parents[1]
STATE_PATH = FIX / "results/ovo-state-memory-v3-2gpu-20261012/predictions.jsonl"
FILL_PATH = FIX.parent / "video2/results/ovo-visual-memory-2gpu-20261011/predictions.jsonl"
THRESHOLDS = (.5, .6, .7, .8, .9, .95)
BIN_EDGES = (0., *THRESHOLDS, 1.)
RESPONSE, SILENCE, STANDBY = "</Response>", "</Silence>", "</Standby>"
TASK_SHORT = {"backward_tracking": "BT", "real_time_visual_perception": "RT",
              "forward_active_responding": "FT"}
MATCH_FIELDS = ("sample_id", "task", "subtask", "video", "start", "end", "question", "options",
                "answer", "eval_mode", "recent_timestamps", "recent_image_sizes", "recent_sampling")


def load(path):
    rows = [json.loads(line) for line in path.open() if line.strip()]
    indexed = {row["sample_id"]: row for row in rows}
    assert len(indexed) == len(rows), f"Duplicate sample IDs: {path}"
    assert all(row["status"] == "ok" for row in rows), f"Non-success results: {path}"
    return rows, indexed


def rate(num, den):
    return num / den if den else None


def p_response(probe):
    return probe["state_probabilities"][RESPONSE]


def state(row, first=False):
    return row["state_rounds"][0 if first else -1].get("state") or "invalid"


def auc(scores, labels):
    """Mann–Whitney AUC with exact tie averaging; no fitted parameters."""
    positives = sum(labels)
    negatives = len(labels) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(zip(scores, labels))
    positive_rank_sum = 0.0
    start = 0
    while start < len(ordered):
        end = start + 1
        while end < len(ordered) and ordered[end][0] == ordered[start][0]:
            end += 1
        average_rank = (start + 1 + end) / 2
        positive_rank_sum += average_rank * sum(label for _, label in ordered[start:end])
        start = end
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def summary(rows, fill):
    n = len(rows)
    final_response = [r for r in rows if state(r) == RESPONSE]
    final_silence = [r for r in rows if state(r) == SILENCE]
    return dict(n=n, correct=sum(r["hit"] for r in rows),
                sample_micro_accuracy=rate(sum(r["hit"] for r in rows), n),
                fill_correct=sum(fill[r["sample_id"]]["hit"] for r in rows),
                initial_states=dict(Counter(state(r, first=True) for r in rows)),
                final_states=dict(Counter(state(r) for r in rows)),
                response_correct=sum(r["hit"] for r in final_response),
                response_n=len(final_response),
                response_accuracy=rate(sum(r["hit"] for r in final_response), len(final_response)),
                response_mean_probability=statistics.mean(p_response(r["state_rounds"][-1])
                                                         for r in final_response) if final_response else None,
                silence_correct=sum(r["hit"] for r in final_silence),
                silence_n=len(final_silence),
                silence_accuracy=rate(sum(r["hit"] for r in final_silence), len(final_silence)),
                silence_mean_probability=statistics.mean(r["state_rounds"][-1]["predicted_probability"]
                                                        for r in final_silence) if final_silence else None,
                stop_reasons=dict(Counter(r["state_stop_reason"] for r in rows)))


def grouped(rows, key, fill):
    groups = defaultdict(list)
    for row in rows:
        groups[key(row)].append(row)
    return {name: summary(group, fill) for name, group in sorted(groups.items())}


def answer_group(row):
    mode = row["eval_mode"]
    answer = str(row["answer"]).strip().lower()
    if mode == "substring_match" and answer in {"yes", "no"}:
        return f"yes_no_target_{answer}"
    if mode == "integer_match":
        return "count_target_zero" if int(answer) == 0 else "count_target_positive"
    return "multiple_choice"


def paired(rows, fill):
    out = Counter()
    for row in rows:
        ours = bool(row["hit"])
        theirs = bool(fill[row["sample_id"]]["hit"])
        out["both_correct" if ours and theirs else "fill_only_correct" if theirs else
            "state_only_correct" if ours else "both_wrong"] += 1
    return dict(n=len(rows), **{name: out[name] for name in
                ("both_correct", "fill_only_correct", "state_only_correct", "both_wrong")},
                net_fill_gain=out["fill_only_correct"] - out["state_only_correct"])


def bins(rows, fill, first=False):
    result = []
    for index, (lo, hi) in enumerate(zip(BIN_EDGES, BIN_EDGES[1:])):
        group = [r for r in rows if lo <= p_response(r["state_rounds"][0 if first else -1])
                 and (p_response(r["state_rounds"][0 if first else -1]) < hi
                      or (hi == 1. and p_response(r["state_rounds"][0 if first else -1]) <= hi))]
        probs = [p_response(r["state_rounds"][0 if first else -1]) for r in group]
        result.append(dict(low=lo, high=hi, high_inclusive=hi == 1., n=len(group),
                           mean_p_response=statistics.mean(probs) if probs else None,
                           correct=sum(r["hit"] for r in group),
                           sample_micro_accuracy=rate(sum(r["hit"] for r in group), len(group)),
                           paired_with_fill=paired(group, fill)))
    return result


def thresholds(rows, fill):
    """Paired endpoint substitution ONLY, not simulation of unobserved stages."""
    out = []
    for threshold in THRESHOLDS:
        divert = [r for r in rows if p_response(r["state_rounds"][0]) < threshold]
        keep = [r for r in rows if p_response(r["state_rounds"][0]) >= threshold]
        pair = paired(divert, fill)
        correct = sum(r["hit"] for r in keep) + sum(fill[r["sample_id"]]["hit"] for r in divert)
        out.append(dict(threshold=threshold, n=len(rows), keep_recent_n=len(keep),
                        divert_to_recorded_fill_n=len(divert), divert_fraction=rate(len(divert), len(rows)),
                        diverted_fill_gains=pair["fill_only_correct"],
                        diverted_fill_losses=pair["state_only_correct"],
                        net_observed_hit_change=pair["net_fill_gain"],
                        subset_endpoint_substitution_correct=correct,
                        subset_endpoint_substitution_accuracy=rate(correct, len(rows)),
                        retained_recent_wrong=sum(not r["hit"] for r in keep),
                        retained_recent_wrong_fill_correct=sum(not r["hit"] and fill[r["sample_id"]]["hit"]
                                                               for r in keep)))
    return out


def discrimination(rows, fill):
    probs = [p_response(r["state_rounds"][0]) for r in rows]
    correct = [bool(r["hit"]) for r in rows]
    fill_gain = [not r["hit"] and bool(fill[r["sample_id"]]["hit"]) for r in rows]
    discordant = [r for r in rows if bool(r["hit"]) != bool(fill[r["sample_id"]]["hit"])]
    return dict(n=len(rows), p_response_auc_for_recent_answer_correct=auc(probs, correct),
                negative_p_response_auc_for_fill_only_correct=auc([-p for p in probs], fill_gain),
                discordant_n=len(discordant),
                negative_p_response_auc_for_fill_wins_on_discordant=auc(
                    [-p_response(r["state_rounds"][0]) for r in discordant],
                    [bool(fill[r["sample_id"]]["hit"]) for r in discordant]))


def example(row, fill):
    return dict(sample_id=row["sample_id"], task=row["task"], subtask=row["subtask"],
                answer_group=answer_group(row), answer=row["answer"], prediction=row["prediction"],
                correct=bool(row["hit"]), fill_prediction=fill[row["sample_id"]]["prediction"],
                fill_correct=bool(fill[row["sample_id"]]["hit"]),
                initial_state=state(row, True), final_state=state(row),
                initial_p_response=p_response(row["state_rounds"][0]),
                final_p_response=p_response(row["state_rounds"][-1]),
                final_predicted_probability=row["state_rounds"][-1]["predicted_probability"],
                history_visual_tokens=row["history_visual_tokens"],
                fill_history_visual_tokens=fill[row["sample_id"]]["history_visual_tokens"])


def main():
    rows, ours = load(STATE_PATH)
    fill_rows, fill = load(FILL_PATH)
    assert ours.keys() == fill.keys(), "Incomplete paired sample set"
    for row in rows:
        assert row["state_rounds"], row["sample_id"]
        assert row["state_rounds"][0]["history_visual_tokens"] == 0
        assert row["history_timestamps"] == row["state_rounds"][-1]["history_timestamps"]
        assert row["history_visual_tokens"] == row["state_rounds"][-1]["history_visual_tokens"]
    mismatches = {r["sample_id"]: [key for key in MATCH_FIELDS if r.get(key) != fill[r["sample_id"]].get(key)]
                  for r in rows}
    mismatches = {sid: fields for sid, fields in mismatches.items() if fields}
    initial_response = [r for r in rows if state(r, True) == RESPONSE]
    assert all(r["history_visual_tokens"] == 0 and len(r["state_rounds"]) == 1 for r in initial_response)
    observed_subset = [r for r in initial_response if r["sample_id"] not in mismatches
                       and fill[r["sample_id"]]["history_visual_tokens"] > 0]
    final_response = [r for r in rows if state(r) == RESPONSE]
    probes = [probe for row in rows for probe in row["state_rounds"]]
    per_task_subset = {task: [r for r in observed_subset if r["task"] == task] for task in sorted(TASK_SHORT)}
    score = dict(
        generated_at=datetime.now(timezone.utc).isoformat(),
        sources={"state": dict(path=str(STATE_PATH), sha256=hashlib.sha256(STATE_PATH.read_bytes()).hexdigest(),
                               rows=len(rows)),
                 "fill": dict(path=str(FILL_PATH), sha256=hashlib.sha256(FILL_PATH.read_bytes()).hexdigest(),
                              rows=len(fill_rows))},
        definitions={
            "probability": "Full-vocabulary next-token P(</Response>), not normalized across states and not trained answer confidence.",
            "correctness": "Recorded final answer hit on the same final visual view, using the unchanged ordinary answering prompt; this is a proxy for evidence usability, not proof of semantic sufficiency.",
            "initial_silence": "Its subsequent answer may use more history. Do not label initial Silence correct/incorrect using that changed-view answer.",
            "endpoint_subset": "Initial Response, no historical tokens in its final answer, paired fill answer has >0 historical tokens, and all listed question/recent metadata agree.",
            "threshold_audit": "Prespecified thresholds only. Within the observed endpoint subset, P(Response)<threshold substitutes the recorded fill answer. This is NOT a rerun or simulation of a modified progressive controller.",
            "missing_counterfactuals": "No recent-only answers for initially non-Response rows; no answers at intermediate probe views; no later state decisions/answers after earlier Response. Those counterfactuals are excluded.",
            "metric": "All correctness rates here are sample micro rates. They are not the benchmark's task/subtask macro aggregate.",
            "labels": "Task and target-answer labels are used only in this post-hoc audit, never in the deployed controller.",
            "inference_limit": "Descriptive associations only; selected endpoints, prompt differences and correlated questions/video streams preclude calibration or causal generalization claims.",
        },
        prespecified_thresholds=list(THRESHOLDS), confidence_bin_edges=list(BIN_EDGES),
        matched_recent_fields=list(MATCH_FIELDS), recent_metadata_mismatches=mismatches,
        all_probes=dict(n=len(probes), states=dict(Counter(p.get("state") or "invalid" for p in probes)),
                        state_probability_mass_min=min(p["state_probability_mass"] for p in probes),
                        state_probability_mass_median=statistics.median(p["state_probability_mass"] for p in probes)),
        overall=summary(rows, fill), by_task=grouped(rows, lambda r: r["task"], fill),
        by_subtask=grouped(rows, lambda r: r["subtask"], fill),
        by_target_answer_kind=grouped(rows, answer_group, fill),
        by_subtask_target_answer_kind=grouped(rows, lambda r: r["subtask"] + ":" + answer_group(r), fill),
        final_response_confidence_bins=bins(final_response, fill),
        final_response_confidence_bins_by_task={task: bins([r for r in final_response if r["task"] == task], fill)
                                                for task in sorted(TASK_SHORT)},
        recent_response_endpoint_subset=dict(
            initial_response_n=len(initial_response), matched_fill_with_history_n=len(observed_subset),
            excluded_no_fill_history=sum(fill[r["sample_id"]]["history_visual_tokens"] == 0 for r in initial_response),
            excluded_metadata_mismatch=sum(r["sample_id"] in mismatches for r in initial_response),
            summary=summary(observed_subset, fill), paired=paired(observed_subset, fill),
            by_task={task: dict(summary=summary(group, fill), paired=paired(group, fill))
                     for task, group in per_task_subset.items()},
            confidence_bins=bins(observed_subset, fill, first=True),
            threshold_endpoint_substitution=thresholds(observed_subset, fill),
            thresholds_by_task={task: thresholds(group, fill) for task, group in per_task_subset.items()},
            discrimination=discrimination(observed_subset, fill),
            discrimination_by_task={task: discrimination(group, fill) for task, group in per_task_subset.items()},
            sample_ids=[r["sample_id"] for r in observed_subset]),
        final_silence_correct_examples=[example(r, fill) for r in sorted(
            (r for r in rows if state(r) == SILENCE and r["hit"]),
            key=lambda r: r["state_rounds"][-1]["predicted_probability"], reverse=True)[:12]],
        recent_response_fill_only_correct_examples=[example(r, fill) for r in sorted(
            (r for r in observed_subset if not r["hit"] and fill[r["sample_id"]]["hit"]),
            key=lambda r: p_response(r["state_rounds"][0]), reverse=True)[:12]],
    )
    (HERE / "state_signal.json").write_text(json.dumps(score, ensure_ascii=False, indent=2) + "\n")
    write_markdown(score)
    print(json.dumps(dict(overall=score["overall"], all_probes=score["all_probes"],
                          endpoint_subset=score["recent_response_endpoint_subset"]["paired"],
                          discrimination=score["recent_response_endpoint_subset"]["discrimination"],
                          recent_metadata_mismatch_n=len(mismatches)), ensure_ascii=False, indent=2))


def pct(value):
    return "—" if value is None else f"{100 * value:.1f}%"


def write_markdown(score):
    s = score["overall"]
    endpoint = score["recent_response_endpoint_subset"]
    lines = ["# V3 状态信号与历史收益审计", "",
             "本报告只读取两份已完成结果，不训练、不运行模型、不选择最优阈值。"
             "所有准确率均为样本 micro 准确率，不是 OVO 的宏平均总分。", "",
             f"共 {s['n']} 个配对样本；{score['all_probes']['n']} 次探测，状态分布为 "
             f"`{score['all_probes']['states']}`。首轮 Response {s['initial_states'].get(RESPONSE, 0)}，"
             f"末轮 Response {s['response_n']}。", "",
             "## 状态不等于同视图最终答案的可靠性", "",
             "只把末轮状态与相同视觉输入上的最终回答对齐。首轮 Silence 后可能加入历史，"
             "不能将加入历史后的正确答案用于证明首轮本来就足够。最终问答提示与状态提示不同，"
             "因此下面说明的是信号与答案行为不一致，不是对真正可回答性的完整标注。", "",
             "| 类别 | Response 正确/总数 | 正确率 | Silence 正确/总数 | 正确率 |",
             "| --- | ---: | ---: | ---: | ---: |"]
    for name, item in [("全部", s), *[(TASK_SHORT[t], x) for t, x in score["by_task"].items()]]:
        lines.append(f"| {name} | {item['response_correct']}/{item['response_n']} | {pct(item['response_accuracy'])} "
                     f"| {item['silence_correct']}/{item['silence_n']} | {pct(item['silence_accuracy'])} |")
    lines += ["", "## 固定置信度分箱：末轮 Response", "",
              "概率是全词表中的 P(Response)，不是回答正确概率，也未将三个状态重新归一化。", "",
              "| P(Response) 范围 | 样本数 | 平均概率 | 最终回答正确率 |",
              "| --- | ---: | ---: | ---: |"]
    for row in score["final_response_confidence_bins"]:
        lines.append(f"| [{row['low']:.2f}, {row['high']:.2f}{']' if row['high_inclusive'] else ')'} "
                     f"| {row['n']} | {pct(row['mean_p_response'])} | {pct(row['sample_micro_accuracy'])} |")
    lines += ["", "## Yes/No、负答案与计数的差异", "",
              "下面按真实答案事后分组；这些标签未输入控制器。Yes/No 题中包括询问现有画面"
              "是否足以回答另一个问题的题目；能确定答案为 No，本身也可以是完整回答。", "",
              "| 目标答案类型 | 样本数 | 首轮 Response | 末轮 Response 正确/总数 | 末轮 Silence 正确/总数 |",
              "| --- | ---: | ---: | ---: | ---: |"]
    for group, row in score["by_target_answer_kind"].items():
        lines.append(f"| {group} | {row['n']} | {row['initial_states'].get(RESPONSE, 0)} "
                     f"| {row['response_correct']}/{row['response_n']} "
                     f"| {row['silence_correct']}/{row['silence_n']} |")
    pair = endpoint["paired"]
    lines += ["", "## 只在可观测的 Recent-only Response 子集上检查提高阈值", "",
              f"首轮 Response 共 {endpoint['initial_response_n']} 题，其中 {endpoint['matched_fill_with_history_n']} "
              "题同时满足：V3 最终回答完全不读历史、原 fill 实际读入历史、题目与 recent 时间戳/尺寸/采样元数据相同。"
              f"{endpoint['excluded_no_fill_history']} 题因 fill 也没有历史而排除；"
              f"{endpoint['excluded_metadata_mismatch']} 题元数据不匹配。特征张量没有逐字节哈希，"
              "这里验证的是记录下来的输入元数据一致。", "",
              f"这 {pair['n']} 题中：两者都对 {pair['both_correct']}；fill 独有正确 {pair['fill_only_correct']}；"
              f"Recent-only 独有正确 {pair['state_only_correct']}；两者都错 {pair['both_wrong']}。", "",
              "固定阈值以下的题直接替换为已经记录的 fill 答案。这个可复算结果只属于上述子集和"
              "两端答案的替换，不是对新阈值渐进控制器的完整模拟，也不能外推为全量新运行分数。", "",
              "| 固定阈值 | 转用 fill 数量/占比 | 新增正确 | 丢失正确 | 净正确变化 | 保留 Recent 中错误/总数 | 其中 fill 能修复 |",
              "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in endpoint["threshold_endpoint_substitution"]:
        lines.append(f"| {row['threshold']:.2f} | {row['divert_to_recorded_fill_n']} / {pct(row['divert_fraction'])} "
                     f"| {row['diverted_fill_gains']} | {row['diverted_fill_losses']} | {row['net_observed_hit_change']:+d} "
                     f"| {row['retained_recent_wrong']}/{row['keep_recent_n']} | {row['retained_recent_wrong_fill_correct']} |")
    lines += ["", "固定阈值分任务净正确变化（仍然只在该子集内）：", "",
              "| 阈值 | BT | RT | FT |", "| --- | ---: | ---: | ---: |"]
    for index, threshold in enumerate(THRESHOLDS):
        values = {TASK_SHORT[task]: values[index]["net_observed_hit_change"]
                  for task, values in endpoint["thresholds_by_task"].items()}
        lines.append(f"| {threshold:.2f} | {values['BT']:+d} | {values['RT']:+d} | {values['FT']:+d} |")
    lines += ["", "概率的排序能力（AUC=0.5 为随机排序，数值仅作描述）：", "",
              "| 子集 | P(Response) 预测 Recent 答对 | −P(Response) 预测 fill 独有正确 | 两端一对一错时 −P(Response) 预测 fill 胜 |",
              "| --- | ---: | ---: | ---: |"]
    for name, row in [("全部", endpoint["discrimination"]),
                      *[(TASK_SHORT[t], x) for t, x in endpoint["discrimination_by_task"].items()]]:
        vals = [row[key] for key in ("p_response_auc_for_recent_answer_correct",
                                     "negative_p_response_auc_for_fill_only_correct",
                                     "negative_p_response_auc_for_fill_wins_on_discordant")]
        lines.append(f"| {name} | " + " | ".join("—" if v is None else f"{v:.3f}" for v in vals) + " |")
    lines += ["", "## 判断与限制", "",
              "提高阈值可以作为独立验证集上的后续对照，但这里不能把它当成已经得到的修复。"
              "高概率 Response 仍包含历史可修复的错误；负答案经常停留在 Silence，"
              "而计数题经常 Response 后仍答错。信号混合了任务语义、是否愿意回答和证据充分性，"
              "不是统一校准的历史需求分数。", "",
              "较高阈值在可观测子集上的收益伴随大比例退回 fill；必须同时检查 RT 历史干扰、"
              "BT/FT 缺失历史和调用成本。不得用本测试集选择阈值再将相同数据作为改进证据。", "",
              "未观测到的首轮非 Response 的 Recent-only 答案、中间预算的答案、以及早停后更大预算"
              "的状态均未估计。JSON 记录了固定分箱、各任务阈值表、精确样本集合、配对统计、"
              "典型样本 ID 与源文件 SHA-256，便于复核。", ""]
    (HERE / "state_signal.md").write_text("\n".join(lines))


if __name__ == "__main__":
    main()
