"""Post-hoc diagnostic only: no policy/threshold tuning and no model changes."""
from collections import Counter, defaultdict
import hashlib,json
from pathlib import Path
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCES={
 'state':ROOT/'results/ovo-state-memory-v3-2gpu-20261012/predictions.jsonl',
 'fill':ROOT.parent/'video2/results/ovo-visual-memory-2gpu-20261011/predictions.jsonl',
 'official_recent16':Path('/mnt/truenas_nfs/dataset/znh1302/OneStreamer-OVO-eval/outputs/OneStreamer-4B/ovobench/ovo-full-recent16-2gpu-20261012/recent16/ovobench/predictions.json')}
rows={name:([json.loads(l) for l in p.read_text().splitlines()] if p.suffix=='.jsonl' else json.loads(p.read_text())) for name,p in SOURCES.items()}
lookup={name:{r['sample_id']:r for r in data} for name,data in rows.items()}
assert all(len(d)==3035 for d in lookup.values())
fields=('sample_id','task','subtask','video','start','end','question','answer','answer_text','options','eval_mode')
for sid,s in lookup['state'].items():
 for name in ['fill','official_recent16']:
  assert all(s[k]==lookup[name][sid][k] for k in fields),(sid,name)
def signatures(items):
 return dict(n=len(items),videos=len({r['video'] for r in items}),video_question_groups=len({(r['video'],r['start'],r['question']) for r in items}))
def numeric_error(r):
 try:a,b=int(r['parsed_output']),int(r['answer'])
 except (ValueError,TypeError):return 'unparsed'
 return 'under' if a<b else 'over' if a>b else 'equal'
rec=[r for r in rows['state'] if r['subtask']=='REC']
rec_loss=[r for r in rec if lookup['fill'][r['sample_id']]['hit'] and not r['hit']]
rec_loss_details=[]
for r in rec_loss:
 f=lookup['fill'][r['sample_id']]
 rec_loss_details.append(dict(sample_id=r['sample_id'],error=numeric_error(r),state_count=r['parsed_output'],fill_count=f['parsed_output'],gold=r['answer'],stop=r['state_stop_reason'],history_frames=r['history_read_frames'],response_probability=r['state_rounds'][-1]['state_probabilities']['</Response>'],bank_dropped=r['bank']['dropped_oldest'],fill_history_frames=f['history_read_frames']))
monotonic={}
for method in ['state','fill','official_recent16']:
 groups=defaultdict(list)
 for r in rows[method]:
  if r['subtask']=='REC':groups[(r['video'],r['start'],r['question'])].append(r)
 violations=[];transitions=0;gold_violations=[]
 for key,items in groups.items():
  items=sorted(items,key=lambda r:r['end'])
  for a,b in zip(items,items[1:]):
   if b['end']<=a['end']:continue
   transitions+=1
   try:av,bv=int(a['parsed_output']),int(b['parsed_output'])
   except (ValueError,TypeError):continue
   if int(b['answer'])<int(a['answer']):gold_violations.append([a['sample_id'],b['sample_id']])
   if bv<av:violations.append(dict(earlier_id=a['sample_id'],later_id=b['sample_id'],earlier_count=av,later_count=bv,earlier_end=a['end'],later_end=b['end'],earlier_gold=a['answer'],later_gold=b['answer']))
 monotonic[method]=dict(groups=len(groups),transitions=transitions,decreasing_predictions=len(violations),decreasing_gold=len(gold_violations),examples=violations[:12])
rt=[r for r in rows['state'] if r['task']=='real_time_visual_perception']
rt_reg=[r for r in rt if lookup['official_recent16'][r['sample_id']]['hit'] and not r['hit']]
rt_gain=[r for r in rt if r['hit'] and not lookup['official_recent16'][r['sample_id']]['hit']]
rt_groups={}
for name,data in [('recent16_correct_state_wrong',rt_reg),('state_correct_recent16_wrong',rt_gain)]:
 rt_groups[name]=dict(n=len(data),state_no_history=sum(r['history_read_frames']==0 for r in data),state_with_history=sum(r['history_read_frames']>0 for r in data),same_recent_timestamps=sum(r['recent_timestamps']==lookup['official_recent16'][r['sample_id']]['recent_timestamps'] for r in data),ids=[r['sample_id'] for r in data])
case_ids=['ovo-2353','ovo-2354','ovo-0598','ovo-0044','ovo-0656','ovo-0732','ovo-1691','ovo-1894']
cases=[]
for sid in case_ids:
 r=lookup['state'][sid];f=lookup['fill'][sid];z=lookup['official_recent16'][sid]
 cases.append(dict(sample_id=sid,subtask=r['subtask'],question=r['question'],options=r['options'],gold=r['answer'],gold_text=r['answer_text'],state=r['prediction'],fill=f['prediction'],official_recent16=z['prediction'],end=r['end'],recent_range=[r['recent_timestamps'][0],r['recent_timestamps'][-1]],state_history_range=[min(r['history_timestamps']),max(r['history_timestamps'])] if r['history_timestamps'] else [],fill_history_range=[min(f['history_timestamps']),max(f['history_timestamps'])] if f['history_timestamps'] else [],state_history_frames=r['history_read_frames'],fill_history_frames=f['history_read_frames'],bank_dropped=r['bank']['dropped_oldest'],probes=[{k:p[k] for k in ('state','history_budget_tokens','history_read_frames','state_probabilities')} for p in r['state_rounds']]))
report=dict(provenance={name:dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for name,p in SOURCES.items()},rec_all=dict(signatures(rec),errors=dict(Counter(numeric_error(r) for r in rec))),rec_regressions=dict(signatures(rec_loss),errors=dict(Counter(numeric_error(r) for r in rec_loss)),recent_only=sum(r['history_read_frames']==0 for r in rec_loss),probability_over_09=sum(r['state_rounds'][-1]['state_probabilities']['</Response>']>=.9 for r in rec_loss),bank_dropped_any=sum(r['bank']['dropped_oldest']>0 for r in rec_loss),details=rec_loss_details),cumulative_count_monotonicity=monotonic,rt_official_recent_comparison=rt_groups,cases=cases,caveats=['Official Recent16 differs in temporal sampling, prompt and feature path: comparisons cannot identify history causality.', 'No same-video/start/question transitions exist in these chunk-path records, so count monotonicity cannot be evaluated without original source-video mapping.', 'Unique chunk paths do not establish independent source videos; no independent-sample significance claims.', 'Case descriptions use stored questions/predictions/timestamps, not manual video-event verification.'])
(HERE/'case_analysis.json').write_text(json.dumps(report,indent=2,ensure_ascii=False)+'\n')
print(json.dumps({k:v for k,v in report.items() if k not in ['provenance','cases','rec_regressions','rt_official_recent_comparison']},ensure_ascii=False,indent=2))
print('REC regressions',{k:v for k,v in report['rec_regressions'].items() if k!='details'})
print('RT comparison',{k:{f:v for f,v in a.items() if f!='ids'} for k,a in rt_groups.items()})
