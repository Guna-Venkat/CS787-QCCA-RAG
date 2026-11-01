"""Retrospective label-format diagnostic; never replaces frozen strict scores."""
import re
from pathlib import Path
import pandas as pd

LEADING_LABEL = re.compile(r'^\s*(SUPPORT|REFUTE|NOT_ENOUGH_INFO)(?=$|[\s.:;,\-])', re.I)


def diagnose(rows, directory):
    records=[]
    for row in rows:
        if row['dataset']!='SciFact':continue
        prediction=row['prediction'].strip()
        match=LEADING_LABEL.match(prediction)
        label=match.group(1).upper() if match else None
        records.append({k:row[k] for k in ['dataset','reader','action_id','query_id','evidence_granularity','selector','budget','compressor']} | dict(
            strict_score=row['task_score'],strict_format_valid=prediction.upper() in ('SUPPORT','REFUTE','NOT_ENOUGH_INFO'),
            leading_label=label,leading_label_available=label is not None,
            leading_label_diagnostic_score=float(label==row['gold']),
            generation_cap_reached=row['generated_tokens']>=128))
    if not records:return pd.DataFrame()
    frame=pd.DataFrame(records)
    summary=frame.groupby(['dataset','reader','action_id','evidence_granularity','selector','budget','compressor'],dropna=False).agg(
        n_queries=('query_id','size'),strict_accuracy=('strict_score','mean'),strict_format_rate=('strict_format_valid','mean'),
        leading_label_rate=('leading_label_available','mean'),leading_label_diagnostic_accuracy=('leading_label_diagnostic_score','mean'),
        cap_hit_rate=('generation_cap_reached','mean')).reset_index()
    directory=Path(directory)
    summary.to_csv(directory/'label_compliance_diagnostic_summary.csv',index=False)
    frame.to_csv(directory/'label_diagnostic_per_query_df.csv',index=False)
    return summary
