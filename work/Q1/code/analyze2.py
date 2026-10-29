import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from config import RESULT

d = pd.read_csv(os.path.join(RESULT, 'daily_features.csv'), parse_dates=['date'])
me = d[(d['month'].between(6, 9)) & (d['year'].between(2021, 2025))].copy().dropna(subset=['PDD_30d'])
thr = me['PDD_30d'].quantile(0.95)
me['y'] = (me['PDD_30d'] >= thr).astype(int)

EXCL = [c for c in d.columns if c.startswith(('T_station', 'T_source', 'T_port', 'PDD'))] + \
       ['date', 'year', 'month', 'doy', 'heatwave', 'no_freeze_days']
EXCL += [c for c in d.columns if c.endswith('_clim')]
feats = [c for c in d.columns if c not in EXCL and me[c].notna().sum() > len(me) * 0.6]

# 与源区温度的相关系数（判断是否只是温度的代理）
corr_t = {}
for c in feats:
    s = me[[c, 'T_source_mean']].dropna()
    corr_t[c] = s[c].corr(s['T_source_mean']) if len(s) > 50 else np.nan

rows = []
for c in feats:
    s = me[[c, 'y']].dropna()
    if s['y'].nunique() < 2 or len(s) < 100:
        continue
    a = roc_auc_score(s['y'], s[c])
    rows.append({'feature': c, 'auc': max(a, 1 - a), 'r_with_Tsource': corr_t.get(c, np.nan)})
t = pd.DataFrame(rows).sort_values('auc', ascending=False).reset_index(drop=True)

# 贪心去共线：|r|>0.7 归为同一簇，保留 AUC 最高者
kept, clusters = [], []
for _, r in t.iterrows():
    f = r['feature']
    s = me[[f] + kept].corr().iloc[0, 1:] if kept else pd.Series(dtype=float)
    if len(kept) and (s.abs() > 0.7).any():
        continue
    kept.append(f)
    clusters.append(f)
t['cluster_rep'] = t['feature'].isin(kept)

print('非温度候选特征 %d 个，去共线后保留 %d 个' % (len(t), len(kept)))
print('\n=== 去共线后的独立特征排序（AUC，融季 PDD30≥p95 为标签）===')
print(t[t['cluster_rep']].head(12)[['feature', 'auc', 'r_with_Tsource']].to_string(index=False, float_format=lambda v: '%.3f' % v))

print('\n=== 被剔除的高相关问题（示例）===')
print(t[~t['cluster_rep']].head(8)[['feature', 'auc', 'r_with_Tsource']].to_string(index=False, float_format=lambda v: '%.3f' % v))

# 事件窗口的 z 值（与前面一致）
EVENT = pd.Timestamp('2026-08-26')
win = d[(d['date'] >= EVENT - pd.Timedelta(days=3)) & (d['date'] <= EVENT)]
ctrl = d[(d['month'].isin([8, 9])) & (d['year'].between(2021, 2025))]
zs = {}
for c in t['feature']:
    x = ctrl[c].dropna()
    if len(x) > 50 and x.std() > 0:
        zs[c] = (win[c].mean() - x.mean()) / x.std()
t['event_z'] = t['feature'].map(zs)
t.to_csv(os.path.join(RESULT, 'feature_final_table.csv'), index=False, encoding='utf-8-sig')

print('\n=== 推荐 Top-5（综合：代理标签 AUC + 事件窗口距平 + 物理可解释性）===')
top = t[t['cluster_rep']].head(6)[['feature', 'auc', 'event_z']]
print(top.to_string(index=False, float_format=lambda v: '%.2f' % v))

# 特征簇结构
print('\n=== 高相关簇（|r|>0.7）===')
corr = me[feats].corr().abs()
used = set()
for c in t['feature'].head(20):
    if c in used:
        continue
    grp = [o for o in t['feature'].head(20) if o != c and corr.loc[c, o] > 0.7]
    used.add(c); used.update(grp)
    if grp:
        print('  %-26s ← %s' % (c, ', '.join(grp)))
