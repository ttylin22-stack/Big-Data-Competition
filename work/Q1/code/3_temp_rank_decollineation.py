"""3_temp_rank_decollineation.py —— 剔除温度类后的独立特征榜单 + 贪心去共线。

目的：证明"非温度特征里，哪些仍能区分热天"。若某特征与源区温度高度相关，
说明它只是温度的另一种写法，不能算独立证据。

输出：result/feature_final_table.csv
"""
import os

import numpy as np
import pandas as pd

import common as K
import config as C

d = K.load_daily()
me, thr = K.build_proxy_label(d)

# 候选池：排除温度类、时间/质量列、气候态列
EXCL = [c for c in d.columns if c.startswith(('T_station', 'T_source', 'T_port', 'PDD'))]
EXCL += list(K.NON_FEATURES) + ['heatwave', 'no_freeze_days_flag']
EXCL += [c for c in d.columns if c.endswith('_clim')]
feats = [c for c in d.columns if c not in EXCL and me[c].notna().sum() > len(me) * 0.6]

# 与源区温度的相关系数（判断是否只是温度的代理）
corr_t = {}
for c in feats:
    s = me[[c, 'T_source_mean']].dropna()
    corr_t[c] = s[c].corr(s['T_source_mean']) if len(s) > 50 else np.nan

t = K.auc_table(me, feats)
t['r_with_Tsource'] = t['feature'].map(corr_t)

# 贪心去共线：按 AUC 降序，与已保留特征 |r|>0.7 的归入同一簇并丢弃
kept = K.dedupe_collinear(me, t['feature'])
t['cluster_rep'] = t['feature'].isin(kept)

print('非温度候选特征 %d 个，去共线后保留 %d 个' % (len(t), len(kept)))
print('\n=== 去共线后的独立特征排序（AUC，融季 PDD30>=p95 为标签）===')
print(t[t['cluster_rep']].head(12)[['feature', 'auc', 'r_with_Tsource']]
      .to_string(index=False, float_format=lambda v: '%.3f' % v))
print('\n=== 被剔除的高相关问题（示例）===')
print(t[~t['cluster_rep']].head(8)[['feature', 'auc', 'r_with_Tsource']]
      .to_string(index=False, float_format=lambda v: '%.3f' % v))

# 回填事件窗 z（与 1_main_analyze 使用同一窗口）
win, ctrl = K.event_window(d), K.control_period(d)
zs = {}
for c in t['feature']:
    r = K.anomaly(d, c, win, ctrl)
    if r:
        zs[c] = r['z']
t['event_z'] = t['feature'].map(zs)
t.to_csv(os.path.join(C.RESULT, 'feature_final_table.csv'), index=False, encoding='utf-8-sig')

print('\n=== 去共线后的前 6 名（AUC + 事件距平）===')
print(t[t['cluster_rep']].head(6)[['feature', 'auc', 'event_z']]
      .to_string(index=False, float_format=lambda v: '%.2f' % v))

# 特征簇结构：谁和谁其实是同一个量
print('\n=== 高相关簇（|r|>0.7）===')
corr = me[feats].corr().abs()
used = set()
for c in t['feature'].head(20):
    if c in used:
        continue
    grp = [o for o in t['feature'].head(20) if o != c and corr.loc[c, o] > C.COLLINEAR_R]
    used.add(c)
    used.update(grp)
    if grp:
        print('  %-26s <- %s' % (c, ', '.join(grp)))
