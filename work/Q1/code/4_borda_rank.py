import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

import common as K
import config as C

d = K.load_daily()
me, thr = K.build_proxy_label(d)

# 候选池：剔除由标签定义的 PDD 系列（避免循环），保留温度类以体现"谁更能解释热力"
POOL = [c for c in K.candidate_features(d) if not c.startswith('PDD')]
POOL = [c for c in POOL if me[c].notna().sum() > len(me) * 0.6 and me[c].std() > 0]

# ---------------- 准则1：代理标签 AUC ----------------
t_auc = K.auc_table(me, POOL)
auc = dict(zip(t_auc['feature'], t_auc['auc']))

# ---------------- 准则2：事件窗距平 ----------------
win, ctrl = K.event_window(d), K.control_period(d)
z = {}
for c in POOL:
    r = K.anomaly(d, c, win, ctrl)
    if r:
        z[c] = abs(r['z'])

# ---------------- 准则3：随机森林重要性 ----------------
X = me[POOL].copy().fillna(me[POOL].median())
rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=5, random_state=0,
                            n_jobs=1, class_weight='balanced')
rf.fit(X, me['y'].values)
imp = pd.Series(rf.feature_importances_, index=POOL)

# ---------------- 合成 ----------------
tab = pd.DataFrame({'auc': pd.Series(auc), 'abs_z': pd.Series(z), 'rf': imp}).dropna()
for col in ['auc', 'abs_z', 'rf']:
    tab['r_%s' % col] = tab[col].rank(ascending=False)
tab['borda'] = tab[['r_auc', 'r_abs_z', 'r_rf']].mean(axis=1)
tab = tab.sort_values('borda').reset_index().rename(columns={'index': 'feature'})

# 去共线：按 Borda 顺序，与已保留特征 |r|>0.7 的丢弃（保留 Borda 更优者）
keep = K.dedupe_collinear(me, tab['feature'])
tab['selected'] = tab['feature'].isin(keep)
tab.to_csv(os.path.join(C.RESULT, 'borda_ranking.csv'), index=False, encoding='utf-8-sig')

print('候选 %d 个 → 去共线后 %d 个' % (len(tab), len(keep)))
print('\n=== Borda 综合排序（AUC + 事件距平 + 随机森林重要性）前 15 ===')
print(tab.head(15)[['feature', 'auc', 'abs_z', 'rf', 'borda', 'selected']]
      .to_string(index=False, float_format=lambda v: '%.3f' % v))

print('\n=== 最终 Top-5（去共线后）===')
print(tab[tab['selected']].head(5)[['feature', 'auc', 'abs_z', 'rf', 'borda']]
      .to_string(index=False, float_format=lambda v: '%.3f' % v))

print('\n=== 降水类特征的位置 ===')
sub = tab[tab['feature'].str.contains('Precip|precip', case=False)]
print(sub[['feature', 'auc', 'abs_z', 'borda']].to_string(index=False, float_format=lambda v: '%.3f' % v))
print('全部 %d 个特征中，降水的 Borda 名次为 %s' % (len(tab), list(sub.index + 1)))
