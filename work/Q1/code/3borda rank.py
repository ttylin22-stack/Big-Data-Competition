import os
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier

from config import RESULT

d = pd.read_csv(os.path.join(RESULT, 'daily_features.csv'), parse_dates=['date'])
me = d[(d['month'].between(6, 9)) & (d['year'].between(2021, 2025))].copy().dropna(subset=['PDD_30d'])
thr = me['PDD_30d'].quantile(0.95)
me['y'] = (me['PDD_30d'] >= thr).astype(int)

POOL = [c for c in d.columns if c not in ('date', 'year', 'month', 'doy', 'y')]
POOL = [c for c in POOL if not c.startswith('PDD')]          # 标签由 PDD 定义，避免循环
POOL = [c for c in POOL if me[c].notna().sum() > len(me) * 0.6 and me[c].std() > 0]

# 准则1：代理标签 AUC
auc = {}
for c in POOL:
    s = me[[c, 'y']].dropna()
    if s['y'].nunique() < 2:
        continue
    a = roc_auc_score(s['y'], s[c])
    auc[c] = max(a, 1 - a)

# 准则2：事件窗口距平
EVENT = pd.Timestamp('2026-08-26')
win = d[(d['date'] >= EVENT - pd.Timedelta(days=3)) & (d['date'] <= EVENT)]
ctrl = d[(d['month'].isin([8, 9])) & (d['year'].between(2021, 2025))]
z = {}
for c in POOL:
    x = ctrl[c].dropna()
    if len(x) > 50 and x.std() > 0 and np.isfinite(win[c].mean()):
        z[c] = abs((win[c].mean() - x.mean()) / x.std())

# 准则3：随机森林重要性
X = me[POOL].copy().fillna(me[POOL].median())
rf = RandomForestClassifier(n_estimators=500, min_samples_leaf=5, random_state=0,
                            n_jobs=-1, class_weight='balanced')
rf.fit(X, me['y'].values)
imp = pd.Series(rf.feature_importances_, index=POOL)

tab = pd.DataFrame({'auc': pd.Series(auc), 'abs_z': pd.Series(z), 'rf': imp}).dropna()
for col in ['auc', 'abs_z', 'rf']:
    tab[f'r_{col}'] = tab[col].rank(ascending=False)
tab['borda'] = tab[['r_auc', 'r_abs_z', 'r_rf']].mean(axis=1)
tab = tab.sort_values('borda').reset_index().rename(columns={'index': 'feature'})

# 去共线（|r|>0.7 保留 Borda 更优者）
corr = me[POOL].corr().abs()
keep = []
for f in tab['feature']:
    if any(corr.loc[f, k] > 0.7 for k in keep):
        continue
    keep.append(f)
tab['selected'] = tab['feature'].isin(keep)
tab.to_csv(os.path.join(RESULT, 'borda_ranking.csv'), index=False, encoding='utf-8-sig')

print('候选 %d 个 → 去共线后 %d 个' % (len(tab), len(keep)))
print('\n=== Borda 综合排序（AUC + 事件距平 + 随机森林重要性）前 15 ===')
print(tab.head(15)[['feature', 'auc', 'abs_z', 'rf', 'borda', 'selected']].to_string(
    index=False, float_format=lambda v: '%.3f' % v))

print('\n=== 最终 Top-5（去共线后）===')
print(tab[tab['selected']].head(5)[['feature', 'auc', 'abs_z', 'rf', 'borda']].to_string(
    index=False, float_format=lambda v: '%.3f' % v))

print('\n=== 降水类特征的位置 ===')
sub = tab[tab['feature'].str.contains('Precip|precip', case=False)]
print(sub[['feature', 'auc', 'abs_z', 'borda']].to_string(index=False, float_format=lambda v: '%.3f' % v))
print('全部 %d 个特征中，降水的 Borda 名次为 %s' % (len(tab), list(sub.index + 1)))
