import os, json
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

from config import RESULT

d = pd.read_csv(os.path.join(RESULT, 'daily_features.csv'), parse_dates=['date'])
print('日表', d.shape, d['date'].min().date(), '→', d['date'].max().date())

EVENT = pd.Timestamp('2026-08-26')
CAND = [c for c in d.columns if c not in ('date', 'year', 'month', 'doy')]
CAND = [c for c in CAND if d[c].notna().sum() > 500]

# 事件窗口：8/23–8/26（事发前 72 小时）
win = d[(d['date'] >= EVENT - pd.Timedelta(days=3)) & (d['date'] <= EVENT)]
ctrl = d[(d['month'].isin([8, 9])) & (d['year'].between(2021, 2025))]
print('事件窗口 %d 天，对照 %d 天（2021-2025 年 8-9 月）' % (len(win), len(ctrl)))

rows = []
for c in CAND:
    x = ctrl[c].dropna()
    if len(x) < 50:
        continue
    mu, sd = x.mean(), x.std()
    xe = win[c].mean()
    if not np.isfinite(xe) or sd == 0:
        continue
    z = (xe - mu) / sd
    pct = (x < xe).mean() * 100
    rows.append({'feature': c, 'event_mean': xe, 'ctrl_mean': mu, 'ctrl_sd': sd,
                 'z': z, 'pct_rank': pct, 'abs_z': abs(z)})
cc = pd.DataFrame(rows).sort_values('abs_z', ascending=False)
cc.to_csv(os.path.join(RESULT, 'case_crossover_ranking.csv'), index=False, encoding='utf-8-sig')
print('\n【分析A】事件前 72 小时 vs 2021-2025 同期气候态（按 |z| 排序，前 15）')
print(cc.head(15)[['feature', 'event_mean', 'ctrl_mean', 'z', 'pct_rank']].to_string(index=False, float_format=lambda v: '%.2f' % v))
print('\n降水相关特征的位置：')
print(cc[cc['feature'].str.contains('Precip|precip')][['feature', 'z', 'pct_rank']].to_string(index=False, float_format=lambda v: '%.2f' % v))

# ---------------- 分析B：代理标签 ----------------
me = d[(d['month'].between(6, 9)) & (d['year'].between(2021, 2025))].copy()
me = me.dropna(subset=['PDD_30d'])
thr = me['PDD_30d'].quantile(0.95)
me['y'] = (me['PDD_30d'] >= thr).astype(int)
print('\n融季样本 %d 天，正样本 %d 天（PDD_30d ≥ %.1f）' % (len(me), me['y'].sum(), thr))

TEMP_LIKE = ['PDD_source', 'PDD_7d', 'PDD_30d', 'T_station_mean', 'T_station_max', 'T_station_min',
             'T_source_mean', 'T_source_max', 'T_source_min', 'T_port_mean', 'T_port_max', 'T_port_min',
             'T_source_mean_anom', 'T_source_anom_7d', 'T_port_mean_anom', 'T_station_mean_anom',
             'no_freeze_days', 'heatwave']
feats_all = [c for c in CAND if c not in ('y',) and me[c].notna().sum() > len(me) * 0.6]
feats_ind = [c for c in feats_all if c not in TEMP_LIKE]

def rank_table(feats, tag):
    out = []
    for c in feats:
        s = me[[c, 'y']].dropna()
        if s['y'].nunique() < 2 or len(s) < 100:
            continue
        auc = roc_auc_score(s['y'], s[c])
        auc = max(auc, 1 - auc)
        out.append({'feature': c, 'auc': auc, 'n': len(s)})
    t = pd.DataFrame(out).sort_values('auc', ascending=False).reset_index(drop=True)
    t['rank'] = t.index + 1
    t.to_csv(os.path.join(RESULT, f'proxy_label_ranking_{tag}.csv'), index=False, encoding='utf-8-sig')
    return t

t_all = rank_table(feats_all, 'all')
t_ind = rank_table(feats_ind, 'independent')
print('\n【分析B】代理标签（融季 PDD30 ≥ p95）的单特征 AUC —— 全部特征前 12')
print(t_all.head(12)[['rank', 'feature', 'auc']].to_string(index=False, float_format=lambda v: '%.3f' % v))
print('\n剔除温度类后的独立特征前 10')
print(t_ind.head(10)[['rank', 'feature', 'auc']].to_string(index=False, float_format=lambda v: '%.3f' % v))

# ---------------- bootstrap 稳定性（按 7 天块重采样） ----------------
rng = np.random.default_rng(42)
idx = np.arange(len(me))
blocks = [idx[i:i + 7] for i in range(0, len(idx), 7)]
counts = {c: 0 for c in feats_all}
NB = 200
for b in range(NB):
    pick = rng.integers(0, len(blocks), len(blocks))
    rows_b = np.concatenate([blocks[i] for i in pick])
    sub = me.iloc[rows_b]
    if sub['y'].sum() < 10:
        continue
    sc = []
    for c in feats_all:
        s = sub[[c, 'y']].dropna()
        if s['y'].nunique() < 2 or len(s) < 100:
            continue
        a = roc_auc_score(s['y'], s[c])
        sc.append((max(a, 1 - a), c))
    sc.sort(reverse=True)
    for _, c in sc[:5]:
        counts[c] += 1
stab = pd.DataFrame([{'feature': k, 'top5_freq': v / NB} for k, v in counts.items()]).sort_values('top5_freq', ascending=False)
stab.to_csv(os.path.join(RESULT, 'top5_stability.csv'), index=False, encoding='utf-8-sig')
print('\n【稳定性】7 天块 bootstrap（%d 次）Top-5 入选频率' % NB)
print(stab.head(12).to_string(index=False, float_format=lambda v: '%.2f' % v))

# ---------------- 多变量模型 ----------------
X = me[feats_ind].copy()
X = X.fillna(X.median())
y = me['y'].values
rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=5, random_state=0, n_jobs=-1, class_weight='balanced')
rf.fit(X, y)
imp = pd.DataFrame({'feature': feats_ind, 'rf_importance': rf.feature_importances_}).sort_values('rf_importance', ascending=False)
imp.to_csv(os.path.join(RESULT, 'rf_importance.csv'), index=False, encoding='utf-8-sig')
print('\n【多变量】随机森林（仅非温度特征）重要性前 10')
print(imp.head(10).to_string(index=False, float_format=lambda v: '%.4f' % v))
print('\n样本内 AUC = %.3f（仅作参考，未做时间外推验证）' % roc_auc_score(y, rf.predict_proba(X)[:, 1]))
