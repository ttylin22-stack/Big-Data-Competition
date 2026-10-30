import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

import common as K
import config as C

d = K.load_daily()
win, ctrl = K.event_window(d), K.control_period(d)
print('日表', d.shape, d['date'].min().date(), '→', d['date'].max().date())
print('事件窗 %d 天（%s ~ %s），对照 %d 天（2021-2025 年 8-9 月）'
      % (len(win), win['date'].min().date(), win['date'].max().date(), len(ctrl)))

# ---------------- 分析 A：事件窗距平 ----------------
cc = K.z_table(d)
cc.to_csv(os.path.join(C.RESULT, 'case_crossover_ranking.csv'), index=False, encoding='utf-8-sig')
print('\n【分析A】事件前 3 天 vs 2021-2025 同期气候态（按 |z| 排序，前 15）')
print(cc.head(15)[['feature', 'event_mean', 'ctrl_mean', 'z', 'pct_rank']]
      .to_string(index=False, float_format=lambda v: '%.2f' % v))
print('\n降水相关特征的位置：')
print(cc[cc['feature'].str.contains('Precip|precip')][['feature', 'z', 'pct_rank']]
      .to_string(index=False, float_format=lambda v: '%.2f' % v))

# ---------------- 分析 B：代理标签 ----------------
me, thr = K.build_proxy_label(d)
print('\n融季样本 %d 天，正样本 %d 天（PDD_30d >= %.1f）' % (len(me), me['y'].sum(), thr))

TEMP_LIKE = ['PDD_source', 'PDD_7d', 'PDD_30d',
             'T_station_mean', 'T_station_max', 'T_station_min',
             'T_source_mean', 'T_source_max', 'T_source_min',
             'T_port_mean', 'T_port_max', 'T_port_min',
             'T_source_mean_anom', 'T_source_anom_7d', 'T_port_mean_anom', 'T_station_mean_anom',
             'no_freeze_days_flag', 'heatwave']
CAND = K.candidate_features(d)
feats_all = [c for c in CAND if me[c].notna().sum() > len(me) * 0.6]
feats_ind = [c for c in feats_all if c not in TEMP_LIKE]

t_all = K.auc_table(me, feats_all)
t_ind = K.auc_table(me, feats_ind)
t_all.to_csv(os.path.join(C.RESULT, 'proxy_label_ranking_all.csv'), index=False, encoding='utf-8-sig')
t_ind.to_csv(os.path.join(C.RESULT, 'proxy_label_ranking_independent.csv'), index=False, encoding='utf-8-sig')
print('\n【分析B】代理标签（融季 PDD30 >= p95）的单特征 AUC —— 全部特征前 12')
print(t_all.head(12)[['rank', 'feature', 'auc']].to_string(index=False, float_format=lambda v: '%.3f' % v))
print('\n剔除温度类后的独立特征前 10')
print(t_ind.head(10)[['rank', 'feature', 'auc']].to_string(index=False, float_format=lambda v: '%.3f' % v))

# ---------------- 稳定性：7 天块 bootstrap ----------------
# 天气有自相关，按天重采样会高估独立性，因此按 7 天一块整体抽。
rng = np.random.default_rng(42)
idx = np.arange(len(me))
blocks = [idx[i:i + 7] for i in range(0, len(idx), 7)]
counts = {c: 0 for c in feats_all}
NB = 200
n_used = 0
for b in range(NB):
    pick = rng.integers(0, len(blocks), len(blocks))
    rows_b = np.concatenate([blocks[i] for i in pick])
    sub = me.iloc[rows_b]
    if sub['y'].sum() < 10:
        continue
    n_used += 1
    t = K.auc_table(sub, feats_all)
    for f in t['feature'].head(5):
        counts[f] += 1
stab = (pd.DataFrame([{'feature': k, 'top5_freq': v / max(n_used, 1)} for k, v in counts.items()])
        .sort_values('top5_freq', ascending=False))
stab.to_csv(os.path.join(C.RESULT, 'top5_stability.csv'), index=False, encoding='utf-8-sig')
print('\n【稳定性】7 天块 bootstrap（%d 次，有效 %d 次）Top-5 入选频率' % (NB, n_used))
print(stab.head(12).to_string(index=False, float_format=lambda v: '%.2f' % v))

# ---------------- 多变量：随机森林 ----------------
X = me[feats_ind].copy().fillna(me[feats_ind].median())
y = me['y'].values
# n_jobs=1：随机森林在受限环境（无命名管道权限）下用多进程会崩，单线程也能在数秒内跑完
rf = RandomForestClassifier(n_estimators=400, min_samples_leaf=5, random_state=0,
                            n_jobs=1, class_weight='balanced')
rf.fit(X, y)
imp = (pd.DataFrame({'feature': feats_ind, 'rf_importance': rf.feature_importances_})
       .sort_values('rf_importance', ascending=False))
imp.to_csv(os.path.join(C.RESULT, 'rf_importance.csv'), index=False, encoding='utf-8-sig')
print('\n【多变量】随机森林（仅非温度特征）重要性前 10')
print(imp.head(10).to_string(index=False, float_format=lambda v: '%.4f' % v))
print('\n样本内 AUC = %.3f（仅作参考，未做时间外推验证）'
      % roc_auc_score(y, rf.predict_proba(X)[:, 1]))
