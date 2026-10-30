"""Q1 公共模块：事件窗、对照期、代理标签、距平、AUC、去共线。

原先这些逻辑在 1main analyze.py / 2temp rank...py / 3borda rank.py 里各写了一遍，
任何一处改动都会让三个脚本给出不同口径的结果。现在统一收敛到这里。

依赖：numpy、pandas、scikit-learn（仅 AUC）
"""
import os

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

import config as C

# 不是特征、不能进入排序的列
NON_FEATURES = ('date', 'year', 'month', 'doy', 'hours_valid_Ta')


# ---------------- 时间口径 ----------------
def event_timestamp():
    """事件发生的完整时刻（含时分）。"""
    return pd.Timestamp('%s %s' % (C.EVENT_DATE, C.EVENT_TIME))


def event_window(d):
    """事件前窗口：默认 08-23 ~ 08-25（不含事件当天）。"""
    start = pd.Timestamp(C.EVENT_DATE) + pd.Timedelta(days=C.WIN_START_DAYS)
    end = pd.Timestamp(C.EVENT_DATE) + pd.Timedelta(days=C.WIN_END_DAYS)
    return d[(d['date'] >= start) & (d['date'] <= end)]


def control_period(d):
    """对照期：2021-2025 年 8-9 月（事件年不参与）。"""
    return d[(d['month'].isin(C.CTRL_MONTHS)) & (d['year'].between(*C.CTRL_YEARS))]


def melt_season(d):
    """融季：2021-2025 年 6-9 月。"""
    return d[(d['month'].between(*C.MELT_MONTHS)) & (d['year'].between(*C.CTRL_YEARS))].copy()


# ---------------- 数据 ----------------
def load_daily():
    """读取日尺度主表。"""
    return pd.read_csv(os.path.join(C.RESULT, 'daily_features.csv'), parse_dates=['date'])


def candidate_features(d, min_obs=500):
    """可作为特征的列：排除时间列/质量列/气候态代理，且非空样本数足够。

    气候态列（*_clim）是"日序的平滑函数"，本质是日历代理而不是观测变量；
    若把它当气象特征参与排序，会得到"7 月底热，所以 7 月底热"的同义反复，
    因此一律排除（距平 *_anom 保留了真实观测信息，可以保留）。
    """
    feats = [c for c in d.columns if c not in NON_FEATURES and not c.endswith('_clim')]
    return [c for c in feats if d[c].notna().sum() > min_obs]


# ---------------- 代理标签 ----------------
def build_proxy_label(d, quantile=None):
    """以"融季 PDD30 >= p95"构造代理标签，返回 (样本表, 阈值)。

    标签由 PDD 定义，因此 PDD 自身不能作为特征（循环性）。
    """
    q = C.PROXY_QUANTILE if quantile is None else quantile
    me = melt_season(d).dropna(subset=['PDD_30d'])
    thr = me['PDD_30d'].quantile(q)
    me['y'] = (me['PDD_30d'] >= thr).astype(int)
    return me, thr


# ---------------- 距平（分析 A） ----------------
def anomaly(d, feature, win=None, ctrl=None):
    """单特征的事件窗标准化距平；样本不足或无法计算时返回 None。"""
    win = event_window(d) if win is None else win
    ctrl = control_period(d) if ctrl is None else ctrl
    x = ctrl[feature].dropna()
    xe = win[feature].mean()
    if len(x) < 50 or not np.isfinite(xe) or x.std() == 0:
        return None
    return {
        'feature': feature,
        'event_mean': xe,
        'ctrl_mean': x.mean(),
        'ctrl_sd': x.std(),
        'z': (xe - x.mean()) / x.std(),
        'pct_rank': (x < xe).mean() * 100,
    }


def z_table(d, features=None):
    """全部候选特征的事件距平表，按 |z| 降序。"""
    win, ctrl = event_window(d), control_period(d)
    features = candidate_features(d) if features is None else features
    rows = [r for r in (anomaly(d, c, win, ctrl) for c in features) if r is not None]
    t = pd.DataFrame(rows)
    t['abs_z'] = t['z'].abs()
    return t.sort_values('abs_z', ascending=False).reset_index(drop=True)


# ---------------- 单特征 AUC（分析 B） ----------------
def auc_table(me, features, min_n=100):
    """以代理标签 y 计算单特征 AUC（取 max(auc, 1-auc) 去掉方向）。"""
    rows = []
    for c in features:
        s = me[[c, 'y']].dropna()
        if s['y'].nunique() < 2 or len(s) < min_n:
            continue
        a = roc_auc_score(s['y'], s[c])
        rows.append({'feature': c, 'auc': max(a, 1.0 - a), 'n': len(s)})
    t = pd.DataFrame(rows).sort_values('auc', ascending=False).reset_index(drop=True)
    t['rank'] = t.index + 1
    return t


# ---------------- 去共线 ----------------
def dedupe_collinear(me, ordered, thr=None):
    """按给定顺序贪心去共线：与已保留特征 |r| > thr 的跳过。"""
    thr = C.COLLINEAR_R if thr is None else thr
    kept = []
    for f in ordered:
        if kept:
            corr = me[[f] + kept].corr().iloc[0, 1:]
            if (corr.abs() > thr).any():
                continue
        kept.append(f)
    return kept
