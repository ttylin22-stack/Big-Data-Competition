# -*- coding: utf-8 -*-
"""问题2（一）：历史灾点地形分位区间 + 触发类型构成 + DEM 空间易发性

本文件对应复核意见的修订点：
  2-2  三个"地形阈值"（沟比降 >0.17 等）实际是灾点自身的 25% 分位数，
       没有非灾点对照，不能叫临界阈值。改名"历史灾点地形分位区间"，
       输出改名 p2_terrain_quantiles.csv，并在输出里明确声明这一点；
       规模判别的 logit（沟比降负系数）提为主结果。
  2-8  把 `类型` 字段（降雨型/冰水混合型/冰雪融水型/冰湖溃决型）落盘为
       p2_trigger_type.csv，供"冰水混合机制"论证直接引用。
  2-4  空间块 CV 显式报告"5 块划分、有效 N 折"，并落盘每折样本数与 AUC。

运行环境：numpy / pandas 必需；geopandas / rasterio / sklearn 缺失时，
对应小节会跳过并提示，属性类结果（分位区间、触发类型）仍可产出。
"""
import os
import struct

import numpy as np
import pandas as pd

from config import RESOURCES, INPUT, RESULT, IMAGE

PT_DIR = 'DataSet_6'
PT_SHP = '西藏日喀则市吉隆县泥石流灾害区历史灾点数据.shp'
PT_DBF = '西藏日喀则市吉隆县泥石流灾害区历史灾点数据.dbf'
DEM_DIR = 'DataSet_7'

NUM_COLS = ['沟比降', '高差', '主沟长', '扇宽', '扇长', '沟口H', '高程Max', 'Area', 'DisFrmFlt']
# 与灾害"冰/雪"相关的触发类型（用于论证冰水混合机制）
ICE_TYPES = ('冰水混合型', '冰雪融水型', '冰湖溃决型')


# =========================================================
# 1. 读取灾点属性（优先 geopandas，缺失时退化为 DBF 解析）
# =========================================================
def read_dbf(path):
    """极简 dBase III 解析：只取字段值与记录，够读属性表。"""
    with open(path, 'rb') as f:
        raw = f.read()
    nrec, hlen, rlen = struct.unpack('<I H H', raw[4:12])
    nfield = (hlen - 33) // 32
    fields = []
    for i in range(nfield):
        off = 32 + 32 * i
        name = raw[off:off + 11].split(b'\x00')[0].decode('gbk', 'replace')
        flen = raw[off + 16]
        fields.append((name, flen))

    for enc in ('gbk', 'utf-8'):
        try:
            rows = []
            for r in range(nrec):
                s = hlen + r * rlen
                rec = raw[s:s + rlen]
                if not rec or rec[0:1] == b'*':
                    continue
                pos, row = 1, {}
                for name, flen in fields:
                    row[name] = rec[pos:pos + flen].decode(enc, 'replace').strip()
                    pos += flen
                rows.append(row)
            return pd.DataFrame(rows)
        except UnicodeDecodeError:
            continue
    raise RuntimeError('DBF 解码失败：%s' % path)


def load_points(resources=RESOURCES):
    """返回带 x/y 的属性表；两种读法数值一致，只影响是否携带几何列。"""
    shp = os.path.join(resources, PT_DIR, PT_SHP)
    dbf = os.path.join(resources, PT_DIR, PT_DBF)
    try:
        import geopandas as gpd
        pts = gpd.read_file(shp).set_crs(4490, allow_override=True).to_crs(4326)
        print('灾点来源：geopandas（含几何）')
    except ImportError:
        pts = read_dbf(dbf)
        print('灾点来源：DBF 解析（未安装 geopandas）')

    pts = pd.DataFrame(pts)
    pts['x'] = [float(v) for v in pts['x_coord']]
    pts['y'] = [float(v) for v in pts['y_coord']]
    for c in NUM_COLS:
        pts[c] = pd.to_numeric(pts[c], errors='coerce')
    return pts


# =========================================================
# 2. 地形分位区间（原"临界阈值"）
# =========================================================
def quantile_table(pts):
    q = {}
    for name, col in [('沟比降', '沟比降'), ('高差_m', '高差'), ('主沟长_m', '主沟长')]:
        q[name] = {'p25': float(pts[col].quantile(.25)),
                   'p50': float(pts[col].quantile(.50)),
                   'p75': float(pts[col].quantile(.75)),
                   'min': float(pts[col].min()),
                   'max': float(pts[col].max())}
    tab = pd.DataFrame(q).T
    tab.index.name = '指标'
    return tab


def trigger_type_table(pts):
    vc = pts['类型'].value_counts()
    tab = pd.DataFrame({'类型': vc.index, 'n': vc.values})
    tab['share'] = (tab['n'] / tab['n'].sum()).round(4)
    tab['ice_snow'] = tab['类型'].isin(ICE_TYPES)
    return tab


def scale_logit(pts):
    """规模判别：P(规模 ≥ 中) ~ log(沟比降) + log(高差) + log(主沟长)，标准化系数。"""
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler
    except ImportError as e:
        print('[跳过] 规模判别 logit：缺少 sklearn（%s）' % e)
        return None, None

    pts = pts.copy()
    pts['large'] = pts['规模'].isin(['中', '大', '特大']).astype(int)
    d = pts[['large', '沟比降', '高差', '主沟长']].dropna()

    # 分位区间的用处：先看两类规模在区间上的位置（不构成判据）
    print('\n按规模分组的地形均值：')
    g = pts.groupby('规模')[['沟比降', '高差', '主沟长']].agg(['mean', 'count']).round(3)
    print(g.to_string())

    X = np.column_stack([np.log(d['沟比降'].clip(1e-3)),
                         np.log(d['高差'].clip(1)),
                         np.log(d['主沟长'].clip(1))])
    Xs = StandardScaler().fit_transform(X)
    lr = LogisticRegression(C=1.0, max_iter=2000).fit(Xs, d['large'])
    auc = roc_auc_score(d['large'], lr.predict_proba(Xs)[:, 1])
    print('\n=== logit: P(规模≥中)（主结果：短陡沟出小灾、长缓沟出大灾）===')
    for nm, coef in zip(['log 沟比降', 'log 高差', 'log 主沟长'], lr.coef_[0]):
        print('  %-12s %+0.3f' % (nm, coef))
    print('  训练集 AUC %.3f（样本内，n=%d）' % (auc, len(d)))

    coef_tab = pd.DataFrame({'term': ['const', 'log_gradient', 'log_drop', 'log_length'],
                             'coef': [lr.intercept_[0]] + list(lr.coef_[0])})
    return coef_tab, auc


# =========================================================
# 3. DEM 空间易发性（rasterio + sklearn）
# =========================================================
def box_sum(a, k, h, w):
    """积分图实现 k×k 邻域均值（边界按有效计数）。"""
    a = np.nan_to_num(a, nan=0.0)
    ii = np.cumsum(np.cumsum(a, 0), 1)
    ii = np.pad(ii, ((1, 0), (1, 0)))
    K = k // 2
    y0 = np.clip(np.arange(h) - K, 0, h)
    y1 = np.clip(np.arange(h) + K + 1, 0, h)
    x0 = np.clip(np.arange(w) - K, 0, w)
    x1 = np.clip(np.arange(w) + K + 1, 0, w)
    out = ii[np.ix_(y1, x1)] - ii[np.ix_(y0, x1)] - ii[np.ix_(y1, x0)] + ii[np.ix_(y0, x0)]
    cnt = np.outer(y1 - y0, x1 - x0)
    return out / np.maximum(cnt, 1)


def dem_susceptibility(pts, resources=RESOURCES, n_neg=600, seed=0):
    try:
        import rasterio
        import rasterio.transform
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler
        from sklearn.cluster import KMeans
        from sklearn.neighbors import NearestNeighbors
    except ImportError as e:
        print('[跳过] DEM 空间易发性：缺少依赖（%s）' % e)
        return None, None

    rng = np.random.default_rng(seed)
    demf = os.path.join(resources, DEM_DIR, 'DEM.tif')
    with rasterio.open(demf) as s:
        dem = s.read(1).astype('float64')
        dem[dem == s.nodata] = np.nan
        tr = s.transform
        h, w = dem.shape
        res_x = abs(tr.a) * 111320 * np.cos(np.radians(28.4))
        res_y = abs(tr.e) * 110540

    gy, gx = np.gradient(np.nan_to_num(dem, nan=np.nanmedian(dem)))
    slope = np.degrees(np.arctan(np.hypot(gx / res_x, gy / res_y)))
    curv = np.gradient(gx, axis=1) + np.gradient(gy, axis=0)
    mean1k = box_sum(np.nan_to_num(dem, nan=np.nanmedian(dem)), 33, h, w)
    mean2 = box_sum(np.nan_to_num(dem, nan=np.nanmedian(dem)) ** 2, 33, h, w)
    std1k = np.sqrt(np.maximum(mean2 - mean1k ** 2, 0))
    tpi = dem - mean1k

    def sample(arr, xs, ys):
        out = []
        for x, y in zip(xs, ys):
            r, c = rasterio.transform.rowcol(tr, x, y)
            out.append(arr[r, c] if (0 <= r < h and 0 <= c < w) else np.nan)
        return np.array(out, float)

    inside = pts['x'].between(85.19, 85.694) & pts['y'].between(28.246, 28.564)
    pos = pts[inside]
    print('\n落在 DEM 内的灾点 %d 个' % len(pos))

    lons = rng.uniform(85.195, 85.69, 20000)
    lats = rng.uniform(28.25, 28.56, 20000)
    z_all = sample(dem, lons, lats)
    pos_xyz = np.column_stack([pos['x'].values * 111320 * np.cos(np.radians(28.4)),
                               pos['y'].values * 110540])
    cand_xy = np.column_stack([lons * 111320 * np.cos(np.radians(28.4)), lats * 110540])
    nn = NearestNeighbors(n_neighbors=1).fit(pos_xyz)
    dist, _ = nn.kneighbors(cand_xy)
    zpos = sample(dem, pos['x'].values, pos['y'].values)
    zlo, zhi = np.nanmin(zpos) - 300, np.nanmax(zpos) + 300
    mask = ((dist[:, 0] > 1000) & (dist[:, 0] < 15000)
            & (z_all > zlo) & (z_all < zhi) & np.isfinite(z_all))
    idx = rng.choice(np.where(mask)[0], size=min(n_neg, int(mask.sum())), replace=False)
    neg_x, neg_y = lons[idx], lats[idx]
    print('候选负样本池 %d，取 %d 个（距灾点 1–15 km，高程 %.0f–%.0f m）'
          % (mask.sum(), len(neg_x), zlo, zhi))

    feat_names = ['elev', 'slope', 'curv', 'std1k', 'tpi']

    def build(xs, ys):
        return np.column_stack([sample(dem, xs, ys), sample(slope, xs, ys),
                                sample(curv, xs, ys), sample(std1k, xs, ys), sample(tpi, xs, ys)])

    Xp, Xn = build(pos['x'].values, pos['y'].values), build(neg_x, neg_y)
    okp, okn = np.isfinite(Xp).all(1), np.isfinite(Xn).all(1)
    X = np.vstack([Xp[okp], Xn[okn]])
    y = np.r_[np.ones(int(okp.sum())), np.zeros(int(okn.sum()))]
    Xs2 = StandardScaler().fit_transform(X)
    lr2 = LogisticRegression(C=0.5, max_iter=3000, class_weight='balanced').fit(Xs2, y)
    print('\n=== 空间易发性 logit 系数（标准化）===')
    for nm, c in zip(feat_names, lr2.coef_[0]):
        print('  %-8s %+0.3f' % (nm, c))
    print('  全样本 AUC %.3f' % roc_auc_score(y, lr2.predict_proba(Xs2)[:, 1]))

    # 空间块 CV：显式记录每一折（含被跳过的），复核意见 2-4
    xy_all = np.column_stack([
        np.r_[pos['x'].values[okp], neg_x[okn]] * 111320 * np.cos(np.radians(28.4)),
        np.r_[pos['y'].values[okp], neg_y[okn]] * 110540])
    fold = KMeans(n_clusters=5, random_state=0, n_init=10).fit_predict(xy_all)

    rows = []
    for k in range(5):
        tr_i, te_i = fold != k, fold == k
        n_test, n_pos_test = int(te_i.sum()), int(y[te_i].sum())
        if len(np.unique(y[te_i])) < 2 or n_test < 20:
            rows.append({'fold': k, 'n_test': n_test, 'n_pos': n_pos_test,
                         'auc': np.nan, 'note': 'skipped'})
            continue
        sc = StandardScaler().fit(X[tr_i])
        m = LogisticRegression(C=0.5, max_iter=3000, class_weight='balanced') \
            .fit(sc.transform(X[tr_i]), y[tr_i])
        auc = roc_auc_score(y[te_i], m.predict_proba(sc.transform(X[te_i]))[:, 1])
        rows.append({'fold': k, 'n_test': n_test, 'n_pos': n_pos_test, 'auc': auc, 'note': ''})

    folds = pd.DataFrame(rows)
    valid = folds.dropna(subset=['auc'])
    print('  空间块 CV：5 块划分，有效 %d 折，AUC = %.3f ± %.3f'
          % (len(valid), valid['auc'].mean(), valid['auc'].std()))
    print(folds.round(3).to_string(index=False))

    coef = pd.DataFrame({'feature': feat_names, 'coef': lr2.coef_[0]})
    return coef, folds


# =========================================================
# 4. 主流程
# =========================================================
def main():
    os.makedirs(RESULT, exist_ok=True)

    pts = load_points()
    print('灾点 %d 个' % len(pts))
    print('\n=== 规模 / 频率 / 类型构成 ===')
    for c in ['规模', '频率', '类型', '有无_Fan']:
        if c in pts.columns:
            print(' ', c, pts[c].value_counts().to_dict())

    # --- 2-2：分位区间（不是临界阈值）---
    qt = quantile_table(pts)
    print('\n=== 历史灾点地形分位区间（84 点自身分位数，非临界阈值）===')
    print('说明：没有非灾点对照，任何分位数都能"覆盖"多数灾点，因此不能称为触发阈值；')
    print('      真正有判别意义的是下面的规模判别 logit。')
    print(qt.round(3).to_string())
    qt.to_csv(os.path.join(RESULT, 'p2_terrain_quantiles.csv'), encoding='utf-8-sig')

    # --- 2-8：触发类型 ---
    ttt = trigger_type_table(pts)
    ice_n = int(ttt.loc[ttt['ice_snow'], 'n'].sum())
    print('\n=== 历史灾点触发类型 ===')
    print(ttt.to_string(index=False))
    print('  冰/雪相关类型（冰水混合、冰雪融水、冰湖溃决）合计 %d/%d = %.0f%%'
          % (ice_n, len(pts), 100.0 * ice_n / len(pts)))
    ttt.to_csv(os.path.join(RESULT, 'p2_trigger_type.csv'), index=False, encoding='utf-8-sig')

    # --- 规模判别 logit（主结果）---
    coef_tab, auc = scale_logit(pts)
    if coef_tab is not None:
        coef_tab.to_csv(os.path.join(RESULT, 'p2_logit_coef.csv'),
                        index=False, encoding='utf-8-sig')

    # --- DEM 空间易发性 ---
    dem_coef, folds = dem_susceptibility(pts)
    if dem_coef is not None:
        dem_coef.to_csv(os.path.join(RESULT, 'p2_dem_logit.csv'),
                        index=False, encoding='utf-8-sig')
        folds.to_csv(os.path.join(RESULT, 'p2_fold_report.csv'),
                     index=False, encoding='utf-8-sig')
        folds.dropna(subset=['auc'])[['auc']].rename(columns={'auc': 'fold_auc'}).to_csv(
            os.path.join(RESULT, 'p2_blockcv.csv'), index=False, encoding='utf-8-sig')

    print('\n完成：结果写入', RESULT)


if __name__ == '__main__':
    main()
