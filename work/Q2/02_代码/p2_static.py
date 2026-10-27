# -*- coding: utf-8 -*-
"""问题2（一）：84 个灾点的静态地形阈值 + DEM 空间易发性模型"""
import os, sys
import numpy as np
import pandas as pd
import rasterio
import geopandas as gpd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


from config import BASE, IN, DATA, FIGS
OUT = DATA
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(0)

PT = os.path.join(BASE, '（6）西藏日喀则市吉隆县泥石流灾害区历史灾点数据',
                  '西藏日喀则市吉隆县泥石流灾害区历史灾点数据',
                  '西藏日喀则市吉隆县泥石流灾害区历史灾点数据.shp')
pts = gpd.read_file(PT).set_crs(4490, allow_override=True).to_crs(4326)
pts['x'] = [float(v) for v in pts['x_coord']]
pts['y'] = [float(v) for v in pts['y_coord']]
for c in ['沟比降', '高差', '主沟长', '扇宽', '扇长', '沟口H', '高程Max', 'Area', 'DisFrmFlt']:
    pts[c] = pd.to_numeric(pts[c], errors='coerce')

print('灾点 %d 个' % len(pts))
print('\n=== 规模 / 频率 / 类型构成 ===')
for c in ['规模', '频率', '类型', '有无_Fan']:
    print(' ', c, pts[c].value_counts().to_dict())

print('\n=== 地形属性分位数（全部 84 点）===')
cols = ['沟比降', '高差', '主沟长', '扇宽', '扇长', 'Area']
q = pts[cols].quantile([0.1, 0.25, 0.5, 0.75, 0.9]).round(3)
print(q.to_string())

print('\n=== 按规模分组的地形均值 ===')
g = pts.groupby('规模')[['沟比降', '高差', '主沟长', '扇长']].agg(['mean', 'count']).round(3)
print(g.to_string())

# ---- 地形阈值建议（用分位数与物理约束） ----
thr = {
    '沟比降': {'p25': float(pts['沟比降'].quantile(.25)), 'p50': float(pts['沟比降'].quantile(.5)),
             'p75': float(pts['沟比降'].quantile(.75))},
    '高差_m': {'p25': float(pts['高差'].quantile(.25)), 'p50': float(pts['高差'].quantile(.5)),
             'p75': float(pts['高差'].quantile(.75))},
    '主沟长_m': {'p25': float(pts['主沟长'].quantile(.25)), 'p50': float(pts['主沟长'].quantile(.5)),
              'p75': float(pts['主沟长'].quantile(.75))},
}
pd.DataFrame(thr).T.to_csv(os.path.join(OUT, 'p2_terrain_thresholds.csv'), encoding='utf-8-sig')

# ---- logit：规模是否 ≥ 中 ----
pts['large'] = pts['规模'].isin(['中', '大', '特大']).astype(int)
d = pts[['large', '沟比降', '高差', '主沟长', '扇长']].dropna()
X = np.column_stack([np.log(d['沟比降'].clip(1e-3)), np.log(d['高差'].clip(1)),
                     np.log(d['主沟长'].clip(1))])
Xs = StandardScaler().fit_transform(X)
lr = LogisticRegression(C=1.0, max_iter=2000).fit(Xs, d['large'])
print('\n=== logit: P(规模≥中) ===')
for nm, coef in zip(['log 沟比降', 'log 高差', 'log 主沟长'], lr.coef_[0]):
    print('  %-12s 系数 %+0.3f' % (nm, coef))
print('  训练集 AUC %.3f (n=%d)' % (roc_auc_score(d['large'], lr.predict_proba(Xs)[:, 1]), len(d)))
pd.DataFrame({'term': ['const'] + ['log_gradient', 'log_drop', 'log_length'],
              'coef': [lr.intercept_[0]] + list(lr.coef_[0])}).to_csv(
    os.path.join(OUT, 'p2_logit_coef.csv'), index=False, encoding='utf-8-sig')

# ---- DEM 派生的空间易发性（22 正样本 + 伪负样本）----
demf = os.path.join(BASE, '（7）2026年8月26日日喀则市吉隆县泥石流灾害区DEM数据\\DEM.tif')
with rasterio.open(demf) as s:
    dem = s.read(1).astype('float64')
    dem[dem == s.nodata] = np.nan
    tr, crs = s.transform, s.crs
    h, w = dem.shape
    res_x = abs(tr.a) * 111320 * np.cos(np.radians(28.4))
    res_y = abs(tr.e) * 110540

def box_sum(a, k):
    """积分图实现 k×k 邻域求和（边界按有效计数）"""
    a = np.nan_to_num(a, nan=0.0)
    ii = np.cumsum(np.cumsum(a, 0), 1)
    ii = np.pad(ii, ((1, 0), (1, 0)))
    K = k // 2
    y0 = np.clip(np.arange(h) - K, 0, h); y1 = np.clip(np.arange(h) + K + 1, 0, h)
    x0 = np.clip(np.arange(w) - K, 0, w); x1 = np.clip(np.arange(w) + K + 1, 0, w)
    out = (ii[np.ix_(y1, x1)] - ii[np.ix_(y0, x1)] - ii[np.ix_(y1, x0)] + ii[np.ix_(y0, x0)])
    cnt = np.outer(y1 - y0, x1 - x0)
    return out / np.maximum(cnt, 1)

gy, gx = np.gradient(np.nan_to_num(dem, nan=np.nanmedian(dem)))
slope = np.degrees(np.arctan(np.hypot(gx / res_x, gy / res_y)))
curv = np.gradient(gx, axis=1) + np.gradient(gy, axis=0)
mean1k = box_sum(np.nan_to_num(dem, nan=np.nanmedian(dem)), 33)
mean2 = box_sum(np.nan_to_num(dem, nan=np.nanmedian(dem)) ** 2, 33)
std1k = np.sqrt(np.maximum(mean2 - mean1k ** 2, 0))
tpi = dem - mean1k

def sample(arr, xs, ys):
    out = []
    for x, y in zip(xs, ys):
        r, c = rasterio.transform.rowcol(tr, x, y)
        out.append(arr[r, c] if (0 <= r < h and 0 <= c < w) else np.nan)
    return np.array(out, float)

inside = (pts['x'].between(85.19, 85.694)) & (pts['y'].between(28.246, 28.564))
pos = pts[inside]
print('\n落在 DEM 内的灾点 %d 个' % len(pos))

# 伪负样本：约束在灾点 1–15 km、同高程带内，避免"高海拔=非灾点"的平凡分离
from sklearn.neighbors import NearestNeighbors
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
mask = (dist[:, 0] > 1000) & (dist[:, 0] < 15000) & (z_all > zlo) & (z_all < zhi) & np.isfinite(z_all)
idx = rng.choice(np.where(mask)[0], size=min(600, mask.sum()), replace=False)
neg_x, neg_y = lons[idx], lats[idx]
print('\n候选负样本池 %d，约束后取 %d 个（距灾点 1–15 km，高程 %.0f–%.0f m）' % (mask.sum(), len(neg_x), zlo, zhi))

feat_names = ['elev', 'slope', 'curv', 'std1k', 'tpi']
def build(xs, ys):
    return np.column_stack([sample(dem, xs, ys), sample(slope, xs, ys), sample(curv, xs, ys),
                            sample(std1k, xs, ys), sample(tpi, xs, ys)])
Xp = build(pos['x'].values, pos['y'].values)
Xn = build(neg_x, neg_y)
okp = np.isfinite(Xp).all(1)
okn = np.isfinite(Xn).all(1)
X = np.vstack([Xp[okp], Xn[okn]])
y = np.r_[np.ones(okp.sum()), np.zeros(okn.sum())]
Xs2 = StandardScaler().fit_transform(X)
lr2 = LogisticRegression(C=0.5, max_iter=3000, class_weight='balanced').fit(Xs2, y)
print('\n=== 空间易发性 logit 系数（标准化） ===')
for nm, c in zip(feat_names, lr2.coef_[0]):
    print('  %-8s %+0.3f' % (nm, c))
print('  全样本 AUC %.3f' % roc_auc_score(y, lr2.predict_proba(Xs2)[:, 1]))

# 空间块交叉验证（KMeans 5 个空间块）
from sklearn.cluster import KMeans
xy_all = np.column_stack([
    np.r_[pos['x'].values[okp], neg_x[okn]] * 111320 * np.cos(np.radians(28.4)),
    np.r_[pos['y'].values[okp], neg_y[okn]] * 110540])
fold = KMeans(n_clusters=5, random_state=0, n_init=10).fit_predict(xy_all)
aucs = []
for k in range(5):
    tr_i, te_i = fold != k, fold == k
    if len(np.unique(y[te_i])) < 2 or te_i.sum() < 20:
        continue
    sc = StandardScaler().fit(X[tr_i])
    m = LogisticRegression(C=0.5, max_iter=3000, class_weight='balanced').fit(sc.transform(X[tr_i]), y[tr_i])
    aucs.append(roc_auc_score(y[te_i], m.predict_proba(sc.transform(X[te_i]))[:, 1]))
print('  空间块 CV AUC = %.3f ± %.3f（%d 折）' % (np.mean(aucs), np.std(aucs), len(aucs)))

pd.DataFrame({'feature': feat_names, 'coef': lr2.coef_[0]}).to_csv(
    os.path.join(OUT, 'p2_dem_logit.csv'), index=False, encoding='utf-8-sig')
pd.DataFrame({'fold_auc': aucs}).to_csv(os.path.join(OUT, 'p2_blockcv.csv'), index=False, encoding='utf-8-sig')
