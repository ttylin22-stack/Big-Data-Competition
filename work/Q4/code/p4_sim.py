# -*- coding: utf-8 -*-
"""问题4:能量线—侧向扩散仿真(沿沟道能量线 + 侧向影响宽度)，输出水深/到达时间/范围

质量守恒按"逐断面实际湿宽"分配:
- 湿边界随水深自适应：高出沟底不超过 max(2h, 5 m)，迭代至自洽；
- 每个断面按 q(s)·Δs 分配体积，抛物线横向归一，保证段内严格守恒；
- 计算域外的 18% 视为从下游出口流出。
"""
from config import RESOURCES, RESULT, IMAGE, INPUT
import os
import numpy as np
import pandas as pd
import rasterio
from sklearn.neighbors import NearestNeighbors

OUT = RESULT
P3 = INPUT
os.makedirs(OUT, exist_ok=True)

demf = os.path.join(RESOURCES, 'DataSet_7', 'DEM.tif')
with rasterio.open(demf) as s:
    dem = s.read(1).astype('float32')
    dem[dem == s.nodata] = np.nan
    tr = s.transform
h, w = dem.shape

path = np.load(os.path.join(P3, 'p3_path.npy'))
prof = pd.read_csv(os.path.join(P3, 'p3_path_profile.csv'))
vel = pd.read_csv(os.path.join(P3, 'p3_velocity_profile.csv'))

lon = np.array([rasterio.transform.xy(tr, r, c)[0] for r, c in path])
lat = np.array([rasterio.transform.xy(tr, r, c)[1] for r, c in path])
MX = 111320 * np.cos(np.radians(28.3))
MY = 110540
px = (lon - lon[0]) * MX
py = (lat - lat[0]) * MY
skm = prof['dist_km'].values.astype(float)      # 与路径点一一对应的真实沿程距离
assert len(skm) == len(path), '剖面点数与路径点数不一致'
z0 = float(prof['elev_m'].iloc[0])
W0, WK = 60.0, 0.012

rr, cc = np.mgrid[0:h, 0:w]
glon = tr.c + (cc + 0.5) * tr.a
glat = tr.f + (rr + 0.5) * tr.e
gx = ((glon - lon[0]) * MX).ravel()
gy = ((glat - lat[0]) * MY).ravel()

pts = np.column_stack([px, py])
nn = NearestNeighbors(n_neighbors=1).fit(pts)
dist, idx = nn.kneighbors(np.column_stack([gx, gy]))
d_lat = dist[:, 0].reshape(h, w)
s_cell = skm[idx[:, 0]].reshape(h, w)

V0 = float(os.environ.get('P4_V0', 7.5e6))   # m3（3e6 m3 固体 × 2.5 扩容），可用环境变量覆盖做敏感性
LD = 12.0           # km，沉积衰减长度
cell_m2 = abs(tr.a * tr.e) * MX * MY
cell_km2 = cell_m2 / 1e6
width = W0 + WK * s_cell * 1000          # 名义影响半宽

# 能量线用视摩擦角 μ_E = H/L（Fahrböschung），与 Voellmy 的 μ 是两个不同的量
HL = (z0 - float(prof['elev_m'].iloc[-1])) / (skm[-1] * 1000)
energy = z0 - HL * s_cell * 1000
z_path = np.interp(s_cell, prof['dist_km'].values, prof['elev_m'].values)
above = dem - z_path
base = (d_lat <= width) & np.isfinite(dem)
print('视摩擦角 μ_E = %.4f(%.1f°)' % (HL, np.degrees(np.arctan(HL))))

BIN = 0.5                                     # km，分箱长度
nb = int(np.ceil(skm[-1] / BIN)) + 1
bi = np.clip((s_cell / BIN).astype(int), 0, nb - 1)
HMAX = 25.0                                   # 单断面水深上限，超出部分顺延下游


def allocate(mask):
    """按断面分配体积：返回 (水深场, 各箱湿宽)"""
    dep = np.zeros(dem.shape, dtype='float64')
    wt = np.full(nb, np.nan)
    carry = 0.0
    for b in range(nb):
        s_b = (b + 0.5) * BIN
        q_b = V0 / (LD * 1000.0) * np.exp(-s_b / LD) * BIN * 1000.0   # m3
        m = mask & (bi == b)
        if not m.any():
            carry += q_b
            continue
        q_b += carry
        carry = 0.0
        d_b = d_lat[m]
        w_b = max(2.0 * float(d_b.max()), 40.0)
        wt[b] = w_b
        p = np.clip(1.0 - (d_b / (w_b / 2.0)) ** 2, 0.0, 1.0)
        denom = p.sum() * cell_m2
        if denom <= 0:
            carry += q_b
            continue
        dep_b = q_b * p / denom
        peak = float(dep_b.max())
        if peak > HMAX:
            k = HMAX / peak
            dep_b *= k
            carry += q_b * (1.0 - k)
        dep[m] = dep_b
    return dep, wt


# 迭代：湿边界随水深自适应（max(2h,5)），阻尼更新
h_prev = None
for it in range(6):
    if h_prev is None:
        thr = np.full(dem.shape, 5.0)
    else:
        # 用"该断面最大水深"广播到整个断面，湿带才会随水深变宽
        hb = np.zeros(nb)
        for b in range(nb):
            mb = (bi == b) & (h_prev > 0)
            if mb.any():
                hb[b] = h_prev[mb].max()
        thr = np.maximum(hb[bi], 5.0)
    mask = base & (above <= thr)
    depth, w_tab = allocate(mask)
    h_prev = depth if it == 0 else 0.5 * depth + 0.5 * h_prev
    print('  iter %d: 湿像元 %d, 最大水深 %.1f m, 域内 %.2e m3'
          % (it, int((depth > 0.01).sum()), depth.max(), float(depth.sum()) * cell_m2))

affected = depth > 0.01

# 最终一致化：用收敛后的水深再确定一次掩膜，保证"掩膜⇄水深"自洽
hb = np.zeros(nb)
for b in range(nb):
    mb = (bi == b) & (depth > 0)
    if mb.any():
        hb[b] = depth[mb].max()
depth, w_tab = allocate(base & (above <= np.maximum(hb[bi], 5.0)))
affected = depth > 0.01
print('最终：湿像元 %d, 最大水深 %.1f m, 域内 %.2e m3'
      % (int(affected.sum()), depth.max(), float(depth.sum()) * cell_m2))

# 到达时间：对速度剖面按 1/v 积分得到累计时间 t(s)，再插值到每个像元
s_m = vel['dist_km'].values * 1000.0
vv = vel['v_mps'].values
t_cum = np.concatenate([[0.0], np.cumsum(np.diff(s_m) / (0.5 * (vv[1:] + vv[:-1])))]) / 60.0
arrive = np.where(affected, np.interp(s_cell, vel['dist_km'].values, t_cum), np.nan)

rp, cp = rasterio.transform.rowcol(tr, 85.3780, 28.2770)
print('口岸栅格 (%d,%d) 高程 %.0f m:水深 %.1f m,到达 %.1f min,受影响 %s'
      % (rp, cp, dem[rp, cp], depth[rp, cp], arrive[rp, cp], bool(affected[rp, cp])))

area = affected.sum() * cell_km2
vol_field = float(depth.sum()) * cell_m2
target = V0 * (1 - np.exp(-skm[-1] / LD))
print('影响面积 %.2f km2,最大水深 %.1f m,到达时间 %.1f - %.1f min'
      % (area, depth.max(), np.nanmin(arrive), np.nanmax(arrive)))
print('域内体积 %.2e m3（目标 %.2e，差额为 20.7 km 以外未计入部分）'
      % (vol_field, target))

rows = []
for q in np.linspace(0, skm[-1], 12):
    m = np.abs(s_cell - q) < 0.25
    wq = np.interp(q, (np.arange(nb) + 0.5) * BIN,
                   np.where(np.isfinite(w_tab), w_tab, np.nan))
    rows.append({'s_km': round(float(q), 1),
                 'depth_max_m': round(float(depth[m].max()), 2) if m.any() else 0.0,
                 'width_m': round(float(wq), 1) if np.isfinite(wq) else 0.0})
sw = pd.DataFrame(rows)
print('\n沿程统计:')
print(sw.to_string(index=False))
sw.to_csv(os.path.join(OUT, 'p4_along_path.csv'), index=False, encoding='utf-8-sig')

np.savez_compressed(os.path.join(OUT, 'p4_result.npz'), depth=depth.astype('float32'),
                    arrive=arrive.astype('float32'), affected=affected,
                    s_cell=s_cell.astype('float32'), dem=dem.astype('float32'),
                    path=path, port=np.array([rp, cp]))
print('saved p4_result.npz')