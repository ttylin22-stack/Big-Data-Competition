# -*- coding: utf-8 -*-
"""问题4：能量线—侧向扩散仿真（沿沟道能量线 + 侧向影响宽度），输出水深/到达时间/范围"""
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
skm = np.linspace(0, float(prof['dist_km'].max()), len(px))
z0 = float(prof['elev_m'].iloc[0])
MU = 0.0395
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

# 质量守恒：沿程体积按指数沉积衰减，断面平均水深 h(s)=V0/(LD*W(s))*exp(-s/LD)
V0 = 7.5e6          # m3（3e6 m3 固体 × 2.5 扩容）
LD = 12.0           # km，沉积衰减长度
width = W0 + WK * s_cell * 1000
h_s = V0 / (LD * 1000.0 * width) * np.exp(-s_cell / LD)

# 能量线用视摩擦角 μ_E = H/L（Fahrböschung），与 Voellmy 的 μ 是两个不同的量
HL = (z0 - float(prof['elev_m'].iloc[-1])) / (skm[-1] * 1000)
energy = z0 - HL * s_cell * 1000
z_path = np.interp(s_cell, prof['dist_km'].values, prof['elev_m'].values)
# 第三条约束：不能爬到沟底以上太高的山坡（否则会把整个河谷两侧都算成淹没区）
above = dem - z_path
affected = ((d_lat <= width) & (dem <= energy) & np.isfinite(dem)
            & (above <= np.maximum(2.0 * h_s, 5.0)))
print('视摩擦角 μ_E = %.4f（%.1f°）' % (HL, np.degrees(np.arctan(HL))))

lateral = np.clip(1.0 - (d_lat / np.maximum(width, 1.0)) ** 2, 0, 1)
depth = np.where(affected, h_s * lateral, 0.0)

# 到达时间：对速度剖面按 1/v 积分得到累计时间 t(s)，再插值到每个像元
s_m = vel['dist_km'].values * 1000.0
vv = vel['v_mps'].values
t_cum = np.concatenate([[0.0], np.cumsum(np.diff(s_m) / (0.5 * (vv[1:] + vv[:-1])))]) / 60.0
arrive = np.where(affected, np.interp(s_cell, vel['dist_km'].values, t_cum), np.nan)

rp, cp = rasterio.transform.rowcol(tr, 85.3780, 28.2770)
print('口岸栅格 (%d,%d) 高程 %.0f m：水深 %.1f m，到达 %.1f min，受影响 %s'
      % (rp, cp, dem[rp, cp], depth[rp, cp], arrive[rp, cp], bool(affected[rp, cp])))

cell_km2 = abs(tr.a * tr.e) * MX * MY / 1e6
area = affected.sum() * cell_km2
print('影响面积 %.2f km2，最大水深 %.1f m，到达时间 %.1f–%.1f min'
      % (area, depth.max(), np.nanmin(arrive), np.nanmax(arrive)))

rows = []
for q in np.linspace(0, skm[-1], 12):
    m = np.abs(s_cell - q) < 0.25
    rows.append({'s_km': round(float(q), 1),
                 'depth_max_m': round(float(depth[m].max()), 2) if m.any() else 0.0,
                 'width_m': round(W0 + WK * q * 1000)})
sw = pd.DataFrame(rows)
print('\n沿程统计：')
print(sw.to_string(index=False))
sw.to_csv(os.path.join(OUT, 'p4_along_path.csv'), index=False, encoding='utf-8-sig')

np.savez_compressed(os.path.join(OUT, 'p4_result.npz'), depth=depth.astype('float32'),
                    arrive=arrive.astype('float32'), affected=affected,
                    s_cell=s_cell.astype('float32'), dem=dem.astype('float32'),
                    path=path, port=np.array([rp, cp]))
print('saved p4_result.npz')
