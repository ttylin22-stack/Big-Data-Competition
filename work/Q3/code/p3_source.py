# -*- coding: utf-8 -*-
"""问题3 三条件判别：地形（T）— 物源（M）— 水源（W）

题面注释把"泥石流形成的基本条件"归为地形、物源、水源三条。本脚本用现有数据把前两条量化，
水源条件取自 p3_forecast 的融水结果，三者共同构成触发判别的门槛：

    触发时刻 t* = min{ t : H(t) ≥ H_c }，前提是 T、M 条件成立

物源估算：沟道两侧 500 m 缓冲内坡度 > 30° 的坡面面积 × 可动层厚（1–3 m）× 启动比例（10–30%）。
这是"潜在可动储量"的上限估计，不是本次事件的实测体积。
"""
import os
import numpy as np
import pandas as pd
import rasterio
from scipy.ndimage import distance_transform_edt

from config import RESOURCES, RESULT

OUT = RESULT
os.makedirs(OUT, exist_ok=True)

# ---------- 地形 ----------
demf = os.path.join(RESOURCES, 'DataSet_7', 'DEM.tif')
with rasterio.open(demf) as s:
    dem = s.read(1).astype('float32')
    dem[dem == s.nodata] = np.nan
    tr = s.transform
dx = abs(tr.a) * 111320 * np.cos(np.radians(28.3))
dy = abs(tr.e) * 110540
cell = dx * dy
z = np.nan_to_num(dem, nan=np.nanmean(dem))
gy, gx = np.gradient(z)
slope = np.degrees(np.arctan(np.hypot(gx / dx, gy / dy)))

path = np.load(os.path.join(OUT, 'p3_path.npy'))
prof = pd.read_csv(os.path.join(OUT, 'p3_path_profile.csv'))
skm, zz = prof['dist_km'].values, prof['elev_m'].values

seg = []
for a, b in [(0, 2), (2, 5), (5, 10), (10, 15), (15, skm[-1])]:
    m = (skm >= a) & (skm <= b)
    if m.sum() < 2:
        continue
    drop = float(zz[m][0] - zz[m][-1])
    length = float((skm[m][-1] - skm[m][0]) * 1000)
    seg.append({'段': '%.1f–%.1f km' % (a, b), '落差_m': round(drop),
                '比降': round(drop / length, 3),
                '坡度_deg': round(float(np.degrees(np.arctan(drop / length))), 1)})
seg_df = pd.DataFrame(seg)

# ---------- 物源 ----------
mask = np.ones(dem.shape, bool)
mask[path[:, 0], path[:, 1]] = False
dist = distance_transform_edt(mask) * dx

rows = []
for r_m in (300, 500, 1000):
    m = dist <= r_m
    for th in (30, 35, 40):
        A = ((slope > th) & m).sum() * cell
        rows.append({'缓冲_m': r_m, '坡度阈值_deg': th,
                     '面积_km2': round(A / 1e6, 2),
                     'V_低_1m10': float('%.2e' % (A * 1.0 * 0.10)),
                     'V_中_2m20': float('%.2e' % (A * 2.0 * 0.20)),
                     'V_高_3m30': float('%.2e' % (A * 3.0 * 0.30))})
src_df = pd.DataFrame(rows)

m500 = dist <= 500
A30 = ((slope > 30) & m500).sum() * cell
A35 = ((slope > 35) & m500).sum() * cell
print('=== 地形条件 ===')
print(seg_df.to_string(index=False))
print('全沟落差 %.0f m，源区 2 km 平均坡度 %.1f°' % (zz[0] - zz[-1], np.nanmean(slope[dist <= 2000])))
print('500 m 缓冲内坡度>30° 面积 %.2f km²（占该带 %.0f%%）'
      % (A30 / 1e6, 100 * A30 / (m500.sum() * cell)))

print('\n=== 物源条件（潜在可动储量） ===')
print('坡度>30°、500 m 缓冲：')
print('  可动层 1 m、启动 10%% -> %.2e m³' % (A30 * 1.0 * 0.10))
print('  可动层 2 m、启动 20%% -> %.2e m³' % (A30 * 2.0 * 0.20))
print('  可动层 3 m、启动 30%% -> %.2e m³' % (A30 * 3.0 * 0.30))
print('坡度>35° 同法：%.2e ~ %.2e m³'
      % (A35 * 1.0 * 0.10, A35 * 3.0 * 0.30))

# ---------- 水源（由 p3_forecast 给出，此处只读结论） ----------
water = pd.DataFrame([{'条件': '水源', '指标': '事件日累积融水占年均', '值': '74.0%'},
                      {'条件': '水源', '指标': '年正积温', '值': '581 ℃·d'},
                      {'条件': '水源', '指标': '事件前 72 h 降水', '值': '0.0 mm'}])

seg_df.to_csv(os.path.join(OUT, 'p3_terrain_segments.csv'), index=False, encoding='utf-8-sig')
src_df.to_csv(os.path.join(OUT, 'p3_source_volume.csv'), index=False, encoding='utf-8-sig')
water.to_csv(os.path.join(OUT, 'p3_water_condition.csv'), index=False, encoding='utf-8-sig')
print('\n已输出 p3_terrain_segments.csv / p3_source_volume.csv / p3_water_condition.csv')
