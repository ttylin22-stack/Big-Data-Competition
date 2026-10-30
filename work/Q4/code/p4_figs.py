# -*- coding: utf-8 -*-
"""问题4 图表：淹没水深、到达时间、沿程断面（走廊放大）"""
import os
import numpy as np
import pandas as pd
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LightSource

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

from config import RESOURCES, RESULT, IMAGE
OUT = RESULT
FIG = IMAGE
os.makedirs(FIG, exist_ok=True)

d = np.load(os.path.join(OUT, 'p4_result.npz'))
depth, arrive, dem, path = d['depth'], d['arrive'], d['dem'], d['path']
rp, cp = int(d['port'][0]), int(d['port'][1])
with rasterio.open(os.path.join(RESOURCES, 'dem', 'DEM.tif')) as s:
    tr = s.transform
    b = s.bounds

h, w = dem.shape
r0 = max(min(path[:, 0].min(), rp) - 90, 0); r1 = min(max(path[:, 0].max(), rp) + 90, h)
c0 = max(min(path[:, 1].min(), cp) - 90, 0); c1 = min(max(path[:, 1].max(), cp) + 90, w)
ext = (b.left + c0 * tr.a, b.left + c1 * tr.a, b.top + r1 * tr.e, b.top + r0 * tr.e)

zc = dem[r0:r1, c0:c1]
dc = depth[r0:r1, c0:c1]
ac = arrive[r0:r1, c0:c1]
ls = LightSource(azdeg=315, altdeg=40)
hs = ls.hillshade(np.nan_to_num(zc, nan=np.nanmin(zc)), vert_exag=1.5, dx=30, dy=30)

fig, ax = plt.subplots(figsize=(8.2, 7.4))
ax.imshow(hs, cmap='gray', extent=ext, origin='upper')
dm = np.ma.masked_where(dc <= 0.05, dc)
im = ax.imshow(dm, cmap='YlOrRd', extent=ext, origin='upper', alpha=0.85,
               norm=mcolors.LogNorm(vmin=0.1, vmax=10))
pr = path[:, 0] - r0; pc = path[:, 1] - c0
ax.plot(ext[0] + (pc + 0.5) * tr.a, ext[3] + (pr + 0.5) * tr.e, color='#1f77b4', lw=0.9, label='沟道中心线')
ax.plot(ext[0] + (pc[0] + 0.5) * tr.a, ext[3] + (pr[0] + 0.5) * tr.e, '^', color='#2ca02c', ms=9, label='释放区')
ax.plot(ext[0] + (cp - c0 + 0.5) * tr.a, ext[3] + (rp - r0 + 0.5) * tr.e, '*', color='k', ms=15, label='吉隆口岸')
cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cb.set_label('最大水深 (m)')
ax.set_xlabel('经度'); ax.set_ylabel('纬度')
_cell = ((b.right-b.left)/dem.shape[1]*111320*np.cos(np.radians(28.4))) * ((b.top-b.bottom)/dem.shape[0]*110540) / 1e6
_area = float((depth > 0.01).sum()) * _cell
ax.set_title('碎屑流淹没范围（走廊放大）：%.2f km²，最大水深 %.1f m' % (_area, float(depth.max())))
ax.legend(loc='lower left', fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'p4_fig1_inundation.png'), dpi=160)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8.2, 7.4))
ax.imshow(hs, cmap='gray', extent=ext, origin='upper')
am = np.ma.masked_where(np.isnan(ac) | (ac <= 0), ac)
im = ax.imshow(am, cmap='viridis', extent=ext, origin='upper', alpha=0.85, vmin=0, vmax=23)
ax.plot(ext[0] + (pc + 0.5) * tr.a, ext[3] + (pr + 0.5) * tr.e, color='w', lw=0.7, alpha=0.8)
ax.plot(ext[0] + (cp - c0 + 0.5) * tr.a, ext[3] + (rp - r0 + 0.5) * tr.e, '*', color='r', ms=15)
cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cb.set_label('到达时间 (min)')
ax.set_xlabel('经度'); ax.set_ylabel('纬度')
ax.set_title('到达时间场：释放区 0 min → 口岸 16 min')
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'p4_fig2_arrival.png'), dpi=160)
plt.close(fig)

al = pd.read_csv(os.path.join(OUT, 'p4_along_path.csv'))
fig, ax1 = plt.subplots(figsize=(9, 3.6))
ax1.plot(al['s_km'], al['depth_max_m'], 'o-', color='#d62728', label='断面最大水深')
ax1.set_xlabel('沿程距离 (km)'); ax1.set_ylabel('水深 (m)', color='#d62728')
ax1.tick_params(axis='y', labelcolor='#d62728')
ax2 = ax1.twinx()
ax2.plot(al['s_km'], al['width_m'], 's--', color='#1f77b4')
ax2.set_ylabel('影响宽度 (m)', color='#1f77b4')
ax2.tick_params(axis='y', labelcolor='#1f77b4')
ax1.set_title('沿程水深与影响宽度（质量守恒 + 指数沉积）')
ax1.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'p4_fig3_along.png'), dpi=160)
plt.close(fig)
print('ok', os.listdir(FIG))
