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
from matplotlib.collections import LineCollection

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

from config import RESOURCES, RESULT, IMAGE
OUT = RESULT
FIG = IMAGE
os.makedirs(FIG, exist_ok=True)

d = np.load(os.path.join(OUT, 'p4_result.npz'))
depth, arrive, dem, path = d['depth'], d['arrive'], d['dem'], d['path']
rp, cp = int(d['port'][0]), int(d['port'][1])
with rasterio.open(os.path.join(RESOURCES, 'DataSet_7', 'DEM.tif')) as s:
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
ax.set_title('碎屑流淹没范围（走廊放大）：%.2f km²,最大水深 %.1f m' % (_area, float(depth.max())))
ax.legend(loc='lower left', fontsize=8)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'p4_fig1_inundation.png'), dpi=160)
plt.close(fig)

# ---- 图 2：到达时间场（中心线加粗，避免在 35 km 幅面上"看不见"）----
fig, ax = plt.subplots(figsize=(8.2, 7.4))
ax.imshow(hs, cmap='gray', extent=ext, origin='upper')

t_max = float(np.nanmax(ac))
am = np.ma.masked_where(np.isnan(ac) | (ac <= 0), ac)
im = ax.imshow(am, cmap='viridis', extent=ext, origin='upper',
               alpha=0.55, vmin=0, vmax=round(t_max))

# 中心线：位置由沟道像元给出，颜色由该像元的到达时间给出
xs_line = ext[0] + (pc + 0.5) * tr.a
ys_line = ext[3] + (pr + 0.5) * tr.e
s_line = np.asarray(d['s_cell'], float)[path[:, 0], path[:, 1]]
t_line = np.asarray(d['arrive'], float)[path[:, 0], path[:, 1]]
ok_line = np.isfinite(t_line)
if ok_line.sum() >= 2:
    t_line = np.interp(s_line, s_line[ok_line], t_line[ok_line])

pts = np.column_stack([xs_line, ys_line])
segs = np.stack([pts[:-1], pts[1:]], axis=1)

ax.plot(xs_line, ys_line, color='k', lw=5.0, alpha=0.55,
        solid_capstyle='round', zorder=3)
lc = LineCollection(segs, cmap='viridis',
                    norm=mcolors.Normalize(0, round(t_max)),
                    linewidths=3.0, zorder=4)
lc.set_array(0.5 * (t_line[:-1] + t_line[1:]))
ax.add_collection(lc)

ax.plot(xs_line[0], ys_line[0], '^', color='#2ca02c', ms=11,
        mec='k', zorder=5, label='释放区')
ax.plot(ext[0] + (cp - c0 + 0.5) * tr.a, ext[3] + (rp - r0 + 0.5) * tr.e,
        '*', color='r', ms=17, mec='k', zorder=5, label='口岸（台地，未淹没）')

cb = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
cb.set_label('到达时间 (min)')
ax.set_xlabel('经度'); ax.set_ylabel('纬度')
ax.set_title(f'到达时间场：0–{t_max:.0f} 分钟'
             f'（粗线为沟道中心线，浅带为影响范围）', fontsize=10)
ax.legend(loc='lower left', fontsize=8)
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
