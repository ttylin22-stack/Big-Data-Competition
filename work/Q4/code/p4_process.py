# -*- coding: utf-8 -*-
"""问题4 过程可视化：时间切片多帧、时空图、过程曲线

基于 p4_result.npz 的静态度场/到达时间场做“运动学切片”：
某像元在 t 时刻被淹没 ⇔ 该像元到达时间 ≤ t。
这不是动力学求解，只是把同一组结果按时间展开，图注中已注明。
"""
import os
import numpy as np
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import LightSource
from matplotlib.collections import LineCollection
from matplotlib.gridspec import GridSpec

from config import RESOURCES, RESULT, IMAGE, INPUT

OUT, FIG = RESULT, IMAGE
os.makedirs(FIG, exist_ok=True)
plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

# ---------- 关键参数（合理取值） ----------
FRAMES = [0, 3, 6, 9, 12, 15, 17]   # 分钟；17 = 沟道末端
DT = 0.25                            # 曲线时间步长（分钟）
DEPTH_MIN, DEPTH_MAX = 0.1, 10.0     # 水深对数色标范围（与图 4-1 一致）
COVER_MARGIN = 90                    # 走廊外扩像元数（与图 4-1/4-2 一致）

d = np.load(os.path.join(OUT, 'p4_result.npz'))
depth, arrive, dem, path = d['depth'], d['arrive'], d['dem'], d['path']
s_cell = d['s_cell']
rp, cp = int(d['port'][0]), int(d['port'][1])

with rasterio.open(os.path.join(RESOURCES, 'dem', 'DEM.tif')) as s:
    tr = s.transform
    b = s.bounds

h, w = dem.shape
r0 = max(min(path[:, 0].min(), rp) - COVER_MARGIN, 0)
r1 = min(max(path[:, 0].max(), rp) + COVER_MARGIN, h)
c0 = max(min(path[:, 1].min(), cp) - COVER_MARGIN, 0)
c1 = min(max(path[:, 1].max(), cp) + COVER_MARGIN, w)
ext = (b.left + c0 * tr.a, b.left + c1 * tr.a,
       b.top + r1 * tr.e, b.top + r0 * tr.e)

demc = dem[r0:r1, c0:c1]
dpc = depth[r0:r1, c0:c1]
arc = arrive[r0:r1, c0:c1]
sc = s_cell[r0:r1, c0:c1]
hs = LightSource(azdeg=315, altdeg=40).hillshade(
    np.nan_to_num(demc, nan=np.nanmin(demc)), vert_exag=1.5, dx=30, dy=30)
pr, pc = path[:, 0] - r0, path[:, 1] - c0

cell_km2 = abs(tr.a * tr.e) * 111320 * np.cos(np.radians(28.3)) * 110540 / 1e6
tmax = float(np.nanmax(arc))

# ---------- 时间序列量 ----------
tgrid = np.linspace(0, tmax, max(int(round(tmax / DT)) + 1, 2))
_keep = [f for f in FRAMES if f < tmax - 1e-6]
FRAMES = sorted(set(_keep + [round(tmax)]))
area_t, vol_t, front_t = [], [], []
for t in tgrid:
    wet = np.isfinite(arc) & (arc <= t)
    area_t.append(wet.sum() * cell_km2)
    vol_t.append(float(dpc[wet].sum()) * cell_km2 * 1e6)   # m3
    front_t.append(float(sc[wet].max()) if wet.any() else 0.0)
area_t, vol_t, front_t = map(np.array, (area_t, vol_t, front_t))
print('t=0   面积 %.3f km2  体积 %.2e m3' % (area_t[0], vol_t[0]))
print('t=17  面积 %.3f km2  体积 %.2e m3  前锋 %.1f km' % (area_t[-1], vol_t[-1], front_t[-1]))

# ---------- 图 4-4：时间切片多帧 ----------
fig = plt.figure(figsize=(13.2, 6.6))
gs = GridSpec(2, 4, figure=fig, wspace=0.14, hspace=0.22)
norm = mcolors.LogNorm(DEPTH_MIN, DEPTH_MAX)
cmap = plt.get_cmap('YlOrRd')
im = None
for i, t in enumerate(FRAMES):
    ax = fig.add_subplot(gs[i // 4, i % 4])
    ax.imshow(hs, cmap='gray', extent=ext, origin='upper')
    wet = np.isfinite(arc) & (arc <= t)
    m = np.ma.masked_where(~wet, np.maximum(dpc, DEPTH_MIN * 0.999))
    im = ax.imshow(m, cmap=cmap, norm=norm, extent=ext, origin='upper', alpha=0.9)
    ax.plot(ext[0] + (pc + 0.5) * tr.a, ext[3] + (pr + 0.5) * tr.e,
            color='k', lw=1.0, alpha=0.5)
    if wet.any():
        j = int(np.argmax(np.where(wet, sc, -1.0)))
        fr, fc = np.unravel_index(j, sc.shape)
        ax.plot(ext[0] + (fc + 0.5) * tr.a, ext[3] + (fr + 0.5) * tr.e,
                'o', color='#00bfff', ms=7, mec='k', zorder=5)
    ax.set_title('t = %.0f min' % t, fontsize=10)
    xt = np.linspace(ext[0], ext[1], 2)
    yt = np.linspace(ext[2], ext[3], 3)
    ax.set_xticks(xt); ax.set_xticklabels(['%.2f' % v for v in xt], fontsize=7)
    ax.set_yticks(yt); ax.set_yticklabels(['%.2f' % v for v in yt], fontsize=7)
    if i % 4 != 0:
        ax.set_yticklabels([])
    if i // 4 != 1:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel('经度', fontsize=8)

cax = fig.add_axes([0.915, 0.2, 0.012, 0.6])
cb = fig.colorbar(im, cax=cax)
cb.set_label('水深 (m)', fontsize=9)
fig.suptitle('淹没演进（运动学切片）：t 为到达时间阈值，蓝点为前锋位置', fontsize=12)
fig.savefig(os.path.join(FIG, 'p4_fig4_frames.png'), dpi=150, bbox_inches='tight')
plt.close(fig)

# ---------- 图 4-5：时空图 ----------
fig, ax = plt.subplots(figsize=(7.6, 5.2))
ax.fill_between(tgrid, 0, front_t, color='#f4a582', alpha=0.75,
                label='已到达区段')
ax.plot(tgrid, front_t, color='#b2182b', lw=2.2, label='前锋位置 s(t)')
ax.axhline(float(np.nanmax(s_cell)), color='k', ls='--', lw=1.0)
ax.axvline(tmax, color='k', ls=':', lw=1.0)
ax.set_xlim(0, tmax)
ax.set_ylim(0, float(np.nanmax(s_cell)))
ax.invert_yaxis()
ax.set_xlabel('时间 (min)')
ax.set_ylabel('沿沟距离 s (km)')
ax.set_title('时空图：前锋推进过程')
ax.grid(alpha=0.25)
ax.legend(loc='lower right', fontsize=9)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'p4_fig5_spacetime.png'), dpi=160)
plt.close(fig)

# ---------- 图 4-6：过程曲线 ----------
fig, ax1 = plt.subplots(figsize=(7.6, 3.9))
l1, = ax1.plot(tgrid, area_t, color='#d62728', lw=2.2, label='湿润面积')
ax1.set_xlabel('时间 (min)')
ax1.set_ylabel('湿润面积 (km²)', color='#d62728')
ax1.tick_params(axis='y', labelcolor='#d62728')
ax1.set_xlim(0, tmax)

ax2 = ax1.twinx()
l2, = ax2.plot(tgrid, vol_t / 1e6, color='#1f77b4', lw=2.2, ls='--', label='存储体积')
ax2.set_ylabel('存储体积（百万 m³）', color='#1f77b4')
ax2.tick_params(axis='y', labelcolor='#1f77b4')

ax1.axvline(tmax, color='k', ls=':', lw=1.0)
ax.text(tmax - 2.8, area_t[-1] * 0.35, '%.0f min' % tmax, fontsize=9)
ax1.grid(alpha=0.25)
ax1.legend(handles=[l1, l2], loc='lower right', fontsize=9)
ax1.set_title('过程曲线：湿润面积与存储体积随时间')
fig.tight_layout()
fig.text(0.5, -0.02,
         '注：存储体积为计算域内体积（%.2f 百万 m³），为设定 V0 的 %.0f%%；'
         '差额被地形掩膜排除在走廊之外' % (vol_t[-1] / 1e6, 100 * vol_t[-1] / 7.5e6),
         ha='center', va='top', fontsize=8, color='#555555')
fig.savefig(os.path.join(FIG, 'p4_fig6_process.png'), dpi=160, bbox_inches='tight')
plt.close(fig)

print('已输出 p4_fig4_frames.png / p4_fig5_spacetime.png / p4_fig6_process.png')
