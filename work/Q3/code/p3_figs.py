# -*- coding: utf-8 -*-
"""问题3 图表"""
import os
import numpy as np
import pandas as pd
import matplotlib
from config import RESOURCES, INPUT, RESULT, IMAGE
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs(IMAGE, exist_ok=True)
EVENT = pd.Timestamp('2026-08-26')

tri = pd.read_csv(os.path.join(RESULT, 'p3_trigger_ensemble.csv'), parse_dates=['date'])
tau = pd.read_csv(os.path.join(RESULT, 'p3_travel_time.csv'))['travel_min'].values
prof = pd.read_csv(os.path.join(RESULT, 'p3_path_profile.csv'))
vel = pd.read_csv(os.path.join(RESULT, 'p3_velocity_profile.csv'))
cov = pd.read_csv(os.path.join(RESULT, 'p3_coverage.csv'))

# 图1 触发时刻集合
fig, ax = plt.subplots(figsize=(10, 3.8))
ax.hist(tri['date'], bins=40, color='#4c72b0', alpha=0.85)
lo, hi = tri['date'].quantile(0.025), tri['date'].quantile(0.975)
ax.axvspan(lo, hi, color='#ff7f0e', alpha=0.18, label='95%% 区间 %s ~ %s' % (lo.date(), hi.date()))
ax.axvline(EVENT, color='k', ls='--', lw=1.5, label='实际事件 2026-08-26')
ax.set_ylabel('集合成员数')
ax.set_title('触发时刻集合（3000 次蒙特卡洛）：区间宽 20 天，事件落在第 98 百分位')
ax.legend(fontsize=8)
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(IMAGE, 'p3_fig1_trigger_ci.png'), dpi=150)
plt.close(fig)

# 图2 覆盖率回测
fig, ax = plt.subplots(figsize=(9, 3.4))
for i, r in cov.iterrows():
    lo_, hi_, ob = pd.Timestamp(r['lo']), pd.Timestamp(r['hi']), pd.Timestamp(r['obs'])
    ax.plot([lo_, hi_], [i, i], lw=7, color='#4c72b0', alpha=0.55,
            solid_capstyle='butt')
    ax.plot(ob, i, 'o', color='#d62728', ms=6, zorder=3)
ax.set_yticks(range(len(cov)))
ax.set_yticklabels(['%d 年' % y for y in cov['year']])
ax.set_title('留一年外回测：5/5 年份的观测触发日落入区间（平均宽 42 天）')
ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
ax.grid(axis='x', alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(IMAGE, 'p3_fig2_coverage.png'), dpi=150)
plt.close(fig)

# 图3 沟道剖面 + 速度
fig, ax1 = plt.subplots(figsize=(11, 3.8))
ax1.plot(prof['dist_km'], prof['elev_m'], color='#8c564b', lw=1.5, label='沟道高程')
ax1.set_xlabel('沿程距离 (km)')
ax1.set_ylabel('高程 (m)', color='#8c564b')
ax1.tick_params(axis='y', labelcolor='#8c564b')
ax2 = ax1.twinx()
ax2.plot(vel['dist_km'], vel['v_mps'], color='#1f77b4', lw=1.2, label='流速')
ax2.set_ylabel('流速 (m/s)', color='#1f77b4')
ax2.tick_params(axis='y', labelcolor='#1f77b4')
ax1.set_title('源区→口岸 24.8 km 沟道纵剖面与 Voellmy 流速（到达口岸 22.8 分钟）')
ax1.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(IMAGE, 'p3_fig3_profile.png'), dpi=150)
plt.close(fig)

# 图4 到达时间分布
fig, ax = plt.subplots(figsize=(7, 3.4))
ax.hist(tau, bins=35, color='#2ca02c', alpha=0.85)
ax.axvline(np.median(tau), color='k', ls='--', lw=1.2)
ax.set_xlabel('到达口岸所需时间 (min)')
ax.set_ylabel('集合成员数')
ax.set_title('传播时间：中位 %.1f 分钟（%.1f–%.1f）'
             % (np.median(tau), np.percentile(tau, 5), np.percentile(tau, 95)))
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(IMAGE, 'p3_fig4_travel.png'), dpi=150)
plt.close(fig)
print('ok', os.listdir(IMAGE))
