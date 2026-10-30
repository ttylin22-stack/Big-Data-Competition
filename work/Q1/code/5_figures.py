import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import common as K
import config as C

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

os.makedirs(C.IMAGE, exist_ok=True)
d = K.load_daily()
ev = pd.Timestamp(C.EVENT_DATE)
win = K.event_window(d)
ctrl = K.control_period(d)

LABELS = {
    'dVWC_mean': '表层-深层\n土壤湿度差',
    'dVWC_shallow_mean': '表层-中层\n土壤湿度差',
    'T_station_min': '夜间最低温\n(站点)',
    'T_source_min': '夜间最低温\n(源区)',
    'LE(W/m2)_sum24': '潜热通量\n日累积',
    'H2O_density(g/m3)_mean': '绝对湿度',
    'shear_alpha_mean': '风切变指数',
    'VPD_kPa_mean': '饱和水汽压差',
    'Rn_sum24': '净辐射\n日累积',
    'Rsd(W/m2)_sum24': '短波辐射\n日累积',
    'T_station_max': '日最高温',
    'T_port_mean': '日均气温\n(口岸线)',
    'T_station_mean': '日均气温\n(站点)',
    'T_source_mean': '日均气温\n(源区线)',
    'T_source_mean_anom': '温度距平',
    'PDD_30d': '30 天累积\n正积温',
}


def label(f):
    return LABELS.get(f, f.replace('_', ' ')[:18])


# ---------------- 图1：三条温度线 ----------------
fig, ax = plt.subplots(figsize=(12, 4.2))
ax.plot(d['date'], d['T_port_mean'], lw=0.6, color='#d62728', label='口岸 1852 m')
ax.plot(d['date'], d['T_station_mean'], lw=0.8, color='#1f77b4', label='观测站 4170 m')
ax.plot(d['date'], d['T_source_mean'], lw=0.8, color='#2ca02c', label='源区冰川 5297 m')
ax.axhline(0, color='gray', lw=0.6, ls=':')
ax.axvline(ev, color='k', lw=1.2, ls='--')
ax.set_ylabel('日均气温 (℃)')
ax.set_title('三条温度线：触发在源区、承灾在口岸（2021-2026；虚线为 2026-08-26 事件）')
ax.legend(loc='lower left', fontsize=8, ncol=3)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(C.IMAGE, 'fig1_temperature_lines.png'), dpi=150)
plt.close(fig)

# ---------------- 图2：日尺度峰值 vs 30 天滚动峰值 ----------------
s26 = d[(d['year'] == 2026) & (d['month'].between(*C.MELT_MONTHS))]
pk_day = s26.loc[s26['PDD_source'].idxmax()]
pk_30 = s26.loc[s26['PDD_30d'].idxmax()]
ev_val = float(d.loc[d['date'] == ev, 'PDD_30d'].iloc[0])
melt = d[(d['month'].between(*C.MELT_MONTHS)) & (d['year'].between(*C.CTRL_YEARS))].dropna(subset=['PDD_30d'])
thr = float(melt['PDD_30d'].quantile(0.95))
pct = float((melt['PDD_30d'] < ev_val).mean() * 100)

fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
ax = axes[0]
ax.plot(d['date'], d['PDD_source'], lw=0.8, color='#e377c2', label='日正积温（日尺度）')
ax.plot([pk_day['date']], [pk_day['PDD_source']], 'v', color='#9467bd', ms=8)
ax.annotate('日尺度最热日 %s\n日均 %.1f ℃' % (pk_day['date'].strftime('%Y-%m-%d'), float(pk_day['T_source_mean'])),
            xy=(pk_day['date'], pk_day['PDD_source']),
            xytext=(pk_day['date'] - pd.Timedelta(days=760), pk_day['PDD_source'] - 3),
            fontsize=8.5, arrowprops=dict(arrowstyle='->', lw=0.8))
ax.axvline(ev, color='k', lw=1.1, ls='--')
ax.set_ylabel('日 PDD$_{source}$ (℃·d)')
ax.set_title('2026 融季热力过程：日尺度峰值 %s，事件 %s' % (pk_day['date'].strftime('%m-%d'), ev.strftime('%m-%d')))
ax.legend(fontsize=8, loc='upper left')
ax.grid(alpha=0.25)

ax2 = axes[1]
ax2.plot(d['date'], d['PDD_30d'], lw=0.9, color='#ff7f0e', label='30 天滚动累积')
ax2.axhline(thr, color='#d62728', ls='--', lw=1, label='2021-2025 融季 95 分位 = %.1f' % thr)
ax2.plot([pk_30['date']], [pk_30['PDD_30d']], 'v', color='#9467bd', ms=8)
ax2.annotate('30 天滚动峰值 %.1f\n%s（比日尺度峰值晚 %d 天，属窗口平滑位移）'
             % (pk_30['PDD_30d'], pk_30['date'].strftime('%Y-%m-%d'),
                (pk_30['date'] - pk_day['date']).days),
             xy=(pk_30['date'], pk_30['PDD_30d']),
             xytext=(pk_30['date'] - pd.Timedelta(days=900), pk_30['PDD_30d'] - 40),
             fontsize=8.5, arrowprops=dict(arrowstyle='->', lw=0.8))
ax2.plot([ev], [ev_val], 'o', color='k', ms=6)
ax2.annotate('事件日 %.1f（第 %.0f 百分位）' % (ev_val, pct), xy=(ev, ev_val),
             xytext=(ev - pd.Timedelta(days=700), ev_val - 60), fontsize=8.5,
             arrowprops=dict(arrowstyle='->', lw=0.8))
ax2.axvline(ev, color='k', lw=1.1, ls='--')
ax2.set_ylabel('PDD$_{30d}$ (℃·d)')
ax2.set_title('30 天滚动累积：峰值日期由窗口平滑决定，不能作为"滞后触发"的证据')
ax2.legend(fontsize=8, loc='upper left')
ax2.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(C.IMAGE, 'fig2_pdd.png'), dpi=150)
plt.close(fig)

# ---------------- 图3：Top-5 事件窗对照 ----------------
borda = pd.read_csv(os.path.join(C.RESULT, 'borda_ranking.csv'))
top5 = borda[borda['selected']].head(5)['feature'].tolist()

fig, axes = plt.subplots(1, 5, figsize=(14, 3.6))
for ax, c in zip(axes, top5):
    x = ctrl[c].dropna()
    mu, sd = x.mean(), x.std()
    ax.boxplot((x - mu) / sd, widths=0.5, showfliers=False)
    ax.scatter(np.ones(len(win)) * 1.15, (win[c] - mu) / sd, color='#d62728', zorder=3, s=22)
    ax.axhline(0, color='gray', lw=0.6, ls=':')
    ax.set_xticks([])
    ax.set_title(label(c), fontsize=10)
    if ax is axes[0]:
        ax.set_ylabel('标准化值 (z)')
fig.suptitle('事件前 3 天（红点，08-23~08-25）与 2021-2025 年 8-9 月气候态（箱线）', fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(C.IMAGE, 'fig3_top5_box.png'), dpi=150)
plt.close(fig)

# ---------------- 图4：Borda 条形图（只画去共线后的代表，避免与最终答案错位） ----------------
tab = borda[borda['selected']].head(10).sort_values('borda')
tab = tab.assign(score=tab['borda'].max() + 1 - tab['borda'])
colors = ['#d62728' if f in top5 else '#4c72b0' for f in tab['feature']]
fig, ax = plt.subplots(figsize=(9, 5))
ax.barh(range(len(tab)), tab['score'], color=colors)
ax.set_yticks(range(len(tab)))
ax.set_yticklabels([f[:32] for f in tab['feature']], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel('综合得分（越长越靠前；红色为最终 Top-5）')
ax.set_title('去共线后的特征关联性综合排序（AUC + 事件距平 + 随机森林）')
ax.grid(axis='x', alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(C.IMAGE, 'fig4_borda.png'), dpi=150)
plt.close(fig)
print('图已输出到', C.IMAGE)
