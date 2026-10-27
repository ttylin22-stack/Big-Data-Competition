import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False

from config import DATA, FIGS
OUT = DATA
FIG = FIGS
os.makedirs(FIG, exist_ok=True)
d = pd.read_csv(os.path.join(OUT, 'daily_features.csv'), parse_dates=['date'])
EVENT = pd.Timestamp('2026-08-26')

# 图1
fig, ax = plt.subplots(figsize=(12, 4.2))
ax.plot(d['date'], d['T_port_mean'], lw=0.6, color='#d62728', label='口岸 1852 m')
ax.plot(d['date'], d['T_station_mean'], lw=0.8, color='#1f77b4', label='观测站 4170 m')
ax.plot(d['date'], d['T_source_mean'], lw=0.8, color='#2ca02c', label='源区冰川 5297 m')
ax.axhline(0, color='gray', lw=0.6, ls=':')
ax.axvline(EVENT, color='k', lw=1.2, ls='--')
ax.text(pd.Timestamp('2021-06-01'), 27, '2026-08-26 冰崩—泥石流', fontsize=9, alpha=0)
ax.set_ylabel('日均气温 (℃)')
ax.set_title('三条温度线：触发在源区、承灾在口岸（2021–2026；虚线为 2026-08-26 事件）')
ax.legend(loc='lower left', fontsize=8, ncol=3)
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig1_temperature_lines.png'), dpi=150)
plt.close(fig)

# 图2：PDD + 峰值 + 滞后
s26 = d[(d['year'] == 2026) & (d['month'].between(6, 9))]
ipk = s26['PDD_30d'].idxmax()
pk_date, pk_val = s26.loc[ipk, 'date'], s26.loc[ipk, 'PDD_30d']
ev_val = d.loc[d['date'] == EVENT, 'PDD_30d'].iloc[0]

fig, ax = plt.subplots(figsize=(12, 4.0))
ax.plot(d['date'], d['PDD_30d'], color='#ff7f0e', lw=0.9, label='源区 30 天累积正积温')
thr = d[(d['month'].between(6, 9)) & (d['year'].between(2021, 2025))]['PDD_30d'].quantile(0.95)
ax.axhline(thr, color='#d62728', ls='--', lw=1, label='2021–2025 融季 95 分位 = %.1f ℃·d' % thr)
ax.plot([pk_date], [pk_val], 'v', color='#9467bd', ms=8)
ax.annotate('峰值 %.1f\n%s' % (pk_val, pk_date.strftime('%Y-%m-%d')),
            xy=(pk_date, pk_val), xytext=(pk_date - pd.Timedelta(days=420), pk_val - 12),
            fontsize=8.5, arrowprops=dict(arrowstyle='->', lw=0.8))
ax.plot([EVENT], [ev_val], 'o', color='k', ms=6)
ax.annotate('事件日 %.1f（第 57 百分位）' % ev_val, xy=(EVENT, ev_val),
            xytext=(EVENT - pd.Timedelta(days=760), ev_val - 55), fontsize=8.5,
            arrowprops=dict(arrowstyle='->', lw=0.8))
ax.annotate('', xy=(EVENT, ev_val + 6), xytext=(pk_date, pk_val + 6),
            arrowprops=dict(arrowstyle='<->', lw=1.0, color='#555555'))
ax.text(pk_date - pd.Timedelta(days=200), pk_val + 9, '滞后 29 天', fontsize=9, color='#555555')
ax.set_ylabel('PDD$_{30d}$ (℃·d)')
ax.set_title('热力累积在 7 月底见顶，事件发生在峰值后 29 天')
ax.legend(fontsize=8, loc='upper left')
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig2_pdd.png'), dpi=150)
plt.close(fig)

# 图3
top5 = ['dVWC_mean', 'T_station_min', 'LE(W/m2)_sum', 'H2O_density(g/m3)_mean', 'shear_alpha_mean']
label5 = ['表层-深层\n土壤湿度差', '夜间最低温', '潜热通量\n累积', '绝对湿度', '风切变指数']
ctrl = d[(d['month'].isin([8, 9])) & (d['year'].between(2021, 2025))]
win = d[(d['date'] >= EVENT - pd.Timedelta(days=3)) & (d['date'] <= EVENT)]
fig, axes = plt.subplots(1, 5, figsize=(14, 3.6))
for ax, c, lb in zip(axes, top5, label5):
    x = ctrl[c].dropna()
    mu, sd = x.mean(), x.std()
    ax.boxplot((x - mu) / sd, widths=0.5, showfliers=False)
    ax.scatter(np.ones(len(win)) * 1.15, (win[c] - mu) / sd, color='#d62728', zorder=3, s=22)
    ax.axhline(0, color='gray', lw=0.6, ls=':')
    ax.set_xticks([])
    ax.set_title(lb, fontsize=10)
    if ax is axes[0]:
        ax.set_ylabel('标准化值 (z)')
fig.suptitle('事件前 3 天（红点）与 2021–2025 年 8–9 月气候态（箱线）', fontsize=11)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig3_top5_box.png'), dpi=150)
plt.close(fig)

# 图4
tab = pd.read_csv(os.path.join(OUT, 'borda_ranking.csv')).head(12).sort_values('borda')
tab = tab.assign(score=tab['borda'].max() + 1 - tab['borda'])
fig, ax = plt.subplots(figsize=(9, 5))
colors = ['#d62728' if f in top5 else '#4c72b0' for f in tab['feature']]
ax.barh(range(len(tab)), tab['score'], color=colors)
ax.set_yticks(range(len(tab)))
ax.set_yticklabels([f[:32] for f in tab['feature']], fontsize=8)
ax.invert_yaxis()
ax.set_xlabel('综合得分（越长越靠前；红色为最终 Top-5）')
ax.set_title('特征关联性综合排序（AUC + 事件距平 + 随机森林）')
ax.grid(axis='x', alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(FIG, 'fig4_borda.png'), dpi=150)
plt.close(fig)
print('ok')
