import os

import numpy as np
import pandas as pd

from config import RESOURCES, INPUT, RESULT, IMAGE
PT_DIR = 'DataSet_6'
PT_SHP = '西藏日喀则市吉隆县泥石流灾害区历史灾点数据.shp'
EVENT = pd.Timestamp('2026-08-26')


def main():
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as e:
        print('[跳过] 出图：缺少 matplotlib（%s）' % e)
        return
    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei']
    plt.rcParams['axes.unicode_minus'] = False

    os.makedirs(IMAGE, exist_ok=True)
    d = pd.read_csv(os.path.join(RESULT, 'p2_daily_index.csv'), parse_dates=['date'])
    ev = d[d['date'] == EVENT].iloc[0]
    base = d[(d['month'].between(6, 9)) & (d['year'].between(2021, 2025))]['TI'].dropna()
    print('事件日 TI = %.2f，融季（2021–2025）第 %.0f 百分位'
          % (ev['TI'], (base < ev['TI']).mean() * 100))
    print('事件日 melt_frac = %.2f，lag = %d 天，入窗 = %s'
          % (ev['melt_frac'], ev['lag'], bool(ev['in_window'])))

    # 图1 地形分位区间
    try:
        import geopandas as gpd
        pts = gpd.read_file(os.path.join(RESOURCES, PT_DIR, PT_SHP))
    except ImportError:
        from p2_static import load_points
        pts = load_points()

    for c in ['沟比降', '高差', '主沟长']:
        pts[c] = pd.to_numeric(pts[c], errors='coerce')

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for ax, c, unit in zip(axes, ['沟比降', '高差', '主沟长'], ['', 'm', 'm']):
        v = pts[c].dropna()
        ax.hist(v, bins=18, color='#4c72b0', alpha=0.85, edgecolor='w')
        for q, ls in [(0.25, ':'), (0.5, '--'), (0.75, ':')]:
            ax.axvline(v.quantile(q), color='#d62728', ls=ls, lw=1)
        ax.set_title('%s（中位 %.2f %s）' % (c, v.median(), unit), fontsize=10)
        ax.set_xlabel(c)
        ax.set_ylabel('灾点数')
        ax.grid(alpha=0.2)
    fig.suptitle('84 个历史灾点的地形属性分布与四分位区间（非临界阈值）', fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGE, 'p2_fig1_terrain_thresholds.png'), dpi=150)
    plt.close(fig)

    # 图2 逐年累积融水
    fig, ax = plt.subplots(figsize=(11, 4))
    for yr, g in d.groupby('year'):
        if yr < 2021:
            continue
        ax.plot(g['date'].dt.dayofyear, g['melt_frac'] * 100, lw=1.0,
                label='%d' % yr, alpha=0.9, color='#d62728' if yr == 2026 else '#4c72b0')
    ax.axhline(70, color='k', ls=':', lw=1)
    ax.text(95, 72, '必要条件 70%', fontsize=9)
    ax.axvline(EVENT.dayofyear, color='k', ls='--', lw=1.0, alpha=0.6)
    ax.set_xlim(90, 300)
    ax.set_xlabel('日序')
    ax.set_ylabel('累积融水 / 多年平均年融水 (%)')
    ax.set_title('逐年累积融水进程：2026 年事件日达 %.0f%%' % (ev['melt_frac'] * 100))
    ax.legend(fontsize=8, ncol=6, loc='lower right')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGE, 'p2_fig2_melt_progress.png'), dpi=150)
    plt.close(fig)

    # 图3 高危窗口（三条件）+ TI 强度
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.fill_between(d['date'], 0, 1, where=d['in_window'].astype(bool),
                    transform=ax.get_xaxis_transform(), color='#d62728',
                    alpha=0.15, step='mid', label='高危窗口（融水≥70%% 且滞后 25–60 天）')
    ax.plot(d['date'], d['TI'], color='#ff7f0e', lw=0.8, label='TI（强度，非判据）')
    ax.axvline(EVENT, color='k', ls='--', lw=1.2, label='2026-08-26')
    ax.set_ylabel('TI / 窗口')
    ax.set_title('高危窗口：三条件判据（红色阴影）与 TI 强度曲线')
    ax.legend(fontsize=8, loc='upper left')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGE, 'p2_fig3_ti.png'), dpi=150)
    plt.close(fig)

    # 图4 尺度敏感性
    sc = pd.read_csv(os.path.join(RESULT, 'p2_scale_sensitivity.csv'))
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    x = np.arange(len(sc))
    ax.errorbar(x, sc['slope_median'],
                yerr=[sc['slope_median'] - sc['slope_p25'],
                      sc['slope_p75'] - sc['slope_median']],
                fmt='o-', capsize=4, color='#2ca02c')
    ax.set_xticks(x)
    ax.set_xticklabels(['%d m' % r for r in sc['resolution_m']])
    ax.set_xlabel('DEM 分辨率')
    ax.set_ylabel('灾点处坡度（°）')
    ax.set_title('空间尺度敏感性：粗化后坡度系统性偏低')
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(os.path.join(IMAGE, 'p2_fig4_scale.png'), dpi=150)
    plt.close(fig)

    print('图已输出：', os.listdir(IMAGE))


if __name__ == '__main__':
    main()