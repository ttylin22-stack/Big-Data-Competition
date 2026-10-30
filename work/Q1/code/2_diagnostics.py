"""2_diagnostics.py —— 支撑报告"局限"与"稳健性"的定量检查。

解决三类"只看主结果看不出来"的问题：
  1) 降水证据的强度：8-9 月本来就有一半以上是无雨日，0 mm 到底说明什么？
  2) 极端热力是否成立：2026 年 8-9 月整体偏暖还是偏冷？热浪标志是否触发？
  3) 0.8 m 土壤湿度传感器的年际稳定性，以及 dVWC 对基线年份/窗口的敏感性。

输出：
  result/context.csv            事件窗与季节的关键上下文指标（键值）
  result/season_context.csv     逐年 8-9 月热力指标
  result/vwc_sensor_check.csv   逐年 8-9 月各层含水量（传感器稳定性）
  result/sensitivity_table.csv  dVWC / LE / VPD 对窗口与基线的敏感性
"""
import os

import numpy as np
import pandas as pd

import common as K
import config as C

d = K.load_daily()
win = K.event_window(d)
ctrl = K.control_period(d)
ev = pd.Timestamp(C.EVENT_DATE)


def window(a, b):
    """以事件日为原点的自定义窗口（a/b 为相对天数）。"""
    return d[(d['date'] >= ev + pd.Timedelta(days=a)) & (d['date'] <= ev + pd.Timedelta(days=b))]


def ctrl_of(y0, y1):
    return d[(d['month'].isin(C.CTRL_MONTHS)) & (d['year'].between(y0, y1))]


# ---------------- 1. 降水证据的强度 ----------------
zero_frac = (ctrl['Precip(mm)_sum'] == 0).mean() * 100
pdd30_event = float(d.loc[d['date'] == ev, 'PDD_30d'].iloc[0])
me, thr = K.build_proxy_label(d)
ctx = [
    ('事件窗天数', str(len(win)), '08-23 ~ 08-25，不含事件当天'),
    ('事件窗日降水合计_mm', float(win['Precip(mm)_sum'].sum()), '站址实测'),
    ('事件日3天累积降水_mm', float(d.loc[d['date'] == ev, 'precip_3d'].iloc[0]), '08-24~08-26；含 08-21~22 的滚动窗均值约 1.2 mm'),
    ('对照期无雨日比例_pct', round(zero_frac, 1), '8-9 月 305 天中日降水为 0 的比例'),
    ('对照期日降水中位数_mm', round(float(ctrl['Precip(mm)_sum'].median()), 2), ''),
    ('事件日PDD30', round(pdd30_event, 2), '℃·d'),
    ('事件日PDD30_百分位', round(float((me['PDD_30d'] < pdd30_event).mean() * 100), 1),
     '相对 2021-2025 融季分布'),
    ('PDD30阈值_p95', round(float(thr), 2), '℃·d'),
]
ctx_df = pd.DataFrame(ctx, columns=['metric', 'value', 'note'])
ctx_df.to_csv(os.path.join(C.RESULT, 'context.csv'), index=False, encoding='utf-8-sig')

# ---------------- 2. 热力背景：逐年 8-9 月 ----------------
rows = []
for y in sorted(d['year'].unique()):
    s = d[(d['year'] == y) & (d['month'].isin(C.CTRL_MONTHS))]
    if not len(s):
        continue
    rows.append({
        'year': y,
        'augsep_T_source_anom': s['T_source_mean_anom'].mean(),
        'augsep_heatwave_days': int(s['heatwave'].sum()),
        'augsep_PDD30_mean': s['PDD_30d'].mean(),
        'season_PDD30_max': d[(d['year'] == y) & (d['month'].between(*C.MELT_MONTHS))]['PDD_30d'].max(),
    })
season = pd.DataFrame(rows).round(2)
season.to_csv(os.path.join(C.RESULT, 'season_context.csv'), index=False, encoding='utf-8-sig')
print('【热力背景】逐年 8-9 月（T 距平 / 热浪日数）')
print(season.to_string(index=False))

early_aug = d[(d['date'] >= ev - pd.Timedelta(days=11)) & (d['date'] <= ev - pd.Timedelta(days=7))]
print('\n事件前 7~11 天（08-15~08-19）温度距平均值 = %.2f ℃' % early_aug['T_source_mean_anom'].mean())
print('事件窗温度距平均值 = %.2f ℃；热浪标志 = %d'
      % (win['T_source_mean_anom'].mean(), int(win['heatwave'].sum())))

# ---------------- 3. 土壤湿度传感器稳定性 ----------------
vwc_cols = ['VWC_0.1m(m3/m3)_mean', 'VWC_0.4m(m3/m3)_mean', 'VWC_0.8m(m3/m3)_mean',
            'dVWC_mean', 'dVWC_shallow_mean']
vrows = []
for y in sorted(d['year'].unique()):
    s = d[(d['year'] == y) & (d['month'].isin(C.CTRL_MONTHS))]
    if not len(s):
        continue
    r = {'year': y}
    r.update({c: s[c].mean() for c in vwc_cols})
    vrows.append(r)
vtab = pd.DataFrame(vrows).round(3)
vtab['VWC_0.8m_yoy_change'] = vtab['VWC_0.8m(m3/m3)_mean'].pct_change() * 100
vtab.to_csv(os.path.join(C.RESULT, 'vwc_sensor_check.csv'), index=False, encoding='utf-8-sig')
print('\n【土壤湿度】逐年 8-9 月各层含水量（0.8 m 年际跳变为传感器稳定性问题）')
print(vtab.to_string(index=False))

# ---------------- 4. 敏感性：窗口 x 基线 ----------------
FEATS = ['dVWC_mean', 'dVWC_shallow_mean', 'VWC_0.8m(m3/m3)_mean',
         'LE(W/m2)_sum24', 'VPD_kPa_mean', 'T_station_min']
WINDOWS = {'08-23~08-25(主)': (-3, -1), '08-22~08-25': (-4, -1), '08-23~08-26': (-3, 0)}
BASELINES = {'2021-2025(主)': (2021, 2025), '2022-2025': (2022, 2025), '2022-2024': (2022, 2024)}
srows = []
for wname, (a, b) in WINDOWS.items():
    w = window(a, b)
    for bname, (y0, y1) in BASELINES.items():
        c = ctrl_of(y0, y1)
        for f in FEATS:
            r = K.anomaly(d, f, w, c)
            if r:
                srows.append({'window': wname, 'baseline': bname, 'feature': f,
                              'z': r['z'], 'pct': r['pct_rank']})
stab = pd.DataFrame(srows)
stab.to_csv(os.path.join(C.RESULT, 'sensitivity_table.csv'), index=False, encoding='utf-8-sig')
print('\n【敏感性】dVWC 的 z 在不同窗口/基线下：')
print(stab[stab['feature'] == 'dVWC_mean'].round(2).to_string(index=False))
print('\n【敏感性】LE 的 z 在不同窗口/基线下：')
print(stab[stab['feature'] == 'LE(W/m2)_sum24'].round(2).to_string(index=False))
print('\n上下文指标已写入', os.path.join(C.RESULT, 'context.csv'))
