# -*- coding: utf-8 -*-
"""问题2（二）：动态热力阈值 + 源区静态异常度 + 综合触发指数与回测 + 尺度敏感性"""
import os
import numpy as np
import pandas as pd
import rasterio
import glob
from rasterio import features
import geopandas as gpd


from config import RESOURCES, INPUT, RESULT, IMAGE

daily = pd.read_csv(os.path.join(INPUT, 'daily_features.csv'), parse_dates=['date'])
EVENT = pd.Timestamp('2026-08-26')

# ---------- 1. DDF 标定 ----------
src = daily[daily['year'].between(2021, 2025)].copy()
pdd_annual = src.groupby('year')['PDD_source'].sum()
pdd_mean = pdd_annual.mean()
DH = 0.649                      # m/a，源区 2015–2020 平均减薄
DDF = DH * 1000 / pdd_mean      # mm w.e./(℃·d)
print('源区年 PDD（2021–2025）: %s' % pdd_annual.round(1).to_dict())
print('年均 PDD = %.1f ℃·d，观测减薄 %.3f m/a → DDF = %.2f mm/(℃·d)' % (pdd_mean, DH, DDF))

annual_melt = pdd_mean * DDF
y26 = daily[daily['year'] == 2026].copy()
y26['cum'] = (y26['PDD_source'] * DDF).cumsum()
ev = y26[y26['date'] == EVENT].iloc[0]
print('\n2026 至事件日累积融水 %.0f mm，占多年平均年融水（%.0f mm）的 %.0f%%'
      % (ev['cum'], annual_melt, ev['cum'] / annual_melt * 100))

rows = []
for yr in range(2021, 2027):
    s = daily[daily['year'] == yr].copy()
    s['cum'] = (s['PDD_source'] * DDF).cumsum()
    rec = {'year': yr}
    for frac in (0.60, 0.70, 0.80):
        hit = s[s['cum'] >= annual_melt * frac]
        rec['date_%d' % int(frac * 100)] = hit['date'].iloc[0] if len(hit) else pd.NaT
    rec['peak_date'] = s.loc[s['PDD_30d'].idxmax(), 'date'] if s['PDD_30d'].notna().any() else pd.NaT
    rows.append(rec)
thr_tab = pd.DataFrame(rows)
print('\n逐年跨过"年融水 60/70/80%"的日期与热力峰值日：')
print(thr_tab.to_string(index=False))
thr_tab.to_csv(os.path.join(RESULT, 'p2_melt_dates.csv'), index=False, encoding='utf-8-sig')

pk_date = y26.loc[y26['PDD_30d'].idxmax(), 'date']
lag = (EVENT - pk_date).days
print('\n2026 热力峰值 %s，事件滞后 %d 天' % (pk_date.date(), lag))

w30 = daily[(daily['date'] > EVENT - pd.Timedelta(days=30)) & (daily['date'] <= EVENT)]
clim = daily[(daily['month'].isin([8, 9])) & (daily['year'].between(2021, 2025))]
print('事件前 30 天：Tmin 均值 %.1f ℃（常年 %.1f），无冻结 %d/30 天'
      % (w30['T_station_min'].mean(), clim['T_station_min'].mean(), int((w30['T_station_min'] > 0).sum())))

# ---------- 2. 源区冰川在缓冲区中的减薄排名 ----------
R3 = os.path.join(RESOURCES, 'DataSet_3')
rgi = gpd.read_file(os.path.join(R3, 'RGI_V6_JilongFlood_Buffer_200km.shp')).to_crs(32645)
rgi['gid'] = np.arange(1, len(rgi) + 1)

f_list = sorted(glob.glob(os.path.join(R3, '13_14_15_rgi60_2015-01-01_2020-01-01', 'dhdt_UTM', '*.tif')))
from rasterio import merge as rmerge
srcs = [rasterio.open(x) for x in f_list]
arr, tr = rmerge.merge(srcs, nodata=-9999, dtype='float32')
for s in srcs:
    s.close()
a = arr[0]
canvas = np.where((a > -100) & (a < 100) & (a != -9999), a, np.nan)

ids = features.rasterize(((g, i) for g, i in zip(rgi.geometry, rgi['gid'])),
                         out_shape=canvas.shape, transform=tr, fill=0, dtype='int32')
ok = np.isfinite(canvas) & (ids > 0)
tot = np.bincount(ids[ok], weights=canvas[ok], minlength=len(rgi) + 1)[1:]
cnt = np.bincount(ids[ok], minlength=len(rgi) + 1)[1:]
rgi['dhdt_1520'] = np.where(cnt > 50, tot / np.maximum(cnt, 1), np.nan)

sub = rgi.dropna(subset=['dhdt_1520']).sort_values('dhdt_1520')
src_ids = ['RGI60-15.04172', 'RGI60-15.04173', 'RGI60-15.04174', 'RGI60-15.04175']
sub['is_src'] = sub['RGIId'].isin(src_ids)
src_mean = sub.loc[sub['is_src'], 'dhdt_1520'].mean()
pct = (sub['dhdt_1520'] < src_mean).mean() * 100
print('\n可算 dhdt 的冰川 %d 条；减薄最快 5 条：' % len(sub))
print(sub.head(5)[['RGIId', 'Area', 'dhdt_1520']].to_string(index=False, float_format=lambda v: '%.3f' % v))
print('源区 4 条冰川平均 dhdt = %.3f m/a，处于第 %.0f 百分位（1 = 减薄最快）' % (src_mean, pct))
sub[['RGIId', 'Area', 'dhdt_1520', 'is_src']].to_csv(os.path.join(RESULT, 'p2_glacier_ranking.csv'),
                                                     index=False, encoding='utf-8-sig')

# ---------- 3. 综合触发指数 ----------
daily['melt_cum'] = np.nan
for yr, g in daily.groupby('year'):
    daily.loc[g.index, 'melt_cum'] = (g['PDD_source'] * DDF).cumsum()
daily['melt_frac'] = daily['melt_cum'] / annual_melt

daily['lag'] = np.nan
for yr, g in daily.groupby('year'):
    if g['PDD_30d'].notna().any():
        pk = g.loc[g['PDD_30d'].idxmax(), 'date']
        daily.loc[g.index, 'lag'] = (g['date'] - pk).dt.days

daily['H'] = np.clip((daily['melt_frac'] - 0.55) / 0.25, 0, 1)
daily['L'] = np.clip((daily['lag'] - 10) / 30, 0, 1) * ((daily['lag'] >= 0) & (daily['lag'] <= 60))
daily['TI'] = daily['H'] * daily['L'] * (daily['PDD_30d'] > 0)

print('\n=== 逐年高危窗口（TI ≥ 0.5）===')
for yr, g in daily.groupby('year'):
    if yr < 2021:
        continue
    al = g[g['TI'] >= 0.5]
    print('  %d: %3d 天  %s ~ %s' % (yr, len(al),
          al['date'].min().date() if len(al) else '-', al['date'].max().date() if len(al) else '-'))
ev = daily[daily['date'] == EVENT].iloc[0]
print('  2026 事件日：melt_frac %.2f，lag %d 天，TI %.2f' % (ev['melt_frac'], ev['lag'], ev['TI']))
print('\n注：TI 是"必要条件"指数，每年 8–9 月都会越过阈值；年度区分靠源区静态异常度。')
daily.to_csv(os.path.join(RESULT, 'p2_daily_index.csv'), index=False, encoding='utf-8-sig')

# ---------- 4. 尺度敏感性 ----------
demf = os.path.join(RESOURCES, 'DataSet_7', 'DEM.tif')
ptdir = os.path.join(RESOURCES, 'DataSet_6')
pts = gpd.read_file(os.path.join(ptdir, '西藏日喀则市吉隆县泥石流灾害区历史灾点数据.shp')).set_crs(4490, allow_override=True)
ax = np.array([float(v) for v in pts['x_coord']])
ay = np.array([float(v) for v in pts['y_coord']])

print('\n=== 空间尺度敏感性 ===')
scale_rows = []
with rasterio.open(demf) as s:
    full = s.read(1).astype('float64')
    full[full == s.nodata] = np.nan
    bx = abs(s.transform.a) * 111320 * np.cos(np.radians(28.4))
    by = abs(s.transform.e) * 110540
    for factor in (1, 3, 8):
        a2 = full[::factor, ::factor]
        gy, gx = np.gradient(np.nan_to_num(a2, nan=np.nanmedian(a2)))
        sl = np.degrees(np.arctan(np.hypot(gx / (bx * factor), gy / (by * factor))))
        vals = []
        for x, y in zip(ax, ay):
            r, c = rasterio.transform.rowcol(s.transform, x, y)
            r //= factor; c //= factor
            if 0 <= r < a2.shape[0] and 0 <= c < a2.shape[1]:
                vals.append(sl[r, c])
        vals = np.array(vals)
        scale_rows.append({'resolution_m': round(bx * factor), 'n': len(vals),
                           'slope_median': float(np.nanmedian(vals)),
                           'slope_p25': float(np.nanpercentile(vals, 25)),
                           'slope_p75': float(np.nanpercentile(vals, 75))})
        print('  ~%3.0f m：灾点坡度中位 %.1f°（IQR %.1f–%.1f）'
              % (bx * factor, np.nanmedian(vals), np.nanpercentile(vals, 25), np.nanpercentile(vals, 75)))
pd.DataFrame(scale_rows).to_csv(os.path.join(RESULT, 'p2_scale_sensitivity.csv'), index=False, encoding='utf-8-sig')
