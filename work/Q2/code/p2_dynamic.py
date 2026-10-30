<<<<<<< Updated upstream
﻿# -*- coding: utf-8 -*-
"""问题2（二）：热力阈值、源区异常度、触发指数与时空尺度检验

本文件对应复核意见的修订点：
  2-1  触发判据改为报告 §4.3 自己声明的三个必要条件
       （累积融水 ≥ 70% 年量、滞后 25–60 天、当日有融雪），
       并显式输出事件日是否入窗；TI 降级为连续强度指标，不再当"是否触发"的硬阈值。
  2-3  事件前 30 天的夜间最低温同时统计台站与源区，落盘 p2_window_tmin.csv。
  2-5  显式验证 melt_frac = ΣPDD / mean(PDD)，与 DDF 无关。
  2-6  对"融水高程 5297 m vs 释放点 3619 m"补敏感性，落盘 p2_sensitivity.csv。
  1-5  对"台站高程 4170 m vs 标准大气 3981 m"补敏感性。
  2-9  DEM 覆盖范围由仿射变换实算，落盘 p2_dem_meta.csv。

运行环境：numpy / pandas 必需；rasterio / geopandas 只在"冰川排名"与"尺度敏感性"
两节使用，缺失时会跳过并给出提示，不影响其余结果。
"""
=======
# -*- coding: utf-8 -*-
>>>>>>> Stashed changes
import os
import glob

import numpy as np
import pandas as pd

from config import RESOURCES, INPUT, RESULT, IMAGE

# ---------------- 可调参数 ----------------
EVENT = pd.Timestamp('2026-08-26')
GLACIER_DH = 0.649          # m/a，源区 2015–2020 观测减薄
LAPSE = 0.0065              # ℃/m
STATION_ELEV = 4170.0       # 台站海拔（原报告口径；标准大气反算为 3981 m，见敏感性）
SOURCE_ELEV = 5297.0        # 冰川中位高程（消融面）
RELEASE_ELEV = 3619.0       # 冰川舌部（释放点，仅用于说明口径差异）

MELT_MIN = 0.70             # 必要条件①：累积融水 ≥ 70% 年量
LAG_MIN, LAG_MAX = 25, 60   # 必要条件②：距热力峰值的滞后窗
TI_THR = 0.50               # 旧判据（仅保留作对照展示）

DEM_DIR = 'DataSet_7'
PT_DIR = 'DataSet_6'
GLACIER_DIR = 'DataSet_3'
DH_SUBDIR = '13_14_15_rgi60_2015-01-01_2020-01-01'


# =========================================================
# 1. 读入与热力指标
# =========================================================
def load_daily(path=None):
    path = path or os.path.join(INPUT, 'daily_features.csv')
    return pd.read_csv(path, parse_dates=['date']).sort_values('date').reset_index(drop=True)


def calibrate_ddf(daily, dh=GLACIER_DH):
    """用观测减薄反算度日因子。注意：所有比例型结论里 DDF 会约掉（见 verify_ddf_cancels）。"""
    src = daily[daily['year'].between(2021, 2025)]
    pdd_annual = src.groupby('year')['PDD_source'].sum()
    pdd_mean = float(pdd_annual.mean())
    ddf = dh * 1000.0 / pdd_mean
    return {'pdd_annual': pdd_annual, 'pdd_mean': pdd_mean, 'ddf': ddf,
            'annual_melt': pdd_mean * ddf}


def add_progress(daily, ddf, annual_melt):
    """按年累计融水、滞后天数、强度指标 TI 与三条件窗口。"""
    daily = daily.copy()
    daily['melt_cum'] = np.nan
    for _, g in daily.groupby('year'):
        daily.loc[g.index, 'melt_cum'] = (g['PDD_source'] * ddf).cumsum()
    daily['melt_frac'] = daily['melt_cum'] / annual_melt

    daily['lag'] = np.nan
    for _, g in daily.groupby('year'):
        if g['PDD_30d'].notna().any():
            pk = g.loc[g['PDD_30d'].idxmax(), 'date']
            daily.loc[g.index, 'lag'] = (g['date'] - pk).dt.days

    # 连续强度指标（0–1），只用于排序/分级，不作为触发判据
    daily['H'] = np.clip((daily['melt_frac'] - 0.55) / 0.25, 0, 1)
    daily['L'] = np.clip((daily['lag'] - 10) / 30, 0, 1) * daily['lag'].between(0, 60)
    daily['TI'] = daily['H'] * daily['L'] * (daily['PDD_30d'] > 0)

    # 三条件判据（复核意见 2-1）
    daily['in_window'] = (
        (daily['melt_frac'] >= MELT_MIN)
        & daily['lag'].between(LAG_MIN, LAG_MAX)
        & (daily['PDD_30d'] > 0)
    )
    return daily


def melt_date_table(daily, ddf, annual_melt):
    """逐年跨过 60/70/80% 年融水的日期 + 热力峰值日。"""
    rows = []
    for yr in sorted(daily['year'].unique()):
        if yr < 2021:
            continue
        s = daily[daily['year'] == yr]
        cum = (s['PDD_source'] * ddf).cumsum()
        rec = {'year': int(yr)}
        for frac in (0.60, 0.70, 0.80):
            hit = s[cum >= annual_melt * frac]
            rec['date_%d' % int(frac * 100)] = hit['date'].iloc[0] if len(hit) else pd.NaT
        rec['peak_date'] = s.loc[s['PDD_30d'].idxmax(), 'date'] if s['PDD_30d'].notna().any() else pd.NaT
        rows.append(rec)
    return pd.DataFrame(rows)


def window_table(daily):
    """逐年高危窗口（三条件同时满足的日期段）。"""
    rows = []
    for yr, g in daily.groupby('year'):
        if yr < 2021:
            continue
        w = g[g['in_window']]
        rows.append({'year': int(yr), 'n_days': int(len(w)),
                     'start': w['date'].min() if len(w) else pd.NaT,
                     'end': w['date'].max() if len(w) else pd.NaT})
    return pd.DataFrame(rows)


def tmin_table(daily, event=EVENT, days=30):
    """事件前 N 天的夜间最低温：台站 vs 源区（源区 = 台站 − 6.5℃/km × 高差）。"""
    w = daily[(daily['date'] > event - pd.Timedelta(days=days)) & (daily['date'] <= event)]
    rows = []
    for name, col in [('台站 2 m', 'T_station_min'), ('源区 5297 m', 'T_source_min')]:
        s = w[col].dropna()
        rows.append({'layer': name, 'n': int(len(s)), 'mean': float(s.mean()),
                     'min': float(s.min()), 'days_above_zero': int((s > 0).sum())})
    return pd.DataFrame(rows)


def verify_ddf_cancels(daily, ddf, annual_melt):
    """验证 melt_frac 与 DDF 无关：Σ(PDD·DDF)/(mean·DDF) 与 ΣPDD/mean 应当逐日相等。"""
    a = add_progress(daily, ddf, annual_melt)['melt_frac']
    cum = daily.groupby('year')['PDD_source'].cumsum()
    mean = daily[daily['year'].between(2021, 2025)].groupby('year')['PDD_source'].sum().mean()
    b = cum / mean
    return float(np.nanmax(np.abs(a - b)))


# =========================================================
# 2. 高程口径敏感性（2-6 / 1-5）
# =========================================================
def _pdd_at(daily, station_elev, source_elev, lapse=LAPSE):
    t = daily['T_station_mean'] - lapse * (source_elev - station_elev)
    return np.maximum(t, 0.0)


def _annual_mean(daily, pdd):
    m = daily['year'].between(2021, 2025)
    return float(pdd[m].groupby(daily.loc[m, 'year']).sum().mean())


def _frac_date(daily, pdd, mean_pdd, frac, year):
    idx = daily.index[daily['year'] == year]
    cum = pdd.loc[idx].cumsum() / mean_pdd
    hit = daily.loc[idx][cum >= frac]
    return hit['date'].iloc[0] if len(hit) else pd.NaT


def _frac_on(daily, pdd, mean_pdd, date):
    idx = daily.index[daily['year'] == date.year]
    cum = pdd.loc[idx].cumsum() / mean_pdd
    return float(cum[daily.loc[idx, 'date'] == date].iloc[0])


def elevation_sensitivity(daily, event=EVENT):
    """换台站海拔 / 换融水高程，看 70% 融水日与事件日进度怎么动。"""
    variants = [
        ('台站4170_源区5297（基准）', STATION_ELEV, SOURCE_ELEV),
        ('台站3981_源区5297（标准大气）', 3981.0, SOURCE_ELEV),
        ('台站4170_源区4997（−300m）', STATION_ELEV, SOURCE_ELEV - 300),
        ('台站4170_源区5597（+300m）', STATION_ELEV, SOURCE_ELEV + 300),
        ('台站4170_源区3619（释放点）', STATION_ELEV, RELEASE_ELEV),
    ]
    rows = []
    for name, se, src in variants:
        pdd = _pdd_at(daily, se, src)
        mean = _annual_mean(daily, pdd)
        rows.append({'variant': name, 'annual_pdd': round(mean, 1),
                     'date_70pct_2026': _frac_date(daily, pdd, mean, 0.70, 2026),
                     'event_melt_frac': round(_frac_on(daily, pdd, mean, event), 3)})
    return pd.DataFrame(rows)


# =========================================================
# 3. 源区冰川在缓冲区中的减薄排名（需要 rasterio / geopandas）
# =========================================================
def glacier_ranking(resources=RESOURCES):
    try:
        import rasterio
        import geopandas as gpd
        from rasterio import features, merge as rmerge
    except ImportError as e:
        print('[跳过] 冰川排名：缺少依赖（%s）' % e)
        return None

    r3 = os.path.join(resources, GLACIER_DIR)
    rgi = gpd.read_file(os.path.join(r3, 'RGI_V6_JilongFlood_Buffer_200km.shp')).to_crs(32645)
    rgi['gid'] = np.arange(1, len(rgi) + 1)

    f_list = sorted(glob.glob(os.path.join(r3, DH_SUBDIR, 'dhdt_UTM', '*.tif')))
    srcs = [rasterio.open(f) for f in f_list]
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
    print('\n可算 dhdt 的冰川 %d 条；源区 4 条平均 %.3f m/a，处于第 %.0f 百分位'
          % (len(sub), src_mean, pct))
    return sub[['RGIId', 'Area', 'dhdt_1520', 'is_src']]


# =========================================================
# 4. DEM 元信息与尺度敏感性（需要 rasterio）
# =========================================================
def dem_meta(resources=RESOURCES):
    try:
        import rasterio
    except ImportError as e:
        print('[跳过] DEM 元信息：缺少依赖（%s）' % e)
        return None
    demf = os.path.join(resources, DEM_DIR, 'DEM.tif')
    with rasterio.open(demf) as s:
        tr, h, w = s.transform, s.height, s.width
    mx = 111320 * np.cos(np.radians(28.4))
    my = 110540
    return pd.DataFrame([{
        'nrows': h, 'ncols': w,
        'res_x_m': round(abs(tr.a) * mx, 2), 'res_y_m': round(abs(tr.e) * my, 2),
        'width_km': round(w * abs(tr.a) * mx / 1000, 2),
        'height_km': round(h * abs(tr.e) * my / 1000, 2),
        'west': round(tr.c, 6), 'north': round(tr.f, 6),
    }])


def scale_sensitivity(resources=RESOURCES):
    try:
        import rasterio
        import geopandas as gpd
    except ImportError as e:
        print('[跳过] 尺度敏感性：缺少依赖（%s）' % e)
        return None

    demf = os.path.join(resources, DEM_DIR, 'DEM.tif')
    pts = gpd.read_file(os.path.join(resources, PT_DIR,
                                     '西藏日喀则市吉隆县泥石流灾害区历史灾点数据.shp')
                        ).set_crs(4490, allow_override=True)
    ax = np.array([float(v) for v in pts['x_coord']])
    ay = np.array([float(v) for v in pts['y_coord']])

    rows = []
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
                r //= factor
                c //= factor
                if 0 <= r < a2.shape[0] and 0 <= c < a2.shape[1]:
                    vals.append(sl[r, c])
            vals = np.array(vals)
            rows.append({'resolution_m': round(bx * factor), 'n': len(vals),
                         'slope_median': float(np.nanmedian(vals)),
                         'slope_p25': float(np.nanpercentile(vals, 25)),
                         'slope_p75': float(np.nanpercentile(vals, 75))})
    return pd.DataFrame(rows)


# =========================================================
# 5. 主流程
# =========================================================
def main():
    os.makedirs(RESULT, exist_ok=True)

    daily = load_daily()
    cal = calibrate_ddf(daily)
    print('源区年 PDD（2021–2025）:', cal['pdd_annual'].round(1).to_dict())
    print('年均 PDD %.1f ℃·d，观测减薄 %.3f m/a → DDF %.3f mm/(℃·d)，年融水 %.0f mm'
          % (cal['pdd_mean'], GLACIER_DH, cal['ddf'], cal['annual_melt']))

    daily = add_progress(daily, cal['ddf'], cal['annual_melt'])
    ev = daily[daily['date'] == EVENT].iloc[0]
    print('\n事件日 %s：融水 %.1f mm（%.1f%% 年量），滞后 %d 天，TI %.3f'
          % (EVENT.date(), ev['melt_cum'], ev['melt_frac'] * 100, ev['lag'], ev['TI']))

    # --- 2-5：DDF 约掉的自检 ---
    err = verify_ddf_cancels(daily, cal['ddf'], cal['annual_melt'])
    print('DDF 约掉自检：max|melt_frac − ΣPDD/mean| = %.2e（≈0 说明比例与 DDF 无关）' % err)

    # --- 融水日期表 ---
    md = melt_date_table(daily, cal['ddf'], cal['annual_melt'])
    print('\n逐年跨过 60/70/80% 年融水的日期：')
    disp = md.copy()
    for c in ['date_60', 'date_70', 'date_80', 'peak_date']:
        disp[c] = disp[c].dt.strftime('%Y-%m-%d').fillna('-')
    print(disp.to_string(index=False))
    md.to_csv(os.path.join(RESULT, 'p2_melt_dates.csv'), index=False, encoding='utf-8-sig')

    # --- 2-1：三条件窗口 ---
    wt = window_table(daily)
    print('\n=== 高危窗口（融水≥%.0f%% 且滞后 %d–%d 天）==='
          % (MELT_MIN * 100, LAG_MIN, LAG_MAX))
    for _, r in wt.iterrows():
        print('  %d: %3d 天  %s ~ %s' % (r['year'], r['n_days'],
              r['start'].date() if pd.notna(r['start']) else '-',
              r['end'].date() if pd.notna(r['end']) else '-'))
    wt.to_csv(os.path.join(RESULT, 'p2_window_table.csv'), index=False, encoding='utf-8-sig')

    row26 = wt[wt['year'] == 2026].iloc[0]
    in_win = bool(daily.loc[ev.name, 'in_window'])
    print('  事件日是否入窗：%s' % ('是' if in_win else '否'))
    if pd.notna(row26['start']):
        print('  2026 窗口起点 %s，距事件 %d 天（这就是可兑现的预警提前量）'
              % (row26['start'].date(), (EVENT - row26['start']).days))

    # 旧判据对照（TI≥0.5），保留透明性
    for yr, g in daily.groupby('year'):
        if yr >= 2021:
            print('  [对照] %d 年 TI≥%.1f 的天数：%d' % (yr, TI_THR, int((g['TI'] >= TI_THR).sum())))

    # --- 2-3：台站 vs 源区夜间最低温 ---
    tt = tmin_table(daily)
    print('\n事件前 30 天夜间最低温：')
    print(tt.round(2).to_string(index=False))
    tt.to_csv(os.path.join(RESULT, 'p2_window_tmin.csv'), index=False, encoding='utf-8-sig')

    # --- 1-5 / 2-6：高程敏感性 ---
    sens = elevation_sensitivity(daily)
    print('\n高程口径敏感性（2026 年 70% 融水日 / 事件日进度）：')
    print(sens.to_string(index=False))
    sens.to_csv(os.path.join(RESULT, 'p2_sensitivity.csv'), index=False, encoding='utf-8-sig')

    # --- 日尺度结果 ---
    keep = ['date', 'year', 'month', 'PDD_source', 'PDD_7d', 'PDD_30d',
            'T_station_mean', 'T_station_min', 'T_source_mean', 'T_source_min',
            'melt_cum', 'melt_frac', 'lag', 'H', 'L', 'TI', 'in_window']
    daily[keep].to_csv(os.path.join(RESULT, 'p2_daily_index.csv'),
                       index=False, encoding='utf-8-sig')

    # --- 冰川排名 ---
    gr = glacier_ranking()
    if gr is not None:
        gr.to_csv(os.path.join(RESULT, 'p2_glacier_ranking.csv'),
                  index=False, encoding='utf-8-sig')

    # --- 2-9：DEM 元信息 ---
    dm = dem_meta()
    if dm is not None:
        print('\nDEM 覆盖：%.2f × %.2f km（像元 %.2f × %.2f m）'
              % (dm.loc[0, 'width_km'], dm.loc[0, 'height_km'],
                 dm.loc[0, 'res_x_m'], dm.loc[0, 'res_y_m']))
        dm.to_csv(os.path.join(RESULT, 'p2_dem_meta.csv'), index=False, encoding='utf-8-sig')

    # --- 尺度敏感性 ---
    sc = scale_sensitivity()
    if sc is not None:
        print('\n空间尺度敏感性（灾点坡度中位数）：')
        print(sc.round(2).to_string(index=False))
        sc.to_csv(os.path.join(RESULT, 'p2_scale_sensitivity.csv'),
                  index=False, encoding='utf-8-sig')

    print('\n完成：结果写入', RESULT)


if __name__ == '__main__':
    main()
