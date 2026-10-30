# -*- coding: utf-8 -*-
"""问题3:预报模型、触发时刻 95% 区间、预警时间、覆盖率、空间距离范围"""
from config import RESOURCES, RESULT, IMAGE, INPUT
import os, heapq
import numpy as np
import pandas as pd
import rasterio

OUT = RESULT
os.makedirs(OUT, exist_ok=True)
rng = np.random.default_rng(7)

daily = pd.read_csv(os.path.join(INPUT, 'daily_features.csv'), parse_dates=['date'])
EVENT = pd.Timestamp('2026-08-26')

demf = os.path.join(RESOURCES, 'DataSet_7', 'DEM.tif')
with rasterio.open(demf) as s:
    dem = s.read(1).astype('float64')
    dem[dem == s.nodata] = np.nan
    tr = s.transform
h, w = dem.shape

def fill_sinks(z):
    z = np.nan_to_num(z, nan=np.nanmax(z))
    out = z.copy()
    visited = np.zeros_like(z, bool)
    heap = []
    for i in range(h):
        for j in (0, w - 1):
            heapq.heappush(heap, (out[i, j], i, j)); visited[i, j] = True
    for j in range(w):
        for i in (0, h - 1):
            if not visited[i, j]:
                heapq.heappush(heap, (out[i, j], i, j)); visited[i, j] = True
    while heap:
        e, i, j = heapq.heappop(heap)
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            ni, nj = i + di, j + dj
            if 0 <= ni < h and 0 <= nj < w and not visited[ni, nj]:
                if out[ni, nj] <= e:
                    out[ni, nj] = e + 1e-4
                visited[ni, nj] = True
                heapq.heappush(heap, (out[ni, nj], ni, nj))
    return out

filled = fill_sinks(dem)

def rc(lon, lat):
    r, c = rasterio.transform.rowcol(tr, lon, lat)
    return int(r), int(c)

src = rc(85.51046, 28.29555)     # 冰川舌部（.nc 掩膜最低点）
port = rc(85.37800, 28.27700)    # 吉隆口岸
print('源区（冰川舌部）栅格 %s,高程 %.0f m' % (src, dem[src]))
print('口岸栅格 %s,高程 %.0f m' % (port, dem[port]))

def d8_path(z, start, goal, stop_r=3, max_steps=40000):
    i, j = start
    path = [(i, j)]
    nb = [(1, 0), (1, 1), (0, 1), (-1, 1), (-1, 0), (-1, -1), (0, -1), (1, -1)]
    for _ in range(max_steps):
        if abs(i - goal[0]) <= stop_r and abs(j - goal[1]) <= stop_r:
            break
        best, bz = None, z[i, j]
        for di, dj in nb:
            ni, nj = i + di, j + dj
            if 0 <= ni < h and 0 <= nj < w and z[ni, nj] < bz:
                bz = z[ni, nj]; best = (ni, nj)
        if best is None:
            break
        i, j = best
        path.append((i, j))
    return path


def nearest_on_path(path, point, cell=(1.0, 1.0)):
    """
    路径上离 point 最近的像元：返回 (索引, 米制距离)

    cell = (行方向像元边长, 列方向像元边长)，单位米，用于把行列号换算成米。
    """
    pts = np.asarray(path, dtype=float) * np.asarray(cell, dtype=float)
    target = np.asarray(point, dtype=float) * np.asarray(cell, dtype=float)
    dist = np.linalg.norm(pts - target, axis=1)
    k = int(dist.argmin())
    return k, float(dist[k])


path = d8_path(filled, src, port)
res_x = abs(tr.a) * 111320 * np.cos(np.radians(28.3))
res_y = abs(tr.e) * 110540

# D8 路径会越过口岸继续下切到 DEM 边界，按"离口岸最近的像元"截断
k_port, offset_m = nearest_on_path(path, port, cell=(res_y, res_x))

print(f'\nD8 路径共 {len(path)} 像元，离口岸最近的是第 {k_port} 个'
      f'（横向偏离 {offset_m:.0f} m)')

path = path[: k_port + 1]
zz = np.array([dem[r, c] for r, c in path])
dd = np.concatenate([[0], np.cumsum(np.hypot(np.diff([p[0] for p in path]) * res_y,
                                             np.diff([p[1] for p in path]) * res_x))]) / 1000
_lon0, _lat0 = rasterio.transform.xy(tr, path[0][0], path[0][1])
_lon1, _lat1 = rasterio.transform.xy(tr, path[-1][0], path[-1][1])
straight = np.hypot((_lon1 - _lon0) * res_x / abs(tr.a), (_lat1 - _lat0) * res_y / abs(tr.e)) / 1000
print('沟道（源区→口岸）%d 像元，沿程 %.1f km(直线 %.1f km,曲折度 %.2f)'
      % (len(path), dd[-1], straight, dd[-1] / straight))
print('  起点 %.0f m → 口岸沟底 %.0f m,落差 %.0f m,H/L = %.3f'
      % (zz[0], zz[-1], zz[0] - zz[-1], (zz[0] - zz[-1]) / (dd[-1] * 1000)))
print(f'  注：口岸台地高程 {dem[port]:.0f} m,高出沟底 {dem[port] - zz[-1]:.0f} m,'
      f'位于横向 {offset_m:.0f} m 处')

# ---- 能量线法：v(s) = sqrt(v0^2 + 2g(Δz(s) - μ s)) ----
HL = (zz[0] - zz[-1]) / (dd[-1] * 1000)

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from voellmy import run_voellmy, calibrate_mu

mu_cal = calibrate_mu(zz, dd)
t_cal, v_cal, ok_cal = run_voellmy(zz, dd, mu=mu_cal)
print('\nVoellmy 标定：μ = %.4f(H/L = %.4f)，ξ=1000、h=8 m、入口速度 15 m/s' % (mu_cal, HL))
print('  到达口岸 %s,耗时 %.1f 分钟，峰值速度 %.1f m/s，抵达速度 %.1f m/s'
      % (ok_cal, t_cal[-1] / 60, v_cal.max(), v_cal[-1]))

N = 2000
tau = []
for _ in range(N):
    mu = rng.uniform(mu_cal * 0.75, mu_cal * 1.25)
    t, _, ok = run_voellmy(zz, dd, mu=mu, xi=rng.uniform(600, 2000),
                           hflow=rng.uniform(5, 12), v0=rng.uniform(8, 25))
    if ok:
        tau.append(t[-1] / 60.0)
tau = np.array(tau)
print('到达时间集合:5 - 95%% = %.1f - %.1f 分钟（中位 %.1f,%d/%d 次能到达）'
      % (np.percentile(tau, 5), np.percentile(tau, 95), np.median(tau), len(tau), N))

# ---- 触发时刻 ----
def trigger_date(s, ddf, annual, crit):
    f = np.cumsum(s.values * ddf) / annual
    idx = np.where(f >= crit)[0]
    return s.index[idx[0]] if len(idx) else pd.NaT

pdd_by_year = {y: daily[daily['year'] == y].set_index('date')['PDD_source'] for y in range(2021, 2027)}
pdd_all = daily[daily['year'].between(2021, 2025)]
annual_pdd = pdd_all.groupby('year')['PDD_source'].sum()
DDF0 = 0.649 * 1000 / annual_pdd.mean()
annual_melt = annual_pdd.mean() * DDF0
print('\nDDF = %.2f mm/(℃·d)，年均融水 %.0f mm' % (DDF0, annual_melt))

obs = {y: trigger_date(pdd_by_year[y], DDF0, annual_melt, 0.70) for y in range(2021, 2026)}

print('\n覆盖率回测(留一年外,400 次集合)')
rows, hit = [], 0
for y in range(2021, 2026):
    others = [z for z in range(2021, 2026) if z != y]
    a_mean = annual_pdd[others].mean()
    ddf = 0.649 * 1000 / a_mean
    am = a_mean * ddf
    sims = []
    for _ in range(400):
        ts = trigger_date(pdd_by_year[y], ddf * rng.normal(1, 0.12), am, rng.uniform(0.66, 0.76))
        if pd.notna(ts):
            sims.append(ts)
    lo = pd.Timestamp(np.percentile([t.timestamp() for t in sims], 2.5), unit='s')
    hi = pd.Timestamp(np.percentile([t.timestamp() for t in sims], 97.5), unit='s')
    cov = lo <= obs[y] <= hi
    hit += cov
    rows.append({'year': y, 'obs': obs[y].date(), 'lo': lo.date(), 'hi': hi.date(),
                 'width_days': (hi - lo).days, 'covered': bool(cov)})
    print('  %d: 观测 %s,区间 %s ~ %s(宽 %d 天)%s'
          % (y, obs[y].date(), lo.date(), hi.date(), (hi - lo).days, '命中' if cov else '未覆盖'))
coverage = hit / 5
print('  覆盖率 %d/5 = %.0f%%，平均区间宽度 %.0f 天'
      % (hit, coverage * 100, np.mean([r['width_days'] for r in rows])))
pd.DataFrame(rows).to_csv(os.path.join(OUT, 'p3_coverage.csv'), index=False, encoding='utf-8-sig')

sims = []
for _ in range(3000):
    ts = trigger_date(pdd_by_year[2026], DDF0 * rng.normal(1, 0.12), annual_melt, rng.uniform(0.66, 0.78))
    if pd.notna(ts):
        sims.append(ts)
arr = np.array([t.timestamp() for t in sims])
lo26 = pd.Timestamp(np.percentile(arr, 2.5), unit='s')
hi26 = pd.Timestamp(np.percentile(arr, 97.5), unit='s')
med26 = pd.Timestamp(np.median(arr), unit='s')
print('\n2026 触发时刻 95%% 区间：%s ~ %s(中位 %s,宽 %d 天)'
      % (lo26.date(), hi26.date(), med26.date(), (hi26 - lo26).days))
print('  实际事件 %s,%s区间内(位于第 %.0f 百分位)'
      % (EVENT.date(), '在' if lo26 <= EVENT <= hi26 else '不在',
         np.mean(arr <= EVENT.timestamp()) * 100))
lead = (EVENT - med26).days
print('  中位触发日与事件日相差 %.0f 天（非区间下界）；事件前最近一次"高危窗口"开始日可由模型给出' % lead)

pd.DataFrame({'date': sims}).to_csv(os.path.join(OUT, 'p3_trigger_ensemble.csv'), index=False, encoding='utf-8-sig')
pd.DataFrame({'travel_min': tau}).to_csv(os.path.join(OUT, 'p3_travel_time.csv'), index=False, encoding='utf-8-sig')
pd.DataFrame({'dist_km': dd, 'elev_m': zz}).to_csv(os.path.join(OUT, 'p3_path_profile.csv'),
                                                   index=False, encoding='utf-8-sig')
pd.DataFrame({'dist_km': np.arange(len(v_cal)) * 0.02, 'v_mps': v_cal}).to_csv(
    os.path.join(OUT, 'p3_velocity_profile.csv'), index=False, encoding='utf-8-sig')
np.save(os.path.join(OUT, 'p3_path.npy'), np.array(path))

summary = {
    'path_km': round(float(dd[-1]), 1), 'straight_km': round(float(straight), 1),
    'drop_m': round(float(zz[0] - zz[-1])), 'HL': round(float(HL), 3),
    'travel_p5': round(float(np.percentile(tau, 5)), 1),
    'travel_p50': round(float(np.median(tau)), 1),
    'travel_p95': round(float(np.percentile(tau, 95)), 1),
    'trigger_lo': str(lo26.date()), 'trigger_hi': str(hi26.date()), 'trigger_med': str(med26.date()),
    'coverage': coverage, 'lead_days': int(lead),
}
pd.DataFrame([summary]).T.rename(columns={0: '值'}).rename_axis('指标').to_csv(
    os.path.join(OUT, 'p3_summary.csv'), encoding='utf-8-sig')
print('\n摘要:', summary)