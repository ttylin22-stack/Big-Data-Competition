import os, glob
import numpy as np
import pandas as pd

from config import RESOURCES, RESULT, TMP
DATA1 = os.path.join(RESOURCES, "DataSet_1")
os.makedirs(RESULT, exist_ok=True)

####处理原始数据####
#载入数据并清洗
def load(kind):
    #先创建所需读取文件的有序列表
    fs = sorted(glob.glob(os.path.join(DATA1, f'{kind}_Gyirong_*.csv')))
    #再创建基于该目录的DataFrame
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    #格式化
    df = df.replace(9999.9, np.nan)
    df['t'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['t']).drop(columns=['Timestamp'])
        ##这个地方把QC全删了，但是有些QC没达到8的地方可能是误差较大的数据，可能要把QC当作数据真实度的权重
    df = df.drop(columns=[c for c in df.columns if c.startswith('QC_')])
    for c in df.columns:
        if c != 't':
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.drop_duplicates('t').sort_values('t')
#将四类数据合并到一起
met = load('MET'); radm = load('RADM'); soil = load('SOIL'); flux = load('FLUX')
print('行数 MET/RADM/SOIL/FLUX:', len(met), len(radm), len(soil), len(flux))
df = met.merge(radm, on='t', how='outer').merge(soil, on='t', how='outer').merge(flux, on='t', how='outer')
df = df.sort_values('t').reset_index(drop=True)
print('合并后', df.shape, df['t'].min(), '→', df['t'].max())


#以下是算法部分
# ---------------- 逐小时物理量 ----------------
T = {z: df[f'Ta_{z}m(degree C)'] for z in (1, 2, 4, 10, 20)}
WS = {z: df[f'WS_{z}m(m/s)'] for z in (1, 2, 4, 10, 20)}
zlev = np.array([1, 2, 4, 10, 20], dtype=float)

Tz = pd.concat([T[z] for z in (1, 2, 4, 10, 20)], axis=1)
Wz = pd.concat([WS[z] for z in (1, 2, 4, 10, 20)], axis=1)

# 温度递减率（1m→20m）与逆温
df['lapse'] = (T[1] - T[20]) / 19.0          # ℃/m，正=随高度递减（正常）
df['inv_str'] = (T[20] - T[1])               # >0 逆温
df['inv_flag'] = (df['inv_str'] > 0.5).astype(float)

# 风切变指数 alpha：log(WS) ~ alpha*log(z)
ws_arr = Wz.values
with np.errstate(all='ignore'):
    ok = (ws_arr > 0.3) & np.isfinite(ws_arr)
    cnt = ok.sum(axis=1)
    logz = np.log(zlev)
    lz = np.where(ok, logz, np.nan)
    lv = np.where(ok, np.log(np.where(ws_arr > 0, ws_arr, np.nan)), np.nan)
    lz_m = np.nanmean(lz, axis=1, keepdims=True)
    lv_m = np.nanmean(lv, axis=1, keepdims=True)
    num = np.nansum((lz - lz_m) * (lv - lv_m), axis=1)
    den = np.nansum((lz - lz_m) ** 2, axis=1)
    alpha = np.where((cnt >= 3) & (den > 1e-6), num / den, np.nan)
df['shear_alpha'] = alpha

# 总体理查森数
g = 9.81
Tm = (T[2] + 273.15)
dTdz = (T[20] - T[1]) / 19.0 * -1.0   # 位温梯度近似，正=稳定
dUdz = (WS[20] - WS[1]) / 19.0
df['Ri'] = np.where(dUdz ** 2 > 1e-8, g / Tm * dTdz / (dUdz ** 2), np.nan)
df.loc[df['Ri'].abs() > 10, 'Ri'] = np.nan

# 辐射
df['Rn'] = df['Rsd(W/m2)'] - df['Rsu(W/m2)'] + df['Rld(W/m2)'] - df['Rlu(W/m2)']
df['albedo'] = np.where(df['Rsd(W/m2)'] > 50, df['Rsu(W/m2)'] / df['Rsd(W/m2)'], np.nan)
df['Lnet'] = df['Rld(W/m2)'] - df['Rlu(W/m2)']

# 土壤
df['dTsoil'] = df['soilT_0.1m(degree C)'] - df['soilT_1.6m(degree C)']
df['dVWC'] = df['VWC_0.1m(m3/m3)'] - df['VWC_0.8m(m3/m3)']
df['G0'] = df['G_0.1m(W/m2)']

# 湍流
df['Bowen'] = np.where(df['LE(W/m2)'].abs() > 1, df['H(W/m2)'] / df['LE(W/m2)'], np.nan)

# 湿度
df['VPD_kPa'] = df['VPD(hPa)'] / 10.0

df['date'] = df['t'].dt.normalize()
df['month'] = df['t'].dt.month
df['year'] = df['t'].dt.year

# ---------------- 三条温度线 ----------------
LAPSE = 0.0065
df['T_station'] = df['Ta_2m(degree C)']
df['T_source'] = df['T_station'] - LAPSE * (5297 - 4170)
df['T_port'] = df['T_station'] + LAPSE * (4170 - 1852)

# ---------------- 日尺度 ----------------
hourly_valid = df[['Ta_2m(degree C)', 'WS_10m(m/s)', 'Rn']].notna().any(axis=1)
df = df[hourly_valid].copy()

agg = {
    'T_station': ['mean', 'max', 'min'],
    'T_source': ['mean', 'max', 'min'],
    'T_port': ['mean', 'max', 'min'],
    'lapse': 'mean', 'inv_str': ['mean', 'max'], 'inv_flag': 'sum',
    'shear_alpha': 'mean', 'Ri': ['mean', 'max'],
    'USTAR(m/s)': ['mean', 'max'], 'TKE(m2/s2)': ['mean', 'max'],
    'Rn': 'sum', 'Rsd(W/m2)': 'sum', 'albedo': 'mean', 'Lnet': 'sum',
    'soilT_0.1m(degree C)': 'mean', 'soilT_1.6m(degree C)': 'mean', 'dTsoil': 'mean',
    'VWC_0.1m(m3/m3)': 'mean', 'VWC_0.8m(m3/m3)': 'mean', 'dVWC': 'mean',
    'G0': 'sum',
    'Precip(mm)': 'sum', 'RH_2m(%)': 'mean', 'VPD_kPa': 'mean',
    'Pressure(hPa)': 'mean', 'WS_10m(m/s)': ['mean', 'max'],
    'LE(W/m2)': 'sum', 'H(W/m2)': 'sum', 'Bowen': 'mean',
    'H2O_density(g/m3)': 'mean',
}
daily = df.groupby('date').agg(agg)
daily.columns = ['_'.join([c for c in tup if c]).strip('_') for tup in daily.columns]
daily = daily.reset_index()
daily['year'] = daily['date'].dt.year
daily['month'] = daily['date'].dt.month

daily['PDD_source'] = np.maximum(daily['T_source_mean'], 0)
daily['PDD_7d'] = daily['PDD_source'].rolling(7, min_periods=5).sum()
daily['PDD_30d'] = daily['PDD_source'].rolling(30, min_periods=20).sum()
daily['precip_3d'] = daily['Precip(mm)_sum'].rolling(3, min_periods=2).sum()

# 气候态（2021-2025，按日序，15 天窗口平滑）
daily['doy'] = daily['date'].dt.dayofyear
clim = daily[daily['year'].between(2021, 2025)].copy()
smooth = {}
for col in ['T_source_mean', 'T_port_mean', 'T_station_mean']:
    s = clim.groupby('doy')[col].mean()
    s = s.reindex(range(1, 367)).interpolate(limit_direction='both')
    s = s.rolling(15, center=True, min_periods=1).mean()
    smooth[col] = s
for col, s in smooth.items():
    daily[f'{col}_clim'] = daily['doy'].map(s)
    daily[f'{col}_anom'] = daily[col] - daily[f'{col}_clim']

daily['T_source_anom_7d'] = daily['T_source_mean_anom'].rolling(7, min_periods=5).mean()

daily['heatwave'] = (daily['T_source_anom_7d'] > 2).astype(int)
daily['no_freeze_days'] = (daily['T_source_min'] > 0).astype(int)

daily.to_csv(os.path.join(RESULT, 'daily_features.csv'), index=False, encoding='utf-8-sig')
df.to_csv(os.path.join(TMP, 'hourly_panel.csv.gz'), index=False, compression='gzip', encoding='utf-8-sig')
print('日表', daily.shape, '→', os.path.join(RESULT, 'daily_features.csv'))
print('列:', list(daily.columns))
print(daily[['date', 'T_source_mean', 'PDD_30d', 'Rn_sum', 'Precip(mm)_sum']].tail(3).to_string(index=False))
