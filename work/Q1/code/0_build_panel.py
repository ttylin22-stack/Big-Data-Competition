"""0_build_panel.py —— 数据管道：原始逐小时观测 -> 日尺度特征表。

输入：resources/DataSet_1/{MET,RADM,SOIL,FLUX}_Gyirong_YYYY.csv
输出：result/daily_features.csv      日尺度主表（1 951 天）
      temporary/washed_and_merged_data.csv  清洗合并后的逐小时表
      temporary/hourly_panel.csv             派生物理量后的逐小时面板

聚合口径（重要）：
  通量/辐射类（LE、H、Rn、Rsd、Lnet、G）是"能量速率"，日累积必须对时间积分。
  直接用 sum 会把缺测小时当成 0，导致缺测多的日子被系统性低估——而 2021-2025
  的 8-9 月有 20% 的日子整天没有 FLUX 观测。因此这里统一按"有效小时归一化"：
      sum24 = sum / n_valid * 24,  且要求 n_valid >= MIN_VALID_HOURS(18)
  降水是真实累积量，仍用 sum（缺测日返回 NaN 而不是 0）。
"""
import glob
import os

import numpy as np
import pandas as pd

import config as C


# ---------------------------------------------
# ---------------- 处理原始数据 ----------------
# ---------------------------------------------
def load(kind):
    """读取某一类（MET/RADM/SOIL/FLUX）全部年份文件并清洗。"""
    # 先创建所需读取文件的有序列表（glob 返回无序，必须排序保证年份顺序）
    fs = sorted(glob.glob(os.path.join(C.DATA1, '%s_Gyirong_*.csv' % kind)))
    # 再纵向拼接
    df = pd.concat([pd.read_csv(f) for f in fs], ignore_index=True)
    # 哨兵值 -> 缺失
    df = df.replace({v: np.nan for v in C.SENTINELS})
    df['t'] = pd.to_datetime(df['Timestamp'], errors='coerce')
    df = df.dropna(subset=['t']).drop(columns=['Timestamp'])
    # 质量控制列整体丢弃（QC=8 的坏值已由 9999.9 哨兵覆盖）；
    # 更严谨的做法是按 QC 分级赋权，见报告"局限"一节。
    df = df.drop(columns=[c for c in df.columns if c.startswith('QC_')])
    for c in df.columns:
        if c != 't':
            df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.drop_duplicates('t').sort_values('t')


# 将四类数据合并到一起
met = load('MET')
radm = load('RADM')
soil = load('SOIL')
flux = load('FLUX')
print('行数 MET/RADM/SOIL/FLUX:', len(met), len(radm), len(soil), len(flux))
df = (met
      .merge(radm, on='t', how='outer')
      .merge(soil, on='t', how='outer')
      .merge(flux, on='t', how='outer')
      )
df = df.sort_values('t').reset_index(drop=True)
print('合并后', df.shape, df['t'].min(), '→', df['t'].max())

os.makedirs(C.TMP, exist_ok=True)
df.to_csv(os.path.join(C.TMP, 'washed_and_merged_data.csv'), index=False, encoding='utf-8-sig')


# ---------------------------------------------
# ---------------- 逐小时物理量 ----------------
# ---------------------------------------------
# 五层温度/风速，用高度作键，避免反复书写长列名
T = {z: df['Ta_%dm(degree C)' % z] for z in (1, 2, 4, 10, 20)}
WS = {z: df['WS_%dm(m/s)' % z] for z in (1, 2, 4, 10, 20)}
zlev = np.array([1, 2, 4, 10, 20], dtype=float)
Wz = pd.concat([WS[z] for z in (1, 2, 4, 10, 20)], axis=1)

# 温度递减率（1m→20m）与逆温
df['TEMP_lapse_rate'] = (T[1] - T[20]) / 19.0                               # ℃/m，正=随高度递减
df['TEMP_inversion'] = (T[20] - T[1])                                       # >0 逆温
df['TEMP_inversion_flag'] = (df['TEMP_inversion'] > 0.5).astype(float)      # 逆温存在性

# 风切变指数 α：u(z) = u1·(z/z1)^α，取对数后 α 为 ln(ws)~ln(z) 的最小二乘斜率
ws_arr = Wz.values
with np.errstate(all='ignore'):
    ok = (ws_arr > 0.3) & np.isfinite(ws_arr)
    valid_data_num = ok.sum(axis=1)
    logz = np.log(zlev)
    lz = np.where(ok, logz, np.nan)
    lv = np.where(ok, np.log(np.where(ws_arr > 0, ws_arr, np.nan)), np.nan)
    lz_m = np.nanmean(lz, axis=1, keepdims=True)
    lv_m = np.nanmean(lv, axis=1, keepdims=True)
    num = np.nansum((lz - lz_m) * (lv - lv_m), axis=1)
    den = np.nansum((lz - lz_m) ** 2, axis=1)
    alpha = np.where((valid_data_num >= 3) & (den > 1e-6), num / den, np.nan)
df['shear_alpha'] = alpha

# 总体理查森数：浮力抑制 / 风切变产生
# 注：严格定义用位温梯度 ∂θ/∂z（θ = T + 9.8 K/km·z）。这里用实测温度梯度，
#     在"接近中性"时符号可能相反；Ri 未进入最终排序，不影响结论。
g = 9.81
Tm = (T[2] + 273.15)
dTdz = (T[20] - T[1]) / 19.0 * -1.0
dUdz = (WS[20] - WS[1]) / 19.0
df['Ri'] = np.where(dUdz ** 2 > 1e-8, g / Tm * dTdz / (dUdz ** 2), np.nan)
df.loc[df['Ri'].abs() > 10, 'Ri'] = np.nan

# 辐射平衡
df['Rn'] = df['Rsd(W/m2)'] - df['Rsu(W/m2)'] + df['Rld(W/m2)'] - df['Rlu(W/m2)']
df['albedo'] = np.where(df['Rsd(W/m2)'] > 50, df['Rsu(W/m2)'] / df['Rsd(W/m2)'], np.nan)
df['Lnet'] = df['Rld(W/m2)'] - df['Rlu(W/m2)']

# 土壤：深度方向梯度比单层绝对值更有信息量
df['dTsoil'] = df['soilT_0.1m(degree C)'] - df['soilT_1.6m(degree C)']
df['dVWC'] = df['VWC_0.1m(m3/m3)'] - df['VWC_0.8m(m3/m3)']
# 浅层版本：0.8 m 传感器年际跳变明显，用 0.1-0.4 m 做稳健性对照
df['dVWC_shallow'] = df['VWC_0.1m(m3/m3)'] - df['VWC_0.4m(m3/m3)']

# 湍流/湿度
df['Bowen'] = np.where(df['LE(W/m2)'].abs() > 1, df['H(W/m2)'] / df['LE(W/m2)'], np.nan)
df['VPD_kPa'] = df['VPD(hPa)'] / 10.0

# 时间辅助列
df['date'] = df['t'].dt.normalize()
df['month'] = df['t'].dt.month
df['year'] = df['t'].dt.year

# 三条温度线：一个站 -> 三个海拔
df['T_station'] = df['Ta_2m(degree C)']
df['T_source'] = df['T_station'] + C.LAPSE * (C.Z_STATION - C.Z_SOURCE)
df['T_port'] = df['T_station'] + C.LAPSE * (C.Z_STATION - C.Z_PORT)


# ---------------------------------------------
# ------------------- 日尺度 -------------------
# ---------------------------------------------
# 扔掉三个主变量全空的时刻（2021 年 1-4 月整段如此，日表因此从 04-23 开始）
hourly_valid = df[['Ta_2m(degree C)', 'WS_10m(m/s)', 'Rn']].notna().any(axis=1)
df = df[hourly_valid].copy()


def sum24(s):
    """能量速率 -> 日累积：按有效小时归一化到 24 h，缺测过多则返回 NaN。"""
    n = s.count()
    if n < C.MIN_VALID_HOURS:
        return np.nan
    return s.sum() / n * 24


def sum_valid(s):
    """真实累积量（降水）：sum，但全缺测返回 NaN 而不是 0。"""
    return s.sum(min_count=1)


agg = {
    # 状态量：取平均/极值
    'T_station': ['mean', 'max', 'min'],
    'T_source': ['mean', 'max', 'min'],
    'T_port': ['mean', 'max', 'min'],
    'TEMP_lapse_rate': 'mean',
    'TEMP_inversion': ['mean', 'max'],
    'TEMP_inversion_flag': 'sum',
    'shear_alpha': 'mean',
    'Ri': ['mean', 'max'],
    'USTAR(m/s)': ['mean', 'max'],
    'TKE(m2/s2)': ['mean', 'max'],
    'albedo': 'mean',
    'soilT_0.1m(degree C)': 'mean',
    'soilT_1.6m(degree C)': 'mean',
    'dTsoil': 'mean',
    'VWC_0.1m(m3/m3)': 'mean',
    'VWC_0.4m(m3/m3)': 'mean',
    'VWC_0.8m(m3/m3)': 'mean',
    'dVWC': 'mean',
    'dVWC_shallow': 'mean',
    'RH_2m(%)': 'mean',
    'VPD_kPa': 'mean',
    'Pressure(hPa)': 'mean',
    'WS_10m(m/s)': ['mean', 'max'],
    'Bowen': 'mean',
    'H2O_density(g/m3)': 'mean',
    # 能量/通量：日累积，按有效小时归一化
    'Rn': sum24,
    'Rsd(W/m2)': sum24,
    'Lnet': sum24,
    'G_0.1m(W/m2)': sum24,
    'LE(W/m2)': sum24,
    'H(W/m2)': sum24,
    # 降水：真实累积量
    'Precip(mm)': sum_valid,
}
daily = df.groupby('date').agg(agg)
# 展平双层列名：('T_station','mean') -> 'T_station_mean'
daily.columns = ['_'.join([c for c in tup if c]).strip('_') for tup in daily.columns]
daily = daily.rename(columns={'Precip(mm)_sum_valid': 'Precip(mm)_sum'})
daily = daily.reset_index()

# 数据完整性标记：当天有几个小时的实测气温（不作为特征，仅用于质量说明）
n_hours = df.groupby('date')['Ta_2m(degree C)'].count().rename('hours_valid_Ta')
daily = daily.merge(n_hours, left_on='date', right_index=True, how='left')
daily['year'] = daily['date'].dt.year
daily['month'] = daily['date'].dt.month

# 积温：日正积温 -> 7/30 天滚动累加
daily['PDD_source'] = np.maximum(daily['T_source_mean'], 0)
daily['PDD_7d'] = daily['PDD_source'].rolling(7, min_periods=5).sum()
daily['PDD_30d'] = daily['PDD_source'].rolling(30, min_periods=20).sum()
daily['precip_3d'] = daily['Precip(mm)_sum'].rolling(3, min_periods=2).sum()

# 气候态：2021-2025 按日序平均，15 天居中平滑（事件年不参与，避免泄漏）
daily['doy'] = daily['date'].dt.dayofyear
clim = daily[daily['year'].between(*C.CTRL_YEARS)].copy()
smooth = {}
for col in ['T_source_mean', 'T_port_mean', 'T_station_mean']:
    s = clim.groupby('doy')[col].mean()
    s = s.reindex(range(1, 367)).interpolate(limit_direction='both')
    s = s.rolling(15, center=True, min_periods=1).mean()
    smooth[col] = s
for col, s in smooth.items():
    daily['%s_clim' % col] = daily['doy'].map(s)
    daily['%s_anom' % col] = daily[col] - daily['%s_clim' % col]

daily['T_source_anom_7d'] = daily['T_source_mean_anom'].rolling(7, min_periods=5).mean()
daily['heatwave'] = (daily['T_source_anom_7d'] > 2).astype(int)
daily['no_freeze_days_flag'] = (daily['T_source_min'] > 0).astype(int)

# 落盘
os.makedirs(C.RESULT, exist_ok=True)
daily.to_csv(os.path.join(C.RESULT, 'daily_features.csv'), index=False, encoding='utf-8-sig')

# 逐小时面板：中间产物，便于回查。若文件被占用（PermissionError）只提示、不中断主流程。
hourly_path = os.path.join(C.TMP, 'hourly_panel.csv')
try:
    df.to_csv(hourly_path, index=False, encoding='utf-8-sig')
except PermissionError:
    print('提示：%s 被占用，已跳过逐小时面板的写出（主表不受影响）' % hourly_path)
print('日表', daily.shape, '→', os.path.join(C.RESULT, 'daily_features.csv'))
print('列:', list(daily.columns))
print(daily[['date', 'T_source_mean', 'PDD_30d', 'Rn_sum24', 'Precip(mm)_sum']].tail(3).to_string(index=False))
