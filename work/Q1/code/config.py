"""Q1 统一配置：路径、事件定义、分析参数、物理常数。

所有脚本只从这里读取常量，改参数只需改本文件，避免同一数字散落在多个脚本里。
"""
import os

# ---------------- 路径（以本文件位置为基准，换机器可直接跑） ----------------
CODE = os.path.dirname(os.path.abspath(__file__))              # .../Q1/code
Q1 = os.path.dirname(CODE)                                     # .../Q1
RESULT = os.path.join(Q1, 'result')
IMAGE = os.path.join(Q1, 'image')
TMP = os.path.join(Q1, 'temporary')
RESOURCES = os.path.abspath(os.path.join(Q1, '..', '..', 'resources'))
DATA1 = os.path.join(RESOURCES, 'DataSet_1')                   # 逐小时观测原始表

# ---------------- 事件定义 ----------------
EVENT_DATE = '2026-08-26'      # 吉隆口岸泥石流发生日（题面）
EVENT_TIME = '10:30'           # 发生时刻（题面）
# 事件窗：事件日 -3 天 ~ 事件日 -1 天，即 08-23 ~ 08-25。
# 刻意不含事件当天，避免把 10:30 之后的"灾后"观测当作前兆。
WIN_START_DAYS = -3
WIN_END_DAYS = -1

CTRL_MONTHS = (8, 9)           # 对照期取汛末 8-9 月
CTRL_YEARS = (2021, 2025)      # 气候态/对照期年份（2026 为事件年，不参与）
MELT_MONTHS = (6, 9)           # 融季
PROXY_QUANTILE = 0.95          # 代理标签分位（融季 PDD30 的上 5%）
COLLINEAR_R = 0.7              # 去共线阈值 |r| > 0.7 视为同一特征簇

# ---------------- 物理常数 ----------------
LAPSE = 0.0065     # 气温递减率 ℃/m
Z_STATION = 4170   # 观测站海拔 m（由气压 618 hPa 反算）
Z_SOURCE = 5297    # 源区中位高程 m（DEM）
Z_PORT = 1852      # 口岸高程 m（DEM）

# ---------------- 数据质量 ----------------
# 日聚合时，通量/辐射类累积量至少需要这么多有效小时，否则记为缺失。
# 原因：sum 会把"缺测"当成"0"，缺测多的日子会被系统性低估。
MIN_VALID_HOURS = 18
SENTINELS = (9999.9, -9999.0)  # 原始数据缺测哨兵值
