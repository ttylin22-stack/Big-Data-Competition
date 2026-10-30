# 问题 1 交付包：吉隆泥石流与高寒山区气象条件的关联性

> 2026 年 MathorCup 大数据挑战赛 赛道 B · 问题 1
> 从 MET / RADM / SOIL / FLUX 四类逐小时观测构建特征，量化与 2026-08-26 吉隆口岸泥石流的关联性，给出 Top-5 因子。

---

## 一、30 秒结论

| 结论 | 证据 |
|---|---|
| **不是降水触发** | 站址事件前 3 天降水 0.0 mm；但 8-9 月有 61.6% 的日子本就无雨，该证据中性（§3.5） |
| **热力条件中位偏上，不是阈值触发** | 事件日 30 天积温 145.4 ℃·d，处融季第 **57** 百分位（阈值 174.0）；2026 年 8-9 月均温距平 **−0.30 ℃**（偏冷）；热浪日数 0 |
| **事件窗是"晴、暖、干、辐射强"** | VPD 第 95 百分位、净辐射第 87、温度距平 +1.17 ℃（第 89） |
| **深层土壤未获补给** | 0.8 m 含水量 0.108（常年 0.167）；但该层传感器年际跳变 40%–50%，结论需谨慎（§5.2） |
| **能量与水相变活跃但未达异常** | 潜热日累积 +26%（第 71 百分位），不是上一版所称的 +65% |

**问题 1 答案（Top-5）**

1. 表层−深层土壤湿度差（dVWC）
2. 日均气温（区域；三条温度线等价）
3. 绝对湿度（水汽密度）
4. 潜热通量日累积（LE）
5. 风切变指数（α）

其中 1、2 是"事件证据"（事件时确异常），3、4、5 是"机制/独立证据"（事件时并不异常，靠对热天的区分力或独立性入选）。事件异常最强的第二名 VPD 因机制得分低（AUC 0.541）未入选，详见报告 §3.6。

完整推导见同级 `问题1_分析报告.md`。

---

## 二、目录结构

```
Q1/
├── 00_README.md                  ← 本文件
├── 问题1_分析报告.md              完整方法、结果、稳健性与局限
├── code/
│   ├── config.py                 路径 / 事件 / 窗口 / 分位 / 物理常数（唯一参数源）
│   ├── common.py                 事件窗、对照期、代理标签、距平、AUC、去共线
│   ├── 0_build_panel.py          逐小时 → 日尺度特征（归一化聚合）
│   ├── 1_main_analyze.py         分析 A（距平）+ B（AUC）+ bootstrap + 随机森林
│   ├── 2_diagnostics.py          降水/季节上下文、传感器检查、窗口×基线敏感性
│   ├── 3_temp_rank_decollineation.py  非温度榜单 + 贪心去共线
│   ├── 4_borda_rank.py           三准则 Borda 合成（最终排序）
│   └── 5_figures.py              出 4 张图
├── result/
│   ├── daily_features.csv         主表，1 951 天 × 61 列（后续问题 2/3 直接用）
│   ├── case_crossover_ranking.csv 事件窗距平排序
│   ├── borda_ranking.csv          综合排序全表（49 项）
│   ├── top5_stability.csv         7 天块 bootstrap 入选频率
│   ├── rf_importance.csv          随机森林重要性
│   ├── feature_final_table.csv    非温度特征 + 去共线标记
│   ├── proxy_label_ranking_*.csv  代理标签单特征 AUC（all / independent）
│   ├── context.csv                事件窗 / 降水 / 积温差值等上下文指标
│   ├── season_context.csv         逐年 8-9 月热力指标
│   ├── vwc_sensor_check.csv       逐年各层含水量（传感器稳定性）
│   └── sensitivity_table.csv      窗口 × 基线敏感性
├── image/
│   ├── fig1_temperature_lines.png 三条温度线
│   ├── fig2_pdd.png               日尺度峰值 vs 30 天滚动峰值
│   ├── fig3_top5_box.png          Top-5 事件窗对照
│   └── fig4_borda.png             Borda 综合排序
└── temporary/
    ├── washed_and_merged_data.csv 合并清洗后的逐小时表（52 584 行）
    └── hourly_panel.csv           派生后的逐小时面板（46 768 行）
```

---

## 三、怎么复现

环境：Python 3.12+；依赖见 `requirements.txt`

```bash
pip install -r requirements.txt
```

```bash
python code/0_build_panel.py                   # → result/daily_features.csv（约 20 s）
python code/1_main_analyze.py                  # 事件距平 + 代理标签 + bootstrap + RF
python code/2_diagnostics.py                   # 上下文、传感器、敏感性
python code/3_temp_rank_decollineation.py      # 非温度榜单 + 去共线
python code/4_borda_rank.py                    # Borda 排序（必须先于出图）
python code/5_figures.py                       # → image/
```

原始数据默认读 `resources/DataSet_1/`，由 `code/config.py` 以文件自身位置为基准相对定位；换机器只需改 `config.py` 里的 `RESOURCES`。

---

## 四、关键参数在哪改

全部集中在 `code/config.py`，脚本里不再有散落的魔法数字。

| 想改什么 | 改哪里 |
|---|---|
| 事件日期/时刻 | `EVENT_DATE` / `EVENT_TIME` |
| 事件窗长度 | `WIN_START_DAYS` / `WIN_END_DAYS`（默认 −3 / −1，不含事件当天） |
| 对照期月份/年份 | `CTRL_MONTHS` / `CTRL_YEARS` |
| 代理标签分位 | `PROXY_QUANTILE`（默认 0.95） |
| 去共线阈值 | `COLLINEAR_R`（默认 0.7） |
| 递减率、站点/源区/口岸海拔 | `LAPSE` / `Z_STATION` / `Z_SOURCE` / `Z_PORT` |
| 日聚合最少有效小时 | `MIN_VALID_HOURS`（默认 18） |

---

## 五、三条温度线（重要）

数据里只有一个观测站，海拔由气压 618 hPa 反算得 **≈4170 m**。按 6.5 ℃/km 递减率外推：

- 源区冰川（5297 m）：`T_station − 7.3 ℃`
- 口岸（1852 m）：`T_station + 15.1 ℃`

**触发在源区、承灾在口岸**，两者相差 22 ℃，混用会把积温算错一倍。三条线互为常数平移（r=1.0），排序中只保留一条代表。

---

## 六、口径与已知局限（详见报告 §5）

1. **日累积口径**：通量/辐射按有效小时归一化到 24 h（有效小时 < 18 记缺失）。不这样做会把缺测当 0，使 2021-2025 年基线偏低（8-9 月有 20% 的日子整天无 FLUX 观测）。
2. **事件窗不含事件当天**：事件发生于 08-26 10:30，窗口取 08-23~08-25，避免灾后观测混入"前兆"。
3. **单事件**：只有 1 场事件，所有关联均为描述性；未做置换检验，z 值不等于显著性。
4. **代理标签循环性**：标签由 PDD 定义，PDD 自身 AUC=1.000 属必然，已从排序中剔除。
5. **土壤湿度传感器**：0.8 m 层年际跳变 40%–50%，dVWC 的幅度随基线年份在 z=1.45~3.19 摆动；改用 0.1−0.4 m 时异常消失。结论应表述为"可能"，不是确证。
6. **站点代表性**：单站位于谷地，与源区（尼泊尔侧）高差 1.1 km、与口岸高差 2.3 km，降水与风场不能外推。
7. **未覆盖**：天气过程（环流型、锋面）与极端事件重现期需要再分析资料或更长序列。

---

## 七、数据来源

- 逐小时观测：国家青藏高原科学数据中心「西藏吉隆高寒山区地气相互作用逐小时综合观测数据集（2021–2026）」
- 事件与灾情：2026 年 MathorCup 赛道 B 题面（2026-08-26 10:30，冰崩—碎屑流—泥石流—山洪链式灾害）
- 未使用：`DataSet_2`（2005-2025 逐日气温）、`DataSet_3~8`（冰川、DEM、土地利用、历史灾点、人口）留给问题 2/3/4
