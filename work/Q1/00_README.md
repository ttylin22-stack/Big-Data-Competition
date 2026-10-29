# 问题 1 交付包：吉隆泥石流与高寒山区气象条件的关联性

> 2026 年 MathorCup 大数据挑战赛 赛道 B · 问题 1
> 内容：从 4 类逐小时观测（MET / RADM / SOIL / FLUX）构建特征、量化关联性、给出 Top-5 因子。
> 本包**自包含**：脚本用相对路径定位数据，换目录、换机器都能直接跑。

---

## 一、30 秒结论

| 结论 | 证据 |
|---|---|
| **不是暴雨触发** | 事件前 72 h 降水 **0.0 mm**；降水在 50 个特征中排第 40、43 位 |
| **是热力累积 + 滞后** | 热力峰值 2026-07-28（181.5 ℃·d，六年最高），事件在 **29 天后**；当天仅 145.4 ℃·d（第 57 百分位） |
| **融水在表层聚集** | 表层 0.1 m 土壤含水量 0.097（常年 0.082），0.8 m 深层 0.108（常年 0.167） |
| **夜间无再冻结** | 事件窗最低温 8.8 ℃（常年 7.6 ℃） |

**问题 1 答案（Top-5）**：

1. 表层−深层土壤湿度差（dVWC）
2. 夜间最低温（源区 Tmin）
3. 潜热通量累积（LE）
4. 绝对湿度（水汽密度）
5. 风切变指数（边界层结构）

完整推导见同级的 `问题1_分析报告.md`。

---

## 二、目录结构

```
Q1/
├── 00_README.md              ← 本文件
├── 问题1_分析报告.md          完整方法、结果表、局限（由原报告目录上移而来）
├── code/
│   ├── config.py             统一路径（改这里就能换数据目录）
│   ├── build_panel.py        逐小时 → 日尺度特征（1 951 天 × 58 指标）
│   ├── analyze.py            事件窗口距平 + 代理标签 AUC + bootstrap
│   ├── analyze2.py           共线性剔除、独立特征筛选
│   ├── analyze3.py           Borda 综合排序（AUC + 距平 + 随机森林）
│   └── figs.py               出 4 张图
├── result/
│   ├── daily_features.csv            主表，后续问题 2/3 直接用
│   ├── case_crossover_ranking.csv    事件前 72 h 的标准化距平排序
│   ├── borda_ranking.csv             综合排序全表（50 个特征）
│   ├── top5_stability.csv            7 天块 bootstrap 入选频率
│   ├── rf_importance.csv             随机森林重要性
│   ├── feature_final_table.csv       去共线后的独立特征表
│   └── proxy_label_ranking_*.csv     代理标签单特征 AUC
├── image/
│   ├── fig1_temperature_lines.png    三条温度线
│   ├── fig2_pdd.png                  热力峰值与 29 天滞后
│   ├── fig3_top5_box.png             Top-5 事件窗对照
│   └── fig4_borda.png                综合排序条形图
└── temporary/
    ├── washed_and_merged_data.csv    合并清洗后的逐小时表（52 584 行 × 63 变量）
    └── hourly_panel.csv              派生物理量后的逐小时面板（46 768 行）
```

---

## 三、怎么复现

环境：Python 3.12；依赖 `numpy pandas scikit-learn matplotlib`

```bash
pip install numpy pandas scikit-learn matplotlib
```

```bash
python code/build_panel.py    # 生成 result/daily_features.csv（约 15 秒）
python code/analyze.py        # 事件距平 + 代理标签 + bootstrap
python code/analyze2.py       # 共线剔除
python code/analyze3.py       # Borda 排序
python code/figs.py           # 出图到 image/
```

原始数据默认读 `resources/DataSet_1/`（由 `code/config.py` 以文件自身位置为基准相对定位）。若数据换了位置，只改 `code/config.py` 里的 `RESOURCES` 一行即可。

---

## 四、关键参数在哪改

| 想改什么 | 改哪里 |
|---|---|
| 事件日期 | `analyze.py` / `analyze3.py` 的 `EVENT = pd.Timestamp(...)` |
| 事件窗口长度 | `EVENT - pd.Timedelta(days=3)`，改成 1 或 7 即为 24 h / 7 天 |
| 对照期范围 | `ctrl = d[(d['month'].isin([8, 9])) & ...]` |
| 递减率（三条温度线） | `build_panel.py` 的 `LAPSE = 0.0065` |
| 源区/口岸海拔 | `build_panel.py` 中 `5297`（源区中位高程）与 `1852`（口岸 DEM 高程） |
| 代理标签分位 | `analyze.py` 中 `quantile(0.95)` |

---

## 五、三条温度线怎么来的（重要）

数据里只有一个观测站，海拔由气压 618 hPa 反算得 **≈4170 m**。按 6.5 ℃/km 递减率外推：

- 源区冰川（5297 m）：`T_station − 7.3 ℃`
- 口岸（1852 m）：`T_station + 15.1 ℃`

**触发在源区、承灾在口岸**，两者相差 22 ℃，混用会直接把 PDD 算错一倍。

---

## 六、已知局限（论文里要写）

1. **样本量**：只有 1 场事件，所有"关联性"都不是监督学习的预测力，而是距平 + 机理一致性。
2. **代理标签循环性**：标签由 PDD 定义，PDD 本身 AUC=1.000 属必然，已从排序中剔除。
3. **尾部不稳定**：bootstrap 显示只有 PDD₃₀/PDD₇ 稳定入选（0.95/0.94），第 3–5 位仅 0.19–0.47。
4. **站点代表性**：站点 4170 m 与口岸 1852 m 相差 2.3 km 高差，温度外推是假设而非实测。

---

## 七、数据来源

- 逐小时观测：国家青藏高原科学数据中心「西藏吉隆高寒山区地气相互作用逐小时综合观测数据集（2021–2026）」
- 日气温：ERA5-Land 日统计（Copernicus CDS），2005–2025，5 县
- 事件与灾情：2026 年 MathorCup 赛道 B 题面

> 本目录即最终版本；早期中间稿已废弃。
