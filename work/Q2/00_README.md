# 问题 2 交付包：触发关键参数的条件区间与评估模型

> 2026 年 MathorCup 大数据挑战赛 赛道 B · 问题 2
> 内容：地形分位区间 → 规模判别 → 物源背景 → 热力三条件窗口 → 时空尺度可靠性。
> 本包自带 `input/daily_features.csv`，脚本用相对路径，可直接运行。

---

## 一、30 秒结论

| 结论 | 数值 |
|---|---|
| 地形分位区间（84 个灾点，**非阈值**） | 沟比降 0.17–0.43、高差 545–2014 m、主沟长 2.1–8.6 km |
| 规模判别（logit，主结果） | 沟比降 **−0.52**、高差 +0.88、主沟长 +1.02，AUC **0.881** |
| 空间易发性（DEM） | AUC **0.836**；5 块划分、有效 4 折 **0.873 ± 0.078** |
| 度日因子（DDF） | 1.12 mm/(℃·d)；**在比例型结论里约掉**，只影响绝对水量 |
| 事件日累积融水 | 480 mm = 年量的 **74%**，滞后峰值 **29 天** |
| 触发类型构成 | 降雨 41 / 冰水混合 25 / 冰雪融水 16 / 冰湖溃决 2（冰雪相关 **51%**） |
| 高危窗口判据 | 融水 ≥ 70% + 滞后 25–60 天 + 有融雪 |
| 2026 年窗口 | **08-23 ~ 08-27**，事件在窗内，**提前 3 天** |
| 源区减薄排名 | −0.717 m/a，缓冲区第 **27 百分位** |
| 高程口径敏感性 | 70% 融水日移动 **≤ 3 天** |

**四个核心判断**：

1. **短陡沟出小灾、长缓沟出大灾** —— 沟比降系数为负，规模由高差与主沟长驱动。
2. **地形分位数不是临界阈值** —— 没有非灾点对照，只能描述分布。
3. **热力判据是季节性窗口** —— 每年 28–36 天，年度区分必须靠物源条件。
4. **源区减薄并不极端**（第 27 百分位），长期减薄只是背景。

---

## 二、目录结构

```
Q2/
├── 00_README.md
├── 问题2_分析报告.md
├── code/
│   ├── config.py        路径配置（自动搜索 resources，可用 BDC_RESOURCES 覆盖）
│   ├── p2_static.py     分位区间 + 触发类型 + 规模 logit + DEM 易发性
│   ├── p2_dynamic.py    DDF + 三条件窗口 + 台站/源区温度 + 高程敏感性
│   └── p2_figs.py       四张图
├── result/
│   ├── p2_terrain_quantiles.csv  分位区间（原 terrain_thresholds）
│   ├── p2_trigger_type.csv       触发类型构成
│   ├── p2_logit_coef.csv         规模判别系数
│   ├── p2_dem_logit.csv          DEM 易发性系数
│   ├── p2_fold_report.csv        空间块每折样本数与 AUC
│   ├── p2_glacier_ranking.csv    冰川减薄排名
│   ├── p2_melt_dates.csv         逐年 60/70/80% 融水日
│   ├── p2_window_table.csv       逐年高危窗口
│   ├── p2_window_tmin.csv        台站/源区夜间最低温
│   ├── p2_sensitivity.csv        高程口径敏感性
│   ├── p2_daily_index.csv        日尺度指标（含 in_window）
│   ├── p2_dem_meta.csv           DEM 尺寸元信息
│   └── p2_scale_sensitivity.csv  分辨率敏感性
├── image/
│   └── p2_fig1–4.png
└── input/
    └── daily_features.csv  问题 1 的日尺度特征表（唯一输入依赖）
```

---

## 三、复现

环境：Python 3.12；`numpy pandas` 必需，`scikit-learn rasterio geopandas matplotlib` 可选（缺失时对应小节自动跳过并提示）。

```bash
pip install numpy pandas scikit-learn rasterio geopandas matplotlib
```

```bash
python code/p2_static.py     # 属性类结果秒出；DEM 部分约 20 s
python code/p2_dynamic.py    # 约 40 s（冰川排名需要 rasterio）
python code/p2_figs.py       # 需要 matplotlib
```

数据根目录默认从 `code/` 向上搜索 `resources`（含 `DataSet_1` 或 `（1）`）；
也可用环境变量指定：

```bat
set BDC_RESOURCES=D:\...\Big_Data_Competition\resources
```

> 图像说明：`image/*.png` 由 `p2_figs.py` 生成。若仓库里的 PNG 仍是旧版
>（图 1 标题含"阈值"、图 3 画的是 0.5 阈值线），说明出图环境缺 matplotlib，
> 请在装好依赖的机器上重跑一次 `python code/p2_figs.py` 覆盖。

---

## 四、关键参数在哪改

| 想改什么 | 位置 | 当前值 |
|---|---|---|
| 伪负样本数量 | `p2_static.py` | 600 |
| 伪负样本缓冲距离 | `p2_static.py` | 1–15 km |
| 观测减薄速率 | `p2_dynamic.py` `GLACIER_DH` | 0.649 m/a |
| 融水比例门槛 | `p2_dynamic.py` `MELT_MIN` | 0.70 |
| 滞后窗 | `p2_dynamic.py` `LAG_MIN/LAG_MAX` | 25 / 60 天 |
| 重采样倍数 | `p2_dynamic.py` | 1 / 3 / 8 |

---

## 五、必须写进论文的局限

1. **没有非灾点对照**：地形只有分位区间，不是临界阈值。
2. **灾点是承灾位置**：DEM 模型找到的是沟口/扇区，不能当作起动位置。
3. **伪负样本影响 AUC**：约束 1–15 km、同高程带后 AUC 从 0.95 降到 0.836。
4. **热力窗口假阳性 100%（按年计）**：不能预报"今年会不会发生"。
5. **2026 窗口只有 5 天**：因为观测止于 08-27，不代表窗口真的短。
6. **未补卫星/天气预报数据**：8 月雨季光学不可用，SAR 超出赛程。
7. **源区减薄不极端**（第 27 百分位），需要短期触发项配合。

---

## 六、与其他包的关系

- 输入 `daily_features.csv` 来自 `../Q1/result/daily_features.csv`（已复制到本包 `input/`）。
- 本包的高危窗口与 DDF 直接支撑 `../Q3` 的触发时刻区间。
