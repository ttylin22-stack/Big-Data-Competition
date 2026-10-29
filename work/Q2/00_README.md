# 问题 2 交付包：触发关键参数的临界条件与评估模型

> 2026 年 MathorCup 大数据挑战赛 赛道 B · 问题 2
> 内容：地形临界条件 → 物源背景 → 热力阈值 → 触发评估模型 → 时间/空间尺度可靠性。
> 本包自包含，脚本用相对路径，可直接运行。

---

## 一、30 秒结论

| 结论 | 数值 |
|---|---|
| 地形阈值（84 个灾点） | 沟比降 > 0.17、高差 > 550 m、主沟长 > 2.1 km |
| 规模判别（logit） | 沟比降 **−0.52**、高差 +0.88、主沟长 +1.02，AUC **0.881** |
| 空间易发性（DEM） | AUC **0.836**，空间块 CV **0.873 ± 0.078** |
| 度日因子（DDF） | **1.12 mm/(℃·d)**（由观测减薄 0.649 m/a 反标定） |
| 事件日累积融水 | 480 mm = 年量的 **74%** |
| 热力峰值滞后 | **29 天**（峰值 07-28，事件 08-26） |
| 源区减薄排名 | −0.717 m/a，缓冲区第 **27 百分位** |
| 触发指数回测 | 每年 8–9 月进入窗口 27–36 天，事件日 TI 第 **78 百分位** |

**三个核心判断**：

1. **短陡沟出小灾、长缓沟出大灾** —— 沟比降系数为负，规模由高差与主沟长驱动。
2. **热力指数是"必要条件"而非"充分条件"** —— 每年都会进入窗口，年度区分必须靠物源条件。
3. **源区减薄并不极端**（第 27 百分位），长期减薄只是背景。

---

## 二、目录结构

```
Q2/
├── 00_README.md
├── 问题2_分析报告.md
├── code/
│   ├── config.py        路径配置
│   ├── p2_static.py     84 点地形阈值 + DEM 空间易发性
│   ├── p2_dynamic.py    DDF 标定 + 冰川排名 + 触发指数 + 尺度敏感性
│   └── p2_figs.py       四张图
├── result/
│   ├── p2_terrain_thresholds.csv   分位数阈值
│   ├── p2_logit_coef.csv           规模判别系数
│   ├── p2_dem_logit.csv            DEM 易发性系数
│   ├── p2_blockcv.csv              空间块 CV 结果
│   ├── p2_glacier_ranking.csv      3430 条冰川减薄排名
│   ├── p2_melt_dates.csv           逐年累积融水关键日期
│   ├── p2_daily_index.csv          日尺度触发指数
│   └── p2_scale_sensitivity.csv    分辨率敏感性
├── image/
│   ├── p2_fig1_terrain_thresholds.png
│   ├── p2_fig2_melt_progress.png
│   ├── p2_fig3_ti.png
│   └── p2_fig4_scale.png
└── input/
    └── daily_features.csv  问题 1 的日尺度特征表（唯一输入依赖）
```

---

## 三、复现

环境：Python 3.12，依赖 `numpy pandas scikit-learn matplotlib rasterio geopandas`

```bash
pip install numpy pandas scikit-learn matplotlib rasterio geopandas
```

```bash
python code/p2_static.py     # 约 20 s
python code/p2_dynamic.py    # 约 40 s
python code/p2_figs.py
```

原始数据默认读 `resources/DataSet_3`（冰川）、`DataSet_6`（历史灾点）、`DataSet_7`（DEM），由 `code/config.py` 的 `RESOURCES` 相对定位。

---

## 四、关键参数在哪改

| 想改什么 | 位置 | 当前值 |
|---|---|---|
| 伪负样本数量 | `p2_static.py` | 600 |
| 伪负样本缓冲距离 | `p2_static.py` | 1–15 km |
| 观测减薄速率 | `p2_dynamic.py` `DH` | 0.649 m/a |
| 触发指数阈值 | `p2_dynamic.py` | TI ≥ 0.5 |
| 重采样倍数 | `p2_dynamic.py` | 1 / 3 / 8 |

---

## 五、必须写进论文的局限

1. **灾点是承灾位置**：DEM 模型找到的是沟口/扇区（低 TPI、低坡度），不能当作起动位置。
2. **伪负样本选择影响 AUC**：约束在 1–15 km、同高程带后 AUC 从 0.95 降到 0.836，说明结果对负样本设计敏感。
3. **热力指数假阳性 100%**（按年计）：不能当"今年会不会发生"的预报。
4. **源区减薄不极端**：第 27 百分位，说明还需要短期触发项。
5. **阈值来自 84 个 2020 年解译的吉隆县灾点**：迁移到其他区域需重新标定。
6. **坡度对分辨率极敏感**：8 倍粗化后低估 33%，跨数据集比较前必须统一重采样。

---

## 六、与其他包的关系

- 输入 `daily_features.csv` 来自 `../Q1/result/daily_features.csv`（已复制一份到本包 `input/`）。
- 本包的 DDF 与触发指数直接支撑 `../Q3` 的触发时刻区间。
