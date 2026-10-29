# 问题 3 交付包：触发时间区间、预警时间、覆盖率与空间范围

> 2026 年 MathorCup 大数据挑战赛 赛道 B · 问题 3
> 内容：预报模型 → 触发时刻 95% 区间 → 覆盖率回测 → 传播时间与空间范围。
> 本包自包含，脚本用相对路径，可直接运行。

---

## 一、30 秒结论

| 结论 | 数值 |
|---|---|
| 触发时刻 95% 区间 | **2026-08-06 ~ 2026-08-26**（中位 08-19，宽 20 天） |
| 事件日位置 | 区间内，第 **98 百分位** |
| 覆盖率（留一年外回测） | **5/5 = 100%**，平均区间宽 **42 天** |
| 沟道长度 / 落差 | 24.8 km / 1998 m，H/L = 0.081 |
| 传播时间（源区→口岸） | **12.7–22.2 分钟**（中位 16.9） |
| 预警提前量 | **7 天**（触发）+ **13–22 分钟**（传播） |
| 空间范围 | 口岸上游 24.8 km 沟道，前 5 km 为主要致灾段 |

**一句话**：模型能给"8 月中到 9 月底"这个窗口，给不了"哪一天"；覆盖率 100% 是靠 42 天宽区间换来的。

---

## 二、目录结构

```
Q3/
├── 00_README.md
├── 问题3_分析报告.md
├── code/
│   ├── config.py        路径配置
│   ├── p3_forecast.py   沟道提取 + Voellmy + 触发区间 + 覆盖率回测
│   ├── voellmy.py       运移模型（含 μ 二分标定）
│   └── p3_figs.py       四张图
├── result/
│   ├── p3_summary.csv             关键指标汇总
│   ├── p3_coverage.csv            留一年外回测
│   ├── p3_trigger_ensemble.csv    3000 次触发时刻集合
│   ├── p3_travel_time.csv         2000 次传播时间集合
│   ├── p3_path_profile.csv        沟道纵剖面
│   ├── p3_velocity_profile.csv    Voellmy 速度剖面
│   └── p3_path.npy                沟道中心线（栅格行列号）
├── image/
│   ├── p3_fig1_trigger_ci.png
│   ├── p3_fig2_coverage.png
│   ├── p3_fig3_profile.png
│   └── p3_fig4_travel.png
└── input/
    └── daily_features.csv  问题 1 的日尺度特征表
```

---

## 三、复现

环境：Python 3.12，依赖 `numpy pandas scikit-learn matplotlib rasterio`

```bash
pip install numpy pandas scikit-learn matplotlib rasterio
```

```bash
python code/p3_forecast.py   # 约 2 分钟（含填洼与两次蒙特卡洛）
python code/p3_figs.py
```

原始数据默认读 `resources/DataSet_7/DEM.tif`（由 `code/config.py` 的 `RESOURCES` 相对定位）。

---

## 四、模型与参数

```
触发时刻 t* = min{ t : Σ PDD·DDF / 年融水 ≥ H_c }
DDF = 1.12 mm/(℃·d)      由 2015–2020 观测减薄 0.649 m/a 标定
年融水 = 649 mm
H_c ∈ [0.66, 0.78]        集合抽样
```

传播模型：

```
Voellmy: a = g sinθ − μ g cosθ − g v²/(ξ h)
μ = 0.0395（二分搜索使流动恰好抵达口岸）
ξ = 1000 m/s²，h = 8 m，v₀ = 15 m/s
```

| 想改什么 | 位置 | 当前值 |
|---|---|---|
| 蒙特卡洛次数 | `p3_forecast.py` | 2000 / 3000 |
| 临界融水比例 | `p3_forecast.py` | 0.66–0.78 |
| Voellmy 参数 | `p3_forecast.py` | ξ=1000, h=8, v₀=15 |
| DDF | `p3_forecast.py` | 1.12 |

---

## 五、必须写进论文的局限

1. **区间不是频率学置信区间**：它是参数不确定性集合，论文里必须明确措辞。
2. **覆盖率必须和区间宽度成对报告**：100% 靠 42 天宽度，只有季节尺度分辨力。
3. **事件落在第 98 百分位**：说明模型捕捉到"偏晚"趋势但低估了触发速度，与问题 2 的"滞后 29 天"一致。
4. **Voellmy 单点标定**：μ 由"恰好抵达"反标定，存在循环论证风险；μ 的 25% 不确定性使传播时间跨度达 1.7 倍。
5. **沟道由 D8 提取**：填洼会削平局部洼地，实际流向可能有偏差。

---

## 六、与其他包的关系

- 输入 `daily_features.csv` 来自 `../Q1/result/`（已复制一份到本包 `input/`）。
- 输出的 `p3_path.npy`、`p3_velocity_profile.csv` 是 `../Q4` 的输入。
