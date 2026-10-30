# 问题 4 论文素材（仿真环境与可视化）

> 用途：论文「问题四」章节的写作底稿，含摘要草稿与正文素材。
> 数据与图件对应 `work/Q4/`；数字与 `result/` 一致。

---

## 一、摘要草稿（约 320 字，可直接改写）

针对泥石流淹没范围与到达时间难以定量的问题，本文构建"**能量线控范围—质量守恒定水深—一维运移定时间**"的准二维仿真框架：以 30 m DEM 为底图，经优先级填洼与 D8 追踪提取沟道，用视摩擦角构造能量线并叠加"高出沟底不超过 2 倍水深（且不小于 5 m）"的地形约束确定淹没边界；沿程体积按指数衰减沉积分配，得到断面水深；到达时间由 Voellmy 速度剖面积分给出。

在释放体积 7.5×10⁶ m³（扩容后）、视摩擦角 0.0905、Voellmy 摩擦系数 0.0057 的参数组合下，仿真得到影响面积 **1.71 km²**、源区最大水深 **10.4 m**、沟道内 **17.0 分钟**抵达口岸段。敏感性分析表明：面积主要由"高出沟底 5 m"的地形阈值控制（体积变化 5 倍仅使面积变化 0.05 km²），而水深与体积近似线性。模型同时指出口岸台地高出沟底 104 m，按现有判据不在淹没范围内；若实际受灾位于台地，需引入漫流或改道机制。

**关键词**：泥石流；能量线；Voellmy 模型；质量守恒；淹没范围；吉隆口岸

---

## 二、正文素材

### 2.1 问题分析

要求构建仿真环境、设定关键参数、给出过程与可视化。数据只有 30 m DEM，没有断面地形与流变参数，事件仅 1 场，因此采用参数少、可校准的**准二维**方案，而非完整浅水流方程。

### 2.2 基本假设

- **H1** 影响范围由能量线（Fahrböschung）控制；
- **H2** 流体沿沟道单向运动，横向按抛物线分布；
- **H3** 体积沿程按指数衰减沉积；
- **H4** 30 m DEM 能代表沟道地形；
- **H5** 不考虑沿途裹挟与堰塞湖溃决。

### 2.3 关键参数

| 参数 | 取值 | 依据 |
|---|---|---|
| 固体体积 | $3.0\times10^6$ m³ | 源区冰量 0.32 km³ 的约 1% |
| 扩容后体积 $V_0$ | $7.5\times10^6$ m³ | 取扩容系数 2.5 |
| 释放位置 | 85.5105°E, 28.2956°N（3619 m） | `.nc` 冰川掩膜最低点 |
| 口岸位置 | 85.3780°E, 28.2770°N（像元 1034, 677，1852 m） | 题面"吉隆口岸"所在；台地高出沟底 104 m |
| 视摩擦角 $\mu_E$ | 0.0905（5.2°） | $\Delta z/L$ |
| Voellmy $\mu$ | 0.0057 | 反标定 |
| 湍流系数 $\xi$ | 1000 m/s² | 文献 |
| 流深 $h$ | 8 m | 文献 |
| 入口速度 $v_0$ | 15 m/s | 上部冰崩段动量 |
| 沉积衰减长度 $L_d$ | 12 km | 使口岸段水深落到 0.3 m 量级 |
| 影响半宽 | $W(s)=60+12s[\mathrm{km}]$ m | 与 84 个灾点扇宽（中位 135 m）同量级 |
| 网格 | 30 m（1816×1148） | DEM 原生 |

### 2.4 模型

#### 2.4.1 沟道

同问题 3：Priority-Flood 填洼 + D8，截断到离口岸最近像元，得 $L=20.7$ km、$\Delta z=1871$ m。

#### 2.4.2 能量线定范围

$$E(s)=z_0-\mu_E s,\qquad \mu_E=\frac{\Delta z}{L}$$

像元判为受影响需同时满足：

$$\underbrace{d\le W(s)}_{\text{横向}},\qquad
\underbrace{z\le E(s)}_{\text{能量线}},\qquad
\underbrace{z-z_{\mathrm{path}}(s)\le \max\{2h(s),5\,\mathrm{m}\}}_{\text{不得爬坡}}$$

第三条是关键：早期版本缺此项，83% 的"淹没像元"位于沟道以上，最高处比沟底高 382 m。

第三条里两个常数都不是拟合值：

- **$2h(s)$**：侧向漫流厚度的上限取两倍流深。取 $1h$ 会把紧邻沟道的低阶地排除，取 $3h$ 以上又会把山坡纳入；
- **5 m**：30 m DEM 在平地上的高程噪声量级（通常 2–5 m），低于此值无法区分"淹没"与"地形起伏"。

#### 2.4.3 质量守恒定水深

$$h(s)=\frac{V_0}{L_d W(s)}\exp\!\left(-\frac{s}{L_d}\right),\qquad
h(x,y)=h(s)\left[1-\left(\frac{d}{W}\right)^2\right]$$

含义：单位长度体积 $\mathrm{d}V/\mathrm{d}s=(V_0/L_d)e^{-s/L_d}$ 除以宽度即断面平均水深。源区 10.4 m，口岸段 0.37 m。

#### 2.4.4 到达时间

由问题 3 的速度剖面按 $1/v$ 积分得累计时间 $t(s)$，再插值到每个像元（早期版本用线性斜坡，已改正）。剖面先做 600 m 滑动平均，消除像素台阶造成的假振荡。

### 2.5 结果

| 指标 | 数值 |
|---|---|
| 影响面积 | 1.71 km² |
| 最大水深 | 10.4 m |
| 口岸台地 | 不在淹没范围内 |
| 到达时间 | 0–17.0 分钟 |
| 沟道长 / 落差 | 20.7 km / 1871 m |

沿程断面：

| $s$ (km) | $h$ (m) | 半宽 (m) |
|---|---|---|
| 0.0 | 10.42 | 60 |
| 3.8 | 4.56 | 105 |
| 7.5 | 2.32 | 150 |
| 13.2 | 0.96 | 218 |
| 16.9 | 0.60 | 263 |
| 20.7 | 0.37 | 308 |

### 2.6 校准与敏感性

**地形约束的效果**

| | 面积 | 高出沟底中位 | 最大 |
|---|---|---|---|
| 无第三条约束 | 8.03 km² | +47 m | +382 m |
| 有约束 | 1.71 km² | −20.5 m | +13.3 m |

**体积敏感性**

| $V_0$ | 面积 | 源区水深 |
|---|---|---|
| 2.5×10⁶ m³ | 1.69 km² | 3.5 m |
| 7.5×10⁶ m³ | 1.71 km² | 10.4 m |
| 12.5×10⁶ m³ | 1.74 km² | 17.4 m |

**结论**：面积由"高出沟底 5 m"这一结构阈值控制（体积变 5 倍面积只变 0.05 km²），水深才随体积线性变化。因此不能把面积当作模型输出。

### 2.7 讨论

1. **口岸台地**：台地高出沟底 104 m、横向 125 m，按现有判据不被淹没。若实际受灾点位于台地，说明发生了漫流、改道或堤防溃决，30 m DEM 无法表达，应作为机制性讨论而非回避。
2. **模型定位**：只回答"到不到、到哪、多久"，不回答冲击力与承灾体损失。
3. **未建模**：堰塞湖溃决、沿途裹挟。

### 2.8 实现细节（供论文取舍）

以下为实现层面的约定，**是否写入论文由论文同学按篇幅决定**，但复核时需要有据可查：

| 项 | 做法 | 说明 |
|---|---|---|
| 经纬度 → 米 | 经度 × 111320·cos(28.3°)，纬度 × 110540 | 研究区纬度跨度小，用局部平面近似 |
| DEM 无效值 | −32767 置为 NaN | 冰川与 DEM 边缘存在无效像元 |
| 像元归属 | KDTree 单邻居：每个像元取到沟道中心线最近点的距离 $d$，并继承该点的沿程距离 $s$ | 保证沿程量连续，不出现断裂 |
| 能量线端点 | $\mu_E$ 由首末点高程差 ÷ 沿程长度直接给出 | 使能量线在口岸端点与地面相切，符合 Fahrböschung 定义 |
| 单元格面积 | 由仿射变换换算，约 **835 m²** | 面积 = 受影响像元数 × 835 m² |
| 沿程统计 | 0–20.7 km 等距 12 个断面，每断面取 ±0.25 km 窗口内最大水深 | 对应 `result/p4_along_path.csv` |
| 速度剖面来源 | 问题 3 的 `p3_velocity_profile.csv`（20 m 步长、600 m 平滑） | 本问不再重复积分 |

---

## 三、图表清单

| 编号 | 文件 | 内容 |
|---|---|---|
| 图 4-1 | `Q4/image/p4_fig1_inundation.png` | 淹没范围与水深（走廊放大） |
| 图 4-2 | `Q4/image/p4_fig2_arrival.png` | 到达时间场 |
| 图 4-3 | `Q4/image/p4_fig3_along.png` | 沿程水深与半宽 |
| 表 4-1 | `Q4/result/p4_along_path.csv` | 沿程断面 |
| 表 4-2 | 正文表 | 参数表、体积敏感性表 |

---

## 四、参考文献

1. O'Callaghan J F, Mark D M. The extraction of drainage networks from digital elevation data. *Computer Vision, Graphics, and Image Processing*, 1984, 28(3): 323–344.
2. Barnes R, Lehman C, Mulla D. Priority-flood: An optimal depression-filling and watershed-labeling algorithm for digital elevation models. *Computers & Geosciences*, 2014, 62: 117–127.
3. Voellmy A. Über die Zerstörungskraft von Lawinen. *Schweizerische Bauzeitung*, 1955, 73: 159–165, 212–217, 246–249, 280–285.
4. Bartelt P, Salm B, Gruber U. Calculating dense-snow avalanche runout using a Voellmy-fluid model with active/passive longitudinal straining. *Journal of Glaciology*, 1999, 45(150): 242–254.
5. Iverson R M. The physics of debris flows. *Reviews of Geophysics*, 1997, 35(3): 245–296.
6. Rickenmann D. Empirical relationships for debris flows. *Natural Hazards*, 1999, 19: 47–77.
7. Hungr O, Leroueil S, Picarelli L. The Varnes classification of landslide types, an update. *Landslides*, 2014, 11: 167–194.
8. Hugonnet R, McNabb R, Berthier E, et al. Accelerated global glacier mass loss in the early twenty-first century. *Nature*, 2021, 592: 726–731.
9. Farinotti D, Huss M, Fürst J J, et al. A consensus estimate for the ice thickness distribution of all glaciers on Earth. *Nature Geoscience*, 2019, 12: 168–173.
10. Gardner A S, Fahnestock M A, Scambos T A. ITS_LIVE regional glacier and ice sheet surface velocities (v2). NSIDC, 2023.

> 卷期页码请在投稿前核对。

---

## 五、写作提示

1. **1.71 km²** 依赖"高出沟底 5 m"这一人为阈值，不是物理定律，必须说明。
2. **口岸不被淹没**与题面"重大伤亡"存在张力，应作为讨论点，而不是隐去。
3. **面积不随体积变**这一点要写清楚，否则读者会以为面积是模型输出。
4. **17.0 分钟**与问题 3 共用同一条速度剖面，两问结论要一致。
5. 堰塞湖溃决未建模，应在"模型评价"中列为缺口。
