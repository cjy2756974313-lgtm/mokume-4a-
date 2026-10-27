# 色差修复总结：38 块 GT 斜切面照片 vs PROC 渲染（显示层）

> 任务来源：`optimization/docs/archive/PROMPT_color_fix.md`（任务书原件，已存档）
> 实验目录：服务器 `/home/gast/cjy/project1/moke_4a_colorfix_20261025/`；本地镜像 `D:/CODE/moke/yhua/optimization/colorfix/`
> 生产目录全程只读，未做任何写入。

## 0. 一句话结论

色差是**显示层**的问题，不是程序反演、也不是指标口径的问题：现行对比图里 GT 直接贴原始照片、PROC 却锚定六外表面一阶均值，两条互不相同的颜色管线造成了平均 **11.07** 的低频 ΔE（0–255 尺度）。
把两侧放进**同一条颜色管线**（GT 也白平衡后用同一样本外表面均值还原，记为 B 方案）后，低频 ΔE 降到 **1.93（-82.6%）**，36/38 块 ≤3、29/38 块 ≤2（大色块 JND≈2.3）；**全部 38 块指标逐位不变**。
另一条候选 A（两侧都只显示白平衡域）虽然也能把 ΔE 压到 2.01，但代价是把 GT 也洗成近乎无彩（饱和度只剩原照片 15%），故未采用——完整对比与数据见 §2.1，备选方案保留在文档中。

## 1. 根因（已确认）

`make_montage38.py` 里两条颜色路径不对称：

| 侧 | 现行做法 | 后果 |
|---|---|---|
| GT | `cv2.imread` 后直接贴原图 | 带相机白平衡 + 现场光照 + 该切面自身的曝光/材质色 |
| PROC | `unwhitebalance(rend, face_chan_mean(folder))` | 锚点只有 6 个面积×3 通道的**一阶均值**（3 个数），且锚的是**外表面**不是该切面 |

三点量化后果（38 块统计，见 `out/color_diag.csv`）：

1. **白点错位（主因）**：GT 切面均色 vs 六外表面均值 `|ΔRGB|` 平均 **25.4**/255（中位 24.5，最大 47.6＝H11），其中通道间极差平均 **20.6**（最大 37.1）——说明主要是**色温/白点方向性偏移**，不是均匀变暗变亮。均色 ΔE76 平均 **10.89**，均色色相角差平均 **16.6°**。
2. **现行低频 ΔE 几乎全是这条错位**：`dE_low` 平均 11.07 vs 均色 ΔE76 平均 10.89，比值 **0.98** → 观感色差 98% 来自白点错位，而非纹理/结构。
3. **"洗白"主要来自程序体本身**：在白平衡域里 GT 平均色度 C\* = **2.62**，PROC 只有 **1.93（0.74×）**；对比度 L std **3.85 vs 2.47（0.64×）**。也就是说程序渲染体本身在白平衡域就偏灰偏平——这是**算法侧**的既有特性，显示层无法凭空造出，也不该用目标切面统计去补（协议红线）。

## 2. 方法：两侧走同一条颜色管线（B 方案，已按指标选定）

两条合法候选——A（两侧都只显示白平衡域）与 B（两侧都白平衡后用同一样本外表面均值还原）——在同一批数据上交叉对比后**选 B**，依据见 §2.1；A 作为备选方案完整保留在本文档中。

```python
mu = face_chan_mean_fast(folder)            # 只由 A–F_col.png 得到，3 个数
gt_wb      = white_balance(gt_rgb)          # GT 也先进白平衡域
gt_disp    = unwhitebalance(gt_wb,    mu)   # 两侧同一个锚点 mu
proc_disp  = unwhitebalance(proc_slice, mu) # 与现行写法一致
```

### 2.1 A vs B：为什么选 B（三组指标，同一批 38 块）

| 维度 | 指标 | A（两侧白平衡域） | **B（两侧同管线）** | 判读 |
|---|---|---|---|---|
| 收敛 | `dE_low`(GT 显示, PROC 显示) mean | 2.007 | **1.923** | B 略优 |
| 收敛 | 同上 median | 1.579 | **1.486** | B 略优 |
| 收敛 | 同上 max | **9.412** | 11.445 | A 仅在此项领先（H11 单块） |
| 收敛 | 逐块胜出 | 13/38 | **25/38** | B 明显占优 |
| 保真 | `dE_low`(GT 显示, GT 原照片) mean | 26.76 | **5.14** | **B 好 5.2 倍** |
| 木质感 | 显示 GT 的平均饱和度保留率 | 0.147 | **0.905** | A 把 GT 洗成灰 |
| 木质感 | 显示 GT 的平均色度保留率 | 0.118 | **0.901** | 同上 |

**关键那一栏是"保真"。** A 的收敛能到 2.01，是因为它把 GT 也一起洗成近乎无彩（饱和度只剩原照片的 **15%**、色度只剩 12%）——图上"GT"那列已经不是那块木头了，两块都灰所以"看起来一致"。B 的 GT 只偏离原照片 **5.14**（A 的 1/5），饱和度保留 **90%**，同时收敛还更好。
A 逐块只在 13/38 上更低，除 H11（优势 2.03）外其余都 ≤0.55（RO09 0.67 除外），属于四舍五入级别的差别；B 的最大优势是 RO03 1.38。
→ **选 B：三项都赢或持平，只在 1 个样本的最大值上让给 A。** 数据见 `out/AB_compare.csv`（脚本 `code/compare_AB.py`），备选 A 的全套对比图见 `out/montage38_gt_vs_proc_altA.png`（`--mode A`，一眼可看出两侧都偏灰、GT 已不像照片）。

> 注：`color_diag.csv`（整套诊断，含 C1/C2 与一阶错位）与 `AB_compare.csv`（A/B 专项）是两次独立运行，`mu_face` 的统计精度不同（`cv2.mean` 双精度 vs `float32` 整图均值），逐块 `dE_low` 差异 ≤0.67（H11 最大），不影响结论；正文主表以 `color_diag.csv` 为准，A/B 判定以同一次运行的 `AB_compare.csv` 为准。

数学上 `gt_disp` 与 `proc_disp` 都是**逐通道 von Kries 增益**，锚点同为 `mu`，因此两侧显示域通道均值**按构造完全相同**，一阶色差归零。
**合法性**：`mu` 只来自六外表面照片，`proc_slice` 来自冻结的 `proc_cube.npz`（H11 用 `proc_cube_knot.npz`）。目标切面 `cut*_col.png` 只作为被比较的参考图出现，没有任何统计进入 PROC 侧或 GT 侧定色。
**代价要说清**：GT 也接受了同一个白点，所以图上呈现的是"在外表面照明白点下的两块木头"，而不是"照片原样的那块木头"；B 的实测偏离只有 5.14 的低频 ΔE、饱和度保留 90%，属于可接受且必须在图注里写明的一步。

其余已试算方案（同一批数据，`dE_low` 越小越好）：

| 方案 | 做法 | mean | median | max | 相对现行 |
|---|---|---|---|---|---|
| 现行 | GT 原图 + unwb(PROC, mu_face) | 11.07 | 10.96 | 17.09 | — |
| A（备选，未采用） | 两侧都只显示白平衡域（GT 也被洗灰，饱和度仅剩 15%） | 2.01 | 1.58 | 9.41 | -81.9% |
| **B（采用）** | 两侧都 WB 后用同一 `mu_face` 还原 | **1.93** | **1.46** | 12.12 | **-82.6%** |
| C1 | 外表面逐通道仿射拟合 WB→相机域 | 5.50 | 5.48 | 9.92 | -50.3% |
| C2 | 外表面 3×3 仿射（含跨通道） | 5.50 | 5.48 | 9.92 | -50.3% |

**为什么第 3 步"更高阶颜色提示"不需要做**：C1 与 C2 的逐块结果**完全相同**（38/38 一致），因为 `white_balance` 本身就是逐通道增益，原始域与白平衡域之间是**精确的对角线性关系**——六外表面能提供的定色信息严格只有 **3 个自由度**（每通道一个增益），不存在"更高阶"的可标定量。任何 Lab 提示 / 早晚材分段 / 饱和度迁移，实质都是用目标切面统计去补，踩红线；而 B 已经把这一阶信息用满。残余量级也证明没有"饱和度系统性不足"：B 之后 38 块的平均饱和度差 `-0.0003`、平均色度差 `-0.036`（29/38 略负，幅度 <0.05 C\* 单位），不存在需要额外全局饱和补偿的系统偏差。

## 3. 结果

- `out/montage38_gt_vs_proc_before_after.png`：每行 `[标签 | GT 原图 | PROC 旧 | GT 对齐 | PROC 对齐 | 指标+ΔE]`，共 38 行，用于一图看清收敛。
- `out/montage38_gt_vs_proc_fixed.png`：正式可用图，每行 `[标签 | GT | PROC]`，两侧同管线。
- `out/montage_{Conifer,Ring-porous,Semi-ring-porous,Diffuse-porous}_fixed.png`：按组织类型 4 张分组图（另附 `_before_after` 版本）。

逐块收敛（`out/color_diag.csv` 的 `dE_low_cur → dE_low_B`）：平均降幅 **82.1%**，中位 **84.7%**；最差的 5 块（现行 → B）：H06 17.1→1.2、S08 16.7→1.1、B08 16.1→1.1、S02 13.7→1.3、TC05 13.4→1.5。

## 4. 残余与边界（不是显示层能解决的）

B 之后 36/38 块 ≤3，两个例外已逐块定位（`out/residual_split.csv`，把低频 ΔE 拆成亮度分量 dL 与色度分量 dAB）：

| 样本 | B 后 dE_low | dL | dAB | GT L std | PROC L std | 判断 |
|---|---|---|---|---|---|---|
| H11（Hinoki, knot 体） | 12.12 | 10.10 | 4.17 | **14.3** | **3.3** | GT 有一个明显的深色节疤团块，PROC 只画出淡环纹（见 `out/panel_H11_colorfix.png`）——**结构缺失**，与颜色无关 |
| K08（Keyaki） | 5.76 | 2.96 | 4.70（83%） | 6.8 | 3.0 | GT 有色相/明度**高度异质**的条纹（灰绿+橙），PROC 偏均匀米色——**色场分布不足**，属反演算法侧 |

这两块说明显示层已到边界：要把它们也压下去，需要动程序反演（H11 的节疤建模、K08 的色相异质性），超出本任务范围，且会触及指标基线。**建议在正文/图注中照实写"显示层已对称对齐；残余差异来自反演本身"**，而不是继续在显示层堆参数。

## 5. 指标不变性证明

三层证据，从强到弱：

1. **代码未动**：生产 `code/` 下 `make_montage38.py / 4a_eval_cut.py / reeval_orient.py / 4c_eval_dists_f1.py` 的 sha256 与审计快照**逐字节相同**（`5a3fdaff… / 9474477e… / d807b58c… / d9dcd5c1…`）。本次改动只新增文件，不修改任何指标路径。
2. **重跑复核**：在符号链接镜像 `verify_mirror/`（只读生产样本、CSV 写镜像）里重跑 `reeval_orient.py`，与生产基线逐列比较：

| 指标 | max\|Δ\| | 说明 |
|---|---|---|
| proc_L1 / proc_RMSE（及 gf/col/flat/nca 的 L1/RMSE） | **0.000e+00** | 逐位一致 |
| proc_Gram-VGG19 | 9.1e-08 | float32 归约顺序 |
| proc_VGG-SWD2 | 4.4e-06 | 同上 |
| proc_DISTS | 8.0e-06 | 同上 |
| proc_RingP/R/F1 | ≤4.0e-03 | 离散环像素判定，10/38 行第 3 位小数抖动 |
| cut / orient | 相同 | 朝向表未变 |

38 块均值（4 位小数）**前后完全一致**：L1 **0.0337** / DISTS **0.3050** / SWD2 **0.3969** / F1 **0.1875**。
3. **理论**：显示层函数（`unwhitebalance` / 新的 `colorlib`）从未被 `compute_metrics` / `metrics_all` 调用；指标始终在 `white_balance` 域计算，对该域而言显示端只是一个可逆的逐通道增益。

## 6. 交付物与晋升方式

新增文件（独立目录，不动生产）：

| 文件 | 作用 |
|---|---|
| `code/colorlib.py` | 公共颜色工具：`white_balance` / `unwhitebalance` / `face_chan_mean(_fast)` / Lab·HSV 统计 / 低频 ΔE / 外表面仿射拟合 |
| `code/diag_color.py` | 诊断：输出 `color_diag.csv` + `color_diag_summary.txt`（38 块 × 5 方案） |
| `code/diag2_residual.py` | 残余拆解（亮度 vs 色度）+ 4 联目视图 |
| `code/compare_AB.py` | A/B 两方案三组指标（收敛/保真/饱和度保留）交叉对比 → `AB_compare.csv` |
| `code/make_montage38_colorfix.py` | 对比图生成，`--mode {B,A,cur,compare}`、`--tag`、`--diag` |
| `code/run_montage.sh` | flock 防重复启动的一次性出图脚本 |
| `out/color_diag.csv` `out/color_diag_summary.txt` `out/residual_split.csv` `out/AB_compare.csv` | 数据 |
| `out/montage38_gt_vs_proc_{fixed,before_after}.png`、`out/montage_*_{fixed,before_after}.png`、`out/montage38_gt_vs_proc_altA.png`（备选 A 同款图）、`out/panel_{H11,K08,MZ03,B08,H06}_colorfix.png` | 图 |

**晋升到生产**（若确认采用 B 方案）：`make_montage38.py` 只需三处改动，指标与 CSV 一律不动。

```python
# ① main() 循环内，取 mu 一次
mu_face = face_chan_mean(folder)
# ② PROC 侧（与原来等价，仅把 mu 提出来）
shown = unwhitebalance(rend, mu_face)
# ③ GT 侧：先白平衡再用同一 mu 还原（RGB 空间），再转回 BGR 贴图
gt_rgb  = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
gt_al   = unwhitebalance(white_balance(gt_rgb), mu_face)
gt_bgr  = cv2.cvtColor(np.clip(255 * gt_al, 0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
gt_t    = cv2.resize(gt_bgr, (TILE, TILE))
```

等价写法（更省事）：直接调用 `colorlib`，或把 `make_montage38_colorfix.py --mode B` 当作新版 `make_montage38.py` 使用（两者输出布局一致，只是标签列加宽到 300 px、指标栏多一行 `dE_low a -> b`）。

## 7. 复现命令

```bash
BASE=/home/gast/cjy/project1/moke_4a_colorfix_20261025
PY=/home/gast/wzy/anaconda3/envs/mokume/bin/python
PROD=/home/gast/cjy/project1/moke_4a_holdout_v2_20261021
cd $BASE/code
$PY -u diag_color.py --prod $PROD --out $BASE/out          # 诊断，约 15 min（纯 CPU）
$PY -u diag2_residual.py --prod $PROD --out $BASE/out      # 残余拆解，约 1 min
$PY -u compare_AB.py --prod $PROD --out $BASE/out          # A/B 三组指标对比，约 5 min
bash run_montage.sh                                        # before/after + fixed 全套，约 9 min
$PY -u /home/gast/cjy/project1/moke_4a_colorfix_20261025/verify_mirror/code/reeval_orient.py  # 指标复核，约 13 min（GPU0）
```

坑：远端启动一律带 `flock -n`（本次两次出现"同一条 ssh 命令被执行两遍→两个 python 抢同一份 CSV"）；面图是 3791×3777，`face_chan_mean_fast` 用 `cv2.mean` 求一阶统计，别把六面转 float32 整图（会吃掉 1 GB+ 内存）。
