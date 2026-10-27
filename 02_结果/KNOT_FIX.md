# KNOT_FIX.md — 节疤渲染修复与 180° 朝向核查（2026-10-23）

> 服务器：`/home/gast/cjy/project1/moke_4a_holdout_v2_20261021/code`
> 背景：2026-10-22 发现 4a 渲染端节疤被硬编码关闭（`4a_proc.py:751` `PhiParams(knot=False)`，全部 `show_knot=False`），且 H11 节疤存在 180° 朝向错位。本次完成修复与验证。

## 一、修复内容

1. **4a_proc.py 加 `-knot` CLI 开关**（0/1，默认 0，不影响主表链路）：
   - `PhiParams(knot=use_knot)`；knot 组参数（deform 8 维 + color_bar 32×3 + ani）进 Stage D 优化器；
   - `show_knot` 贯通 `run_optimize` / `grid_search_sizes` / `build_volume` / 外表面渲染与误差对照；
   - **接续生长场已拟合的节疤参数**：phi 初值按 `knot_deformations` / `knot_color_bar-0.5` / `(ani-0.5)/5` 的逆映射设置，保证首次 apply 后渲染与 gf 拟合结果一致（不从零开始）；
   - knot 模式输出加 `_knot` 后缀（`proc_cube_knot.npz` 等），**不覆盖主表结果**；
   - 守卫：gf_params 无节疤参数时明确报错（需先拷掩膜重跑 Step3）。
2. **掩膜归档**：官方仓库 H11 的 `C_knot.png`/`F_knot.png` 已拷入 v2 `Samples/H11/`。H11 的 `gf_params.pkl` 本身已含完整节疤参数（knot_origin/direction/density/deformations 均为拟合值，Step3 当时有掩膜），**无需重跑 Step3**。CN03 官方 Samples 无 `*_knot.png`（昨日记忆有误：38 块中仅 H11 有节疤掩膜）。

## 二、H11 节疤消融（cut2，白平衡域；`eval_knot_h11.json` / `proc_vis/knot_ablation.png`）

| 配置 | L1 ↓ | VGG-SWD2 ↓ | DISTS ↓ | Gram ↓ |
|---|---|---|---|---|
| knot OFF rot0（主表现状） | 0.1009 | 0.3189 | 0.4295 | 0.00016 |
| knot OFF rot180 | 0.0919 | 0.3194 | 0.4070 | 0.00016 |
| knot ON rot0 | 0.1074 | 0.3143 | 0.4135 | 0.00016 |
| **knot ON rot180** | **0.0828** | **0.2988** | **0.3689** | **0.00014** |

- **knot ON + rot180 全面最优**：L1 −18%、SWD2 −6.3%、DISTS −14.1%（相对主表现状）。四联图确认节疤环形结构出现在与照片一致的位置（左下）。
- 因素分解：朝向修正贡献 L1 −0.0090；正确朝向下开节疤再贡献 −0.0091。knot ON rot0 反而变差（0.1074）——错位节疤是负贡献，印证朝向问题必须先修。

## 三、全 38 块 180° 朝向核查（`check_rot180.py` → `rot180_check.json`）

- **6/38 样本 rot180 显著更优**（delta>0.002）：SG16（L1 0.0763→0.0378，**几乎减半**）、H11（−0.0090）、H14（−0.0068）、K08（−0.0059）、SG15（−0.0030）、RO09（−0.0023）。
- 其余 32 块 |delta|<0.002（噪声级），且 13 块 rot0 明确更优（IC04 −0.0146 等）→ **非系统性翻转，是逐块朝向二义性**（cut 切面照片拍摄朝向与体坐标系的每块组合不同；结构张量法对 180° 不敏感，故 check_orient.py 发现不了）。

## 四、待决策（评价口径，需与组里统一）

1. **样本级朝向表**：建议以"渲染与 GT 的结构对应"逐块登记 rot0/rot180（6 块需翻转），写入 `notes/eval_geometry.md`，**所有方法（PROC/NCA/col/gf/flat 及后续基线）共用同一朝向表**，不得按方法各自选优。主表是否按朝向表重评 38 块待定（预计仅 6 块指标变化，SG16 变化最大）。
2. H11 正式结果建议采用 knot ON + 朝向表（当前最优配置），并在表 4 节疤实验中说明。

## 五、遗留

- CN03 无节疤掩膜，其 cut2 的 180° 嫌疑（delta +0.001，噪声级）无需处理。
- NCA 侧如需表 4 对比，可用 `4b_nca_train.py` 权重 + 相同朝向表重渲。
