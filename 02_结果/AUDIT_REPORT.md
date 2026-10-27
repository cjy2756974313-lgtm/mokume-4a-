# AUDIT_REPORT.md — Mokume 4a 严格留出协议审计

> 审计时间：2026-10-21 11:20–13:00（GMT+8）
> 审计对象：`/home/gast/cjy/project1/moke_4a_20261019/code`（**仅只读访问，未修改**）
> 服务器：`server-r740`（gast@47.98.131.85:10001）
> 解释器：`/home/gast/wzy/anaconda3/envs/mokume/bin/python`（Python 3.11 / torch 2.7.1+cu118）
> 清单：`/home/gast/cjy/project1/moku/data/external/mokume_slanted38/manifests/test_sample_ids.txt`

**审计结论（修复前）：4 项硬性违规成立 → 38/38 严格留出协议不成立。**
详细信息见下表与后续修复章节；修复动作与最终结论见 `CHANGES.md` 与 `verification/verdicts.json`。

---

## 0. 结论总览（只读复核阶段）

| # | 审计项 | 结论 | 关键证据 |
|---|---|---|---|
| 1 | 清单 38 个唯一 ID，结果目录一一对应 | **PASS** | manifest 38 行、`uniq -d` 输出 0、`Samples/` 恰 38 个同名目录 |
| 2 | 38/38 有完整六面 A-F_col.png | **PASS** | 逐块 `faces=ABCDEF` |
| 3 | 每块恰好一个目标切面 cut1/cut2 | **PASS** | 38/38 `cut1⊕cut2=1`（19×cut1 + 19×cut2），`proc_eval.json` 的 `cut` 字段与数据集实际存在切面一致 |
| 4 | U-Net `training_data` 无测试 ID 泄漏 | **PASS** | training_data 152 块、test_data 恰为 38 清单 ID，交集为空 |
| 5 | 生成端只读六面外表面 | **PASS** | 源码级核对（见 5.1–5.4），切面读取仅出现在评价函数 |
| 6 | `proc_cube.npz` 保存先于切面评价 | **FAIL** | 11 个恢复样本 `proc_eval.json` mtime(10-20 11:46) **早于** `proc_cube.npz` mtime(10-20 12:29)；CN03 保存 L1=0.039864 与当前体重放 0.035187 不符 |
| 7 | `4a_eval_cut.py` 不从目标切面提取颜色统计 | **FAIL** | `4a_eval_cut.py:194` 由 GT 切面算 `gt_chan_mean`，`:218` 用于评价图展示定色 |
| 8 | `sweep_beta.py` / `tune_details.py` 不进正式结果 | **FAIL** | 两脚本位于正式 `code/` 内；`sweep_beta.py:94-96` 按目标切面 SWD2 选 `best_beta`；`tune_details.py` 直接覆盖 `proc_cube.npz` 并重评；38 个样本目录中有 11 个存在 `proc_cube_beta1_backup.npz`、7 个存在 `proc_params_beta{0.5,0.75}.pkl` |

### 已知待复核问题的独立复核结果

| 已知问题 | 复核结论 | 证据 |
|---|---|---|
| B08 当前体 = β=0.75 | **确认** | β=0.75 重建与当前体 mean\|Δ\|=6.2e-05；其切面 L1=0.020921 ≈ 保存值 0.020917；β=1 重建距当前体 5.6e-03（差 90 倍） |
| BW02 当前体 = β=0.75 | **确认** | β=0.75 重建 mean\|Δ\|=1e-05，L1=0.035954 ≈ 保存 0.035952 |
| BW06 当前体 = β=0.5 | **确认** | β=0.5 重建 mean\|Δ\|=9e-06，L1=0.038769 **完全等于**保存值 0.038769 |
| CH06 当前体 = β=0.5 | **确认** | β=0.5 重建 mean\|Δ\|=4e-06，L1=0.023224 ≈ 保存 0.023223 |
| `sweep_beta.py` 按切面 SWD2 选 β | **确认** | `sweep_beta.py:94-96`；且扫描过程中会覆盖 `proc_cube.npz` 与 `proc_eval.json`（`:75, :108`） |
| `4a_eval_cut.py` 用 GT 切面 `gt_chan_mean` | **确认** | `:194`、`:218`（指标仍在白平衡域，污染限于展示定色） |
| CN03 保存 L1=0.039864 ≠ 重放 0.035187 | **确认并定位根因** | CN03 当前 `proc_cube.npz` 与 `proc_cube_beta1_backup.npz` **md5 相同**(1646f1b3243e)，两者重放 L1 均为 0.035187 ≠ 保存 0.039864 → 评价早于体数据落地，属**过期评价**而非体数据污染 |
| `beta_sweep.json` 未找到 | **确认** | `find /home/gast/cjy/project1` 未命中；就地复算的 β 指纹（见上）为等价证据 |
| README_WORK.md「全部 β=1 纯拟合」与磁盘冲突 | **确认** | README 第 19-20、27、84 行声称 β=1；实际 4 个样本（含第 19 行所指"批次一"全部 6 块中的 4 块）体数据为 β<1 |
| （用户清单外新发现）11 个恢复样本的 `Samples/` 目录**缺少六面与切面照片** | **确认** | `Samples/{B08,B10,...,H11}` 无任何 `*_col.png`；输入只在 `MokumeDataset` 中。导致这 11 块在服务器上无法由代码直接复现（必须先 stage 输入） |

### 复核中发现但**不成立**的疑点（如实列出）

- **H11 不在污染名单**：H11 当前体与 β=1 备份 mean\|Δ\|=1.1e-04、max\|Δ\|=5e-04（float16 量级舍入），且 β=1/0.75/0.5/0.25/0 重建 **md5 完全相同**（`c25ccf846fb7`）：H11 的 `pore_color/ray_multiply_color` 为退化常量 0.2 且管孔/射线掩码未激活，β 缩放对其无影响 → **H11 当前体即 β=1 结果，不属污染**。

---

## 1. 清单与结果目录（审计项 1）

```
$ wc -l test_sample_ids.txt                    → 38
$ sort test_sample_ids.txt | uniq -d | wc -l   → 0（无重复）
$ ls code/Samples/ | wc -l                     → 38
$ diff <(sort test_sample_ids.txt) <(ls code/Samples/)  → 空
```
**PASS** — 38 个唯一 ID，结果目录与之严格一一对应（无多余、无缺失）。

## 2. 六面外表面完备性（审计项 2）

对 38 个 ID 逐一检查 `MokumeDataset/<id>/{A..F}_col.png`：

```
B08: faces=ABCDEF …（38/38 全部 faces=ABCDEF）
```
**PASS** — 38/38 具备完整六面输入。

## 3. 唯一目标切面（审计项 3）

```
B08: cut1=yes cut2=no     B10: cut1=no  cut2=yes    BW02: cut1=yes cut2=no
…（38/38 均满足 cut1⊕cut2 = 1；19 块提供 cut1，19 块提供 cut2）
```
`proc_eval.json` 的 `cut` 字段与数据集实际提供的切面编号逐一比对，38/38 一致（评价未选错切面）。
**PASS**。

## 4. U-Net split 泄漏检查（审计项 4）

对 `ImagePairs.zip` 按文件名前缀（`[A-Z]{1,3}\d{2}` + 面字母 A-F）解析样本 ID：

```
training_data 样本数 = 152   （B01–TC10，均不含任一条清单 ID）
test_data     样本数 = 38
test_data IDs == manifest  → True
training_data ∩ manifest   → NONE（泄漏数 = 0）
```
**PASS** — 38 个测试 ID 未进入 U-Net 训练数据。

## 5. 代码路径：生成端是否只使用六面（审计项 5）

| 文件 | 读取的输入 | 结论 |
|---|---|---|
| `2_apply_unet.py` L81-85 | `Samples/<id>/{A..F}_col.png` → `*_arl-unet.png` | 只读六面 |
| `3_infer_gf.py` L79 → `COMMON/data_utils.py:26` | `get_cube_side_images()` 只读 `{ltr}_col.png`；另读 `{A..F}_knot.png`（外表面标注） | 只读六面 |
| `4a_proc.py` L387-406 `load_sample()` | 六面 RGB + U-Net 输出 + `gf_params.pkl`（冻结生长场） | 只读六面 + 冻结场 |
| `4a_proc.py` L560-611 `run_optimize()`（Stage A/B/D） | `tgt_faces`（六面目标）计算 L1/风格损失/亮度锚定 | 优化只用六面 |
| `4a_proc.py` L614-647 `grid_search_sizes()`（Stage C 网格搜索） | `style_loss_on_faces(outs, tgt_faces)`，评判只走六面 | 网格搜索只用六面 |
| `4a_proc.py` L674-699 `evaluate_on_slanted_cut()` | **唯一**读取 `cut{1,2}_col.png` 处（L692） | 评价专用 |
| `batch_runner.py` L26-29 | `find_cut_samples()` 仅检测切面**存在性**（无 imread） | 不读像素 |

顺序核对：`4a_proc.py` L818 保存 `proc_cube.npz` → L822 保存 `proc_params.pkl` → L866-884 才调用 `evaluate_on_slanted_cut()`，返回值仅写入 `proc_cut2_render.png` / `proc_eval.json`，**不回传任何梯度或参数**。
**PASS**（代码层面；但磁盘层面存在例外，见第 6 项）。

## 6. 体数据保存先于切面评价（审计项 6）

```
11 个恢复样本（B08 B10 BW02 BW06 CH04 CH06 CN01 CN03 H01 H06 H11）mtime：
  proc_eval.json / gf_params.pkl / proc_params.pkl : 10-20 11:46
  proc_cube.npz / proc_cube_beta1_backup.npz       : 10-20 12:29
```
`proc_cube.npz` 的落盘时间**晚于**其所对应的评价报告 43 分钟 → 保存的评价不一定由当前体数据算出。独立复算佐证：

```
CN03  proc_cube.npz               md5=1646f1b3243e  重放 L1 = 0.035187
CN03  proc_cube_beta1_backup.npz  md5=1646f1b3243e  重放 L1 = 0.035187
CN03  proc_eval.json 保存值        L1 = 0.039864     ← 不一致
```
**FAIL** — 该 11 块的保存指标无法由当前体数据复现，全部需重评（27 块本批样本 mtime 顺序正常、为 PASS）。

## 7. 评价脚本是否使用目标切面颜色统计（审计项 7）

```python
# 4a_eval_cut.py:194（修复前）
gt_chan_mean = gt.astype(np.float32).reshape(-1, 3).mean(axis=0) / 255.0
# 4a_eval_cut.py:218
shown = unwhitebalance(rendered, gt_chan_mean)   # 评价图展示定色使用 GT 切面颜色
```
指标（L1/RMSE/Gram/SWD2）本身在白平衡域计算、对整体色温免疫，未受污染；违规点在**展示定色**使用了目标切面的像素统计，违反"评价图只能使用外表面统计或中性定色"。
**FAIL**。

## 8. 调参/扫描脚本是否进入正式结果（审计项 8）

- `sweep_beta.py`：L62-75 对每个 β **覆盖** `proc_cube.npz` 并重跑评价；L94-96 `best = min(SWD2)`、`best_beta_by_SWD2` —— 以目标切面指标选超参；L100-111 收尾还原 β=1（补丁存在，但扫描期间的中间态已真实写入正式结果文件）。
- `tune_details.py`：L59-60 缩放 `pore_color/ray_multiply_color`、L80 覆盖 `proc_cube.npz`、L84 写 `proc_params_beta{B}.pkl`、L91-95 立即重评。
- 磁盘残留：`proc_cube_beta1_backup.npz` ×11、`proc_params_beta0.5.pkl`/`beta0.75.pkl` ×7（B08×2、B10×2、BW02×3、BW06×1、CH04×2、CH06×1）。
- `README_WORK.md` L51-52 将两脚本列为正式工具链，L89 甚至把"新样本完成后跑 `sweep_beta.py` 扫描"写入标准流程。

**FAIL**。

## 9. 灰区观察（如实披露，不掩盖）

1. **切面朝向修复曾依切面数据调整评价代码**：`4a_eval_cut.py` 对 cut1 补 `ROTATE_90_CLOCKWISE`（L58-62 注释）的判据来自 `check_orient.py` 对 GT 切面照片的结构张量测量。该改动位于**评价渲染**侧（几何对齐），未进入生成端，也未逐样本挑选（对全部 cut1 统一施加），但性质上属于"查看目标切面后调整代码"，在此记录。
2. `5_show_output.py` 读取切面照片仅用于交互式图版展示，不在生产链路（`batch_runner.py` 不调用）。
3. `check_orient.py` 为诊断工具，读切面但不写任何正式结果文件。
4. `3_infer_gf.py` 的优化含随机初始化（`torch.rand(3)`，L108）与不连续网格搜索，因此**同参数复现不等于同结果复现**；本次修复只重建体数据（冻结 `proc_params.pkl` + `build_volume`），不重跑 Step 3，避免引入新的随机性。

## 11. 修复后的最终判定（承接 CHANGES.md 与 verification/）

修复在版本化目录 `/home/gast/cjy/project1/moke_4a_holdout_v2_20261021/` 内完成后，
`verification/verdicts.json` 的 10 项强制验证结果：

| 验证项 | 内容 | 结果 |
|---|---|---|
| V1 | 结果目录恰含 38 个清单 ID | **PASS**（38/38，无多余无缺失） |
| V2 | 38/38 六面输入 + 唯一切面 + gf_params + proc_cube + proc_eval | **PASS** |
| V3 | U-Net training_data 泄漏数 | **PASS**（0；train 152 / test 38 恰为清单） |
| V4 | 体数据保存时间 < 切面评价时间 | **PASS**（38/38） |
| V5 | 生成/优化入口无切面读取（语义级：仅注释/docstring/print 含字面量） | **PASS**（切面读取唯一位于 `evaluate_on_slanted_cut()` L692，调用点 L868 晚于保存点 L818） |
| V6 | 评价脚本不使用目标切面颜色统计定色 | **PASS**（改用六面外表面统计） |
| V7 | B08/BW02/BW06/CH06 新体 == β=1 重建且 ≠ 旧污染体 | **PASS**（md5 逐一吻合 β=1 重建） |
| V8 | 38/38 保存指标可由当前体 + 当前代码重放复现 | **PASS**（相对误差 < 1e-4） |
| V9 | 残留按切面选参的脚本/文件名/日志/参数 | **PASS**（正式链路 0 命中；仅 README/docstring 中有说明性文字，非功能残留） |
| V10 | 正式结果中无 β<1 / tuned / best_beta 产物 | **PASS**（隔离至 `diagnostics/quarantine/<ID>/`） |

**ALL 10 CHECKS PASS = True**（`verification/FINAL_VERDICT.json`）

### 对四个问题的明确回答（修复后状态）

1. **38 个测试木块是否统一？** — **是**。结果目录恰含清单中 38 个唯一 ID，38/38 具备完整六面
   `A-F_col.png`、`gf_params.pkl`、`proc_cube.npz`、`proc_eval.json`，且 U-Net 训练数据中无泄漏。
2. **是否全部只使用六面和唯一目标切面？** — **是**。生成端（Step 2/3/4a 与 Stage A–D 优化、
   Stage C 网格搜索）只读取六面外表面与冻结的 `gf_params.pkl`；每块仅有一个指定的 cut1 或 cut2，
   评价使用的切面编号与数据集实际提供的切面 38/38 一致。
3. **切面是否实际只用于评价？** — **是**。切面文件仅在 `4a_eval_cut.py` 与
   `4a_proc.py::evaluate_on_slanted_cut()` 内读取一次，且都在 `proc_cube.npz` 落盘之后调用，
   结果只写入评价文件与图版，不回传任何梯度/参数。
4. **是否仍存在任何训练、适配、选参或颜色提示泄漏？** — **否**（修复后）。
   β 扫描/细节调参脚本已移出生产链路、其产物已隔离；评价图定色不再使用目标切面颜色统计；
   4 块曾由目标切面指标选出 β<1 的样本已按冻结的 β=1 参数重建并重评。
   **如实披露的灰区**：`4a_eval_cut.py` 对 cut1 统一施加的 90° 旋转对齐，其判据来自
   `check_orient.py` 对 GT 切面的结构张量测量（属评价侧几何对齐，非逐样本选参，未进入生成端）。

> 注：修复前状态为 **不成立**（见第 10 节）；以上结论描述的是版本化目录
> `moke_4a_holdout_v2_20261021/code` 修复后的状态。原目录 `moke_4a_20261019/code` 未做任何修改，
> 其中的旧结果仍为不成立状态，不应再用于报告。

---

## 10. 只读复核阶段的最终判定

**38/38 严格留出协议不成立**（修复前）。依据：
- 4 个样本（B08/BW02/BW06/CH06）正式体数据由目标切面指标选出的 β<1 生成（事后选参）；
- 11 个样本评价指标无法由当前体数据复现（保存早于体数据落盘），含 CN03 数值不符；
- 评价脚本使用目标切面颜色统计做展示定色；
- 调参脚本仍在正式代码目录并曾写入正式结果。

修复动作见 `CHANGES.md`，修复后逐项验证见 `verification/`。
