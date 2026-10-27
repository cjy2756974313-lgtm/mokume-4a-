# CHANGES.md — Mokume 4a 严格留出协议修复记录

> 执行时间：2026-10-21 11:20 – 13:20（GMT+8）
> 服务器：`server-r740`（gast@47.98.131.85:10001）
> 原目录：`/home/gast/cjy/project1/moke_4a_20261019/code` — **全程未修改、未覆盖、未删除任何文件**
> 新目录：`/home/gast/cjy/project1/moke_4a_holdout_v2_20261021/`（版本化结果目录，全部修复动作在此进行）

---

## 0. 总原则

| 约束 | 落实情况 |
|---|---|
| 不直接覆盖原结果 | ✅ 原目录 `moke_4a_20261019/code` 自始至终只读访问（全部操作为 `cp/scp` 出、脚本在副本上执行） |
| 不删除原始数据 | ✅ 未执行任何 `rm` 删除数据；调参产物为**移动**到 `diagnostics/quarantine/`（原目录副本中仍保留） |
| 不停止/重启正在运行的任务 | ✅ 未 kill 任何他人任务；仅新建后台作业（自建的审计进程除外） |
| 读取只读 | ✅ 审计阶段全部为只读命令与只读脚本 |

## 1. 创建版本化结果目录（修复要求 1）

```
mkdir /home/gast/cjy/project1/moke_4a_holdout_v2_20261021
cp -a /home/gast/cjy/project1/moke_4a_20261019/code \
      /home/gast/cjy/project1/moke_4a_holdout_v2_20261021/code     # 7.5 GB 完整快照（含 Samples）
```
- 原始目录保持原样，作为"污染前状态"的完整快照（要求中的"完整快照"）。
- 后续所有修复/重生成/重评均在 `moke_4a_holdout_v2_20261021/code` 内进行。
- 备份位置：`/home/gast/cjy/project1/moke_4a_20261019/code`（原目录，未动）

## 2. 补齐 11 个恢复样本的输入（发现的新问题）

审计发现 11 个恢复样本（`B08 B10 BW02 BW06 CH04 CH06 CN01 CN03 H01 H06 H11`）的 `Samples/` 目录
**不含任何 `*_col.png`**（六面与切面照片当初只留在本地，未上传）。已在新目录中按
`batch_runner.stage_inputs()` 的同一规则从 `MokumeDataset` 补齐：

```
for id in <38>:  cp -n $MokumeDataset/$id/*_col.png $NEW/Samples/$id/    # 六面 + 目标切面
                 cp -n $MokumeDataset/$id/*_knot.png $NEW/Samples/$id/   # 外表面节子标注
```
作用：使 38/38 具备"完整六面输入 + 唯一切面"，满足验证项 2；并使评价脚本可运行（需要 GT 切面）。

## 3. 标记 4 个污染样本无效并重新生成（修复要求 2、3）

**判定依据**（`audit_20261021/beta_match_result.json`，β 重建指纹比对）：

| 样本 | 原体数据匹配 | 与当前体 mean\|Δ\| | 该 β 切面 L1 vs 保存值 | β=1 重建距当前体 |
|---|---|---|---|---|
| B08 | **β=0.75** | 6.2e-05 | 0.020921 ≈ 保存 0.020917 | 5.6e-03（差 90×） |
| BW02 | **β=0.75** | 1e-05 | 0.035954 ≈ 保存 0.035952 | 1.7e-03 |
| BW06 | **β=0.5** | 9e-06 | 0.038769 **=** 保存 0.038769 | 1.3e-02 |
| CH06 | **β=0.5** | 4e-06 | 0.023224 ≈ 保存 0.023223 | 4.9e-03 |
| H11 | β=1（退化参数，各 β 重建 md5 相同 `c25ccf846fb7`） | 1.1e-04（float16 舍入） | 0.100874 = 保存 0.100874 | — **不属污染** |

**处理**：在新目录中，用**冻结的 `proc_params.pkl`（β=1 纯拟合参数，未被任何脚本改写）**
经 `4a_proc.build_volume()` 重新生成全部 38 个样本的 `proc_cube.npz`：

```
python regen_cubes.py --root <new>/code --workers 3      # 日志：regen_cubes_log.json
```
- **27 个本批样本**：重建后 md5 与原有体 **完全一致**（`changed=false`）→ 证明它们本来就是 β=1 纯拟合，且重建可确定复现。
- **4 个污染样本**：新体 md5 = 独立复核得到的 β=1 重建 md5（B08 `0fae18a47382`、BW02 `3f093f7d9dde`、
  BW06 `00d1e2fc2e9d`、CH06 `3d917574e166`），且 ≠ 原污染体 → **不再与任何 β<1 候选匹配**。
- **11 个恢复样本中其余 6 块**（B10/CH04/CN01/CN03/H01/H06）：原体即 β=1，重建后数值等价
  （md5 差异仅来自本地/服务器浮点舍入，mean\|Δ\|≈1e-5）。

**原结果标记无效**：`AUDIT_REPORT.md` + `README_WORK.md`（新目录内已修订）明确声明
B08/BW02/BW06/CH06 的旧体数据与旧评价无效；旧评价文件保留在原目录副本中未删除。

## 4. 隔离调参产物（修复要求 4）

```
mv <new>/code/sweep_beta.py <new>/code/tune_details.py  ->  <new>/code/diagnostics/
rm -f <new>/code/__pycache__/{sweep_beta,tune_details}.*
mv <new>/code/Samples/<ID>/proc_cube_beta1_backup.npz   ->  <new>/code/diagnostics/quarantine/<ID>/
mv <new>/code/Samples/<ID>/proc_params_beta*.pkl        ->  <new>/code/diagnostics/quarantine/<ID>/
```
涉及样本：B08 B10 BW02 BW06 CH04 CH06 CN01 CN03 H01 H06 H11（11 个）。
`verification/v9_v10_residual_scan.json` 确认：正式结果树中 β 相关残留 = 0，正式代码根目录下调参脚本 = 0。

## 5. 修改 `4a_eval_cut.py`：禁止目标切面颜色统计（修复要求 5）

| 位置 | 修改前 | 修改后 |
|---|---|---|
| `4a_eval_cut.py:194` | `gt_chan_mean = gt.astype(np.float32).reshape(-1,3).mean(axis=0)/255.0`（GT 切面统计） | 取 `A-F_col.png` 六面外表面各自通道均值的平均（`:193-203`）；六面缺失时退化为中性定色 `[0.5,0.5,0.5]` |
| `:218` | `shown = unwhitebalance(rendered, gt_chan_mean)` | 同上，但输入已改为外表面统计 |
| `:299-300` | `note` 字段 | 追加"展示定色使用六面外表面统计（严格留出协议修复 2026-10-21）" |

- 指标计算（L1/RMSE/Gram-VGG19/VGG-SWD2）仍使用 GT 切面真值，但始终在**白平衡域**比较
  （双方各自归一化），不含任何来自切面的颜色统计量 —— 符合"指标计算仍可使用 GT"。
- 验证：`verification/v6_eval_color.json` → `uses_gt_cut_color_stats=false`、`uses_outer_face_stats=true`。

## 6. 移出调参脚本（修复要求 6）

`4a_proc.py`、`3_infer_gf.py`、`2_apply_unet.py`、`batch_runner.py` 未调用两者；
移入 `diagnostics/` 后，正式链路（`batch_runner.py`）不可能触达；`refresh_summary.py` 亦无引用。

## 7. 重新生成全部评价、汇总与对比图（修复要求 7）

```
python reeval_all.py --root <new>/code        # 38/38 用修复后 4a_eval_cut.py 重评，rc=0、零失败
python refresh_summary.py -write              # 新 summary.json（38 条）
```
- 重评顺序：体数据（regen）→ 评价（reeval），`verification/v4_timestamps.json` 确认 **38/38 cube_mtime < eval_mtime**。
- 对比图 `Samples/<ID>/proc_vis/cut_eval_compare.png`：**38/38 已重新生成**。
- CN03 过期评价修复：L1 0.039864 → **0.035187**（与当前 β=1 体数据重放一致）；SWD2 0.5352 → 0.5197。
- 4 个污染样本新指标（β=1 纯拟合，不再含事后选参收益）：

| 样本 | 旧 L1 / SWD2（β<1，作废） | 新 L1 / SWD2（β=1，正式） |
|---|---|---|
| B08 | 0.020917 / 0.2333 | **0.023414 / 0.2900** |
| BW02 | 0.035952 / 0.2890 | **0.036585 / 0.2886** |
| BW06 | 0.038769 / 0.3195 | **0.048756 / 0.5372** |
| CH06 | 0.023223 / 0.2957 | **0.026739 / 0.4432** |
| CN03（过期评价） | 0.039864 / 0.5352 | **0.035187 / 0.5197** |
| H11（未污染） | 0.100874 / 0.3171 | **0.100874 / 0.3189** |

（BW06/CH06/B08 数值变差，正是"按目标切面选 β"带来的乐观偏差被移除后的真实水平。）

## 8. 其他文件改动清单

| 文件（新目录内） | 改动 | 原因 |
|---|---|---|
| `code/4a_eval_cut.py` | 展示定色改用外表面统计 | 修复要求 5 |
| `code/sweep_beta.py`、`code/tune_details.py` | 移动至 `code/diagnostics/` | 修复要求 6 |
| `code/refresh_summary.py` | 缺失 `col_cube`/`gf_cube` 的样本不再抛异常，缺失列显示 `-` | 11 个恢复样本在服务器上无 Step 3 体数据（仅本地有），原脚本 `KeyError` 无法汇总 |
| `code/README_WORK.md` | 修订与磁盘状态冲突的表述（批次状态、β=1 声明、脚本索引、协议纪律、下一步） | 修复要求 8 |
| `code/Samples/<ID>/proc_cube.npz` | 由冻结 β=1 参数重建（38 个） | 修复要求 3 |
| `code/Samples/<ID>/proc_eval.json`、`proc_vis/*` | 重新评价（38 个） | 修复要求 7 |
| `code/batch_logs/summary.json` | 重新汇总为 38 条 | 修复要求 7 |

## 9. 未改动事项（说明）

- **未重跑 Step 2（U-Net）/ Step 3（生长场拟合）**：`3_infer_gf.py` 含随机初始化与不连续网格搜索，
  重跑会引入新随机性、且需数小时/样本；本次修复只重建体数据（冻结参数 + 确定性 `build_volume`），
  保证"可复现"而非"再优化"。
- **11 个样本的消融列（col/gf）缺失**：其 `col_cube.npz`/`gf_cube.npz` 未上传到服务器。
  PROC 主指标 38/38 完整；消融列 n=27。详见 `RESULTS_38.md`。
- `H11` 未列入污染名单（复核证实为退化参数，β 缩放对其无影响）。
