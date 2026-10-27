# Mokume 4a 项目路径索引（PATHS.md）

> 更新：2026-10-26（全部 4a 资产已并入单一根目录 `D:/CODE/moke/yhua/`）。单一入口：找程序、找数据、找结果，先查本文。
> 三条工作线：**A 生产基准**（冻结，只读）｜ **B 优化实验**（已收敛，五轮全部未过判据）｜ **C 论文配套**（分层/口径/核对）。
> 根目录说明见 `../README.md`。

## 0. 一页速查

| 要什么 | 去哪 |
|---|---|
| 生产基准数字 / 逐样本明细 | 本地 `deliverables/RESULTS_38.md` + `reeval_orient_summary.csv`（服务器 v2 同名） |
| 跑 4a 流水线的代码 | 本地 `D:/CODE/moke/yhua/code_audit/`（冻结副本）；服务器 `v2/code/`（生产） |
| 优化结论与判定（唯一入口） | `optimization/docs/OPTIMIZATION_MASTER_2026-10-26.md` |
| 优化工作区导航/口径/复现 | `optimization/WORKSPACE_GUIDE.md` |
| 优化纪律/判据/弱点清单 | `D:/CODE/.workbuddy/memory/OPTIMIZATION_HANDOFF.md` |
| 评价口径（朝向表/计时/F₁ 定义） | 服务器 `v2/notes/eval_geometry.md`、`v2/notes/timing.json` |
| 合规验证证据 | 服务器 `v2/verification/`（V1–V10 JSON）+ 本地 `deliverables/FINAL_VERDICT.json` |

## 1. 本地机器（D:/CODE）

### 1.1 代码（`D:/CODE/moke/yhua/code_audit/` —— 冻结基线代码副本，与服务器 v2/code 同步）

| 文件 | 作用 |
|---|---|
| `2_apply_unet.py` | Step2：U-Net 年轮提取（六面照片 → `*_arl-unet.png`） |
| `3_infer_gf.py` | Step3：生长场拟合（400 iter，含 knot 掩膜分支） |
| `4a_proc.py` | **Step4a 主脚本**：程序反演。已含 `-knot` 开关（2026-10-23） |
| `4a_eval_cut.py` | 基础评价（L1/RMSE/Gram/SWD2），唯一读斜切面的地方 |
| `4c_eval_dists_f1.py` | DISTS + 生长带 P/R/F₁ 指标重放（2026-10-23） |
| `reeval_orient.py` | **统一口径重评**（朝向表 + H11 knot 体，全指标）——胜出判据用这个 |
| `check_rot180.py` / `check_rot180_multi.py` | 180° 朝向核查（单立方体 / 多立方体共识） |
| `eval_knot_h11.py` | H11 节疤消融四格评价 |
| `build_species_map.py` | A1b 材种分层登记 + 分层统计 |
| `make_montage38.py` | 38 块 GT vs PROC 总览/分组对比图生成 |
| `DISTS_local.py` + `weights.pt` | DISTS 离线本地化实现 |
| `batch_runner.py` | 批跑编排（step2→3→4a→eval） |
| `check_orient.py` / `sweep_beta.py` / `tune_details.py` / `refresh_summary.py` | 旧工具（sweep/tune 已移出生产链路） |
| `COMMON/` | 官方共享库（procedural_wood_function / ProceduralParameters / data_utils / unet / loss_utils） |

### 1.2 结果与文档快照（`D:/CODE/moke/yhua/deliverables/`）

| 文件 | 内容 |
|---|---|
| `RESULTS_38.md` | **主结果表**（38 块 × 朝向 × 全指标 + 计时节 + 新旧对比） |
| `reeval_orient_summary.csv/json` | 统一口径逐样本全指标（基准真值来源） |
| `reeval_dists_f1_summary.csv` | DISTS/F₁ 首轮重放明细（rot0 版，已被上者取代） |
| `eval_geometry.md` | 评价几何口径：切面坐标、90° 修正、180° 朝向表、计时（服务器 notes/ 为权威） |
| `timing.json` | 表 3 计时口径与数值 |
| `KNOT_FIX.md` / `eval_knot_h11.json` / `H11_knot_ablation.png` | 节疤修复与 H11 消融 |
| `rot180_check.json`（+ 服务器 `rot180_multi.json`） | 朝向证据 |
| `species_map.csv/json`、`stratified_proc.json`、`review_ray_montage.png` | A1b 分层登记、分层统计、射线目检图 |
| `SECTION4.4_完成度核对.md` | §4.4 完成度核对（2026-10-24 版） |
| `AUDIT_REPORT.md`、`CHANGES.md`、`FINAL_VERDICT.json`、`verdicts.json`、`v*.json` | 留出协议审计与验证证据 |

### 1.3 优化实验（`D:/CODE/moke/yhua/optimization/`）

优化文档/数据/代码统一归并于此（`README.md` 为目录索引，`WORKSPACE_GUIDE.md` 为使用说明）：

- **结论入口**：`optimization/docs/OPTIMIZATION_MASTER_2026-10-26.md`（Round 1 / R2-A / R3-1 / Gate 2A / P3 / P3-1 全部结果与判定）
- **设计依据**：`optimization/docs/optimization_framework_rethink_2026-10-25.md`（P0–P5 / Gate 0–3）
- **过程存档**：`optimization/docs/archive/`（round1、round2_r2a、10-25 状态快照、色差任务书；只读）
- **本轮实验证据**：`optimization/p1/docs/`（p0_p1_prep、r3_1_results、g2a_results、p3_pore_diagnosis）
- `optimization/data/` — rl6 全量 + R2-A 三变体逐样本 CSV
- `optimization/p1/` — P0/P1 主分支：`code/`（`4a_proc_p1.py`、`COMMON/`、审计与汇总工具）、`run_r3_1_matrix.sh`、`out/`（r31 全部结论数据）
- `optimization/r2a/` — R2-A 实验：`code/4a_proc_r2a.py`、`audit_external_gate.py`（**F₁ 审计仍在使用**）、`unet_trained_model.pt`、`results*`、`local/Samples/CN03`
- `optimization/colorfix/` — **GT vs PROC 显示色差修复**：`color_fix_summary.md`（总报告）、`code/`、`out/`。服务器对应目录 `/home/gast/cjy/project1/moke_4a_colorfix_20261025/`
- `trash_20261026/`（与本目录同级）— 2026-10-26 清理回收站，可删可还原

### 1.4 其他本地资产

| 路径 | 内容 |
|---|---|
| `D:/CODE/moke/yhua/` | **4a 全部本地资产的唯一根目录**（`README.md` 为总索引）：`optimization/`、`deliverables/`、`code_audit/`、`trash_20261026/` |
| `D:/CODE/moke/MokumeDataset/` | 数据集纯净源（190 块，4.5G） |
| `D:/CODE/moke/mokumeproject-main/` | 官方代码库副本 + `Samples/`（H11/CN03 官方产物、H11 knot 掩膜） |
| `D:/CODE/moke/hub/` | 预训练权重（vgg16/vgg19、unet_trained_model.pt） |
| `D:/CODE/moke/ORGANIZE_PLAN.md` | 本地 moke 目录清点记录 |
| `D:/CODE/.workbuddy/tmp/nca/` | 4b NCA 训练脚本（`4b_nca_train.py`）与 knot 实验脚本 |
| `D:/CODE/.workbuddy/tmp/pdfout/20261013 纹理 CJ(1).md` | CJ 论文 §4.4 提取文本（核对原文用） |
| `D:/CODE/.workbuddy/memory/MEMORY.md` | 项目长期记忆（新对话自动读） |
| `D:/CODE/.workbuddy/memory/OPTIMIZATION_HANDOFF.md` | 优化交接（纪律/基线/判据） |
| `D:/CODE/.workbuddy/memory/YYYY-MM-DD.md` | 每日工作日志 |

## 2. 服务器 R740（gast@47.98.131.85:10001）

连接：`ssh -i /d/BaiduNetdiskDownload/r740_gast -p 10001 gast@47.98.131.85`
python：`/home/gast/wzy/anaconda3/envs/mokume/bin/python`

### 2.1 生产目录（`/home/gast/cjy/project1/moke_4a_holdout_v2_20261021/` —— **冻结，只读**）

| 路径 | 内容 |
|---|---|
| `code/` | 生产代码（与本地 code_audit 同步）+ 38 块 `Samples/`（proc/col/gf/nca 立方体、gf_params、评价 JSON）+ `batch_logs/`（含 H11.knot.log 计时证据） |
| `code/reeval_orient_summary.csv` | 统一口径基准明细（服务器权威版） |
| `code/montage38_gt_vs_proc.png`、`code/montage_{环孔材,半环孔材,散孔材,针叶材}.png` | **38 块 GT vs PROC 评价对比图**（2026-10-25 生成；脚本 `code/make_montage38.py`） |
| `code/Samples/<ID>/proc_vis/cut_eval_compare.png` | 每块旧口径五联对比图（GT/GF/COL/PROC/NCA，rot0） |
| `notes/montage/` | 对比图归档副本（总览 + 分组 4 张） |
| `notes/` | **口径文档权威位置**：eval_geometry.md、timing.json、species_map、stratified_proc.json、review_ray_montage.png |
| `verification/` | V1–V10 合规验证 JSON + hashes.json |
| 根目录 | RESULTS_38.md、KNOT_FIX.md、SECTION4.4_完成度核对.md、AUDIT_REPORT.md、CHANGES.md、regen/reeval 日志 |

### 2.2 实验总目录（`/home/gast/cjy/moke_4a_r31_20261025/` —— 全部 4a 实验，未晋升）

- `code/`、`logs/`（R3-1 / Gate 2A / P3-1 的原始 job 日志，按 `<cfg>_<ID>_s<seed>.log` 命名）、`faces_{r31,g2a,p31}/`、`Samples/*/proc_params_*`
- `archive_round2a_20261025/` — R2-A 归档（原 `moke_4a_r2a_20261025`，2026-10-26 并入；`Samples/` 的 96 个输入 PNG 已删，可从生产 `code/Samples/` `cp` 或软链恢复）
- 目录说明：该目录 `README.md` 与 `archive_round2a_20261025/README.md`
- 状态：五轮干预均未过晋升判据（见 `optimization/docs/OPTIMIZATION_MASTER_2026-10-26.md`），该目录结果不进主表；生产晋升前的中转区。

### 2.2b 色差修复实验目录（`/home/gast/cjy/project1/moke_4a_colorfix_20261025/` —— 显示层，未晋升）

`code/`（colorlib、diag_color、diag2_residual、make_montage38_colorfix、run_montage.sh）、`out/`（诊断 CSV + 全套对比图）、`logs/`、`verify_mirror/`（符号链接镜像，只读生产样本、用于重跑 `reeval_orient.py` 做指标不变性复核）。本地镜像 `optimization/colorfix/`。

### 2.3 其他服务器资产

| 路径 | 内容 |
|---|---|
| `/home/gast/cjy/project1/moke_4a_20261019/` | v1 旧批跑目录（Step3 计时证据 batch_logs；结果已被 v2 取代，勿引用其指标） |
| `/home/gast/cjy/project1/moku/` | 官方仓库 + 数据集服务器副本（MokumeDataset 191 项；ImagePairs U-Net 训练/测试划分证据） |
| `/home/gast/.cache/torch` | TORCH_HOME（无外网，预训练权重缓存） |

## 3. 权威与同步关系

- **基准数字唯一权威**：`reeval_orient_summary.csv`（本地 deliverables 与服务器 v2/code 双副本，内容一致）。
- **口径文档权威**：服务器 `v2/notes/`；本地 deliverables 为快照。
- **代码**：本地 `code_audit` = 服务器 `v2/code`（除 `__pycache__`/日志外同步）；改代码先本地，胜出后 scp 上传。
- **纪律**：生产 v2 冻结只读；优化一律走独立实验目录；晋升条件见 `OPTIMIZATION_HANDOFF.md` §3。
