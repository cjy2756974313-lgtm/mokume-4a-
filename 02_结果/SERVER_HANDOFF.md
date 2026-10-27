# 服务器路径交接（SERVER_HANDOFF）

> 面向接手人。仅服务器 R740 上的路径与操作要点。2026-10-25。

## 0. 连接与环境

```
ssh -i /d/BaiduNetdiskDownload/r740_gast -p 10001 gast@47.98.131.85
```
- python：`/home/gast/wzy/anaconda3/envs/mokume/bin/python`
- GPU 序号 ≠ nvidia-smi：**必须** `CUDA_DEVICE_ORDER=PCI_BUS_ID`
- 预训练权重缓存（无外网）：`TORCH_HOME=/home/gast/.cache/torch`，脚本里传 `-torch_home`
- 后台任务：`setsid nohup python ... > log 2>&1 </dev/null & echo OK`（启动即返回）；轮询本地 sleep 后另开连接；pgrep 用 `[x]xx.py` 字符类避免匹配 ssh 自身；长任务加 `flock -n /tmp/xxx.lock` 防双跑

## 1. 目录总览

| 目录 | 性质 |
|---|---|
| `/home/gast/cjy/project1/moke_4a_holdout_v2_20261021/` | **生产主目录（冻结，只读）** |
| `/home/gast/cjy/moke_4a_r2a_20261025/` | 优化实验目录（未晋升，结果不进主表） |
| `/home/gast/cjy/project1/moke_4a_20261019/` | v1 旧批跑（仅 batch_logs 有 Step3 计时价值） |
| `/home/gast/cjy/project1/moku/` | 官方仓库 + 数据集副本 |
| `/home/gast/.cache/torch` | 预训练权重缓存 |

## 2. 生产主目录 `.../moke_4a_holdout_v2_20261021/`

### 程序（`code/`）

| 文件 | 作用 |
|---|---|
| `2_apply_unet.py` | Step2 年轮提取（U-Net） |
| `3_infer_gf.py` | Step3 生长场拟合 |
| `4a_proc.py` | **Step4a 程序反演**（含 `-knot` 开关） |
| `4a_eval_cut.py` | 基础评价（L1/RMSE/Gram/SWD2） |
| `4c_eval_dists_f1.py` | DISTS + 生长带 P/R/F₁ |
| `reeval_orient.py` | **统一口径重评（基准/晋升判据用）** |
| `check_rot180_multi.py` | 180° 朝向共识判定 |
| `eval_knot_h11.py` | H11 节疤消融 |
| `build_species_map.py` | 材种分层登记 |
| `make_montage38.py` | GT vs PROC 对比图生成 |
| `batch_runner.py` | 批跑编排（step2→3→4a→eval） |
| `COMMON/` | 官方共享库 |
| `DISTS_local.py`+`weights.pt` | DISTS 离线实现 |

### 成果（数据/指标）

| 路径 | 内容 |
|---|---|
| `code/Samples/<ID>/` | 38 块：`proc_cube.npz`（H11 另有 `proc_cube_knot.npz`）、`col/gf/nca_cube.npz`、`gf_params.pkl`、`proc_eval.json`、`proc_eval_orient.json`、`cutN_arl-unet.png` |
| `code/reeval_orient_summary.csv/json` | **基准指标明细（权威）** |
| `code/reeval_dists_f1_summary.csv` | DISTS/F₁ 首轮重放（rot0 版） |
| `code/rot180_multi.json` / `rot180_check.json` | 朝向判定证据 |
| `code/species_map.csv/json` / `stratified_proc.json` / `review_ray_montage.png` | 材种分层 + 射线目检 |

### 成果（评价图）

| 路径 | 内容 |
|---|---|
| `code/Samples/<ID>/proc_vis/cut_eval_compare.png` | 每块旧口径五联图（GT/GF/COL/PROC/NCA，rot0） |
| `code/montage38_gt_vs_proc.png` | **38 块统一口径总览**（GT vs PROC，含指标标注） |
| `code/montage_{环孔材,半环孔材,散孔材,针叶材}.png` | 按组织类型分组对比图 |
| `notes/montage/` | 上述对比图归档副本 |

### 文档与证据

| 路径 | 内容 |
|---|---|
| `RESULTS_38.md` | 主结果表（38 块 × 全指标 + 计时） |
| `KNOT_FIX.md` | 节疤修复报告 |
| `SECTION4.4_完成度核对.md` | §4.4 完成度核对 |
| `AUDIT_REPORT.md` / `CHANGES.md` | 留出协议审计与修复记录 |
| `notes/eval_geometry.md` | **评价几何口径**（切面坐标、90°/180° 朝向表、计时） |
| `notes/timing.json` | 表 3 计时口径与数值 |
| `notes/PATHS.md` | 全项目路径索引（本地+服务器） |
| `notes/species_map.*`、`stratified_proc.json` | 分层登记 |
| `verification/` | V1–V10 合规验证 JSON + hashes.json |
| `regen_cubes_log.json` / `reeval_log.json` / `nca_train_H11.log` / `code/batch_logs/` | 计时/日志证据 |

## 3. 权威与注意事项

- **基准数字唯一权威**：`code/reeval_orient_summary.csv`。
- **口径文档权威**：`notes/eval_geometry.md` + `notes/timing.json`。
- **生产目录只读**：任何优化结果先在独立目录（如 r2a）验证，晋升判据见 `code/../` 上级的 `OPTIMIZATION_HANDOFF.md`（本地 `D:/CODE/.workbuddy/memory/OPTIMIZATION_HANDOFF.md`）。
- 指标口径：白平衡域；朝向表 H11/H14/MP07/N07/SG15/SG16 评价时 rot180，其余 rot0；H11 用 knot 体；cut1 补 90°。
- 材种代码对照见 `code/species_map.csv`（官方 Zenodo 17 材种）。
