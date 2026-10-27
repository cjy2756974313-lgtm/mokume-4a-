# -*- coding: utf-8 -*-
"""build_species_map.py — A1b：38 块测试集的材种组织类型分层 + 射线可见性登记。

数据源：Mokume 官方 Zenodo「Wood Species Codes」表（17 材种）。
组织类型按木材解剖学归类；边界情形单独标注（不强行并入三类）。
射线可见性：先登记材种级先验（木材学常识），per-sample 人工确认状态 TBD，
  配套 review_ray_montage.png（38 块 C 面照片拼图）供逐块目检。

产出：
  notes/species_map.json / species_map.csv   —— 逐块登记
  notes/stratified_proc.json                 —— PROC 指标按组织类型分层统计
"""
import os
import sys
import csv
import json

HERE = os.path.dirname(os.path.abspath(__file__))

# ---------------- 官方材种代码表（Zenodo 15588748） ----------------
SPECIES = {
    "B":  ("Beech",        "Fagus spp.",           "B10"),
    "BW": ("Black walnut", "Juglans nigra",        "BW02"),
    "CH": ("Cherry",       "Prunus serotina",      "CH04"),
    "CN": ("Kuri",         "Castanea crenata",     "CN01"),
    "H":  ("Hinoki",       "Chamaecyparis obtusa", "H01"),
    "IC": ("Icho",         "Ginkgo biloba",        "IC04"),
    "K":  ("Keyaki",       "Zelkova serrata",      "K04"),
    "KR": ("Kurumi",       "Juglans mandshurica",  "KR01"),
    "MP": ("Maple",        "Acer spp.",            "MP04"),
    "MZ": ("Mizume",       "Betula grossa",        "MZ03"),
    "N":  ("Nara",         "Quercus crispula",     "N01"),
    "NR": ("Nire",         "Ulmus davidiana",      "NR04"),
    "P":  ("Platanus",     "Platanus occidentalis","P06"),
    "RO": ("Red oak",      "Quercus rubra",        "RO03"),
    "S":  ("Sakura",       "Prunus spp.",          "S02"),
    "SG": ("Sugi",         "Cryptomeria japonica", "SG03"),
    "TC": ("Tochinoki",    "Aesculus turbinata",   "TC04"),
}

# ---------------- 组织类型归类（木材解剖学） ----------------
# 环孔材：早材导管显著大于晚材，年轮界肉眼分明
RING_POROUS = {"K", "N", "NR", "RO", "CN"}
# 半环孔材：环孔→散孔过渡（胡桃属典型）
SEMI_RING_POROUS = {"BW", "KR"}
# 散孔材：导管全年轮均匀
DIFFUSE_POROUS = {"B", "CH", "MP", "MZ", "P", "S", "TC"}
# 针叶材/无孔材：裸子植物（银杏为无孔材，与针叶材同归"无孔"组）
CONIFER = {"H", "SG", "IC"}

def anatomy(code):
    if code in RING_POROUS:
        return "环孔材", ""
    if code in SEMI_RING_POROUS:
        return "半环孔材", "Juglans 属典型半环孔；三类合并时建议随散孔材或单独成组"
    if code in DIFFUSE_POROUS:
        return "散孔材", ""
    return "针叶材", "含银杏（无孔材，裸子植物）" if code == "IC" else ""

# ---------------- 射线可见性（材种级先验；逐块确认 TBD） ----------------
# 三档：可见 / 弱可见 / 不可见；依据木材学常识（径切面/弦切面射线花纹）
RAY_PRIOR = {
    "N": "可见",   "RO": "可见",  # 栎属宽射线，肉眼显著
    "B": "可见",                # 山毛榉径面射线斑纹
    "MZ": "可见",               # 真樺径面射线纺锤纹
    "K": "可见",   "CN": "可见", # 榉/栗径面射线
    "P": "可见",                # 悬铃木弦面连锁射线（蛇纹）
    "TC": "弱可见", "NR": "弱可见",
    "CH": "弱可见", "S": "弱可见",
    "MP": "弱可见",
    "BW": "弱可见", "KR": "弱可见",
    "H": "不可见", "SG": "不可见", "IC": "不可见",  # 针叶/无孔材射线窄小
}

def main():
    samples_root = os.path.join(HERE, "Samples")
    ids = sorted(d for d in os.listdir(samples_root)
                 if os.path.isdir(os.path.join(samples_root, d))
                 and d[0].isalpha())
    rows = []
    for sid in ids:
        code = "".join(c for c in sid if c.isalpha())
        # H/SG 是单字母码，其余两位：取开头连续字母
        if sid.startswith("BW"): code = "BW"
        elif sid.startswith("CH"): code = "CH"
        elif sid.startswith("CN"): code = "CN"
        elif sid.startswith("IC"): code = "IC"
        elif sid.startswith("KR"): code = "KR"
        elif sid.startswith("MP"): code = "MP"
        elif sid.startswith("MZ"): code = "MZ"
        elif sid.startswith("NR"): code = "NR"
        elif sid.startswith("RO"): code = "RO"
        elif sid.startswith("SG"): code = "SG"
        elif sid.startswith("TC"): code = "TC"
        else: code = sid[0]
        name, sci, _ = SPECIES[code]
        anat, note = anatomy(code)
        rows.append({
            "sample": sid, "species_code": code, "species": name,
            "scientific": sci, "anatomy": anat, "anatomy_note": note,
            "ray_visibility_prior": RAY_PRIOR[code],
            "ray_visibility_confirmed": "TBD",  # 待逐块目检（review_ray_montage.png）
        })

    out_csv = os.path.join(HERE, "species_map.csv")
    with open(out_csv, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(HERE, "species_map.json"), "w", encoding="utf-8") as f:
        json.dump(rows, f, indent=2, ensure_ascii=False)
    print(f"species_map: {len(rows)} 块 -> {out_csv}")

    # ---------------- 分层统计（PROC 统一口径重评结果） ----------------
    reeval = os.path.join(HERE, "reeval_orient_summary.csv")
    metrics = {}
    if os.path.isfile(reeval):
        for r in csv.DictReader(open(reeval)):
            metrics[r["sample"]] = r
    keys = ["proc_L1", "proc_RMSE", "proc_DISTS", "proc_VGG-SWD2", "proc_RingF1"]
    groups = {}
    for row in rows:
        g = row["anatomy"]
        groups.setdefault(g, []).append(row["sample"])
    strat = {}
    for g, sids in groups.items():
        stat = {"n": len(sids), "samples": sids}
        for k in keys:
            vals = []
            for s in sids:
                v = metrics.get(s, {}).get(k)
                if v and v != "nan":
                    vals.append(float(v))
            stat[k] = round(sum(vals) / len(vals), 4) if vals else None
        strat[g] = stat
        print(g, {k: stat[k] for k in keys}, "n=", len(sids))
    # 三类合并视图（半环孔随散孔；供表 3 分层用）
    merged = {}
    for g in strat:
        key = "散孔材(含半环孔)" if g == "半环孔材" else g
        merged.setdefault(key, []).extend(strat[g]["samples"])
    strat3 = {}
    for g, sids in merged.items():
        stat = {"n": len(sids)}
        for k in keys:
            vals = [float(metrics[s][k]) for s in sids
                    if metrics.get(s, {}).get(k) and metrics[s][k] != "nan"]
            stat[k] = round(sum(vals) / len(vals), 4) if vals else None
        strat3[g] = stat
    out = {"anatomy_4way": strat, "anatomy_3way_merged": strat3,
           "note": "半环孔材=Juglans属(BW/KR)；银杏IC为无孔材归针叶组；"
                   "射线可见性为材种级先验，逐块确认见 review_ray_montage.png（TBD）"}
    with open(os.path.join(HERE, "stratified_proc.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    print("stratified_proc.json written")


if __name__ == "__main__":
    main()
