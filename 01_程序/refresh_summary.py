# -*- coding: utf-8 -*-
"""
refresh_summary.py —— 用各样本当前的 proc_eval.json 回填 batch_logs/summary.json。

背景：batch_runner.py 只在当次运行结束的瞬间写一次 summary。之后若做了
朝向修复、beta 调整等事后重评价，summary 就变成旧值；sweep_beta.py 结束后
还会把 proc_eval.json 留在最后一个 beta 上，更容易失真。
本脚本以磁盘上的 proc_eval.json 为准，重新汇总，是「当前真实状态」的单一入口。

用法：
  python refresh_summary.py            # 刷新全部已评价样本
  python refresh_summary.py -print     # 只打印表格，不写文件
"""
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
SAMPLES_DIR = os.path.join(HERE, "Samples")
LOG_DIR = os.path.join(HERE, "batch_logs")
SUMMARY = os.path.join(LOG_DIR, "summary.json")

# 斜切面编号：与 MokumeDataset 中的 cut1_col.png / cut2_col.png 对应
CUT_OF = {}


def detect_cut(sid):
    """从 proc_eval.json 自身读取 cut 编号，避免手写表出错。"""
    p = os.path.join(SAMPLES_DIR, sid, "proc_eval.json")
    with open(p, encoding="utf-8") as f:
        return int(json.load(f)["cut"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("-write", action="store_true",
                    help="写回 summary.json（默认只打印）")
    args = ap.parse_args()

    sids = sorted(d for d in os.listdir(SAMPLES_DIR)
                  if os.path.isfile(os.path.join(SAMPLES_DIR, d, "proc_eval.json")))

    rows = []
    for sid in sids:
        with open(os.path.join(SAMPLES_DIR, sid, "proc_eval.json"),
                  encoding="utf-8") as f:
            ev = json.load(f)
        if "metrics" not in ev:
            print(f"[skip] {sid}: proc_eval.json 无 metrics 键（非完整报告）")
            continue
        m = ev["metrics"]
        p = m["PROC (4a 程序纹理)"]
        c = m["col_cube (仅颜色)"]
        g = m["gf_cube (生长场切片)"]
        rows.append((sid, detect_cut(sid), p, c, g))

    hdr = ("%-6s %-4s %-9s %-9s %-9s | %-9s %-9s" %
           ("id", "cut", "P_L1", "P_Gram", "P_SWD2", "col_SWD2", "gf_SWD2"))
    print(hdr)
    print("-" * len(hdr))
    tot = {"p": [0.0] * 3, "c": [0.0] * 3, "g": [0.0] * 3}
    for sid, cut, p, c, g in rows:
        print("%-6s %-4d %-9.4f %-9.5f %-9.4f | %-9.4f %-9.4f" %
              (sid, cut, p["L1"], p["Gram-VGG19"], p["VGG-SWD2"],
               c["VGG-SWD2"], g["VGG-SWD2"]))
        for k, x in (("p", p), ("c", c), ("g", g)):
            tot[k][0] += x["L1"]
            tot[k][1] += x["Gram-VGG19"]
            tot[k][2] += x["VGG-SWD2"]

    n = len(rows)
    print("-" * len(hdr))
    print("n = %d 均值：" % n)
    for k, lab in (("p", "PROC"), ("c", "col  "), ("g", "gf   ")):
        print("  %s L1 %.4f   Gram %.5f   SWD2 %.4f" %
              (lab, tot[k][0] / n, tot[k][1] / n, tot[k][2] / n))
    wins = sum(1 for _, _, p, c, _ in rows if p["VGG-SWD2"] < c["VGG-SWD2"])
    print("  PROC 在 SWD2 上胜出 col 消融: %d/%d" % (wins, n))

    if args.write:
        summary = []
        if os.path.isfile(SUMMARY):
            with open(SUMMARY, encoding="utf-8") as f:
                try:
                    summary = json.load(f)
                except Exception:
                    summary = []
        for sid, cut, p, c, g in rows:
            with open(os.path.join(SAMPLES_DIR, sid, "proc_eval.json"),
                      encoding="utf-8") as f:
                ev = json.load(f)
            # 保留原有的 status / minutes 字段
            old = next((r for r in summary if r["id"] == sid), {})
            ev = dict(ev)
            rec = {"id": sid, "cut": cut, "eval": ev,
                   "status": old.get("status", "ok"),
                   "minutes": old.get("minutes", 0.0)}
            summary = [r for r in summary if r["id"] != sid] + [rec]
        os.makedirs(LOG_DIR, exist_ok=True)
        with open(SUMMARY, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=1)
        print("\n已刷新 -> %s" % SUMMARY)
    else:
        print("\n（只读模式；加 -write 才会写回 summary.json）")


if __name__ == "__main__":
    main()
