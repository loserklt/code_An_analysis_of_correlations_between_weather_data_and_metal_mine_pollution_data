# -*- coding: utf-8 -*-
# Quick-look summary & why-log10 visuals for three sites (12 indicators, code 37 removed)
# Requirements: pandas, numpy, matplotlib, openpyxl (for reading .xlsx)

import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
DATA_DIR ='.\\'  # 修改为你的路径
OUT_DIR  = os.path.join(DATA_DIR, "quicklook")
os.makedirs(OUT_DIR, exist_ok=True)

FILES = {
    "S35629": os.path.join(DATA_DIR, "S35629_monthly.csv"),
    "S35636": os.path.join(DATA_DIR, "S35636_monthly.csv"),
    "Nenthead": os.path.join(DATA_DIR, "Nenthead_monthly.csv"),
}
MAPPING_XLSX = os.path.join(DATA_DIR, "指标_编号名称对应表.xlsx")

# 仅保留的 12 指标（剔除 37）
TARGET_CODES = [50, 52, 61, 106, 108, 183, 3408, 6051, 6450, 6452, 6455, 6460]
# 演示 log10 的两个核心变量：总锌、硫酸盐
DEMO_CODES = [3408, 183]
# 可选：关键指标横向对比（论文表）
KEY_CODES = [3408, 50, 108, 183]  # Zn, Pb, Cd, SO4

# ---------------- Helpers ----------------
def load_mapping(path):
    df = pd.read_excel(path)
    df["parameter_shortname"] = pd.to_numeric(df["parameter_shortname"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["parameter_shortname"]).copy()
    code2name = dict(zip(df["parameter_shortname"], df["parameter_name"]))
    code2unit = dict(zip(df["parameter_shortname"], df["unit_symbol"]))
    return code2name, code2unit

def pick_indicator_columns(df, target_codes):
    out = []
    for c in df.columns:
        m = re.match(r"^(\d+)_", str(c))
        if m and int(m.group(1)) in target_codes:
            out.append(c)
    return out

def summarise_site(df, site_name, code2name):
    """生成每站点 12 指标的概要表，并保存 CSV"""
    ind_cols = pick_indicator_columns(df, TARGET_CODES)
    rows = []
    for c in ind_cols:
        v = pd.to_numeric(df[c], errors="coerce")
        n = int(v.notna().sum())
        miss = 100.0 * (1 - n/len(df)) if len(df)>0 else np.nan
        pct_le0 = 100.0 * (v.fillna(0) <= 0).sum() / len(v) if len(v)>0 else np.nan
        rows.append({
            "code": int(re.match(r"^(\d+)_", c).group(1)),
            "column": c,
            "n": n,
            "missing_%": round(miss, 1),
            "min": float(np.nanmin(v.values)) if n>0 else np.nan,
            "median": float(np.nanmedian(v.values)) if n>0 else np.nan,
            "mean": float(np.nanmean(v.values)) if n>0 else np.nan,
            "p95": float(np.nanpercentile(v.values, 95)) if n>0 else np.nan,
            "skewness": float(pd.Series(v).skew(skipna=True)) if n>1 else np.nan,
            "%<=0": round(pct_le0, 1),
        })
    summ = pd.DataFrame(rows)
    summ["name"] = summ["code"].map(code2name)
    summ = summ[["code","name","column","n","missing_%","min","median","mean","p95","skewness","%<=0"]]
    summ.to_csv(os.path.join(OUT_DIR, f"{site_name}_summary_12vars.csv"), index=False)
    return summ, ind_cols

def plot_histograms(df, site_name, code, code2name):
    """原值 vs log10 直方图（每图单独文件，matplotlib only）"""
    cols = [c for c in df.columns if c.startswith(f"{code}_")]
    if not cols:
        return
    col = cols[0]
    v = pd.to_numeric(df[col], errors="coerce")
    v_pos = v[v > 0]

    # raw
    fig, ax = plt.subplots(figsize=(8,5))
    ax.hist(v.dropna().values, bins=30)
    ax.set_title(f"{site_name} — {code2name.get(code, str(code))} (raw)")
    ax.set_xlabel("concentration(µg/l)")
    ax.set_ylabel("count")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, f"{site_name}_{code}_hist_raw.png"), dpi=200)
    plt.close(fig)

    # log10
    if len(v_pos) > 0:
        fig, ax = plt.subplots(figsize=(8,5))
        ax.hist(np.log10(v_pos.values), bins=30)
        ax.set_title(f"{site_name} — {code2name.get(code, str(code))} (log10)")
        ax.set_xlabel("log10(concentration)(µg/l)")
        ax.set_ylabel("count")
        fig.tight_layout()
        fig.savefig(os.path.join(OUT_DIR, f"{site_name}_{code}_hist_log10.png"), dpi=200)
        plt.close(fig)

def build_key_table(summ_dict, code2name, out_csv, out_tex=None):
    """
    从三个站点的 summary 中，抽取 KEY_CODES 的 mean/median/p95/skewness
    生成一个横向对比表（CSV + 可选 LaTeX 三线表）。
    """
    rows = []
    for site, summ in summ_dict.items():
        for code in KEY_CODES:
            r = summ.loc[summ["code"] == code, ["mean","median","p95","skewness"]]
            if r.empty:
                rows.append([site, code, code2name.get(code, str(code)), np.nan, np.nan, np.nan, np.nan])
            else:
                rows.append([site, code, code2name.get(code, str(code)),
                             float(r["mean"].values[0]),
                             float(r["median"].values[0]),
                             float(r["p95"].values[0]),
                             float(r["skewness"].values[0])])
    out = pd.DataFrame(rows, columns=["site","code","name","mean","median","p95","skewness"])
    out.to_csv(out_csv, index=False)

    if out_tex:
        # 生成简洁 LaTeX 三线表（booktabs）
        def fmt(x):
            return "—" if pd.isna(x) else f"{x:.2f}"
        lines = []
        lines.append("\\begin{table}[ht]\n\\centering")
        lines.append("\\caption{Key indicators summary across sites}\n")
        lines.append("\\begin{tabular}{llrrrr}\n\\toprule")
        lines.append("Site & Indicator & Mean & Median & P95 & Skewness \\\\\n\\midrule")
        for _, r in out.iterrows():
            lines.append(f"{r['site']} & {r['name']} & {fmt(r['mean'])} & {fmt(r['median'])} & {fmt(r['p95'])} & {fmt(r['skewness'])} \\\\")
        lines.append("\\bottomrule\n\\end{tabular}\n\\end{table}")
        with open(out_tex, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    return out

# ---------------- Main ----------------
def main():
    code2name, code2unit = load_mapping(MAPPING_XLSX)

    summaries = {}
    for site, path in FILES.items():
        df = pd.read_csv(path, parse_dates=["date"])
        summ, cols = summarise_site(df, site, code2name)
        summaries[site] = summ
        # 两个核心变量：总锌 & 硫酸盐
        for code in DEMO_CODES:
            plot_histograms(df, site, code, code2name)

    # 关键指标（Zn, Pb, Cd, SO4）横向对比表（CSV + 可选 LaTeX）
    out_csv = os.path.join(OUT_DIR, "key_indicators_across_sites.csv")
    out_tex = os.path.join(OUT_DIR, "key_indicators_across_sites.tex")  # 若不需要 LaTeX，可设为 None
    build_key_table(summaries, code2name, out_csv, out_tex)

    print("Done. See outputs under:", OUT_DIR)

if __name__ == "__main__":
    main()
