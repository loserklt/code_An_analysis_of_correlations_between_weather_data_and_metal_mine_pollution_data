# -*- coding: utf-8 -*-
"""
Correlation & Partial Correlation (with log10) for three sites (12 indicators; code 37 removed)
- Inputs (default /mnt/data/):
    S35629_monthly.csv
    S35636_monthly.csv
    Nenthead_monthly.csv
    指标_编号名称对应表.xlsx   # columns: parameter_shortname, parameter_name, unit_symbol
- Outputs: /mnt/data/corr_outputs/
    <site>_corr_pearson_raw.csv/.png
    <site>_corr_spearman_raw.csv/.png
    <site>_corr_pearson_log10.csv/.png
    <site>_corr_spearman_log10.csv/.png
    <site>_partial_corr_log10_R_T_season.csv/.png
"""

import os, re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

# ---------------- CONFIG ----------------
DATA_DIR = ".\\"
OUT_DIR  = os.path.join(DATA_DIR, "corr_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

FILES = {
    "S35629": os.path.join(DATA_DIR, "S35629_monthly.csv"),
    "S35636": os.path.join(DATA_DIR, "S35636_monthly.csv"),
    "Nenthead": os.path.join(DATA_DIR, "Nenthead_monthly.csv"),
}
MAPPING_XLSX = os.path.join(DATA_DIR, "指标_编号名称对应表.xlsx")

# 12 指标（已剔除 37=流速）
TARGET_CODES = [50, 52, 61, 106, 108, 183, 3408, 6051, 6450, 6452, 6455, 6460]

# 热力图配色（不使用蓝/紫）
CMAP = "RdYlGn"  # 负相关=Red，0=Yellow，正相关=Green
VMIN, VMAX = -1.0, 1.0

# ---------------- Helpers ----------------
def load_mapping(path):
    m = pd.read_excel(path)
    m["parameter_shortname"] = pd.to_numeric(m["parameter_shortname"], errors="coerce").astype("Int64")
    m = m.dropna(subset=["parameter_shortname"]).copy()
    code2name = dict(zip(m["parameter_shortname"], m["parameter_name"]))
    code2unit = dict(zip(m["parameter_shortname"], m["unit_symbol"]))
    return code2name, code2unit

def pick_indicator_columns(df):
    cols = []
    for c in df.columns:
        m = re.match(r"^(\d+)_", str(c))
        if m and int(m.group(1)) in TARGET_CODES:
            cols.append(c)
    return cols

def code_of(col):
    return int(re.match(r"^(\d+)_", str(col)).group(1))

def nice_label(col, code2name):
    code = code_of(col)
    nm = code2name.get(code, str(code))
    return f"{nm} ({code})"

def add_log10_numeric(df, ind_cols):
    out = df.copy()
    for c in ind_cols:
        v = pd.to_numeric(out[c], errors="coerce")  # 强制数值
        out[f"log10_{c}"] = np.where(v > 0, np.log10(v.astype(float)), np.nan)
    return out

def month_dummies(idx):
    m = idx.month
    d = pd.get_dummies(m.astype(int), prefix="m", drop_first=True).astype(float)
    d.index = idx
    return d

# 覆盖原有的 save_heatmap
def save_heatmap(matrix, title, out_png,
                 cmap="coolwarm", vmin=-1.0, vmax=1.0,
                 annotate=True, fmt="{:.2f}", text_thresh=0.55,
                 tick_fontsize=9, title_fontsize=12):
    """
    标注版热力图：
    - cmap="coolwarm"（红=正相关，蓝=负相关，白≈0）
    - 在格子中央写入相关系数；绝对值较大时用白字提升对比度
    - 其余保持不变
    """
    data = np.array(matrix.values, dtype=float)
    data = np.ma.masked_invalid(data)  # NaN 做掩码

    # 图幅大小随矩阵规模自适应，略放大以容纳标注
    h = max(6, 0.7 * len(matrix.index))
    w = max(8, 0.7 * len(matrix.columns))
    fig, ax = plt.subplots(figsize=(w, h))

    im = ax.imshow(
        data,
        aspect="auto",
        interpolation="nearest",
        cmap=cmap,
        norm=TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax),
    )

    # 轴刻度
    ax.set_xticks(range(len(matrix.columns)))
    ax.set_xticklabels(matrix.columns, rotation=90, fontsize=tick_fontsize)
    ax.set_yticks(range(len(matrix.index)))
    ax.set_yticklabels(matrix.index, fontsize=tick_fontsize)
    ax.set_title(title, fontsize=title_fontsize)

    # 颜色条
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Correlation", rotation=270, labelpad=12)

    # 在格子里标注数值
    if annotate:
        nrows, ncols = matrix.shape
        for i in range(nrows):
            for j in range(ncols):
                val = matrix.iat[i, j]
                if np.isnan(val):
                    continue
                # 绝对值较大时用白色字体，避免和底色混淆
                txt_color = "white" if abs(val) >= text_thresh else "black"
                ax.text(
                    j, i, fmt.format(val),
                    ha="center", va="center",
                    fontsize=max(7, tick_fontsize-1),
                    color=txt_color
                )

    fig.tight_layout()
    fig.savefig(out_png, dpi=220)
    plt.close(fig)


def to_numeric_df(df):
    """将 DataFrame 全部转为数值，非数值转 NaN。"""
    out = df.copy()
    for c in out.columns:
        out[c] = pd.to_numeric(out[c], errors="coerce")
    return out

def partial_corr_residual(X, Y, Z):
    """
    基于残差法的偏相关：
      先回归 X~Z、Y~Z，取残差，再做 Pearson 相关。
    强制把所有列转 float，并剔除 Z 中常数/全空列，避免 dtype('O') 报错与奇异矩阵。
    """
    df = pd.concat([X, Y, Z], axis=1)
    df = to_numeric_df(df).dropna()
    if df.shape[0] < 3:
        return np.nan

    x = df.iloc[:, 0].to_numpy(dtype=float)
    y = df.iloc[:, 1].to_numpy(dtype=float)
    z = df.iloc[:, 2:].to_numpy(dtype=float)

    if z.size == 0:  # 无控制变量，退化为普通相关
        return float(np.corrcoef(x, y)[0, 1])

    # 剔除常数或方差为 0 的控制变量
    keep = np.nanstd(z, axis=0) > 0
    z = z[:, keep]
    if z.size == 0:
        return float(np.corrcoef(x, y)[0, 1])

    Zc = np.column_stack([np.ones(z.shape[0]), z])  # 加常数项
    bx, *_ = np.linalg.lstsq(Zc, x, rcond=None)
    by, *_ = np.linalg.lstsq(Zc, y, rcond=None)
    rx = x - Zc @ bx
    ry = y - Zc @ by

    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx**2).sum()) * np.sqrt((ry**2).sum())
    return float((rx * ry).sum() / denom) if denom > 0 else np.nan

def partial_corr_matrix(df_log, ind_log_cols, controls):
    mat = pd.DataFrame(index=ind_log_cols, columns=ind_log_cols, dtype=float)
    for i, a in enumerate(ind_log_cols):
        for j, b in enumerate(ind_log_cols):
            if i == j:
                mat.loc[a, b] = 1.0
            elif pd.isna(mat.loc[a, b]):
                r = partial_corr_residual(df_log[a], df_log[b], controls)
                mat.loc[a, b] = r
                mat.loc[b, a] = r
    return mat

# ---------------- per-site pipeline ----------------
def analyze_site(site_name, csv_path, code2name):
    print(f"[{site_name}] loading …")
    df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date")
    ind_cols = pick_indicator_columns(df)
    if not ind_cols:
        print(f"[{site_name}] no indicator columns found (check column names).")
        return

    # ---------- 原值相关 ----------
    corr_pearson_raw = df[ind_cols].apply(pd.to_numeric, errors="coerce").corr(method="pearson")
    corr_spearman_raw = df[ind_cols].apply(pd.to_numeric, errors="coerce").corr(method="spearman")

    labels_raw = [nice_label(c, code2name) for c in ind_cols]
    corr_pearson_raw.index = labels_raw; corr_pearson_raw.columns = labels_raw
    corr_spearman_raw.index = labels_raw; corr_spearman_raw.columns = labels_raw

    corr_pearson_raw.to_csv(os.path.join(OUT_DIR, f"{site_name}_corr_pearson_raw.csv"))
    corr_spearman_raw.to_csv(os.path.join(OUT_DIR, f"{site_name}_corr_spearman_raw.csv"))

    save_heatmap(corr_pearson_raw, f"{site_name} — Pearson (raw)", os.path.join(OUT_DIR, f"{site_name}_corr_pearson_raw.png"))
    save_heatmap(corr_spearman_raw, f"{site_name} — Spearman (raw)", os.path.join(OUT_DIR, f"{site_name}_corr_spearman_raw.png"))

    # ---------- log10 相关 ----------
    dfL = add_log10_numeric(df, ind_cols)
    ind_log_cols = [c for c in (f"log10_{ic}" for ic in ind_cols) if c in dfL.columns]

    corr_pearson_log = dfL[ind_log_cols].corr(method="pearson")
    corr_spearman_log = dfL[ind_log_cols].corr(method="spearman")

    labels_log = [f"log10 {nice_label(c.replace('log10_', ''), code2name)}" for c in ind_log_cols]
    corr_pearson_log.index = labels_log; corr_pearson_log.columns = labels_log
    corr_spearman_log.index = labels_log; corr_spearman_log.columns = labels_log

    corr_pearson_log.to_csv(os.path.join(OUT_DIR, f"{site_name}_corr_pearson_log10.csv"))
    corr_spearman_log.to_csv(os.path.join(OUT_DIR, f"{site_name}_corr_spearman_log10.csv"))

    save_heatmap(corr_pearson_log, f"{site_name} — Pearson (log10)", os.path.join(OUT_DIR, f"{site_name}_corr_pearson_log10.png"))
    save_heatmap(corr_spearman_log, f"{site_name} — Spearman (log10)", os.path.join(OUT_DIR, f"{site_name}_corr_spearman_log10.png"))

    # ---------- 偏相关：log10 | 控制 R、T、季节 ----------
    dfL_idx = dfL.set_index("date")
    controls = pd.DataFrame(index=dfL_idx.index)
    if "rainfall_mm" in dfL_idx.columns:
        controls["R"] = pd.to_numeric(dfL_idx["rainfall_mm"], errors="coerce")
    if "temperature_C" in dfL_idx.columns:
        controls["T"] = pd.to_numeric(dfL_idx["temperature_C"], errors="coerce")
    controls = pd.concat([controls, month_dummies(dfL_idx.index)], axis=1)
    controls = to_numeric_df(controls).dropna(axis=1, how="all")

    # 仅对存在的 log 列做偏相关
    part = partial_corr_matrix(dfL_idx, ind_log_cols, controls)

    labels_part = [f"log10 {nice_label(c.replace('log10_', ''), code2name)}" for c in ind_log_cols]
    part.index = labels_part; part.columns = labels_part
    part.to_csv(os.path.join(OUT_DIR, f"{site_name}_partial_corr_log10_R_T_season.csv"))

    save_heatmap(part, f"{site_name} — Partial Corr (log10 | R, T, season)",
                 os.path.join(OUT_DIR, f"{site_name}_partial_corr_log10_R_T_season.png"))

    print(f"[{site_name}] done ✓  (files saved to {OUT_DIR})")

# ---------------- Run all ----------------
def main():
    code2name, code2unit = load_mapping(MAPPING_XLSX)
    for site, path in FILES.items():
        analyze_site(site, path, code2name)
    print("All outputs saved to:", OUT_DIR)

if __name__ == "__main__":
    main()
