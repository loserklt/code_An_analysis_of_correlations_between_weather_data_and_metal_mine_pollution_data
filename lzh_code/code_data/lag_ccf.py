# -*- coding: utf-8 -*-
"""
Lag analysis (CCF) for metals vs rainfall/temperature at 3 sites
- Uses monthly data already prepared (same as correlation section)
- Metals: 12 indicators (code 37 removed)
- Transforms metals by log10 (>0) then z-score; meteorology z-score
- Computes raw CCF and prewhitened CCF (ARIMA on met series) within +/- MAX_LAG
- Outputs per-site plots and CSV summaries

Inputs (default /mnt/data/):
  S35629_monthly.csv
  S35636_monthly.csv
  Nenthead_monthly.csv

Outputs: /mnt/data/lag_outputs/
"""

import os, re, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA

# ----------------- CONFIG -----------------
DATA_DIR = ".\\"
OUT_DIR  = os.path.join(DATA_DIR, "lag_outputs")
os.makedirs(OUT_DIR, exist_ok=True)

FILES = {
    "S35629": os.path.join(DATA_DIR, "S35629_monthly.csv"),
    "S35636": os.path.join(DATA_DIR, "S35636_monthly.csv"),
    "Nenthead": os.path.join(DATA_DIR, "Nenthead_monthly.csv"),
}

# 12 指标（已去掉 37）
TARGET_CODES = [50, 52, 61, 106, 108, 183, 3408, 6051, 6450, 6452, 6455, 6460]

# 最大滞后（月）
MAX_LAG = 12

# 预白化 ARIMA 阶数（对气象序列）
ARIMA_ORDER = (1, 0, 0)  # 可按需要改成 (p, d, q)

# ------------------------------------------

def pick_indicator_columns(df):
    cols = []
    for c in df.columns:
        m = re.match(r"^(\d+)_", str(c))
        if m and int(m.group(1)) in TARGET_CODES:
            cols.append(c)
    return cols

def log10_series(s):
    v = pd.to_numeric(s, errors="coerce")
    v = v.where(v > 0, np.nan)
    return np.log10(v)

def zscore(s):
    s = pd.to_numeric(s, errors="coerce")
    m, sd = s.mean(skipna=True), s.std(skipna=True)
    if sd is None or sd == 0 or np.isnan(sd):
        return s * np.nan
    return (s - m) / sd

def ccf(y, x, max_lag=12):
    """
    计算 y 与 x 的跨相关（y_t vs x_{t-k}）
    正滞后 k 的定义：气象领先金属 k 个月（x 先动）
    返回 (lags, rho)
    """
    # 对齐、去缺失
    df = pd.concat([y, x], axis=1, join="inner").dropna()
    if len(df) < 5:
        return np.arange(-max_lag, max_lag+1), np.full(2*max_lag+1, np.nan)

    y = zscore(df.iloc[:,0]); x = zscore(df.iloc[:,1])
    y = y.values.astype(float); x = x.values.astype(float)
    n = len(y)

    r = []
    lags = range(-max_lag, max_lag+1)
    for k in lags:
        if k >= 0:
            # corr( y_t , x_{t-k} ), t = k..n-1
            yk = y[k:]
            xk = x[:n-k]
        else:
            # k<0: corr( y_t , x_{t-k} ) = corr( y_{-k:} , x_{:n+k} )
            k2 = -k
            yk = y[:n-k2]
            xk = x[k2:]
        if len(yk) < 3:
            r.append(np.nan)
        else:
            r.append( float(np.corrcoef(yk, xk)[0,1]) )
    return np.array(list(lags)), np.array(r)

def prewhiten(y, x, order=(1,0,0)):
    """
    预白化：先对 x(气象)拟合 ARIMA，取残差 e_x；
    用同样滤波器作用到 y 得 e_y（与 x 同阶滤波），
    返回 (e_y, e_x) 以用于 CCF。
    """
    df = pd.concat([y, x], axis=1).dropna()
    if len(df) < 10:
        return None, None
    x0 = df.iloc[:,1]

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            model = ARIMA(x0, order=order)
            res = model.fit()
        except Exception:
            return None, None

    ex = res.resid  # e_x
    # 将同样的 AR 部分应用到 y（简单起见：只用AR系数过滤）
    y0 = df.iloc[:,0].astype(float)
    arparams = getattr(res, "arparams", np.array([]))
    if arparams.size == 0:
        # 若没有 AR 部分，则退化为原值
        ey = y0
    else:
        # y_filtered_t = y_t - sum(phi_i * y_{t-i})
        ey = y0.copy()
        for i, phi in enumerate(arparams, start=1):
            ey.iloc[i:] = ey.iloc[i:] - phi * y0.shift(i).iloc[i:]
        ey = ey.dropna()
        ex = ex.loc[ey.index]

    # 标准化
    return zscore(ey), zscore(ex)

def plot_ccf(site, metal_label, xname, lags, r, out_png, title_suffix=""):
    # 有效样本近似置信带 ±1.96/sqrt(N)
    N = np.sum(~np.isnan(r))
    ci = 1.96 / np.sqrt(N) if N and N > 0 else np.nan

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axhline(0, color="k", lw=1)

    if not np.isnan(ci):
        ax.fill_between([lags.min(), lags.max()], ci, -ci, color="#cccccc", alpha=0.3)

    # 这里不再使用 use_line_collection
    markerline, stemlines, baseline = ax.stem(lags, r)
    plt.setp(markerline, marker='o', markersize=4, color="#1f77b4")
    plt.setp(stemlines, linewidth=1, color="#1f77b4")
    plt.setp(baseline, color="k", linewidth=0.8)

    ax.set_xlim(lags.min()-0.5, lags.max()+0.5)
    ax.set_xlabel("Lag (months)  [positive = meteorology leads]")
    ax.set_ylabel("CCF")
    ax.set_title(f"{site} — {metal_label} vs {xname} {title_suffix}")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200)
    plt.close(fig)


def best_lag(lags, r):
    """返回 |r| 最大的 (lag, r)；若全是 NaN 返回 (np.nan, np.nan)"""
    if r is None or np.all(np.isnan(r)):
        return np.nan, np.nan
    idx = np.nanargmax(np.abs(r))
    return lags[idx], r[idx]

def nice_metal_label(col):
    # label 示例：'log10 6455_S29' -> 'Zn (6455)'（若你有映射表，可在此替换更友好的名字）
    # 这里仅简化为 'code' 抽取
    m = re.match(r"(?:log10_)?(\d+)_", str(col))
    code = m.group(1) if m else str(col)
    return f"metal {code}"

def analyze_site(site_name, csv_path):
    print(f"[{site_name}] loading …")
    df = pd.read_csv(csv_path, parse_dates=["date"]).sort_values("date")
    df = df.set_index("date")

    # 指标列
    ind_cols = pick_indicator_columns(df.reset_index())
    ind_cols = [c for c in ind_cols if c in df.columns]

    # 气象列
    has_rain = "rainfall_mm" in df.columns
    has_temp = "temperature_C" in df.columns

    if not ind_cols or (not has_rain and not has_temp):
        print(f"[{site_name}] skip (no indicators or no met)")
        return

    # 预处理：对金属做 log10，然后 z-score；对气象直接 z-score
    dfL = df.copy()
    for c in ind_cols:
        dfL[f"log10_{c}"] = log10_series(dfL[c])

    # 输出汇总
    rows_raw = []
    rows_pw  = []

    for c in ind_cols:
        metal = dfL[f"log10_{c}"]
        if metal.notna().sum() < 6:
            continue
        metal_label = f"log10 {c}"

        for xname in ["rainfall_mm", "temperature_C"]:
            if xname not in dfL.columns:
                continue
            # --- raw CCF ---
            lags, r = ccf(zscore(metal), zscore(dfL[xname]), max_lag=MAX_LAG)
            out_png = os.path.join(OUT_DIR, f"{site_name}_{c}_vs_{xname}_RAW.png")
            plot_ccf(site_name, metal_label, xname, lags, r, out_png, title_suffix="(RAW)")
            lag_b, r_b = best_lag(lags, r)
            rows_raw.append([site_name, c, xname, lag_b, r_b])

            # --- prewhitened CCF ---
            ey, ex = prewhiten(zscore(metal), zscore(dfL[xname]), order=ARIMA_ORDER)
            if ey is not None and ex is not None:
                lags_pw, r_pw = ccf(ey, ex, max_lag=MAX_LAG)
                out_png_pw = os.path.join(OUT_DIR, f"{site_name}_{c}_vs_{xname}_PWHITE.png")
                plot_ccf(site_name, metal_label, xname, lags_pw, r_pw, out_png_pw, title_suffix="(Prewhitened)")
                lag_bp, r_bp = best_lag(lags_pw, r_pw)
            else:
                lags_pw = np.arange(-MAX_LAG, MAX_LAG+1)
                r_pw = np.full_like(lags_pw, np.nan, dtype=float)
                lag_bp, r_bp = np.nan, np.nan
            rows_pw.append([site_name, c, xname, lag_bp, r_bp])

    # 保存 CSV
    if rows_raw:
        pd.DataFrame(rows_raw, columns=["site","indicator_code","met","best_lag","ccf_at_best"])\
          .to_csv(os.path.join(OUT_DIR, f"{site_name}_ccf_bestlags_RAW.csv"), index=False)
    if rows_pw:
        pd.DataFrame(rows_pw, columns=["site","indicator_code","met","best_lag","ccf_at_best"])\
          .to_csv(os.path.join(OUT_DIR, f"{site_name}_ccf_bestlags_PWHITE.csv"), index=False)

    print(f"[{site_name}] done. Plots & CSV saved to {OUT_DIR}")

def main():
    for site, path in FILES.items():
        if os.path.exists(path):
            analyze_site(site, path)
        else:
            print(f"Missing file: {path}")

if __name__ == "__main__":
    main()
