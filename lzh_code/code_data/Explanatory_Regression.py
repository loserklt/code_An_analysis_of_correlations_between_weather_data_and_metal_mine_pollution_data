# -*- coding: utf-8 -*-
"""
Section 4.4 Explanatory Regressions (final)
- OLS with HAC(Newey–West) robust SEs
- Seasonal dummies by month
- Lags for rainfall/temperature: fixed from best-lag table if available; otherwise BIC grid search on {0,1,2,3}
Outputs (./reg44_outputs):
  - <site>_reg44_coeffs.csv                   # for thesis tables
  - <site>_<indicator>_lrX_ltY_fit.png        # observed vs fitted
  - <site>_<indicator>_lrX_ltY_summary.txt    # full statsmodels summary + Ljung-Box p
  - ALL_sites_reg44_coeffs.csv                # combined table
"""

import os, re, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox

# -------------------- CONFIG --------------------
INPUTS = {
    "S35629": "S35629_monthly.csv",
    "S35636": "S35636_monthly.csv",
    "Nenthead": "Nenthead_monthly.csv",
}

BESTLAG_CANDIDATES = [
    os.path.join(".", "combined_ccf_bestlags_ge0p30_FIXED2.csv"),
    "combined_ccf_bestlags_ge0p30.csv",
]

MAP_XLSX     = "指标_编号名称对应表.xlsx"   # 可缺省
OUTDIR       = "reg44_outputs"
os.makedirs(OUTDIR, exist_ok=True)

# 指标识别（列名中包含这些编号通常就是金属/离子列）
CODE_WHITELIST = ["50","52","61","106","108","183","3408","6051","6450","6452","6455","6460"]
# 兜底关键词（若你的列名没写编号）
COL_HINTS      = ["_S29","_s29","_S36","_s36","NE","Nent","S356","S29","S36"]

# HAC 滞后、BIC 网格
HAC_MAXLAGS  = 3
BIC_LAG_GRID = (0, 1, 2, 3)

# -------------------- Utils --------------------
def read_mapping(xlsx_path):
    if not os.path.exists(xlsx_path):
        return {}
    try:
        dfm = pd.read_excel(xlsx_path)
        cols = {c.lower(): c for c in dfm.columns}
        idcol   = cols.get("parameter_shortname") or cols.get("parameter_id") or list(dfm.columns)[0]
        namecol = cols.get("parameter_name")      or cols.get("name")            or list(dfm.columns)[1]
        mp = {str(r[idcol]).strip(): str(r[namecol]).strip() for _, r in dfm.iterrows()}
        return mp
    except Exception as e:
        print("映射表读取失败，将不显示友好名称。", e)
        return {}

NAME_MAP = read_mapping(MAP_XLSX)

def pretty_name(col):
    m = re.search(r"(\d{2,4})", str(col))
    code = m.group(1) if m else None
    if code and code in NAME_MAP:
        return f"{NAME_MAP[code]} ({code})"
    return f"{col}" if code is None else f"{col} ({code})"

def load_site(path):
    df = pd.read_csv(path)
    # 时间列
    dcol_candidates = [c for c in df.columns if c.lower().startswith("date")]
    if not dcol_candidates:
        raise KeyError(f"{path}: 未找到以 'date' 开头的时间列")
    dcol = dcol_candidates[0]
    df[dcol] = pd.to_datetime(df[dcol], errors="coerce")
    df = df.sort_values(dcol).reset_index(drop=True)

    # 指标列：包含编号，且不是气象
    met_cols = [c for c in df.columns
                if any(code in c for code in CODE_WHITELIST)
                and ("rain" not in c.lower()) and ("temp" not in c.lower())]
    if len(met_cols) < 8:
        for c in df.columns:
            if any(h in c for h in COL_HINTS) and ("rain" not in c.lower()) and ("temp" not in c.lower()):
                if c not in met_cols:
                    met_cols.append(c)
    if not met_cols:
        raise KeyError(f"{path}: 未识别到指标列，请检查列名是否含编号(如 3408/6455/52)")

    # 气象列
    rcols = [c for c in df.columns if "rain" in c.lower()]
    tcols = [c for c in df.columns if "temp" in c.lower()]
    if not rcols or not tcols:
        raise KeyError(f"{path}: 未找到 rain*/temp* 列")
    rcol, tcol = rcols[0], tcols[0]
    return df, dcol, met_cols, rcol, tcol

def log_transform(df, cols):
    out = {}
    for c in cols:
        if re.search(r"\bph\b", c, flags=re.I):   # pH 不取对数
            out[f"log10 {c}"] = pd.to_numeric(df[c], errors="coerce")
        else:
            x = pd.to_numeric(df[c], errors="coerce")
            x = x.replace([np.inf, -np.inf], np.nan).where(x > 0, np.nan)
            out[f"log10 {c}"] = np.log10(x)
    return pd.DataFrame(out)

def month_dummies(dates):
    mm = pd.Index(dates).month
    d = pd.get_dummies(mm.astype(int), prefix="m", drop_first=True)
    d.index = pd.Index(dates)
    return d

def shift_series(s, k):
    return s.shift(int(k)) if k != 0 else s

def _prep_numeric_design(y_series: pd.Series, X_df: pd.DataFrame):
    """
    清洗 -> 数值化 -> 对齐，返回 (y_arr, X_arr, cols, kept_index)
    解决：dtype=object 报错 & values/index 长度不匹配
    """
    y = pd.to_numeric(y_series, errors="coerce")
    X = X_df.apply(pd.to_numeric, errors="coerce")
    X = X.replace([np.inf, -np.inf], np.nan)
    y = y.replace([np.inf, -np.inf], np.nan)

    Z = pd.concat([y.rename("__y__"), X], axis=1).dropna()
    if Z.empty or Z.shape[0] < 24:
        return None, None, None, None

    kept_index = Z.index
    y_clean = Z["__y__"]
    X_clean = Z.drop(columns=["__y__"])

    # 删除零方差列
    nunique = X_clean.nunique(dropna=True)
    keep_cols = nunique[nunique > 1].index.tolist()
    X_clean = X_clean[keep_cols]

    y_arr = y_clean.to_numpy(dtype=float)
    X_arr = X_clean.to_numpy(dtype=float)
    return y_arr, X_arr, X_clean.columns.tolist(), kept_index

# ---------- best-lag 兼容读入 ----------
def _extract_code_from_col(colname: str):
    m = re.search(r"(\d{2,4})", str(colname))
    return int(m.group(1)) if m else None

def load_and_normalize_bestlag(candidates):
    path = None
    for p in candidates:
        if os.path.exists(p):
            path = p; break
    if path is None:
        print("[提示] 未找到最佳滞后表，将对所有指标使用 BIC 网格搜索。")
        return None
    df = pd.read_csv(path)
    if "best_lag" not in df.columns and "best_lag_pwhite" in df.columns:
        df["best_lag"] = df["best_lag_pwhite"]
    if "ccf" not in df.columns and "ccf_pwhite" in df.columns:
        df["ccf"] = df["ccf_pwhite"]
    if "Driver" not in df.columns and "met" in df.columns:
        df["Driver"] = df["met"].map({"rainfall_mm":"Rainfall","temperature_C":"Temperature"}).fillna(df["met"])
    if "indicator_code" in df.columns:
        df["indicator_code"] = pd.to_numeric(df["indicator_code"], errors="coerce").astype("Int64")
    need = {"site","Driver","best_lag"}
    missing = need - set(df.columns)
    if missing:
        raise KeyError(f"best-lag 文件缺少列: {missing}")
    print(f"[OK] 使用最佳滞后表: {path}")
    return df

def pick_fixed_lags(bestlag_df: pd.DataFrame, site: str, indicator_colname: str):
    """返回 (lag_rain, lag_temp) 或 None。优先按 (site, indicator_code, Driver) 精确匹配。"""
    if bestlag_df is None:
        return None
    sub = bestlag_df[bestlag_df["site"] == site].copy()
    code = _extract_code_from_col(indicator_colname)
    if code is not None and "indicator_code" in sub.columns:
        sub = sub[(sub["indicator_code"] == code)]

    def _get_lag(_drv_key):
        g = sub[sub["Driver"].str.contains(_drv_key, case=False, na=False)]
        if len(g):
            g2 = g.copy()
            if "ccf" in g2.columns:
                g2 = g2.iloc[g2["ccf"].abs().argsort()[::-1]]  # |ccf| 最大优先
            return int(g2["best_lag"].iloc[0])
        return None

    lr = _get_lag("Rain")
    lt = _get_lag("Temp")
    if (lr is None) and (lt is None):
        return None
    return (0 if lr is None else lr, 0 if lt is None else lt)

# -------------------- Core: regression per site --------------------
def grid_BIC(y, X_base, r, t, lag_grid=BIC_LAG_GRID):
    """小网格搜索 (lr, lt) 以最小化 BIC；返回 (bic_min, lr_best, lt_best)。"""
    best = (np.inf, 0, 0)
    for lr in lag_grid:
        for lt in lag_grid:
            X = X_base.copy()
            X["rain_lag"] = shift_series(r, lr)
            X["temp_lag"] = shift_series(t, lt)
            XX = sm.add_constant(X, has_constant="add")
            y_arr, X_arr, cols, kept_idx = _prep_numeric_design(y, XX)
            if y_arr is None or len(y_arr) < 24:
                continue
            mod = sm.OLS(y_arr, X_arr, hasconst=True).fit(
                cov_type="HAC", cov_kwds={"maxlags": HAC_MAXLAGS}
            )
            if mod.bic < best[0]:
                best = (mod.bic, lr, lt)
    return best

def run_expl_reg(site_name, csv_path, bestlag_df=None, lag_grid=BIC_LAG_GRID):
    df, dcol, met_cols, rcol, tcol = load_site(csv_path)
    D = df[[dcol, rcol, tcol] + met_cols].copy().sort_values(dcol)
    D.index = D[dcol]

    logM = log_transform(D, met_cols)
    Xbase = month_dummies(D.index)
    r = pd.to_numeric(D[rcol], errors="coerce")
    t = pd.to_numeric(D[tcol], errors="coerce")

    results_rows = []
    for raw_col, log_col in zip(met_cols, logM.columns):
        y = logM[log_col].rename(log_col)
        # 先尝试用最佳滞后表固定；否则走 BIC 网格
        fixed = pick_fixed_lags(bestlag_df, site_name, raw_col) if bestlag_df is not None else None
        if fixed is None:
            bic_min, lr, lt = grid_BIC(y, Xbase, r, t, lag_grid=lag_grid)
            if np.isinf(bic_min):
                continue
        else:
            lr, lt = fixed

        # ——统一拟合块（带 kept_idx，避免长度不匹配）——
        X = Xbase.copy()
        X["rain_lag"] = shift_series(r, lr)
        X["temp_lag"] = shift_series(t, lt)
        XX = sm.add_constant(X, has_constant="add")

        y_arr, X_arr, cols, kept_idx = _prep_numeric_design(y, XX)
        if y_arr is None or len(y_arr) < 24:
            continue

        mod = sm.OLS(y_arr, X_arr, hasconst=True).fit(
            cov_type="HAC", cov_kwds={"maxlags": HAC_MAXLAGS}
        )
        bic = mod.bic

        # 结果（只保留 const / rain_lag / temp_lag 到主表）
        coefs, ses, tvals, pvals = mod.params, mod.bse, mod.tvalues, mod.pvalues
        r2, r2a, n = mod.rsquared, mod.rsquared_adj, int(mod.nobs)
        name_to_pos = {name: i for i, name in enumerate(cols)}
        for term in ["const", "rain_lag", "temp_lag"]:
            if term in name_to_pos:
                i = name_to_pos[term]
                results_rows.append({
                    "site": site_name,
                    "indicator": pretty_name(raw_col),
                    "y": y.name,
                    "lag_rain": lr, "lag_temp": lt,
                    "term": term,
                    "coef": coefs[i], "se": ses[i], "t": tvals[i], "p": pvals[i],
                    "R2": r2, "R2_adj": r2a, "BIC": bic, "n": n
                })

        # 可视化：观测 vs 拟合（索引用 kept_idx 对齐）
        out_prefix = os.path.join(OUTDIR, f"{site_name}_{y.name}_lr{lr}_lt{lt}")
        fitted = pd.Series(mod.fittedvalues, index=kept_idx, name="fitted")
        y_used  = y.loc[kept_idx]
        ZZ = pd.concat([y_used, fitted], axis=1)

        plt.figure(figsize=(8,3))
        plt.plot(ZZ.index, ZZ[y.name], label="Observed")
        plt.plot(ZZ.index, ZZ["fitted"], label="Fitted")
        plt.title(f"{site_name} — {pretty_name(raw_col)} | lags (R={lr}, T={lt})")
        plt.legend(); plt.tight_layout()
        plt.savefig(out_prefix+"_fit.png", dpi=200); plt.close()

        # 残差 Ljung–Box（同样对齐）
        resid = pd.Series(mod.resid, index=kept_idx)
        lb_p = acorr_ljungbox(resid, lags=[6,12], return_df=True).iloc[-1]["lb_pvalue"]
        with open(out_prefix+"_summary.txt","w",encoding="utf-8") as f:
            f.write(mod.summary().as_text()+"\n")
            f.write(f"\nLjung-Box p (lag 12) = {lb_p:.4f}\n")

    resdf = pd.DataFrame(results_rows)
    resdf.to_csv(os.path.join(OUTDIR, f"{site_name}_reg44_coeffs.csv"), index=False, encoding="utf-8-sig")
    return resdf

# -------------------- Run all --------------------
def main():
    bestlag_df = load_and_normalize_bestlag(BESTLAG_CANDIDATES)
    all_res = []
    for site, path in INPUTS.items():
        if not os.path.exists(path):
            print(f"[跳过] {site} 未找到：{path}")
            continue
        print(f"== 运行 {site} ==")
        res = run_expl_reg(site, path, bestlag_df=bestlag_df, lag_grid=BIC_LAG_GRID)
        all_res.append(res)

    if all_res:
        big = pd.concat(all_res, axis=0, ignore_index=True)
        big.to_csv(os.path.join(OUTDIR, "ALL_sites_reg44_coeffs.csv"), index=False, encoding="utf-8-sig")
        print("完成。输出目录：", OUTDIR)
    else:
        print("未生成结果，请检查输入文件路径。")

if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    main()
