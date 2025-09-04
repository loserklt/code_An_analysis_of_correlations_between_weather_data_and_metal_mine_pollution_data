
"""
ts_multifeature_forecast_local_short_compat.py
- Short-sample configuration + sklearn compatibility (handles old versions without `squared` arg).
- Reads two CSVs from the SAME folder as this script:
    S35629_monthly_forecast.csv
    Nenthead_monthly_forecast.csv
- Outputs to ./ts_multi_outputs_short/
"""

import pandas as pd, numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json, warnings
warnings.filterwarnings("ignore", category=FutureWarning)

# ------------------ CONFIG (short-sample) ------------------
HERE = Path(__file__).resolve().parent
S29_FILE = "S35629_monthly_forecast.csv"
NE_FILE  = "Nenthead_monthly_forecast.csv"
OUTDIR = HERE / "ts_multi_outputs_short"
OUTDIR.mkdir(parents=True, exist_ok=True)

TARGET_NAME = "Lead"   # change to 'Zinc' to run Zn

# shorter lags to keep more rows after dropna()
LAGS_AR     = [1,2,3,6]
LAGS_RAIN   = [1,2,3,6]
LAGS_TEMP   = [1,2,3,6]
LAGS_OTHERS = [1,2,3]   # conservative for other metals

MIN_TRAIN   = 12        # much smaller for S35629
TEST_H      = 1

DATE_ALIASES = ['date','Date','Sampling Date','sampling_date','datetime','timestamp','time','month_date']

# ------------------ helpers ------------------
def rmse_compat(y_true, y_pred):
    """Return RMSE; compatible with old sklearn (no `squared` arg)."""
    try:
        return mean_squared_error(y_true, y_pred, squared=False)
    except TypeError:
        return np.sqrt(mean_squared_error(y_true, y_pred))

def normalize_columns(df):
    df = df.copy()
    df.columns = [str(c).replace('\\ufeff','').strip() for c in df.columns]
    return df

def find_or_build_date(df, fname):
    df = normalize_columns(df.copy())
    # direct match
    for c in df.columns:
        if str(c).strip() in DATE_ALIASES or any(k in str(c).lower() for k in ['date','time','timestamp']):
            df['date'] = pd.to_datetime(df[c], errors='coerce')
            if df['date'].notna().any():
                return df
    # year+month
    ycol = next((c for c in df.columns if str(c).lower() in ['year','yr','yyyy']), None)
    mcol = next((c for c in df.columns if str(c).lower() in ['month','mm','mon']), None)
    if ycol and mcol:
        y = pd.to_numeric(df[ycol], errors='coerce').astype('Int64')
        m = pd.to_numeric(df[mcol], errors='coerce').astype('Int64')
        df['date'] = pd.to_datetime(dict(year=y, month=m, day=1), errors='coerce')
        return df
    # diagnostics
    print("[ERROR] No date column found in:", fname)
    print("Columns:", list(df.columns))
    print(df.head())
    raise KeyError("No date column")

def ensure_monthly(df):
    df = df.sort_values('date')
    df['ym'] = df['date'].dt.to_period('M')
    m = df.groupby('ym').mean(numeric_only=True).reset_index()
    m['date'] = m['ym'].dt.to_timestamp('M')
    m = m.drop(columns=['ym'])
    keep = ['date'] + m.select_dtypes(include=[np.number]).columns.tolist()
    return m[keep]

def log10_with_eps(s: pd.Series):
    s = s.astype(float)
    pos = s[s>0]
    eps = np.nanpercentile(pos, 5)*0.1 if len(pos) else 1e-6
    return np.log10(s + eps)

def make_lags(s, lags, prefix):
    df = pd.DataFrame(index=s.index)
    for k in lags:
        df[f"{prefix}_lag{k}"] = s.shift(k)
    return df

def expanding_backtest(y, X, min_train=12, horizon=1):
    preds = []
    idx = y.index
    # safety guard: do not exceed available length
    min_train = max(6, min(min_train, len(y)-horizon-1))
    if len(y) <= min_train + horizon:
        return pd.DataFrame(columns=['y_true','y_pred']), {"RMSE_log10":np.nan,"MAE_log10":np.nan,"MAPE_level_%":np.nan}

    for t in range(min_train, len(y)-horizon+1):
        y_tr = y.iloc[:t]; X_tr = X.iloc[:t, :]
        y_te = y.iloc[t:t+horizon]; X_te = X.iloc[t:t+horizon, :]

        splits = max(3, min(4, max(2, len(y_tr)//8)))
        tscv = TimeSeriesSplit(n_splits=splits)

        lasso = Pipeline([("scaler", StandardScaler()),
                          ("model", LassoCV(cv=tscv, random_state=42, max_iter=5000))])
        lasso.fit(X_tr, y_tr)
        coef = lasso.named_steps['model'].coef_
        keep = np.where(np.abs(coef)>1e-8)[0]
        if len(keep)==0:
            # fallback: select top-5 by abs corr
            corrs = []
            for i in range(X_tr.shape[1]):
                v = pd.concat([X_tr.iloc[:,i], y_tr], axis=1).dropna()
                c = 0.0 if len(v)<8 else abs(v.iloc[:,0].corr(v.iloc[:,1]))
                corrs.append(c)
            keep = np.argsort(corrs)[-5:]

        Xtr_sel = X_tr.iloc[:, keep]
        Xte_sel = X_te.iloc[:, keep]

        ridge = Pipeline([("scaler", StandardScaler()),
                          ("model", Ridge(alpha=1.0, random_state=42))])
        ridge.fit(Xtr_sel, y_tr)
        yhat = ridge.predict(Xte_sel)

        preds.append({"date": idx[t+horizon-1], "y_true": float(y_te.iloc[horizon-1]), "y_pred": float(yhat[-1])})

    pred_df = pd.DataFrame(preds).set_index("date")
    if len(pred_df)==0:
        return pred_df, {"RMSE_log10":np.nan,"MAE_log10":np.nan,"MAPE_level_%":np.nan}
    rmse = rmse_compat(pred_df['y_true'], pred_df['y_pred'])
    mae  = mean_absolute_error(pred_df['y_true'], pred_df['y_pred'])
    mape = np.mean(np.abs((10**pred_df['y_pred'] - 10**pred_df['y_true']) / np.maximum(1e-6,10**pred_df['y_true']))) * 100.0
    return pred_df, {"RMSE_log10": rmse, "MAE_log10": mae, "MAPE_level_%": mape}

# ------------------ load ------------------
print("[INFO] Reading:", HERE / S29_FILE)
print("[INFO] Reading:", HERE / NE_FILE)
s29 = pd.read_csv(HERE / S29_FILE)
ne  = pd.read_csv(HERE / NE_FILE)
print("[DEBUG] S35629 columns:", list(s29.columns))
print("[DEBUG] Nenthead columns:", list(ne.columns))

s29 = find_or_build_date(s29, S29_FILE)
ne  = find_or_build_date(ne,  NE_FILE)
s29 = ensure_monthly(s29)
ne  = ensure_monthly(ne)

REQUIRED = ['rainfall_mm','temperature_C', TARGET_NAME]
for col in REQUIRED:
    if col not in s29.columns:
        raise ValueError(f"S35629 missing column: {col}")
    if col not in ne.columns:
        raise ValueError(f"Nenthead missing column: {col}")

num_cols = s29.select_dtypes(include=[np.number]).columns.tolist()
exclude  = set(['rainfall_mm','temperature_C', TARGET_NAME])
others   = [c for c in num_cols if c not in exclude]

def build_design(df):
    df = df.copy().set_index('date')
    y = log10_with_eps(df[TARGET_NAME])
    X = make_lags(y, LAGS_AR, 'AR')
    X = pd.concat([X, make_lags(df['rainfall_mm'], LAGS_RAIN, 'rain')], axis=1)
    X = pd.concat([X, make_lags(df['temperature_C'], LAGS_TEMP, 'temp')], axis=1)
    for c in others:
        if c in df.columns:
            X = pd.concat([X, make_lags(df[c], LAGS_OTHERS, c)], axis=1)
    data = pd.concat([y.rename('y'), X], axis=1).dropna()
    return data

train = build_design(s29)
test  = build_design(ne)

print(f"[INFO] Train rows after lags/dropna: {len(train)}  | Test rows: {len(test)}")

# ------------------ backtest ------------------
pred_df, metrics = expanding_backtest(train['y'], train.drop(columns=['y']), min_train=MIN_TRAIN, horizon=TEST_H)
pred_df.to_csv(OUTDIR / f"S35629_{TARGET_NAME}_backtest_preds.csv")
with open(OUTDIR / f"S35629_{TARGET_NAME}_backtest_metrics.json","w") as f:
    json.dump(metrics, f, indent=2)

# final fit on full training with feature selection
splits = max(3, min(4, max(2, len(train)//8)))
tscv = TimeSeriesSplit(n_splits=splits)
lasso = Pipeline([("scaler", StandardScaler()), ("model", LassoCV(cv=tscv, random_state=42, max_iter=5000))])
X_full = train.drop(columns=['y']); y_full = train['y']
lasso.fit(X_full, y_full)
coef = lasso.named_steps['model'].coef_
keep_idx = np.where(np.abs(coef)>1e-8)[0]
if len(keep_idx)==0:
    corrs = []
    for i in range(X_full.shape[1]):
        v = pd.concat([X_full.iloc[:,i], y_full], axis=1).dropna()
        c = 0.0 if len(v)<8 else abs(v.iloc[:,0].corr(v.iloc[:,1]))
        corrs.append(c)
    keep_idx = np.argsort(corrs)[-6:]

selected_cols = X_full.columns[keep_idx].tolist()
pd.Series(selected_cols).to_csv(OUTDIR / f"S35629_{TARGET_NAME}_selected_features.csv", index=False, header=False)

ridge = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0, random_state=42))])
ridge.fit(X_full[selected_cols], y_full)

# in-sample fit plot
fig = plt.figure()
plt.plot(y_full.index, y_full.values, label="Observed (log10)")
plt.plot(y_full.index, ridge.predict(X_full[selected_cols]), label="Fitted (log10)")
plt.title(f"S35629 — {TARGET_NAME}: In-sample fit")
plt.legend(); plt.tight_layout()
fig.savefig(OUTDIR / f"S35629_{TARGET_NAME}_fit.png", dpi=200); plt.close(fig)

# ------------------ migration ------------------
common = [c for c in selected_cols if c in test.columns]
X_te_sel = test[common].dropna()
y_te = test['y'].loc[X_te_sel.index]
yhat = ridge.predict(X_te_sel)

pred_mig = pd.DataFrame({"date": X_te_sel.index, "y_true": y_te.values, "y_pred": yhat}).set_index("date")
pred_mig.to_csv(OUTDIR / f"S35629_to_Nenthead_{TARGET_NAME}_migration_preds.csv")

rmse = rmse_compat(pred_mig['y_true'], pred_mig['y_pred']) if len(pred_mig)>0 else np.nan
mae  = mean_absolute_error(pred_mig['y_true'], pred_mig['y_pred']) if len(pred_mig)>0 else np.nan
mape = np.mean(np.abs((10**pred_mig['y_pred'] - 10**pred_mig['y_true']) / np.maximum(1e-6,10**pred_mig['y_true']))) * 100.0 if len(pred_mig)>0 else np.nan

with open(OUTDIR / f"S35629_to_Nenthead_{TARGET_NAME}_metrics.json","w") as f:
    json.dump({"RMSE_log10": rmse, "MAE_log10": mae, "MAPE_level_%": mape}, f, indent=2)

fig = plt.figure()
plt.plot(pred_mig.index, pred_mig['y_true'], label="Observed (log10)")
plt.plot(pred_mig.index, pred_mig['y_pred'], label="Predicted (log10)")
plt.title(f"Migration: S35629→Nenthead — {TARGET_NAME}")
plt.legend(); plt.tight_layout()
fig.savefig(OUTDIR / f"S35629_to_Nenthead_{TARGET_NAME}_migration_fit.png", dpi=200); plt.close(fig)

print("Done. Outputs in:", OUTDIR)
