# -*- coding: utf-8 -*-
"""
Minimal recalibration demo:
- Train on source (S35629), predict on target (Nenthead)
- Then use a small recent window of target data to recalibrate:
  mode='intercept'  -> y_adj = y_pred + b
  mode='affine'     -> y_adj = a * y_pred + b
Outputs:
  - metrics before/after recalibration (log10 与 还原值两种)
  - 迁移对比图（前/后再标定）
Author: you
"""

import json
import os
from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LassoCV, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler

# ---------------------- 参数区（按需改） ----------------------
SOURCE_CSV = "S35629_monthly_forecast.csv"
TARGET_CSV = "Nenthead_monthly_forecast.csv"
TARGET_NAME = "Lead"                 # 预测的金属名称，如 "Lead" 或 "Zinc"
METEO_COLS = ["rainfall_mm", "temperature_C"]
LAGS_Y = [1, 2, 3, 6]                 # 自回归滞后
LAGS_M = [1, 2, 3, 6]                 # 气象滞后
LOG_EPS = 1e-9                        # 防止 log10(0)
MIN_TRAIN = 24                        # 源站点最小训练期（月）
CAL_N = 12                            # 目标站点用于再标定的最近月份个数
RECALI_MODE = "affine"               # 'intercept' 或 'affine'
OUT_DIR = "./recalib_outputs"         # 输出目录
RANDOM_STATE = 42

# ---------------------- 工具函数 ----------------------
def ensure_dir(d):
    if not os.path.exists(d):
        os.makedirs(d)

def to_datetime_index(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    assert date_col in df.columns, f"'{date_col}' not in columns: {df.columns}"
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    return df.set_index(date_col).sort_index()

def safe_log10(x: pd.Series) -> pd.Series:
    return np.log10(np.maximum(x.astype(float), LOG_EPS))

def inv_log10(x: np.ndarray) -> np.ndarray:
    return np.power(10.0, x)

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mape_level(y_true_level, y_pred_level, eps=1e-6) -> float:
    yt = np.maximum(y_true_level, eps)
    return float(np.mean(np.abs((y_pred_level - yt) / yt)) * 100.0)

def build_lag_features(df: pd.DataFrame,
                       target: str,
                       meteo_cols: List[str],
                       lags_y: List[int],
                       lags_m: List[int],
                       use_log=True) -> pd.DataFrame:
    """
    返回含有 y（目标列，log10）与全滞后特征的 DataFrame（dropna 后对齐）
    """
    d = df.copy()
    # 目标列 -> log10
    if use_log:
        d[f"log10_{target}"] = safe_log10(d[target])
        y_col = f"log10_{target}"
    else:
        y_col = target

    # 自回归滞后
    for L in lags_y:
        d[f"{y_col}_lag{L}"] = d[y_col].shift(L)

    # 气象滞后（气象使用原始值）
    for m in meteo_cols:
        for L in lags_m:
            d[f"{m}_lag{L}"] = d[m].shift(L)

    cols = [y_col] + [c for c in d.columns if c.endswith(tuple([f"lag{L}" for L in set(lags_y + lags_m)]))]
    d = d[cols].dropna().copy()
    d.rename(columns={y_col: "y"}, inplace=True)
    return d

@dataclass
class SimpleARX:
    """ Lasso 选特征 + Ridge 拟合（线性可解释） """
    alphas: Tuple[float, ...] = (0.001, 0.01, 0.1, 1.0)
    ridge_alpha: float = 1.0
    scaler: StandardScaler = None
    lasso: LassoCV = None
    ridge: Ridge = None
    selected_: List[str] = None
    feat_cols_: List[str] = None

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.feat_cols_ = list(X.columns)
        # 标准化 + Lasso 选特征（防止小样本过拟合）
        self.scaler = StandardScaler().fit(X)
        Xz = self.scaler.transform(X)
        self.lasso = LassoCV(alphas=self.alphas, cv=min(5, max(2, len(y)//6)), random_state=RANDOM_STATE).fit(Xz, y)
        mask = np.abs(self.lasso.coef_) > 1e-6
        if mask.sum() == 0:
            # 若一个都没选上，保底用所有特征
            self.selected_ = self.feat_cols_
        else:
            self.selected_ = list(np.array(self.feat_cols_)[mask])
        # Ridge 在选中的特征上再拟合
        Xsel = X[self.selected_].values
        Xsel_z = StandardScaler().fit_transform(Xsel)  # 为稳定，再做一次局部标准化
        self.ridge = Ridge(alpha=self.ridge_alpha, random_state=RANDOM_STATE).fit(Xsel_z, y)
        self._sel_scaler = StandardScaler().fit(Xsel)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy()
        if self.selected_ is None:
            # 未训练，返回全零
            return np.zeros(len(X))
        Xsel = X[self.selected_].values
        Xsel_z = self._sel_scaler.transform(Xsel)
        return self.ridge.predict(Xsel_z)

# ---------------------- 主流程 ----------------------
def main():
    ensure_dir(OUT_DIR)

    # 读入并设定时间索引
    src = to_datetime_index(pd.read_csv(SOURCE_CSV))
    tgt = to_datetime_index(pd.read_csv(TARGET_CSV))

    # 构造特征（log10）
    src_feat = build_lag_features(src, TARGET_NAME, METEO_COLS, LAGS_Y, LAGS_M, use_log=True)
    tgt_feat = build_lag_features(tgt, TARGET_NAME, METEO_COLS, LAGS_Y, LAGS_M, use_log=True)

    # 源站点训练数据
    if len(src_feat) < MIN_TRAIN:
        print(f"[WARN] Source training rows only {len(src_feat)} < MIN_TRAIN={MIN_TRAIN}")
    train_src = src_feat.copy()

    X_src, y_src = train_src.drop(columns=["y"]), train_src["y"]
    model = SimpleARX()
    model.fit(X_src, y_src)

    # 目标站点直接迁移预测
    X_tgt, y_tgt = tgt_feat.drop(columns=["y"]), tgt_feat["y"]
    y_pred_mig = model.predict(X_tgt)  # 这是 log10 尺度

    # 评估（迁移前）
    metrics_before = evaluate_metrics(y_tgt.values, y_pred_mig)
    print("[INFO] Migration (before recalibration):", json.dumps(metrics_before, indent=2))

    # ------- 最小再标定：从目标站点最近 CAL_N 个月学习偏移/尺度 -------
    # 对齐可用索引
    common_idx = y_tgt.index.intersection(tgt_feat.index)
    # 可用段
    y_true_full = y_tgt.loc[common_idx]
    y_pred_full = pd.Series(y_pred_mig, index=X_tgt.index).loc[common_idx]

    # 取最近 CAL_N 个月作为标定集（若不足则全用）
    if CAL_N <= 0 or CAL_N > len(common_idx):
        cal_idx = common_idx
    else:
        cal_idx = common_idx[-CAL_N:]

    y_cal_true = y_true_full.loc[cal_idx].values
    y_cal_pred = y_pred_full.loc[cal_idx].values

    if RECALI_MODE.lower() == "intercept":
        # y_adj = y_pred + b
        b = float(np.mean(y_cal_true - y_cal_pred))
        a = 1.0
    elif RECALI_MODE.lower() == "affine":
        # y_true = a * y_pred + b
        # 最小二乘拟合
        A = np.vstack([y_cal_pred, np.ones_like(y_cal_pred)]).T
        sol, _, _, _ = np.linalg.lstsq(A, y_cal_true, rcond=None)
        a, b = float(sol[0]), float(sol[1])
    else:
        raise ValueError("RECALI_MODE must be 'intercept' or 'affine'.")

    # 应用校正
    y_pred_after = a * y_pred_full.values + b
    metrics_after = evaluate_metrics(y_true_full.values, y_pred_after)
    print("[INFO] Migration (after  recalibration):", json.dumps(metrics_after, indent=2))
    print(f"[INFO] Recalibration params: mode={RECALI_MODE}, a={a:.4f}, b={b:.4f}, calib_n={len(cal_idx)}")

    # ------- 可视化：迁移前/后对比（log10） -------
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(y_true_full.index, y_true_full.values, label="Observed (log10)", color="C0")
    ax.plot(y_pred_full.index, y_pred_full.values, label="Predicted (before, log10)", color="C1", alpha=0.6)
    ax.plot(y_true_full.index, y_pred_after, label="Predicted (after, log10)", color="C3")
    ax.axvspan(y_true_full.index.min(), y_true_full.index.max(), color="0.95", zorder=-1)
    ax.set_title(f"Migration {os.path.splitext(SOURCE_CSV)[0]} → {os.path.splitext(TARGET_CSV)[0]} — {TARGET_NAME} (log10)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"migration_{TARGET_NAME}_before_after_log10.png"), dpi=200)

    # 也给一张“还原值”对比图
    lvl_true = inv_log10(y_true_full.values)
    lvl_pred_before = inv_log10(y_pred_full.values)
    lvl_pred_after = inv_log10(y_pred_after)

    fig2, ax2 = plt.subplots(figsize=(10, 6))
    ax2.plot(y_true_full.index, lvl_true, label="Observed (level)", color="C0")
    ax2.plot(y_pred_full.index, lvl_pred_before, label="Predicted (before, level)", color="C1", alpha=0.6)
    ax2.plot(y_true_full.index, lvl_pred_after, label="Predicted (after, level)", color="C3")
    ax2.set_title(f"Migration {os.path.splitext(SOURCE_CSV)[0]} → {os.path.splitext(TARGET_CSV)[0]} — {TARGET_NAME} (level)")
    ax2.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"migration_{TARGET_NAME}_before_after_level.png"), dpi=200)

    # 保存指标
    with open(os.path.join(OUT_DIR, f"migration_{TARGET_NAME}_metrics_before.json"), "w") as f:
        json.dump(metrics_before, f, indent=2)
    with open(os.path.join(OUT_DIR, f"migration_{TARGET_NAME}_metrics_after.json"), "w") as f:
        json.dump(metrics_after, f, indent=2)

    print("[DONE] Plots & metrics saved to:", OUT_DIR)


def evaluate_metrics(y_true_log, y_pred_log) -> dict:
    """ 同时给出 log10 尺度与还原值（水平）的指标 """
    rmse_log = rmse(y_true_log, y_pred_log)
    mae_log = mean_absolute_error(y_true_log, y_pred_log)

    y_true_level = inv_log10(y_true_log)
    y_pred_level = inv_log10(y_pred_log)

    rmse_lvl = rmse(y_true_level, y_pred_level)
    mae_lvl = mean_absolute_error(y_true_level, y_pred_level)
    mape_lvl = mape_level(y_true_level, y_pred_level)

    return {
        "log10": {"RMSE": rmse_log, "MAE": mae_log},
        "level": {"RMSE": rmse_lvl, "MAE": mae_lvl, "MAPE_%": mape_lvl}
    }


if __name__ == "__main__":
    main()
