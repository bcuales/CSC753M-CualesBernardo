from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# Approximate LSST extinction coefficients (A_lambda / E(B-V))
# Commonly used values (e.g., Schlafly+Finkbeiner style coefficients).
LSST_EXT_COEFF = {
    "u": 4.239,
    "g": 3.303,
    "r": 2.285,
    "i": 1.698,
    "z": 1.263,
    "y": 1.088,
}


def _find_data_dir() -> Path:
    kaggle = Path("/kaggle/input/mallorn-astronomical-classification-challenge")
    if kaggle.exists():
        return kaggle

    here = Path(__file__).resolve().parent
    candidates = [here / "mallorn-astronomical-classification-challenge", here]
    for c in candidates:
        if (c / "train_log.csv").exists() and (c / "test_log.csv").exists():
            return c
    raise FileNotFoundError(
        "Could not locate competition data directory. "
        "Set MALLORN_DATA_DIR to the folder containing train_log.csv/test_log.csv."
    )


def _split_dirs(data_dir: Path) -> list[Path]:
    splits = sorted(data_dir.glob("split_*"))
    if not splits:
        raise FileNotFoundError(f"No split_* folders found under: {data_dir}")
    return splits


def _flatten_columns(columns: pd.MultiIndex) -> list[str]:
    out: list[str] = []
    for metric, flt in columns.to_flat_index():
        out.append(f"{flt}__{metric}")
    return out


def _safe_skew(s: pd.Series) -> float:
    v = s.dropna()
    if len(v) < 3:
        return 0.0
    return float(v.skew())


def _safe_kurt(s: pd.Series) -> float:
    v = s.dropna()
    if len(v) < 4:
        return 0.0
    return float(v.kurt())


def _weighted_mean(x: pd.Series, err: pd.Series) -> float:
    w = 1.0 / (err.replace(0, np.nan) ** 2)
    num = (w * x).sum(skipna=True)
    den = w.sum(skipna=True)
    if not np.isfinite(den) or den == 0:
        return float(x.mean(skipna=True))
    return float(num / den)


def extract_lightcurve_features(lightcurves: pd.DataFrame, ebv_map: pd.Series | None = None) -> pd.DataFrame:
    """Aggregate time-series rows into per-object engineered features.

    Expected columns: object_id, Time (MJD), Flux, Flux_err, Filter
    ebv_map: optional Series indexed by object_id providing EBV.
    """
    df = lightcurves.rename(columns={"Time (MJD)": "time"}).copy()

    df["Flux"] = pd.to_numeric(df["Flux"], errors="coerce")
    df["Flux_err"] = pd.to_numeric(df["Flux_err"], errors="coerce")
    df["time"] = pd.to_numeric(df["time"], errors="coerce")
    df["Filter"] = df["Filter"].astype(str)

    if ebv_map is not None:
        df = df.join(ebv_map.rename("EBV"), on="object_id")
        df["EBV"] = pd.to_numeric(df["EBV"], errors="coerce").fillna(0.0)

        coeff = df["Filter"].map(LSST_EXT_COEFF).fillna(0.0)
        # De-redden flux: F_corr = F * 10^(0.4 * A_lambda), A_lambda = coeff * EBV
        corr = np.power(10.0, 0.4 * coeff.to_numpy(dtype=np.float64) * df["EBV"].to_numpy(dtype=np.float64))
        df["Flux_corr"] = df["Flux"].to_numpy(dtype=np.float64) * corr
        df["Flux_err_corr"] = df["Flux_err"].to_numpy(dtype=np.float64) * corr
    else:
        df["Flux_corr"] = df["Flux"]
        df["Flux_err_corr"] = df["Flux_err"]

    # Avoid division warnings
    err = df["Flux_err"].replace(0, np.nan)
    errc = df["Flux_err_corr"].replace(0, np.nan)

    df["snr"] = df["Flux"] / err
    df["snr_corr"] = df["Flux_corr"] / errc

    # Normalize time per (object, filter) to improve numerical stability
    g0 = df.groupby(["object_id", "Filter"], sort=False)["time"]
    df["t0"] = df["time"] - g0.transform("min")

    df["t2"] = df["t0"] * df["t0"]
    df["tf"] = df["t0"] * df["Flux_corr"]

    gf = df.groupby(["object_id", "Filter"], sort=False)

    # Basic stats + quantiles
    q = gf["Flux_corr"].quantile([0.05, 0.25, 0.75, 0.95]).unstack(level=-1)
    q.columns = ["q05", "q25", "q75", "q95"]

    base = gf.agg(
        n_obs=("Flux_corr", "size"),
        flux_mean=("Flux_corr", "mean"),
        flux_std=("Flux_corr", "std"),
        flux_min=("Flux_corr", "min"),
        flux_max=("Flux_corr", "max"),
        flux_median=("Flux_corr", "median"),
        snr_max=("snr_corr", "max"),
        snr_mean=("snr_corr", "mean"),
        time_min=("t0", "min"),
        time_max=("t0", "max"),
        pos_frac=("Flux_corr", lambda s: float(np.mean(s > 0))),
        n_pos=("Flux_corr", lambda s: int(np.sum(s > 0))),
        n_3sig=("snr_corr", lambda s: int(np.sum(s > 3))),
    )

    # Weighted mean per filter
    wmean = gf.apply(lambda g: _weighted_mean(g["Flux_corr"], g["Flux_err_corr"]))
    wmean.name = "flux_wmean"

    # Skew/kurt per filter
    skew = gf["Flux_corr"].apply(_safe_skew).rename("flux_skew")
    kurt = gf["Flux_corr"].apply(_safe_kurt).rename("flux_kurt")

    # Peak timing & simple rise/decay rates
    def _peak_features(g: pd.DataFrame) -> pd.Series:
        gg = g.sort_values("t0")
        t_start = float(gg["t0"].iloc[0])
        t_end = float(gg["t0"].iloc[-1])
        f_start = float(gg["Flux_corr"].iloc[0])
        f_end = float(gg["Flux_corr"].iloc[-1])
        idx_max = int(gg["Flux_corr"].to_numpy().argmax())
        t_peak = float(gg["t0"].iloc[idx_max])
        f_peak = float(gg["Flux_corr"].iloc[idx_max])
        eps = 1e-6
        rise = (f_peak - f_start) / max(t_peak - t_start, eps)
        decay = (f_end - f_peak) / max(t_end - t_peak, eps)
        return pd.Series({"t_peak": t_peak, "f_start": f_start, "f_end": f_end, "rise_rate": rise, "decay_rate": decay})

    peak = gf.apply(_peak_features)

    # Linear slope in corrected flux vs t0
    sums = gf.agg(
        sum_t=("t0", "sum"),
        sum_f=("Flux_corr", "sum"),
        sum_tt=("t2", "sum"),
        sum_tf=("tf", "sum"),
    )
    denom = base["n_obs"] * sums["sum_tt"] - (sums["sum_t"] ** 2)
    numer = base["n_obs"] * sums["sum_tf"] - sums["sum_t"] * sums["sum_f"]
    slope = np.where(denom != 0, numer / denom, 0.0)
    slope = pd.Series(slope, index=base.index, name="flux_slope")

    agg = (
        base.join(q, how="left")
        .join(wmean, how="left")
        .join(skew, how="left")
        .join(kurt, how="left")
        .join(peak, how="left")
        .join(slope, how="left")
    )

    agg["time_span"] = agg["time_max"] - agg["time_min"]
    agg["iqr"] = agg["q75"] - agg["q25"]
    agg["amp"] = agg["q95"] - agg["q05"]

    wide = agg.unstack("Filter")
    wide.columns = _flatten_columns(wide.columns)  # type: ignore[arg-type]
    wide = wide.reset_index()

    # Overall (all filters)
    go = df.groupby("object_id", sort=False)
    overall = go.agg(
        n_obs_all=("Flux_corr", "size"),
        flux_mean_all=("Flux_corr", "mean"),
        flux_std_all=("Flux_corr", "std"),
        flux_min_all=("Flux_corr", "min"),
        flux_max_all=("Flux_corr", "max"),
        snr_max_all=("snr_corr", "max"),
        n_3sig_all=("snr_corr", lambda s: int(np.sum(s > 3))),
        time_span_all=("t0", lambda s: float(np.nanmax(s) - np.nanmin(s)) if len(s) else 0.0),
    ).reset_index()

    out = wide.merge(overall, on="object_id", how="left")

    # Simple color-like differences between filter medians (when present)
    # (uses corrected flux medians; missing values become 0 after later fill)
    for a, b, name in [("g", "r", "gr"), ("r", "i", "ri"), ("i", "z", "iz"), ("z", "y", "zy"), ("u", "g", "ug")]:
        ca = f"{a}__flux_median"
        cb = f"{b}__flux_median"
        if ca in out.columns and cb in out.columns:
            out[f"color_{name}"] = out[ca] - out[cb]

    return out


def _prep_meta(log_df: pd.DataFrame) -> pd.DataFrame:
    df = log_df.copy()
    for col in ["Z", "Z_err", "EBV"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Z_err" in df.columns:
        df["Z_err_missing"] = df["Z_err"].isna().astype(np.int8)
        df["Z_err"] = df["Z_err"].fillna(0.0)

    # Keep split for feature-building, but also one-hot it for the model
    if "split" in df.columns:
        split_dummies = pd.get_dummies(df["split"].astype("string"), prefix="split", dummy_na=False)
        df = pd.concat([df.drop(columns=["split"], errors="ignore"), split_dummies], axis=1)

    df = df.drop(columns=[c for c in ["SpecType", "English Translation"] if c in df.columns], errors="ignore")
    return df


def build_feature_table(data_dir: Path, kind: str, log_df: pd.DataFrame) -> pd.DataFrame:
    feats: list[pd.DataFrame] = []
    # Map EBV per object (for extinction correction)
    ebv_map = log_df.set_index("object_id")["EBV"] if "EBV" in log_df.columns else None

    for split_dir in _split_dirs(data_dir):
        path = split_dir / f"{kind}_full_lightcurves.csv"
        if not path.exists():
            raise FileNotFoundError(f"Missing expected file: {path}")
        lc = pd.read_csv(path)
        feats.append(extract_lightcurve_features(lc, ebv_map=ebv_map))

    out = pd.concat(feats, axis=0, ignore_index=True)
    out = out.drop_duplicates(subset=["object_id"], keep="first")
    return out


@dataclass(frozen=True)
class CVResult:
    oof_proba: np.ndarray
    best_threshold: float
    cv_f1_at_best_threshold: float
    best_iterations_mean: int | None


def _f1_score_np(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    # Manual F1 (avoids reliance on scipy internals)
    y_true = y_true.astype(np.int8)
    y_pred = y_pred.astype(np.int8)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    if tp == 0:
        return 0.0
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    if precision + recall == 0:
        return 0.0
    return float(2 * precision * recall / (precision + recall))


def _find_best_threshold(y_true: np.ndarray, proba: np.ndarray) -> tuple[float, float]:
    vals = np.unique(proba)
    if vals.size > 3000:
        qs = np.linspace(0.0, 1.0, 3000)
        vals = np.quantile(proba, qs)

    best_thr = 0.5
    best_f1 = -1.0
    for thr in vals:
        pred = (proba >= float(thr)).astype(np.int8)
        f1 = _f1_score_np(y_true, pred)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr, float(best_f1)


def _make_model(y: np.ndarray, seed: int = 42):
    pos = float(np.sum(y == 1))
    neg = float(np.sum(y == 0))
    scale_pos_weight = (neg / max(pos, 1.0))

    try:
        from xgboost import XGBClassifier  # type: ignore

        return XGBClassifier(
            n_estimators=6000,
            learning_rate=0.02,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            min_child_weight=1.0,
            gamma=0.0,
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=seed,
            n_jobs=max(os.cpu_count() or 2, 2),
            scale_pos_weight=scale_pos_weight,
        )
    except Exception:
        from sklearn.ensemble import HistGradientBoostingClassifier

        return HistGradientBoostingClassifier(
            learning_rate=0.05,
            max_depth=7,
            max_iter=800,
            l2_regularization=0.0,
            random_state=seed,
        )


def _fit_with_optional_early_stopping(model, X_tr, y_tr, X_va, y_va):
    # XGBClassifier supports early stopping; sklearn fallback does not.
    if model.__class__.__name__ == "XGBClassifier":
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            verbose=False,
            early_stopping_rounds=200,
        )
        best_iter = getattr(model, "best_iteration", None)
        if best_iter is None:
            best_iter = getattr(model, "best_iteration_", None)
        return model, (int(best_iter) if best_iter is not None else None)

    model.fit(X_tr, y_tr)
    return model, None


def cross_validated_oof_proba(X: pd.DataFrame, y: np.ndarray, n_splits: int = 5, seed: int = 42) -> CVResult:
    from sklearn.model_selection import StratifiedKFold

    X_np = X.to_numpy(dtype=np.float32)
    oof = np.zeros(len(y), dtype=np.float32)

    best_iters: list[int] = []

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_np, y), start=1):
        model = _make_model(y[tr_idx], seed=seed + fold)
        model, best_iter = _fit_with_optional_early_stopping(model, X_np[tr_idx], y[tr_idx], X_np[va_idx], y[va_idx])
        proba = model.predict_proba(X_np[va_idx])[:, 1]
        oof[va_idx] = proba.astype(np.float32)
        if best_iter is not None:
            best_iters.append(best_iter)
        print(f"fold {fold}/{n_splits}: done")

    thr, f1 = _find_best_threshold(y, oof)
    mean_best_iter = int(np.mean(best_iters)) if best_iters else None
    return CVResult(oof_proba=oof, best_threshold=thr, cv_f1_at_best_threshold=f1, best_iterations_mean=mean_best_iter)


def train_full_and_predict(X_train: pd.DataFrame, y: np.ndarray, X_test: pd.DataFrame, seed: int = 42, n_estimators: int | None = None) -> np.ndarray:
    model = _make_model(y, seed=seed)
    if model.__class__.__name__ == "XGBClassifier" and n_estimators is not None:
        try:
            model.set_params(n_estimators=max(int(n_estimators), 50))
        except Exception:
            pass

    model.fit(X_train.to_numpy(dtype=np.float32), y)
    return model.predict_proba(X_test.to_numpy(dtype=np.float32))[:, 1]


def main() -> None:
    data_dir = Path(os.environ.get("MALLORN_DATA_DIR", "")) if os.environ.get("MALLORN_DATA_DIR") else _find_data_dir()
    print(f"Using data_dir: {data_dir}")

    train_log = pd.read_csv(data_dir / "train_log.csv")
    test_log = pd.read_csv(data_dir / "test_log.csv")

    train_meta = _prep_meta(train_log)
    test_meta = _prep_meta(test_log)

    print("Building lightcurve features (train)...")
    train_feats = build_feature_table(data_dir, kind="train", log_df=train_log)
    print("Building lightcurve features (test)...")
    test_feats = build_feature_table(data_dir, kind="test", log_df=test_log)

    train = train_meta.merge(train_feats, on="object_id", how="left")
    test = test_meta.merge(test_feats, on="object_id", how="left")

    if "target" not in train.columns:
        raise ValueError("train_log.csv must contain the target column")

    y = train["target"].to_numpy(dtype=np.int8)

    X_train = train.drop(columns=["object_id", "target"], errors="ignore")
    X_test = test.drop(columns=["object_id"], errors="ignore")

    # Align columns and fill missing with 0
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0.0)

    # Replace any remaining NaNs
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape} | Pos rate: {y.mean():.4f}")

    cv = cross_validated_oof_proba(X_train, y, n_splits=5, seed=42)
    print(f"CV best threshold: {cv.best_threshold:.5f}")
    print(f"OOF F1 @ best threshold: {cv.cv_f1_at_best_threshold:.5f}")
    if cv.best_iterations_mean is not None:
        print(f"Mean best_iteration (XGB): {cv.best_iterations_mean}")

    print("Training full model + predicting test...")
    test_proba = train_full_and_predict(X_train, y, X_test, seed=42, n_estimators=cv.best_iterations_mean)
    test_pred = (test_proba >= cv.best_threshold).astype(np.int8)

    sub = pd.DataFrame({"object_id": test_log["object_id"].astype(str), "prediction": test_pred})
    out_path = Path("submission.csv")
    sub.to_csv(out_path, index=False)
    print(f"Wrote: {out_path.resolve()}")


if __name__ == "__main__":
    main()
