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


def _skew_np(x: np.ndarray) -> float:
    x = x.astype(np.float64, copy=False)
    if x.size < 3:
        return 0.0
    m = float(np.mean(x))
    s = float(np.std(x))
    if not np.isfinite(s) or s == 0.0:
        return 0.0
    z = (x - m) / s
    return float(np.mean(z**3))


def _kurt_np(x: np.ndarray) -> float:
    x = x.astype(np.float64, copy=False)
    if x.size < 4:
        return 0.0
    m = float(np.mean(x))
    s = float(np.std(x))
    if not np.isfinite(s) or s == 0.0:
        return 0.0
    z = (x - m) / s
    return float(np.mean(z**4) - 3.0)


def _resample_linear(t: np.ndarray, x: np.ndarray, n_grid: int = 64) -> tuple[np.ndarray, float]:
    """Resample (t, x) to a uniform grid via linear interpolation.

    Returns (x_grid, dt) where dt is the grid spacing in the same units as t.
    """
    t = t.astype(np.float64, copy=False)
    x = x.astype(np.float64, copy=False)
    if t.size < 2:
        return np.full(n_grid, float(x[0]) if x.size else 0.0, dtype=np.float64), 1.0

    t0 = float(np.nanmin(t))
    t1 = float(np.nanmax(t))
    if not np.isfinite(t0) or not np.isfinite(t1) or t1 <= t0:
        return np.full(n_grid, float(np.nanmean(x)) if x.size else 0.0, dtype=np.float64), 1.0

    order = np.argsort(t)
    tt = t[order]
    xx = x[order]

    # de-duplicate time points (np.interp expects increasing x)
    uniq_t, uniq_idx = np.unique(tt, return_index=True)
    uniq_x = xx[uniq_idx]

    grid = np.linspace(t0, t1, n_grid)
    x_grid = np.interp(grid, uniq_t, uniq_x)
    dt = (t1 - t0) / max(n_grid - 1, 1)
    return x_grid.astype(np.float64, copy=False), float(dt)


def _linear_regression_1d(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
    """Return (slope, intercept, r2) for y ~ a + b x."""
    x = x.astype(np.float64, copy=False)
    y = y.astype(np.float64, copy=False)
    if x.size < 2:
        return 0.0, float(np.mean(y) if y.size else 0.0), 0.0

    xm = float(np.mean(x))
    ym = float(np.mean(y))
    xv = x - xm
    yv = y - ym
    den = float(np.sum(xv * xv))
    if not np.isfinite(den) or den == 0.0:
        return 0.0, ym, 0.0
    slope = float(np.sum(xv * yv) / den)
    intercept = float(ym - slope * xm)
    y_hat = intercept + slope * x
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - ym) ** 2))
    r2 = 0.0 if ss_tot == 0.0 else float(1.0 - ss_res / ss_tot)
    return slope, intercept, r2


def _threshold_for_target_positives(proba: np.ndarray, target_pos: int) -> float:
    """Pick a threshold that yields approximately `target_pos` positives."""
    p = np.asarray(proba, dtype=np.float64)
    n = p.size
    if n == 0:
        return 0.5
    target_pos = int(np.clip(target_pos, 0, n))
    if target_pos == 0:
        return float(np.nanmax(p) + 1e-12)
    if target_pos == n:
        return float(np.nanmin(p) - 1e-12)
    # kth largest => threshold at (n - target_pos)
    kth = int(n - target_pos)
    thr = float(np.partition(p, kth)[kth])
    return thr


def _print_threshold_sweep(proba: np.ndarray, targets: list[int]) -> None:
    p = np.asarray(proba, dtype=np.float64)
    print("Threshold sweep (target positives -> threshold):")
    for t in targets:
        thr = _threshold_for_target_positives(p, int(t))
        pred = (p >= thr)
        print(f"  {int(t):5d} -> thr={thr:.6f} | pos={int(pred.sum())}")


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
    def _wmean_apply(g: pd.DataFrame) -> float:
        return _weighted_mean(g["Flux_corr"], g["Flux_err_corr"])

    try:
        wmean = gf.apply(_wmean_apply, include_groups=False)
    except TypeError:
        wmean = gf.apply(_wmean_apply)
    wmean.name = "flux_wmean"

    # Skew/kurt per filter
    skew = gf["Flux_corr"].apply(_safe_skew).rename("flux_skew")
    kurt = gf["Flux_corr"].apply(_safe_kurt).rename("flux_kurt")

    # Peak timing + physics-ish shape/fit features inspired by top solutions
    def _advanced_features(g: pd.DataFrame) -> pd.Series:
        gg = g.sort_values("t0")
        t = gg["t0"].to_numpy(dtype=np.float64)
        f = gg["Flux_corr"].to_numpy(dtype=np.float64)
        e = gg["Flux_err_corr"].to_numpy(dtype=np.float64)
        snr = gg["snr_corr"].to_numpy(dtype=np.float64)

        t_start = float(t[0])
        t_end = float(t[-1])
        f_start = float(f[0])
        f_end = float(f[-1])
        if f.size and np.isfinite(f).any():
            f_peak_idx = int(np.argmax(np.where(np.isfinite(f), f, -np.inf)))
            idx_max = f_peak_idx
            t_peak = float(t[idx_max])
            f_peak = float(f[idx_max]) if np.isfinite(f[idx_max]) else 0.0
        else:
            idx_max = 0
            t_peak = float(t[0]) if t.size else 0.0
            f_peak = 0.0

        eps = 1e-6
        rise = (f_peak - f_start) / max(t_peak - t_start, eps)
        decay = (f_end - f_peak) / max(t_end - t_peak, eps)

        # Negative flux diagnostics (use significance when possible)
        e_safe = np.where(np.isfinite(e) & (e > 0), e, np.nan)
        neg_sig = f < (-3.0 * e_safe)
        neg_sig_frac = float(np.nanmean(neg_sig)) if neg_sig.size else 0.0
        neg_frac = float(np.mean(f < 0.0)) if f.size else 0.0

        # Physical binning relative to the peak (captures rise/decay morphology)
        # (chosen to be coarse + stable; avoids overfitting)
        rel = t - t_peak
        edges = np.array([-1e18, -40, -20, -10, -5, 0, 5, 10, 20, 40, 1e18], dtype=np.float64)
        bin_means: dict[str, float] = {}
        for i in range(len(edges) - 1):
            m = (rel >= edges[i]) & (rel < edges[i + 1])
            if not np.any(m):
                bin_means[f"phys_bin_{i}_mean"] = 0.0
                bin_means[f"phys_bin_{i}_n"] = 0.0
            else:
                vals = f[m]
                bin_means[f"phys_bin_{i}_mean"] = float(np.nanmean(vals)) if np.isfinite(vals).any() else 0.0
                bin_means[f"phys_bin_{i}_n"] = float(np.sum(m))

        # Power-law decay fit post-peak: log(F) ~ a + alpha log(t)
        post = (rel > 0) & (f > 0) & (snr > 2)
        if np.sum(post) >= 3:
            x = np.log(rel[post] + 1.0)
            y = np.log(f[post])
            alpha, intercept, r2 = _linear_regression_1d(x, y)
            powerlaw_alpha = alpha
            powerlaw_r2 = r2
            powerlaw_alpha_err = float(abs(alpha - (-5.0 / 3.0)))
        else:
            powerlaw_alpha = 0.0
            powerlaw_r2 = 0.0
            powerlaw_alpha_err = 0.0

        # Fireball rise fit pre-peak: log(F) ~ a + beta log(dt)
        pre = (rel < 0) & (f > 0) & (snr > 2)
        if np.sum(pre) >= 3:
            dt = -rel[pre]
            x = np.log(dt + 1.0)
            y = np.log(f[pre])
            beta, intercept2, r2b = _linear_regression_1d(x, y)
            rise_beta = beta
            rise_r2 = r2b
            rise_beta_err = float(abs(beta - 2.0))
        else:
            rise_beta = 0.0
            rise_r2 = 0.0
            rise_beta_err = 0.0

        # Template matching chi^2 (very lightweight): rise~t^2, decay~t^{-5/3}
        # Scale template to peak flux.
        if f.size >= 3 and np.isfinite(f_peak) and np.isfinite(t_peak):
            tmpl = np.ones_like(rel, dtype=np.float64)
            pre_m = rel < 0
            post_m = rel > 0
            tmpl[pre_m] = (np.abs(rel[pre_m]) + 1.0) ** 2.0
            tmpl[post_m] = (rel[post_m] + 1.0) ** (-5.0 / 3.0)
            f_hat = f_peak * tmpl
            w = np.where(np.isfinite(e_safe) & (e_safe > 0), 1.0 / (e_safe**2), 0.0)
            num = np.sum(w * (f - f_hat) ** 2)
            den = np.sum(w) + 1e-12
            template_chisq_tde = float(num / den)
        else:
            template_chisq_tde = 0.0

        # Resampled (interpolated) stats + FFT features (captures quasi-periodic variability)
        xg, dt_grid = _resample_linear(t, f, n_grid=64)
        xg0 = xg - float(np.mean(xg))
        resampled_skew = _skew_np(xg)
        resampled_kurt = _kurt_np(xg)
        if dt_grid > 0:
            spec = np.fft.rfft(xg0)
            pwr = (spec.real**2 + spec.imag**2).astype(np.float64, copy=False)
            pwr_mean = float(np.mean(pwr))
            if pwr.size > 1:
                k = int(np.argmax(pwr[1:]) + 1)
                dom_power = float(pwr[k])
                dom_freq = float(k / (xg0.size * dt_grid))
                dom_power_ratio = float(dom_power / (float(np.sum(pwr)) + 1e-12))
            else:
                dom_freq = 0.0
                dom_power_ratio = 0.0
        else:
            pwr_mean = 0.0
            dom_freq = 0.0
            dom_power_ratio = 0.0

        return pd.Series(
            {
                "t_peak": t_peak,
                "f_peak": f_peak,
                "f_start": f_start,
                "f_end": f_end,
                "rise_rate": float(rise),
                "decay_rate": float(decay),
                "neg_sig_frac": neg_sig_frac,
                "neg_frac": neg_frac,
                "powerlaw_alpha": float(powerlaw_alpha),
                "powerlaw_r2": float(powerlaw_r2),
                "powerlaw_alpha_err": float(powerlaw_alpha_err),
                "fireball_beta": float(rise_beta),
                "fireball_r2": float(rise_r2),
                "fireball_beta_err": float(rise_beta_err),
                "template_chisq_tde": template_chisq_tde,
                "resampled_skew": float(resampled_skew),
                "resampled_kurt": float(resampled_kurt),
                "fft_mean_power": float(pwr_mean),
                "fft_dom_freq": float(dom_freq),
                "fft_dom_power_ratio": float(dom_power_ratio),
                **bin_means,
            }
        )

    try:
        adv = gf.apply(_advanced_features, include_groups=False)
    except TypeError:
        adv = gf.apply(_advanced_features)

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
        .join(adv, how="left")
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

    # Color evolution proxies via physical bins (min/max over bins)
    # Example: g-r difference computed per bin where both are present.
    for a, b, nm in [("g", "r", "gr"), ("r", "i", "ri"), ("i", "z", "iz")]:
        diffs: list[pd.Series] = []
        for i in range(10):
            ca = f"{a}__phys_bin_{i}_mean"
            cb = f"{b}__phys_bin_{i}_mean"
            if ca in out.columns and cb in out.columns:
                diffs.append(out[ca] - out[cb])
        if diffs:
            mat = np.vstack([d.to_numpy(dtype=np.float64) for d in diffs])
            valid = ~np.isnan(mat)
            # Avoid RuntimeWarnings on all-NaN columns by using where+initial.
            vmin = np.nanmin(mat, axis=0, where=valid, initial=0.0)
            vmax = np.nanmax(mat, axis=0, where=valid, initial=0.0)
            cnt = np.sum(valid, axis=0)
            vmean = np.where(cnt > 0, np.nansum(mat, axis=0) / np.maximum(cnt, 1), 0.0)
            out[f"color_{nm}_bin_min"] = vmin
            out[f"color_{nm}_bin_max"] = vmax
            out[f"color_{nm}_bin_mean"] = vmean

    return out


def _prep_meta(log_df: pd.DataFrame) -> pd.DataFrame:
    df = log_df.copy()
    for col in ["Z", "Z_err", "EBV"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    if "Z_err" in df.columns:
        df["Z_err_missing"] = df["Z_err"].isna().astype(np.int8)
        df["Z_err"] = df["Z_err"].fillna(0.0)

    # Keep raw split for grouped CV; do NOT one-hot encode it.
    # Using split as a feature can overfit to split-specific artifacts.
    if "split" in df.columns:
        df["split"] = df["split"].astype("string")

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
    best_threshold_median: float
    cv_f1_at_best_threshold: float
    cv_f1_at_median_threshold: float
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


def cross_validated_oof_and_test_proba(
    X: pd.DataFrame,
    y: np.ndarray,
    X_test: pd.DataFrame,
    groups: np.ndarray | None = None,
    n_splits: int = 5,
    seed: int = 42,
) -> tuple[CVResult, np.ndarray]:
    from sklearn.model_selection import StratifiedKFold

    X_np = X.to_numpy(dtype=np.float32)
    X_test_np = X_test.to_numpy(dtype=np.float32)
    oof = np.zeros(len(y), dtype=np.float32)
    test_sum = np.zeros(X_test_np.shape[0], dtype=np.float32)

    best_iters: list[int] = []
    fold_thresholds: list[float] = []

    if groups is not None:
        try:
            from sklearn.model_selection import StratifiedGroupKFold  # type: ignore

            splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
            split_iter = splitter.split(X_np, y, groups)
        except Exception:
            # Fallback: group-only split (not stratified)
            from sklearn.model_selection import GroupKFold

            splitter = GroupKFold(n_splits=n_splits)
            split_iter = splitter.split(X_np, y, groups)
    else:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        split_iter = splitter.split(X_np, y)

    for fold, (tr_idx, va_idx) in enumerate(split_iter, start=1):
        model = _make_model(y[tr_idx], seed=seed + fold)
        model, best_iter = _fit_with_optional_early_stopping(model, X_np[tr_idx], y[tr_idx], X_np[va_idx], y[va_idx])
        proba = model.predict_proba(X_np[va_idx])[:, 1]
        oof[va_idx] = proba.astype(np.float32)
        fold_thr, _ = _find_best_threshold(y[va_idx], proba)
        fold_thresholds.append(float(fold_thr))

        test_sum += model.predict_proba(X_test_np)[:, 1].astype(np.float32)
        if best_iter is not None:
            best_iters.append(best_iter)
        print(f"fold {fold}/{n_splits}: done")

    thr, f1 = _find_best_threshold(y, oof)
    thr_med = float(np.median(np.asarray(fold_thresholds, dtype=np.float64))) if fold_thresholds else thr
    f1_med = _f1_score_np(y, (oof >= thr_med).astype(np.int8))
    mean_best_iter = int(np.mean(best_iters)) if best_iters else None
    cv = CVResult(
        oof_proba=oof,
        best_threshold=float(thr),
        best_threshold_median=float(thr_med),
        cv_f1_at_best_threshold=float(f1),
        cv_f1_at_median_threshold=float(f1_med),
        best_iterations_mean=mean_best_iter,
    )
    test_mean = test_sum / float(n_splits)
    return cv, test_mean


def _parse_seeds() -> list[int]:
    """Parse ensemble seeds from env.

    - If MALLORN_SEEDS is set (comma-separated ints), use it.
    - Else use MALLORN_N_SEEDS (default 3) starting from base seed 42.
    """
    raw = os.environ.get("MALLORN_SEEDS", "").strip()
    if raw:
        out: list[int] = []
        for part in raw.split(","):
            part = part.strip()
            if not part:
                continue
            out.append(int(part))
        return out or [42]

    n = int(os.environ.get("MALLORN_N_SEEDS", "3").strip() or "3")
    n = max(1, min(n, 10))
    return [42 + i for i in range(n)]


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

    # Optional grouped CV (recommended when you suspect split-specific artifacts).
    use_group_cv = os.environ.get("MALLORN_GROUP_CV", "0").strip().lower() in {"1", "true", "yes"}
    groups = train["split"].to_numpy() if (use_group_cv and "split" in train.columns) else None

    # drop split from the feature matrix; we only use it as a grouping variable
    X_train = train.drop(columns=["object_id", "target", "split"], errors="ignore")
    X_test = test.drop(columns=["object_id", "split"], errors="ignore")

    # Align columns and fill missing with 0
    X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0.0)

    # Replace any remaining NaNs
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    X_test = X_test.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    print(f"Train shape: {X_train.shape} | Test shape: {X_test.shape} | Pos rate: {y.mean():.4f}")

    seeds = _parse_seeds()
    print(f"Ensembling over seeds: {seeds} | GroupCV: {use_group_cv}")

    oof_sum = np.zeros(len(y), dtype=np.float64)
    test_sum = np.zeros(X_test.shape[0], dtype=np.float64)
    med_thrs: list[float] = []
    best_iters: list[int] = []

    for s in seeds:
        cv, test_proba_s = cross_validated_oof_and_test_proba(X_train, y, X_test, groups=groups, n_splits=5, seed=int(s))
        print(f"seed {s}: OOF F1 best={cv.cv_f1_at_best_threshold:.5f} | median_thr={cv.best_threshold_median:.5f}")
        oof_sum += cv.oof_proba.astype(np.float64)
        test_sum += test_proba_s.astype(np.float64)
        med_thrs.append(float(cv.best_threshold_median))
        if cv.best_iterations_mean is not None:
            best_iters.append(int(cv.best_iterations_mean))

    oof_mean = (oof_sum / float(len(seeds))).astype(np.float32)
    test_proba = (test_sum / float(len(seeds))).astype(np.float32)

    thr_global, f1_global = _find_best_threshold(y, oof_mean)
    thr_med = float(np.median(np.asarray(med_thrs, dtype=np.float64))) if med_thrs else float(thr_global)
    f1_med = _f1_score_np(y, (oof_mean >= thr_med).astype(np.int8))
    print(f"Ensemble OOF best threshold: {thr_global:.5f} | OOF F1: {f1_global:.5f}")
    print(f"Ensemble median threshold: {thr_med:.5f} | OOF F1: {f1_med:.5f}")
    if best_iters:
        print(f"Mean best_iteration (XGB): {int(np.mean(best_iters))}")

    # Robust choice used by multiple top solutions: median of fold-wise thresholds
    thr_mode = os.environ.get("MALLORN_THR_MODE", "best").strip().lower()
    if thr_mode in {"median", "med"}:
        chosen_thr = thr_med
    else:
        # default: maximize OOF F1
        chosen_thr = float(thr_global)

    # Optional: override threshold by targeting a specific number of positives.
    # This is a common leaderboard-tuning trick for F1 competitions.
    target_pos = os.environ.get("MALLORN_TARGET_POS", "").strip()
    if target_pos:
        try:
            chosen_thr = _threshold_for_target_positives(test_proba, int(target_pos))
            print(f"Using MALLORN_TARGET_POS={int(target_pos)} => threshold={chosen_thr:.6f}")
        except Exception:
            print("Warning: invalid MALLORN_TARGET_POS; ignoring.")

    # Print a small sweep around typical ranges seen in top writeups.
    _print_threshold_sweep(test_proba, targets=[350, 400, 415, 450, 500, 550, 600])
    test_pred = (test_proba >= chosen_thr).astype(np.int8)
    print(f"Predicted positives in test: {int(test_pred.sum())} (threshold={chosen_thr:.5f})")

    sub = pd.DataFrame({"object_id": test_log["object_id"].astype(str), "prediction": test_pred})
    out_path = Path("submission.csv")
    sub.to_csv(out_path, index=False)
    print(f"Wrote: {out_path.resolve()}")


if __name__ == "__main__":
    main()
