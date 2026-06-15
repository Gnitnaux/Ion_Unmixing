"""
PLSR Unmixing — multi-output PLSR for simultaneous Cu/Fe/Zn quantification
in SERS mixtures.

Adapted from Transfer-Learning-Assisted-SERS (CA_Paper_PLSR_Unmixing).
Uses nested 3-fold (outer) × 2-fold (inner) stratified group CV.
Two response modes: "concentration" (nM) and "ratio" (0-1).

X.T.Liu 20260615
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import r2_score, mean_squared_error

from src.utils import read_data, preprocess_data

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

ANALYTES = ["Cu", "Fe", "Zn"]
CONC_COLS = ["conc_Cu", "conc_Fe", "conc_Zn"]
VALID_MIXTURES = [
    "BA", "Cu", "Fe", "Zn",
    "Cu+Fe", "Cu+Zn", "Fe+Zn",
    "Cu+Fe+Zn",
]
N_OUTER = 3
N_INNER = 2
MAX_COMP = 10
RANDOM_STATE = 2026

MODEL_FILENAME = "plsr_unmixing.joblib"


# ---------------------------------------------------------------------------
# Helpers: data filtering & mixture labelling
# ---------------------------------------------------------------------------

def _make_mixture_label(conc_cu, conc_fe, conc_zn):
    """Determine mixture type from Cu/Fe/Zn concentrations (nM).

    Returns one of VALID_MIXTURES (e.g. "Cu+Fe", "Zn", "BA").
    """
    parts = []
    if conc_cu > 0:
        parts.append("Cu")
    if conc_fe > 0:
        parts.append("Fe")
    if conc_zn > 0:
        parts.append("Zn")
    return "BA" if len(parts) == 0 else "+".join(parts)


def _make_group_id(group_number, conc_cu, conc_fe, conc_zn):
    """Create a unique group identifier string."""
    return f"G{group_number}|{conc_cu}|{conc_fe}|{conc_zn}"


def _filter_mix_conc(intensities, concentrations, group_numbers, mixtures,
                     mix_only=False, present_conc_range=None):
    """Filter spectra by mixture type and present-component concentration.

    Args:
        intensities: list of 1D np.ndarrays (spectra).
        concentrations: list of [Cu_nM, Fe_nM, Zn_nM].
        group_numbers: list of int.
        mixtures: list of str (mixture labels).
        mix_only: if True, keep only binary & ternary mixtures.
        present_conc_range: (min, max) in nM. If set, a spectrum is kept
            only if every present component's concentration is in [min, max].

    Returns:
        Filtered copies of all five inputs.
    """
    n_orig = len(intensities)
    keep = np.ones(n_orig, dtype=bool)

    if mix_only:
        keep &= np.isin(mixtures, ["Cu+Fe", "Cu+Zn", "Fe+Zn", "Cu+Fe+Zn"])

    if present_conc_range is not None:
        lo, hi = present_conc_range
        for i in range(n_orig):
            if not keep[i]:
                continue
            for val in concentrations[i]:
                if val > 0 and not (lo <= val <= hi):
                    keep[i] = False
                    break

    n_removed = (~keep).sum()
    if n_removed > 0:
        keep_idx = np.where(keep)[0]
        print(f"  filter(mix_only={mix_only}, range={present_conc_range}): "
              f"kept {len(keep_idx)}/{n_orig} spectra "
              f"({len(set(np.array(group_numbers)[keep_idx]))} groups)")
        intensities = [intensities[i] for i in keep_idx]
        concentrations = [concentrations[i] for i in keep_idx]
        group_numbers = [group_numbers[i] for i in keep_idx]
        mixtures = [mixtures[i] for i in keep_idx]

    return intensities, concentrations, group_numbers, mixtures


# ---------------------------------------------------------------------------
# Spectral normalization
# ---------------------------------------------------------------------------

def _peak_normalize(raman_shift, intensities, peak_position=250, peak_range=20):
    """Normalize spectra by dividing each by its max intensity near peak_position.

    Unlike min-max scaling, this preserves concentration-dependent intensity
    differences that are essential for PLSR quantification.

    Args:
        raman_shift: 1D np.ndarray of Raman shift values.
        intensities: 2D np.ndarray (n_samples, n_features).
        peak_position: center of the reference peak window (cm-1).
        peak_range: half-width of the reference peak window (cm-1).

    Returns:
        2D np.ndarray of peak-normalized spectra.
    """
    peak_indices = np.where(
        (raman_shift >= peak_position - peak_range) &
        (raman_shift <= peak_position + peak_range)
    )[0]

    if len(peak_indices) == 0:
        raise ValueError(
            f"No Raman shift points in peak range "
            f"{peak_position - peak_range}-{peak_position + peak_range} cm-1."
        )

    normalized = intensities.copy()
    for i in range(intensities.shape[0]):
        peak_max = np.max(intensities[i, peak_indices])
        if peak_max > 0:
            normalized[i, :] = intensities[i, :] / peak_max

    return normalized


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def _rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))


def _safe_r2(y_true, y_pred):
    if len(np.unique(y_true)) <= 1:
        return np.nan
    return r2_score(y_true, y_pred)


# ---------------------------------------------------------------------------
# Ratio helpers
# ---------------------------------------------------------------------------

def _make_ratio_targets(y_conc):
    """Convert concentration (nM) to ratios summing to 1."""
    y = np.asarray(y_conc, dtype=float)
    total = y.sum(axis=1, keepdims=True)
    return np.divide(y, total, out=np.zeros_like(y), where=total > 0)


def _normalize_ratio_pred(pred_ratio):
    """Clip negative values to 0 and L1-normalize each row to 1."""
    z = np.maximum(np.asarray(pred_ratio, dtype=float), 0.0)
    s = z.sum(axis=1, keepdims=True)
    bad = s.squeeze() <= 1e-12
    if np.any(bad):
        z[bad, :] = 1.0 / z.shape[1]
        s = z.sum(axis=1, keepdims=True)
    return z / s


# ---------------------------------------------------------------------------
# Stratified group-based fold assignment
# ---------------------------------------------------------------------------

def _group_folds(group_table, n_splits=3, random_state=2026):
    """Assign each concentration group to a fold, stratified by mixture type.

    Within each mixture type, groups are shuffled then round-robin assigned.
    This guarantees each fold has ≈equal proportions of each mixture type.
    """
    rng = np.random.default_rng(random_state)
    group_table = group_table.copy()
    group_table["fold"] = -1

    for mix in VALID_MIXTURES:
        idx = group_table.index[group_table["mixture"] == mix].to_numpy()
        if len(idx) == 0:
            continue
        rng.shuffle(idx)
        for i, row_idx in enumerate(idx):
            group_table.loc[row_idx, "fold"] = i % n_splits

    leftover = group_table["fold"] < 0
    if leftover.any():
        missed = group_table.loc[leftover, "mixture"].unique()
        raise ValueError(f"Fold assignment missed groups: {missed}")
    return group_table


# ---------------------------------------------------------------------------
# Inner CV: select n_components on the outer-training set
# ---------------------------------------------------------------------------

def _select_n_components(X_train, Y_train, train_gids, train_meta, response_mode):
    """2-fold inner CV to pick best n_components.

    Uses group-level RMSE: predictions are averaged within each group first,
    then RMSE is computed across groups. This prevents groups with many
    replicates from dominating the selection.
    """
    inner_folds = _group_folds(
        train_meta, n_splits=N_INNER,
        random_state=RANDOM_STATE + 99
    )
    inner_lookup = dict(zip(inner_folds["group_id"], inner_folds["fold"]))
    inner_sample_fold = np.array([inner_lookup[g] for g in train_gids])

    max_comp = min(MAX_COMP, X_train.shape[1],
                   max(1, X_train.shape[0] - 1))
    rows = []

    for n_comp in range(1, max_comp + 1):
        oof = np.zeros_like(Y_train, dtype=float)

        for f in range(N_INNER):
            vmask = inner_sample_fold == f
            tmask = ~vmask

            pls = Pipeline([
                ("scaler", StandardScaler()),
                ("pls", PLSRegression(n_components=n_comp, scale=False))
            ])
            pls.fit(X_train[tmask], Y_train[tmask])
            pred = pls.predict(X_train[vmask])
            if response_mode == "ratio":
                pred = _normalize_ratio_pred(pred)
            oof[vmask] = pred

        # Aggregate to group-level and compute RMSE
        tmp = pd.DataFrame({
            "gid": train_gids,
            "t0": Y_train[:, 0], "t1": Y_train[:, 1], "t2": Y_train[:, 2],
            "p0": oof[:, 0], "p1": oof[:, 1], "p2": oof[:, 2],
        })
        grp = tmp.groupby("gid", as_index=False).agg({
            "t0": "first", "t1": "first", "t2": "first",
            "p0": "mean",  "p1": "mean",  "p2": "mean",
        })
        yt_g = grp[["t0", "t1", "t2"]].to_numpy(dtype=float)
        yp_g = grp[["p0", "p1", "p2"]].to_numpy(dtype=float)
        if response_mode == "ratio":
            yp_g = _normalize_ratio_pred(yp_g)

        score = _rmse(yt_g.reshape(-1), yp_g.reshape(-1))
        rows.append({"n_components": n_comp, "group_level_rmse": score})

    result = pd.DataFrame(rows)
    min_rmse = result["group_level_rmse"].min()
    # RMSE within 1% → prefer fewer components
    best_rows = result[result["group_level_rmse"] <= min_rmse * 1.01]
    best_n = int(best_rows["n_components"].min())
    return best_n


# ---------------------------------------------------------------------------
# Outer CV
# ---------------------------------------------------------------------------

def _run_outer_cv(X, Y, group_ids, df_model, response_mode):
    """3-fold outer CV returning OOF predictions for every sample.

    For each fold:
      1. Inner 2-fold CV on the outer-training data → select best_n.
      2. Train PLSR(best_n) on the full outer-training set.
      3. Predict the held-out outer-test set.
    """
    oof_pred = np.zeros_like(Y, dtype=float)
    selected_rows = []

    for fold in range(N_OUTER):
        test_mask = df_model["outer_fold"].to_numpy() == fold
        train_mask = ~test_mask

        X_tr = X[train_mask]; Y_tr = Y[train_mask]
        X_te = X[test_mask]

        train_gids = group_ids[train_mask]
        train_meta = (
            df_model.loc[train_mask,
                         ["group_id", "mixture",
                          "conc_Cu", "conc_Fe", "conc_Zn"]]
            .drop_duplicates("group_id")
            .reset_index(drop=True)
        )

        best_n = _select_n_components(
            X_train=X_tr, Y_train=Y_tr,
            train_gids=train_gids, train_meta=train_meta,
            response_mode=response_mode
        )

        pls = Pipeline([
            ("scaler", StandardScaler()),
            ("pls", PLSRegression(n_components=best_n, scale=False))
        ])
        pls.fit(X_tr, Y_tr)
        pred = pls.predict(X_te)
        if response_mode == "ratio":
            pred = _normalize_ratio_pred(pred)
        oof_pred[test_mask] = pred

        n_tr_g = df_model.loc[train_mask, "group_id"].nunique()
        n_te_g = df_model.loc[test_mask, "group_id"].nunique()
        selected_rows.append({
            "response_mode": response_mode,
            "outer_fold": fold + 1,
            "selected_n_components": best_n,
            "n_train_groups": n_tr_g,
            "n_test_groups": n_te_g,
        })
        print(f"  Fold {fold + 1}: best_n={best_n}, "
              f"train_groups={n_tr_g}, test_groups={n_te_g}")

    return oof_pred, pd.DataFrame(selected_rows)


# ---------------------------------------------------------------------------
# Result table builders
# ---------------------------------------------------------------------------

def _build_tables(df_model, oof_pred, Y_conc, response_mode):
    """Build sample-level and group-level result dataframes."""
    cols = ["outer_fold", "group_id", "mixture",
            "conc_Cu", "conc_Fe", "conc_Zn"]
    # Preserve group_number if present
    if "group_number" in df_model.columns:
        cols.insert(3, "group_number")
    sample_df = df_model[cols].copy()

    if response_mode == "concentration":
        for j, a in enumerate(ANALYTES):
            sample_df[f"pred_conc_{a}"] = oof_pred[:, j]
    else:
        true_ratio = _make_ratio_targets(Y_conc)
        pred_ratio = _normalize_ratio_pred(oof_pred)
        for j, a in enumerate(ANALYTES):
            sample_df[f"true_ratio_{a}"] = true_ratio[:, j]
            sample_df[f"pred_ratio_{a}"] = pred_ratio[:, j]

    agg = {
        "outer_fold": "first", "mixture": "first",
        "conc_Cu": "first", "conc_Fe": "first", "conc_Zn": "first",
    }
    if response_mode == "concentration":
        for a in ANALYTES:
            agg[f"pred_conc_{a}"] = "mean"
    else:
        for a in ANALYTES:
            agg[f"true_ratio_{a}"] = "first"
            agg[f"pred_ratio_{a}"] = "mean"

    group_df = sample_df.groupby("group_id", as_index=False).agg(agg)

    if response_mode == "ratio":
        cols = [f"pred_ratio_{a}" for a in ANALYTES]
        pr = _normalize_ratio_pred(
            group_df[cols].to_numpy(dtype=float))
        for j, a in enumerate(ANALYTES):
            group_df[f"pred_ratio_{a}"] = pr[:, j]

    return sample_df, group_df


def _continuous_summary(result_df, response_mode, level_name):
    """MAE, RMSE, bias, R² per analyte."""
    if response_mode == "concentration":
        true_cols = [f"conc_{a}" for a in ANALYTES]
        pred_cols = [f"pred_conc_{a}" for a in ANALYTES]
    else:
        true_cols = [f"true_ratio_{a}" for a in ANALYTES]
        pred_cols = [f"pred_ratio_{a}" for a in ANALYTES]

    yt = result_df[true_cols].to_numpy(dtype=float)
    yp = result_df[pred_cols].to_numpy(dtype=float)

    row = {
        "response_mode": response_mode, "level": level_name,
        "n": len(result_df),
        "global_mae": np.mean(np.abs(yp - yt)),
        "global_rmse": _rmse(yt.reshape(-1), yp.reshape(-1)),
    }
    for j, a in enumerate(ANALYTES):
        row[f"{a}_mae"] = np.mean(np.abs(yp[:, j] - yt[:, j]))
        row[f"{a}_rmse"] = _rmse(yt[:, j], yp[:, j])
        row[f"{a}_bias"] = np.mean(yp[:, j] - yt[:, j])
        # R² computed only on samples where the analyte is actually present
        # (true > 0), otherwise zeros dominate the variance.
        mask_present = yt[:, j] > 0
        if mask_present.sum() > 1:
            row[f"{a}_r2"] = _safe_r2(yt[mask_present, j],
                                       yp[mask_present, j])
        else:
            row[f"{a}_r2"] = np.nan
    return pd.DataFrame([row])


def _range_by_true_value(result_df, response_mode, level_name):
    """Per-true-value prediction statistics (group-level)."""
    rows = []
    for a in ANALYTES:
        if response_mode == "concentration":
            tc = f"conc_{a}"; pc = f"pred_conc_{a}"; unit = "nM"
        else:
            tc = f"true_ratio_{a}"; pc = f"pred_ratio_{a}"; unit = "ratio"

        tvals = np.sort(np.unique(
            np.round(result_df[tc].to_numpy(dtype=float), 6)))
        for tv in tvals:
            sub = result_df[
                np.isclose(result_df[tc], tv, atol=1e-6)].copy()
            if len(sub) == 0:
                continue
            p = sub[pc].to_numpy(dtype=float)
            rows.append({
                "response_mode": response_mode, "level": level_name,
                "target": a, "unit": unit, "true_value": tv,
                "n": len(sub),
                "pred_min": np.min(p), "pred_mean": np.mean(p),
                "pred_median": np.median(p), "pred_max": np.max(p),
                "mae": np.mean(np.abs(p - tv)),
                "bias": np.mean(p - tv),
            })
    return pd.DataFrame(rows)


def _print_per_group_mse(grp_df, tag=""):
    """Compute per-group MSE across Cu/Fe/Zn, print sorted low→high."""
    rows = []
    for _, r in grp_df.iterrows():
        t0 = r.get("true_Cu", r.get("conc_Cu", 0))
        t1 = r.get("true_Fe", r.get("conc_Fe", 0))
        t2 = r.get("true_Zn", r.get("conc_Zn", 0))
        p0 = r.get("pred_Cu", r.get("pred_conc_Cu", 0))
        p1 = r.get("pred_Fe", r.get("pred_conc_Fe", 0))
        p2 = r.get("pred_Zn", r.get("pred_conc_Zn", 0))
        t = np.array([t0, t1, t2], dtype=float)
        p = np.array([p0, p1, p2], dtype=float)
        mse = mean_squared_error(t, p)
        rows.append((r["group_id"], mse, t, p))
    rows.sort(key=lambda x: x[1])

    print(f"\n{'=' * 90}")
    print(f"Per-group MSE ranking ({tag}):")
    print(f"{'Group':<35s} {'MSE':>8s}  "
          f"{'True(Cu/Fe/Zn)':>30s}  {'Pred(Cu/Fe/Zn)':>30s}")
    print("-" * 100)
    for gid, mse, t, p in rows:
        ts = f"[{t[0]:.1f}, {t[1]:.1f}, {t[2]:.1f}]"
        ps = f"[{p[0]:.1f}, {p[1]:.1f}, {p[2]:.1f}]"
        print(f"{gid:<35s} {mse:8.2f}  {ts:>30s}  {ps:>30s}")
    print("=" * 90)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _plot_group_errorbars(sample_df, response_mode):
    """Per-group mean±SD error-bar plot."""
    os.makedirs("visualization", exist_ok=True)
    colors = ['#E74C3C', '#3498DB', '#2ECC71']

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), facecolor="white")
    for j, (ax, a) in enumerate(zip(axes, ANALYTES)):
        if response_mode == "concentration":
            true_col = f"conc_{a}"; pred_col = f"pred_conc_{a}"
            unit = "nM"
        else:
            true_col = f"true_ratio_{a}"; pred_col = f"pred_ratio_{a}"
            unit = "ratio"

        grp = sample_df.groupby("group_id").agg(
            true=(true_col, "first"), mean=(pred_col, "mean"),
            sd=(pred_col, "std")).reset_index()
        grp["sd"] = grp["sd"].fillna(0)

        t = grp["true"].to_numpy(); m = grp["mean"].to_numpy()
        s = grp["sd"].to_numpy()
        ax.errorbar(t, m, yerr=s, fmt='o', color=colors[j],
                    capsize=3, markersize=6, markeredgecolor='k',
                    markeredgewidth=0.5)
        mx = max(t.max(), (m + s).max()) * 1.1
        if mx <= 0:
            mx = 1
        ax.plot([0, mx], [0, mx], 'r--', lw=1)
        ax.set_xlim(-0.2, mx); ax.set_ylim(-0.2, mx)
        ax.set_xlabel(f"True {a} ({unit})")
        ax.set_ylabel(f"Predicted {a} ({unit})")
        mask = t > 0
        if mask.sum() > 1:
            rmse_val = np.sqrt(mean_squared_error(t[mask], m[mask]))
            r2_val = r2_score(t[mask], m[mask])
            ax.set_title(f"{a}: RMSE={rmse_val:.3f} {unit}, R²={r2_val:.3f}")
        ax.grid(alpha=0.25)
    fig.suptitle(f"Group-level {response_mode} prediction (mean±SD)",
                 fontsize=14)
    fig.tight_layout()
    fname = (f"visualization/"
             f"PLSR_Unmixing_group_{response_mode}_with_bar.png")
    fig.savefig(fname, dpi=600)
    plt.show(block=False)
    plt.pause(5)
    plt.close(fig)


def _plot_pred_vs_true(result_df, response_mode, level_name):
    """OOF true-vs-predicted scatter plots for Cu, Fe, Zn."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), facecolor="white")

    for ax, a in zip(axes, ANALYTES):
        if response_mode == "concentration":
            true = result_df[f"conc_{a}"].to_numpy(dtype=float)
            pred = result_df[f"pred_conc_{a}"].to_numpy(dtype=float)
            xlbl = f"True {a} (nM)"; ylbl = f"Predicted {a} (nM)"
            tit = f"{a} concentration"
        else:
            true = result_df[f"true_ratio_{a}"].to_numpy(dtype=float)
            pred = result_df[f"pred_ratio_{a}"].to_numpy(dtype=float)
            xlbl = f"True {a} ratio"; ylbl = f"Predicted {a} ratio"
            tit = f"{a} ratio"

        ax.scatter(true, pred, s=50, alpha=0.75,
                   edgecolors="black", linewidths=0.3)

        vmin = min(true.min(), pred.min())
        vmax = max(true.max(), pred.max())
        pad = max(0.5 if response_mode == "concentration" else 0.05,
                  0.12 * (vmax - vmin))
        ax.plot([vmin - pad, vmax + pad], [vmin - pad, vmax + pad],
                linestyle="--", linewidth=1.2, color="red")
        ax.set_xlim(vmin - pad, vmax + pad)
        ax.set_ylim(vmin - pad, vmax + pad)
        ax.set_xlabel(xlbl); ax.set_ylabel(ylbl)
        ax.set_title(tit)
        ax.grid(alpha=0.25)

    fig.suptitle(
        f"{level_name.capitalize()}-level {response_mode} prediction (OOF)",
        fontsize=14)
    plt.tight_layout()
    os.makedirs("visualization", exist_ok=True)
    fname = (f"visualization/"
             f"PLSR_Unmixing_{level_name}_{response_mode}.png")
    fig.savefig(fname, dpi=600)
    plt.show(block=False)
    plt.pause(5)
    plt.close(fig)


# ===========================================================================
# Main entry point
# ===========================================================================

def run_plsr_unmixing(data_dir, model_dir, plot=True,
                      mix_only=False, present_conc_range=None,
                      peak_position=920, peak_range=20,
                      cut_range=(0, 2000), airpls_lambda=1e7,
                      airpls_polyorder=3, airpls_max_iters=150):
    """Multi-output PLSR unmixing for Cu/Fe/Zn quantification in SERS mixtures.

    Uses nested 3-fold (outer) × 2-fold (inner) stratified group CV to
    simultaneously predict Cu, Fe, Zn concentrations (or ratios) from SERS
    spectra. Both single-component and mixture data are used for training.

    Args:
        data_dir (str): Path to the data directory containing txt files.
        model_dir (str): Directory for saving results.
        plot (bool): Whether to generate diagnostic plots.
        mix_only (bool): If True, keep only binary & ternary mixtures.
        present_conc_range (tuple): (min, max) in nM for present component
            concentrations. Spectra where any present component falls outside
            this range are excluded.
        peak_position (int): Raman shift of the reference peak for
            normalization (cm-1).
        peak_range (int): Half-width of the reference peak window (cm-1).
        cut_range (tuple): (min, max) Raman shift range to retain.
        airpls_lambda (float): Smoothness for airPLS baseline correction.
        airpls_polyorder (int): Difference penalty order for airPLS.
        airpls_max_iters (int): Max iterations for airPLS.

    Returns:
        dict: Payload with OOF predictions, metrics, and split information.
    """
    print("=" * 60)
    print("PLSR Unmixing — Multi-output PLSR (Cu/Fe/Zn)")
    print("=" * 60)

    # ---- 1. Read & preprocess ------------------------------------------------
    print("\n[1/6] Reading data ...")
    group_numbers, concentrations, raman_shift, intensities = read_data(
        data_dir
    )

    # airPLS baseline correction WITHOUT min-max normalization.
    # Peak-normalization (below) preserves concentration-dependent intensity
    # differences that are essential for PLSR quantification.
    cut_raman, cut_intensities = preprocess_data(
        raman_shift, intensities,
        cut_range=cut_range,
        lamb=airpls_lambda,
        polyorder=airpls_polyorder,
        max_iters=airpls_max_iters,
        plot=False,
        minmax_normalize=False
    )

    # Convert to 2D numpy array
    X_preprocessed = np.array(cut_intensities)

    # Determine mixture labels
    mixtures = np.array([
        _make_mixture_label(c[0], c[1], c[2])
        for c in concentrations
    ])

    # Apply optional filtering
    cut_intensities_list, concentrations_list, group_numbers_list, mixtures_list = \
        _filter_mix_conc(
            [row for row in X_preprocessed],
            concentrations, group_numbers, mixtures,
            mix_only=mix_only, present_conc_range=present_conc_range
        )

    X_preprocessed = np.array(cut_intensities_list)
    concentrations_arr = np.array(concentrations_list)
    group_numbers_arr = np.array(group_numbers_list)
    mixtures_arr = np.array(mixtures_list)

    # ---- 2. Peak-normalize (preserving concentration information) ------------
    print("\n[2/6] Peak normalization ...")
    X = _peak_normalize(cut_raman, X_preprocessed,
                        peak_position=peak_position,
                        peak_range=peak_range)
    Y_conc = concentrations_arr.copy()

    # Build group IDs
    group_ids = np.array([
        _make_group_id(group_numbers_arr[i],
                       concentrations_arr[i, 0],
                       concentrations_arr[i, 1],
                       concentrations_arr[i, 2])
        for i in range(len(group_numbers_arr))
    ])

    group_table = pd.DataFrame({
        "group_id": group_ids,
        "mixture": mixtures_arr,
        "conc_Cu": concentrations_arr[:, 0],
        "conc_Fe": concentrations_arr[:, 1],
        "conc_Zn": concentrations_arr[:, 2],
    }).drop_duplicates("group_id").reset_index(drop=True)

    print(f"  Spectra: {X.shape[0]}, features: {X.shape[1]}")
    print(f"  Groups: {len(group_table)}, "
          f"Raman: {cut_raman[0]:.0f}-{cut_raman[-1]:.0f} cm-1")
    for mix in VALID_MIXTURES:
        n = (mixtures_arr == mix).sum()
        if n > 0:
            ng = (group_table["mixture"] == mix).sum()
            print(f"    {mix:12s}: {n:5d} spectra, {ng:3d} groups")

    # ---- 3. Assign outer folds -----------------------------------------------
    print("\n[3/6] Assigning 3-fold splits "
          "(stratified by mixture, grouped by concentration) ...")
    outer_folds = _group_folds(
        group_table, n_splits=N_OUTER,
        random_state=RANDOM_STATE)
    fold_lookup = dict(zip(outer_folds["group_id"], outer_folds["fold"]))

    df_model = pd.DataFrame({
        "group_id": group_ids, "mixture": mixtures_arr,
        "group_number": group_numbers_arr,
        "conc_Cu": concentrations_arr[:, 0],
        "conc_Fe": concentrations_arr[:, 1],
        "conc_Zn": concentrations_arr[:, 2],
    })
    df_model["outer_fold"] = (
        df_model["group_id"].map(fold_lookup).astype(int))
    for f in range(N_OUTER):
        n_grp = (df_model["outer_fold"] == f).sum()
        print(f"  Fold {f}: {n_grp} spectra")

    # ---- 4. Concentration mode -----------------------------------------------
    print("\n[4/6] Outer CV — concentration mode ...")
    oof_conc, sel_conc = _run_outer_cv(
        X, Y_conc, group_ids, df_model, response_mode="concentration")

    # ---- 5. Ratio mode -------------------------------------------------------
    print("\n[5/6] Outer CV — ratio mode ...")
    Y_ratio = _make_ratio_targets(Y_conc)
    oof_ratio, sel_ratio = _run_outer_cv(
        X, Y_ratio, group_ids, df_model, response_mode="ratio")

    # ---- 6. Build tables & print results ------------------------------------
    print("\n[6/6] Building result tables ...")
    sample_conc, group_conc = _build_tables(
        df_model, oof_conc, Y_conc, "concentration")
    sample_ratio, group_ratio = _build_tables(
        df_model, oof_ratio, Y_conc, "ratio")

    all_sel = pd.concat([sel_conc, sel_ratio], ignore_index=True)

    sum_parts = []
    for mode, gdf in [("concentration", group_conc),
                       ("ratio", group_ratio)]:
        sum_parts.append(_continuous_summary(gdf, mode, "group"))
    summary_df = pd.concat(sum_parts, ignore_index=True)

    range_parts = []
    for mode, gdf in [("concentration", group_conc),
                       ("ratio", group_ratio)]:
        range_parts.append(_range_by_true_value(gdf, mode, "group"))
    range_df = pd.concat(range_parts, ignore_index=True)

    print("\n--- Per-fold n_components ---")
    print(all_sel.to_string(index=False))
    print("\n--- Continuous Prediction Error (group-level) ---")
    with pd.option_context("display.max_columns", 120,
                           "display.width", 200,
                           "display.float_format", "{:.4f}".format):
        print(summary_df.to_string(index=False))
    print("\n--- Group-level by True Value ---")
    with pd.option_context("display.max_rows", 200,
                           "display.float_format", "{:.4f}".format):
        print(range_df.to_string(index=False))

    # ---- 7. Plot -------------------------------------------------------------
    if plot:
        print("\nGenerating plots ...")
        for mode, gdf in [("concentration", group_conc),
                           ("ratio", group_ratio)]:
            _plot_pred_vs_true(gdf, mode, "group")
        _plot_group_errorbars(sample_conc, "concentration")
        _plot_group_errorbars(sample_ratio, "ratio")
        _print_per_group_mse(group_conc, "PLSR conc")
        _print_per_group_mse(group_ratio, "PLSR ratio")

    # ---- 8. Save -------------------------------------------------------------
    os.makedirs(model_dir, exist_ok=True)

    # Export per-spectrum concentration predictions to CSV
    reports_dir = "reports"
    os.makedirs(reports_dir, exist_ok=True)
    csv_export = sample_conc[[
        "group_number", "conc_Cu", "pred_conc_Cu",
        "conc_Fe", "pred_conc_Fe",
        "conc_Zn", "pred_conc_Zn"
    ]].rename(columns={
        "conc_Cu": "Cu_true(nM)", "pred_conc_Cu": "Cu_pred(nM)",
        "conc_Fe": "Fe_true(nM)", "pred_conc_Fe": "Fe_pred(nM)",
        "conc_Zn": "Zn_true(nM)", "pred_conc_Zn": "Zn_pred(nM)",
    })
    csv_path = os.path.join(reports_dir, "plsr_unmixing_predictions.csv")
    csv_export.to_csv(csv_path, index=False)
    print(f"\nPer-spectrum predictions saved to {csv_path}")

    payload = {
        "selected_n_components": all_sel,
        "summary": summary_df,
        "range_by_true": range_df,
        "group_concentration": group_conc,
        "group_ratio": group_ratio,
        "sample_concentration": sample_conc,
        "sample_ratio": sample_ratio,
        "raman_shift": cut_raman,
    }
    model_path = os.path.join(model_dir, MODEL_FILENAME)
    joblib.dump(payload, model_path)
    print(f"\nResults saved to {model_path}")
    print("=" * 60)
    print("PLSR Unmixing completed.")
    print("=" * 60)

    return payload
