#!/usr/bin/env python3
"""Build report-site pages for the fut-qyas Python feature zoo.

The report repository's existing generators consume crypto alpha-replayer
artifacts. This adapter consumes fut-qyas panel parquet files, the daily IC
table, and the persisted predict_test result instead.
"""
from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq


CSS = """
:root{--bg:#0f172a;--panel:#111827;--panel2:#1e293b;--line:#334155;
--txt:#e2e8f0;--muted:#94a3b8;--gold:#fbbf24;--blue:#38bdf8;--green:#34d399}
*{box-sizing:border-box}body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;
margin:0;background:var(--bg);color:var(--txt);line-height:1.45}header{padding:22px 30px;
background:linear-gradient(90deg,#1a202c,#2d3748);border-bottom:1px solid var(--line)}
h1{font-size:22px;margin:6px 0}.meta,.note{color:var(--muted);font-size:12px}.back{color:#90cdf4;
text-decoration:none;font-size:13px}.wrap{padding:20px 24px 50px}.cards{display:grid;
grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:12px}.card,section{background:var(--panel);
border:1px solid var(--line);border-radius:10px;padding:15px}.card .k{color:var(--muted);font-size:11px;
text-transform:uppercase;letter-spacing:.8px}.card .v{font-size:22px;font-weight:700;color:var(--gold)}
section{margin-top:16px;overflow-x:auto}h2{font-size:16px;margin:0 0 10px;color:#7dd3fc}
table{border-collapse:collapse;width:100%;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;
font-size:11px}th,td{padding:6px 8px;border-bottom:1px solid #243244;text-align:right;white-space:nowrap}
th{position:sticky;top:0;background:var(--panel2);color:#cbd5e1}th:first-child,td:first-child{text-align:left}
.pos{color:var(--green)}.neg{color:#f87171}details{margin-top:12px}summary{cursor:pointer;color:var(--gold)}
pre{white-space:pre-wrap;font:12px/1.5 ui-monospace,SFMono-Regular,Menlo,monospace;color:#cbd5e1}
.warn{border-left:3px solid var(--gold);padding:9px 12px;background:#251f13;color:#fde68a;font-size:12px}
"""

META = {"ext", "mid", "micro", "cost_rt", "day"}


def page(title: str, subtitle: str, body: str) -> str:
    return ("<!doctype html><html lang='en'><head><meta charset='utf-8'>"
            "<meta name='viewport' content='width=device-width,initial-scale=1'>"
            f"<title>{html.escape(title)}</title><style>{CSS}</style></head><body>"
            f"<header><a class='back' href='index.html'>&larr; All reports</a>"
            f"<h1>{html.escape(title)}</h1><div class='meta'>{html.escape(subtitle)}</div>"
            f"</header><div class='wrap'>{body}</div></body></html>")


def cards(items):
    return "<div class='cards'>" + "".join(
        f"<div class='card'><div class='k'>{html.escape(k)}</div>"
        f"<div class='v'>{html.escape(str(v))}</div></div>" for k, v in items) + "</div>"


def table(frame: pd.DataFrame, decimals=5) -> str:
    def fmt(v):
        if pd.isna(v):
            return "—"
        if isinstance(v, (float, np.floating)):
            return f"{v:.{decimals}g}"
        return str(v)

    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in frame.columns)
    rows = []
    for row in frame.itertuples(index=False, name=None):
        cells = []
        for v in row:
            s = fmt(v)
            cls = ""
            if isinstance(v, (float, np.floating)) and np.isfinite(v):
                cls = " class='pos'" if v > 0 else (" class='neg'" if v < 0 else "")
            cells.append(f"<td{cls}>{html.escape(s)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<table><thead><tr>{head}</tr></thead><tbody>{''.join(rows)}</tbody></table>"


def feature_columns(panel_paths):
    schemas = [set(pq.ParquetFile(p).schema.names) for p in panel_paths]
    union = set.union(*schemas)
    return sorted(c for c in union if c not in META and not c.startswith("fwd"))


def sampled_features(panel_paths, features, stride):
    chunks = []
    offset = 0
    for path in panel_paths:
        parquet = pq.ParquetFile(path)
        available = [c for c in features if c in parquet.schema.names]
        for batch in parquet.iter_batches(columns=available, batch_size=65536):
            frame = batch.to_pandas().reindex(columns=features).astype("float32", copy=False)
            # A global deterministic stride, preserved across parquet batches.
            start = (-offset) % stride
            chunks.append(frame.iloc[start::stride])
            offset += len(frame)
    return pd.concat(chunks, ignore_index=True)


def build_feature_report(fut_root: Path, out_dir: Path, stride: int):
    source = fut_root / "output"
    panels = sorted(source.glob("au0001_panel_*.parquet"))
    if not panels:
        raise SystemExit(f"no au0001 panels under {source}")
    features = feature_columns(panels)
    total_rows = sum(pq.ParquetFile(p).metadata.num_rows for p in panels)
    trading_days = len(set().union(*(
        set(pd.read_parquet(p, columns=["day"])["day"].unique()) for p in panels)))
    sample = sampled_features(panels, features, stride)
    finite = np.isfinite(sample.to_numpy())
    vals = sample.to_numpy(float, copy=False)
    safe = np.where(finite, vals, np.nan)
    std = np.nanstd(safe, axis=0)
    minimum = np.nanmin(safe, axis=0); maximum = np.nanmax(safe, axis=0)
    max_abs = np.maximum(np.abs(minimum), np.abs(maximum))
    stats = pd.DataFrame({
        "feature": features,
        "mean": np.nanmean(safe, axis=0), "std": std,
        "min": minimum, "p01": np.nanquantile(safe, .01, axis=0),
        "median": np.nanquantile(safe, .50, axis=0),
        "p99": np.nanquantile(safe, .99, axis=0), "max": maximum,
        "x_sigma": np.divide(max_abs, std, out=np.full_like(std, np.nan), where=std > 0),
        "kurtosis": sample.kurt(axis=0, numeric_only=True).reindex(features).to_numpy(),
        "zero_pct": 100 * np.mean(vals == 0, axis=0),
        "nonfinite_pct": 100 * np.mean(~finite, axis=0),
    })

    ic = pd.read_csv(source / "au0001_ic_3m.csv", skipinitialspace=True)
    ic.columns = ic.columns.str.strip()
    ic["col"] = ic["col"].str.strip(); ic["h"] = ic["h"].str.strip()
    ic10 = ic[ic.h == "10s"].copy()
    stats = stats.merge(ic10[["col", "iclag", "t_iclag", "icR", "decay"]],
                        how="left", left_on="feature", right_on="col").drop(columns="col")
    directional = stats.t_iclag.abs() >= 3
    regime = stats.icR.abs() >= .10
    stats["ic_role"] = np.select(
        [directional & regime, directional, regime],
        ["both", "directional", "regime"], default="weak/unclear")
    stats["model_input"] = np.select(
        [(stats.x_sigma > 30) | (stats["kurtosis"].abs() > 75),
         (stats.x_sigma > 15) | (stats["kurtosis"].abs() > 25)],
        ["RANK-GAUSS REQUIRED", "review transform"], default="plain scaler safe")
    top = ic10.assign(abs_t=ic10.t_iclag.abs()).nlargest(25, "abs_t")
    top = top[["col", "iclag", "t_iclag", "icR", "decay", "bounce", "days"]]
    full = ic[["col", "h", "iclag", "t_iclag", "icR", "decay", "bounce", "days"]]
    full = full.assign(abs_t=full.t_iclag.abs()).sort_values(["h", "abs_t"], ascending=[True, False])
    full = full.drop(columns="abs_t")

    months = f"{panels[0].stem[-6:]}–{panels[-1].stem[-6:]}"
    body = cards([
        ("panel rows", f"{total_rows:,}"), ("trading days", trading_days),
        ("Python features", len(features)), ("distribution sample", f"{len(sample):,}"),
        ("IC features", ic.col.nunique()), ("IC horizons", "1 / 10 / 60 s"),
        ("rank-gauss required", int((stats.model_input == "RANK-GAUSS REQUIRED").sum())),
    ])
    body += (f"<section><h2>Method and coverage</h2><div class='warn'>Distribution moments and "
             f"quantiles use every {stride}th snapshot ({len(sample):,} rows) to bound memory. "
             "The IC table is the stored daily evaluation: fwdlag delays entry by one 500 ms "
             "snapshot and is the trusted directional label. The distribution covers all six "
             "panels; the IC artifact has 60 contributing days for most columns. Following "
             "the crypto_alpha memory, xσ&gt;30 or |kurtosis|&gt;75 is marked RANK-GAUSS "
             "REQUIRED; xσ&gt;15 or |kurtosis|&gt;25 is marked for review. IC role is an "
             "audit label: |daily t|≥3 is directional and |icR|≥0.10 is regime.</div></section>")
    body += f"<section><h2>Top 25 directional features at 10 seconds</h2>{table(top)}</section>"
    ordered = stats.assign(_risk=np.select(
        [stats.model_input == "RANK-GAUSS REQUIRED", stats.model_input == "review transform"],
        [2, 1], default=0)).sort_values(["_risk", "x_sigma"], ascending=False).drop(columns="_risk")
    body += ("<section><h2>Feature distributions, roles and model-input guidance</h2>"
             + table(ordered, 6) + "</section>")
    body += ("<section><details><summary>All IC rows (feature × horizon)</summary>"
             + table(full) + "</details></section>")
    rendered = page("AUPY — Python feature distribution & IC",
                    f"au0001 · panels {months} · fut-qyas Python zoo", body)
    path = out_dir / "aupy_feature_distribution.html"
    path.write_text(rendered)
    return path


def _fr(pred, target):
    sd = np.std(pred)
    return np.cov(pred, target, ddof=0)[0, 1] / sd if sd > 0 else np.nan


def _conviction_table(pred, target):
    frame = pd.DataFrame({"pred": pred, "target": target}).dropna()
    frame["tier"] = pd.qcut(frame.pred.abs(), 10, labels=False, duplicates="drop") + 1
    rows = []
    for tier, group in frame.groupby("tier", sort=True):
        p = group.pred.to_numpy(); y = group.target.to_numpy()
        sp, sy = np.sign(p), np.sign(y); moved = y != 0
        ic = np.corrcoef(p, y)[0, 1] if np.std(p) > 0 and np.std(y) > 0 else np.nan
        rows.append({
            "tier": int(tier), "n": len(group), "abs_pred_min": np.abs(p).min(),
            "abs_pred_mean": np.abs(p).mean(), "abs_pred_max": np.abs(p).max(),
            "real_mean": y.mean(), "real_median": np.median(y),
            "dir_return_bp": np.mean(sp * y) * 1e4, "FR_bp": _fr(p, y) * 1e4,
            "IC*sd(alpha)_bp": ic * np.std(p) * 1e4, "ic_pearson": ic,
            "ic_spearman": pd.Series(p).corr(pd.Series(y), method="spearman"),
            "P/N_ratio": np.sum(p > 0) / max(np.sum(p < 0), 1),
            "hit": np.mean(sp == sy), "hit+0": np.mean((sp == sy) | ~moved),
            "hit_move": np.mean((sp == sy)[moved]) if moved.any() else np.nan,
        })
    return pd.DataFrame(rows)


def build_model_report(fut_root: Path, out_dir: Path):
    artifacts = fut_root / "output" / "au0001_predict_test_10s_walk_forward"
    required = ["summary.json", "oos_predictions.parquet", "folds.csv",
                "model_metrics.csv", "net_edge.csv", "episodes.csv",
                "cost_sensitivity.csv", "feature_importance.csv"]
    missing = [name for name in required if not (artifacts / name).is_file()]
    if missing:
        raise SystemExit(f"missing predict_test report artifacts: {', '.join(missing)}")
    summary = json.loads((artifacts / "summary.json").read_text())
    pred = pd.read_parquet(artifacts / "oos_predictions.parquet")
    metrics = pd.read_csv(artifacts / "model_metrics.csv")
    folds = pd.read_csv(artifacts / "folds.csv")
    net = pd.read_csv(artifacts / "net_edge.csv")
    episodes = pd.read_csv(artifacts / "episodes.csv")
    cost = pd.read_csv(artifacts / "cost_sensitivity.csv")
    importance = pd.read_csv(artifacts / "feature_importance.csv")
    lgbm = metrics[metrics.model == "lightgbm"].iloc[0]
    ep1 = episodes[episodes.cost_multiple == 1].iloc[0]
    body = cards([
        ("OOS test days", summary["oos_days"]), ("walk-forward folds", summary["fold_count"]),
        ("LightGBM OOS IC", f"{lgbm.oos_ic:+.4f}"), ("daily IC t", f"{lgbm.daily_t:+.2f}"),
        ("1× spread net/trade", f"{ep1.net_per_trade_bp:+.4f} bp"),
        ("positive days", f"{ep1.positive_days:.0%}"),
    ])
    body += ("<section><h2>Validation design</h2><p>Expanding-window evaluation: "
             "20 initial training days, then 10-day validation blocks. Each later fold adds "
             "all prior days to training. The final block has four days, producing 84 OOS "
             f"days across the {summary['available_days']}-day panel. The saved OOS parquet "
             "contains every prediction and is the authority for the tables below.</p>"
             "<div class='warn'>The episode result deducts "
             "quoted spread but still excludes passive-fill adverse selection, explicit fees, "
             "size/impact and capacity. Treat it as a signal robustness report, not a deployable "
             "PnL forecast.</div></section>")
    body += f"<section><h2>Model headline metrics</h2>{table(metrics, 6)}</section>"
    body += f"<section><h2>Fold definitions</h2>{table(folds)}</section>"

    fold_perf = []
    for fold, group in pred.groupby("fold", sort=True):
        for model, col in (("ridge", "ridge_pred"), ("lightgbm", "lightgbm_pred")):
            p = group[col].to_numpy(); y = group.target.to_numpy()
            fold_perf.append({"fold": fold, "model": model, "n": len(group),
                              "IC": np.corrcoef(p, y)[0, 1],
                              "FR_bp": _fr(p, y) * 1e4,
                              "hit": np.mean(np.sign(p) == np.sign(y))})
    body += f"<section><h2>Per-fold OOS stability</h2>{table(pd.DataFrame(fold_perf), 6)}</section>"
    for model, col in (("Ridge", "ridge_pred"), ("LightGBM", "lightgbm_pred")):
        conviction = _conviction_table(pred[col].to_numpy(), pred.target.to_numpy())
        body += (f"<section><h2>{model} — IC by |prediction| decile</h2>"
                 "<p class='note'>FR = IC×sd(return), in bps. hit excludes no special cases; "
                 "hit+0 counts zero moves as non-adverse; hit_move evaluates moved rows only.</p>"
                 f"{table(conviction, 6)}</section>")
    body += f"<section><h2>Net edge on tradeable rows</h2>{table(net, 6)}</section>"
    body += f"<section><h2>Non-overlapping episodes</h2>{table(episodes, 6)}</section>"
    body += f"<section><h2>Cost sensitivity</h2>{table(cost, 6)}</section>"
    top_sign = importance.nlargest(25, "sign_gain").copy()
    top_sign["sign_gain_pct"] = 100 * top_sign.sign_gain / importance.sign_gain.sum()
    top_sign["magnitude_gain_pct"] = 100 * top_sign.magnitude_gain / importance.magnitude_gain.sum()
    body += f"<section><h2>Top 25 model features</h2>{table(top_sign, 6)}</section>"
    mag_ic = summary["magnitude_oos_ic"]
    body += (f"<section><h2>Magnitude model</h2><p>OOS IC against |fwd_10s|: "
             f"<b>{mag_ic:+.4f}</b>. This is a regime/volatility forecast, not directional "
             "accuracy.</p></section>")
    rendered = page("AUPY — Python feature model report",
                    "au0001 · 10-second fwdlag target · expanding walk-forward", body)
    path = out_dir / "aupy_model.html"
    path.write_text(rendered)
    return path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fut-root", type=Path,
                    default=Path("/home/guanyang/work/OnDev/fut-qyas"))
    ap.add_argument("--out-dir", type=Path, default=Path.cwd())
    ap.add_argument("--sample-stride", type=int, default=100)
    args = ap.parse_args()
    if args.sample_stride < 1:
        raise SystemExit("--sample-stride must be positive")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    paths = [build_feature_report(args.fut_root, args.out_dir, args.sample_stride),
             build_model_report(args.fut_root, args.out_dir)]
    for path in paths:
        print(f"wrote {path} ({path.stat().st_size / 1024:.1f} KiB)")


if __name__ == "__main__":
    main()
