#!/usr/bin/env python3
"""
NIFTY50 implied volatility surface pipeline (offline, bhavcopy-based).

Steps:
  1. Load NSE F&O bhavcopy CSVs (old or UDiFF format), keep index options.
  2. Build OTM quotes, invert Black-Scholes for implied vol.
  3. Quote-level no-arbitrage filter (monotonicity + convexity in strike).
  4. Fit a smile per expiry: SVI (parametric) and quadratic polynomial (baseline),
     with a robust MAD-based outlier pass.
  5. Validate fits: RMSE / R^2, butterfly (density) and calendar violations.
  6. Build the surface across expiries and save CSVs and a 3D plot.

Usage:
  python3 nifty_iv_surface.py --bhav_dir ./bhav --date 2026-11-13 --plot
  python3 nifty_iv_surface.py --bhav_dir ./bhav --spot_csv nifty_spot.csv --r 0.066 --q 0.013
"""
import argparse
import glob
import json
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.optimize import brentq, least_squares

CFG = dict(
    symbol="NIFTY",
    r=0.065,              # continuously compounded risk-free rate (override with --r)
    q=0.013,              # dividend yield (override with --q)
    min_days=3,           # drop expiries with fewer calendar days than this
    min_price=0.05,       # drop quotes below this price (Rs)
    min_oi=1,             # drop quotes with zero open interest
    min_quotes=6,         # minimum quotes per expiry to fit a smile
    mad_cut=3.0,          # robust outlier cutoff, in MAD units
    arb_tol_frac=1e-5,    # no-arb tolerance as a fraction of spot
    grid_x=np.linspace(-0.25, 0.25, 201),  # log(K/S) grid for surfaces
    grid_k=np.linspace(-0.25, 0.25, 201),  # log-moneyness grid for butterfly checks
)


# ============================================================
# 1. Black-Scholes pricing and implied volatility
# ============================================================
def _ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(S, K, T, sig, is_call, r, q):
    sT = sig * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sig * sig) * T) / sT
    d2 = d1 - sT
    if is_call:
        return S * math.exp(-q * T) * _ncdf(d1) - K * math.exp(-r * T) * _ncdf(d2)
    return K * math.exp(-r * T) * _ncdf(-d2) - S * math.exp(-q * T) * _ncdf(-d1)


def implied_vol(price, S, K, T, is_call, r, q):
    """NIFTY options are European, so Black-Scholes inversion applies."""
    dF = S * math.exp(-q * T)
    dK = K * math.exp(-r * T)
    lo, hi = (max(dF - dK, 0.0), dF) if is_call else (max(dK - dF, 0.0), dK)
    if not (lo + 1e-8 < price < hi - 1e-8):
        return np.nan
    f = lambda s: bs_price(S, K, T, s, is_call, r, q) - price
    try:
        return brentq(f, 1e-4, 5.0, xtol=1e-10, maxiter=200)
    except (ValueError, RuntimeError):
        return np.nan


# ============================================================
# 2. Loading bhavcopy and spot
# ============================================================
def read_bhavcopy(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]

    if "TckrSymb" in df.columns:  # UDiFF format (newer)
        out = pd.DataFrame({
            "symbol": df["TckrSymb"].astype(str).str.strip(),
            "itype": df["FinInstrmTp"].astype(str).str.strip(),
            "date": pd.to_datetime(df["TradDt"]),
            "expiry": pd.to_datetime(df["XpryDt"]),
            "strike": pd.to_numeric(df["StrkPric"], errors="coerce"),
            "opt_type": df["OptnTp"].astype(str).str.strip().str.upper(),
            "price": pd.to_numeric(df["SttlmPric"], errors="coerce"),
            "oi": pd.to_numeric(df["OpnIntrst"], errors="coerce"),
            "spot": (pd.to_numeric(df["UndrlygPric"], errors="coerce")
                     if "UndrlygPric" in df.columns else np.nan),
        })
        out = out[out.itype == "IDO"]  # index options

    else:  # legacy format
        out = pd.DataFrame({
            "symbol": df["SYMBOL"].astype(str).str.strip(),
            "itype": df["INSTRUMENT"].astype(str).str.strip(),
            "date": pd.to_datetime(df["TIMESTAMP"].astype(str).str.strip()),
            "expiry": pd.to_datetime(df["EXPIRY_DT"].astype(str).str.strip()),
            "strike": pd.to_numeric(df["STRIKE_PR"], errors="coerce"),
            "opt_type": df["OPTION_TYP"].astype(str).str.strip().str.upper(),
            "price": pd.to_numeric(df["SETTLE_PR"], errors="coerce"),
            "oi": pd.to_numeric(df["OPEN_INT"], errors="coerce"),
            "spot": np.nan,
        })
        out = out[out.itype == "OPTIDX"]

    return out[["symbol", "date", "expiry", "strike", "opt_type", "price", "oi", "spot"]]


def load_bhavcopies(folder):
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise SystemExit(f"No CSV files found in {folder}")
    return pd.concat([read_bhavcopy(f) for f in files], ignore_index=True)


def attach_spot(df, spot_csv=None):
    if spot_csv:
        sp = pd.read_csv(spot_csv, parse_dates=["date"])
        sp = sp.rename(columns={"close": "spot_ext"})[["date", "spot_ext"]]
        df = df.merge(sp, on="date", how="left")
        df["spot"] = df["spot"].fillna(df["spot_ext"])
        df = df.drop(columns="spot_ext")
    missing = int(df["spot"].isna().sum())
    if missing:
        print(f"warning: {missing} rows have no spot price and were dropped")
    return df.dropna(subset=["spot"])


# ============================================================
# 3. Quote construction and no-arbitrage cleaning
# ============================================================
def build_quotes(day, S, r, q):
    """OTM quotes with implied vols, for one trading day and one spot level S."""
    d = day[(day.price >= CFG["min_price"]) & (day.oi >= CFG["min_oi"])].copy()
    d["days"] = (d.expiry - d.date).dt.days
    d = d[d.days >= CFG["min_days"]].copy()
    if d.empty:
        return d

    d["tte"] = d.days / 365.0
    d["is_call"] = d.opt_type.eq("CE")
    d["S"] = S
    d["k"] = np.log(d.strike / (S * np.exp((r - q) * d.tte)))  # log-forward moneyness

    # OTM options only: calls for K >= F, puts for K < F (most liquid, most stable IV)
    d = d[np.where(d.is_call, d.k >= 0, d.k < 0)].copy()

    d["iv"] = [implied_vol(p, S, K, T, c, r, q)
               for p, K, T, c in zip(d.price, d.strike, d.tte, d.is_call)]
    d = d.dropna(subset=["iv"]).copy()

    # Call-equivalent price via put-call parity, used for the no-arb checks
    d["C"] = np.where(d.is_call, d.price,
                      d.price + S * np.exp(-q * d.tte) - d.strike * np.exp(-r * d.tte))
    d["w"] = d.iv ** 2 * d.tte  # total implied variance
    return d


def no_arb_filter(d, passes=3):
    """
    Removes quotes that violate, within each expiry:
      - Monotonicity: call price must not rise with strike.
      - Convexity (butterfly): call price must be convex in strike.
    Repeats a few passes because removing one bad quote can expose another.
    """
    d = d.copy()
    keep = pd.Series(True, index=d.index)
    for _ in range(passes):
        bad = []
        for _, g in d[keep].groupby("expiry"):
            g = g.sort_values("strike")
            K, C, idx = g.strike.values, g.C.values, g.index.values
            tol = CFG["arb_tol_frac"] * g.S.iloc[0]
            for i in range(len(K) - 1):
                if C[i + 1] > C[i] + tol:
                    bad.append(idx[i + 1])
            for i in range(1, len(K) - 1):
                lam = (K[i + 1] - K[i]) / (K[i + 1] - K[i - 1])
                if C[i] > lam * C[i - 1] + (1 - lam) * C[i + 1] + tol:
                    bad.append(idx[i])
        if not bad:
            break
        keep.loc[bad] = False
    return d[keep]


# ============================================================
# 4. Smile models: SVI (parametric) and polynomial baseline
# ============================================================
def svi_w(p, k):
    a, b, rho, m, s = p
    return a + b * (rho * (k - m) + np.sqrt((k - m) ** 2 + s * s))


def fit_svi(k, w):
    p0 = [max(float(w.min()) * 0.5, 1e-4), 0.1, -0.3, 0.0, 0.1]
    lb = [-1.0, 1e-6, -0.999, -1.0, 1e-4]
    ub = [2.0, 5.0, 0.999, 1.0, 2.0]

    def resid(p):
        a, b, rho, m, s = p
        # Penalty keeps total variance non-negative in the wings
        wing = min(0.0, a + b * s * math.sqrt(max(1 - rho * rho, 0.0)))
        return np.append(svi_w(p, k) - w, 1e3 * wing)

    return least_squares(resid, p0, bounds=(lb, ub), max_nfev=2000).x


def fit_model(k, w, model):
    if model == "svi":
        p = fit_svi(k, w)
        return (lambda x, p=p: svi_w(p, x)), p
    if model.startswith("poly"):
        c = np.polyfit(k, w, int(model[4:]))
        return (lambda x, c=c: np.polyval(c, x)), c
    raise ValueError(f"unknown model {model}")


def fit_smile(g, model):
    """Fit, drop MAD outliers, refit."""
    k, w = g.k.values, g.w.values
    f, p = fit_model(k, w, model)
    res = w - f(k)
    med = np.median(res)
    mad = 1.4826 * np.median(np.abs(res - med)) + 1e-12
    keep = np.abs(res - med) < CFG["mad_cut"] * mad
    if keep.sum() >= CFG["min_quotes"] and not keep.all():
        g = g[keep]
        f, p = fit_model(g.k.values, g.w.values, model)
    return g, f, p


def smile_metrics(g, tte, f):
    """RMSE in vol points and R^2 on implied vol."""
    iv_fit = np.sqrt(np.maximum(f(g.k.values), 1e-12) / tte)
    iv_mkt = g.iv.values
    err = (iv_fit - iv_mkt) * 100
    ss_res = np.sum(err ** 2)
    ss_tot = np.sum((iv_mkt * 100 - (iv_mkt * 100).mean()) ** 2)
    rmse = float(np.sqrt(np.mean(err ** 2)))
    r2 = float(1 - ss_res / ss_tot) if ss_tot > 0 else np.nan
    return rmse, r2


def butterfly_viol(f, grid_k):
    """Fraction of grid points where Gatheral's density condition g(k) >= 0 fails."""
    raw = f(grid_k)
    w = np.maximum(raw, 1e-12)
    w1 = np.gradient(w, grid_k)
    w2 = np.gradient(w1, grid_k)
    g = (1 - grid_k * w1 / (2 * w)) ** 2 - (w1 ** 2 / 4) * (1 / w + 0.25) + w2 / 2
    return float(np.mean((g < -1e-8) | (raw <= 0)))


def fit_day(d, model):
    models = {}
    for exp, g in d.groupby("expiry"):
        if len(g) < CFG["min_quotes"]:
            continue
        tte = float(g.tte.iloc[0])
        g, f, p = fit_smile(g, model)
        rmse, r2 = smile_metrics(g, tte, f)
        models[exp] = dict(tte=tte, f=f, params=p, n=len(g),
                           rmse=rmse, r2=r2, bfly=butterfly_viol(f, CFG["grid_k"]))
    return models


def calendar_viol(models, x, r, q):
    """
    Total variance must not decrease with expiry at the same strike.
    Compares at fixed strike by shifting log-forward moneyness per expiry.
    Returns the fraction of grid points violating, for each adjacent expiry pair.
    """
    exps = sorted(models, key=lambda e: models[e]["tte"])
    out = []
    for a, b in zip(exps[:-1], exps[1:]):
        ta, tb = models[a]["tte"], models[b]["tte"]
        wa = models[a]["f"](x - (r - q) * ta)
        wb = models[b]["f"](x - (r - q) * tb)
        out.append(float(np.mean(wb < wa - 1e-8)))
    return out


# ============================================================
# 5. Surface construction and output
# ============================================================
def build_surface(models, x, r, q, n_T=50):
    exps = sorted(models, key=lambda e: models[e]["tte"])
    T = np.array([models[e]["tte"] for e in exps])
    W = np.array([models[e]["f"](x - (r - q) * models[e]["tte"]) for e in exps])
    Tg = np.linspace(T[0], T[-1], n_T)
    Wg = np.array([[np.interp(t, T, W[:, j]) for j in range(len(x))] for t in Tg])
    IVg = np.sqrt(np.maximum(Wg, 1e-12) / Tg[:, None])
    return Tg, IVg


def plot_surface(Tg, x, IVg, title, out_png):
    X, Tm = np.meshgrid(x, Tg)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(projection="3d")
    ax.plot_surface(X, Tm, IVg * 100, cmap="viridis", linewidth=0, alpha=0.9)
    ax.set_xlabel("log(K/S)")
    ax.set_ylabel("time to expiry (yrs)")
    ax.set_zlabel("implied vol (%)")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    plt.show()


def save_params(models, path):
    rows = []
    for e, m in sorted(models.items(), key=lambda kv: kv[1]["tte"]):
        rows.append(dict(
            expiry=e.date(), tte=round(m["tte"], 5), n_quotes=m["n"],
            rmse_volpts=round(m["rmse"], 4), r2=round(m["r2"], 5),
            butterfly_viol_pct=round(100 * m["bfly"], 3),
            params=json.dumps(np.asarray(m["params"]).tolist()),
        ))
    pd.DataFrame(rows).to_csv(path, index=False)


def process_day(day, S, r, q):
    d = build_quotes(day, S, r, q)
    if d.empty:
        return None
    n_raw = len(d)
    d = no_arb_filter(d)
    removed_pct = 100 * (1 - len(d) / n_raw)

    svi = fit_day(d, "svi")
    poly = fit_day(d, "poly2")

    row = dict(spot=round(S, 2), quotes_raw=n_raw, quotes_clean=len(d),
               pct_removed=round(removed_pct, 2),
               expiries_svi=len(svi), expiries_poly2=len(poly))
    for name, m in [("svi", svi), ("poly2", poly)]:
        rm = [v["rmse"] for v in m.values()]
        bf = [v["bfly"] for v in m.values()]
        cal = calendar_viol(m, CFG["grid_x"], r, q) if len(m) > 1 else []
        row[f"{name}_rmse_volpts"] = round(float(np.mean(rm)), 4) if rm else np.nan
        row[f"{name}_bfly_pct"] = round(100 * float(np.mean(bf)), 3) if bf else np.nan
        row[f"{name}_cal_pct"] = round(100 * float(np.mean(cal)), 3) if cal else np.nan
    return dict(row=row, quotes=d, svi=svi, poly2=poly)


# ============================================================
# 6. Main
# ============================================================
def main():
    ap = argparse.ArgumentParser(description="NIFTY50 IV surface from bhavcopy")
    ap.add_argument("--bhav_dir", required=True, help="folder with bhavcopy CSVs")
    ap.add_argument("--spot_csv", help="optional CSV with columns date,close")
    ap.add_argument("--symbol", default=CFG["symbol"])
    ap.add_argument("--r", type=float, default=CFG["r"], help="risk-free rate (cont. comp.)")
    ap.add_argument("--q", type=float, default=CFG["q"], help="dividend yield")
    ap.add_argument("--date", help="YYYY-MM-DD; default: process every date")
    ap.add_argument("--plot", action="store_true", help="plot the surface of the last processed day")
    ap.add_argument("--out", default="iv_output", help="output folder")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    raw = load_bhavcopies(args.bhav_dir)
    raw = raw[raw.symbol == args.symbol]
    raw = attach_spot(raw, args.spot_csv)
    raw = raw.dropna(subset=["strike", "price", "expiry"])
    if raw.empty:
        raise SystemExit("No usable rows after filtering. Check --symbol and the input format.")

    dates = [pd.Timestamp(args.date)] if args.date else sorted(pd.to_datetime(raw.date.unique()))

    summary, last = [], None
    for dt in dates:
        day = raw[raw.date == dt]
        if day.empty:
            continue
        S = float(day.spot.median())
        res = process_day(day, S, args.r, args.q)
        if res is None:
            continue
        tag = f"{pd.Timestamp(dt):%Y-%m-%d}"
        row = dict(date=tag, **res["row"])
        summary.append(row)
        print(f"{tag}: raw={row['quotes_raw']} clean={row['quotes_clean']} "
              f"removed={row['pct_removed']}%  SVI RMSE={row['svi_rmse_volpts']} vol pts")

        res["quotes"].to_csv(os.path.join(args.out, f"quotes_{tag}.csv"), index=False)
        save_params(res["svi"], os.path.join(args.out, f"svi_params_{tag}.csv"))
        Tg, IVg = build_surface(res["svi"], CFG["grid_x"], args.r, args.q)
        pd.DataFrame(IVg, index=np.round(Tg, 5), columns=np.round(CFG["grid_x"], 4)) \
            .to_csv(os.path.join(args.out, f"surface_{tag}.csv"))
        last = (tag, Tg, IVg)

    if not summary:
        raise SystemExit("No dates produced a result.")

    pd.DataFrame(summary).to_csv(os.path.join(args.out, "summary.csv"), index=False)
    print(f"\nSaved results to {args.out}/ (summary.csv has one row per day)")

    if args.plot and last:
        tag, Tg, IVg = last
        plot_surface(Tg, CFG["grid_x"], IVg,
                     f"{args.symbol} implied vol surface (SVI), {tag}",
                     os.path.join(args.out, f"surface_{tag}.png"))


if __name__ == "__main__":
    main()