#!/usr/bin/env python3
"""
Tomorrow's Movers scanner (NSE end-of-day data). For learning only.
Finds liquid stocks likely to move a lot next day (direction is NOT predicted) and builds a
password-protected dashboard (results/dashboard.html). Needs dashboard_template.html in this folder.

Usage:
  python scanner.py                 # nightly run: update data, rank movers, save results, build dashboard
  python scanner.py --check SONACOMS  # raw NSE numbers for one stock (compare with your chart)
  python scanner.py --backfill 60   # save movers lists for the last 60 past days (fills the History tab)
  python scanner.py --backtest 40   # test the rules on the last 40 trading days
  python scanner.py --study         # research: which features predict next-day moves (uses all cached data)
  python scanner.py --history 750   # download about 3 years of NSE data (slow once), for longer backtests
  python scanner.py --offline       # skip downloading, use cached files only
  python scanner.py --source yahoo  # force Yahoo Finance instead of NSE files

Data: tries NSE Bhavcopy first; if NSE downloads fail it switches to Yahoo Finance automatically.
Needs: pip install pandas numpy requests yfinance cryptography
First run downloads ~260 days of NSE bhavcopy files (takes a few minutes); later runs add one day.
"""
import argparse, base64, json, os, sys, time
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# ---- Rules (starting values from the BRD; tune after backtest) ----
CFG = dict(
    min_turnover_cr=10,    # 20-day average turnover, in crore rupees
    min_price=50,
    range_days=6,          # coiled range window (5-7 days)
    max_range_pct=6.0,     # max high-to-low range as % of price
    min_close_pos=0.70,    # close must sit in top 30% of the range
    top_n=15,
    min_score=40,          # hide weak picks
    min_avg_range_pct=1.0, # 20-day average daily range; removes liquid funds / debt ETFs
    exclude_words=("ETF", "BEES", "LIQUID"),  # funds are not stocks
    history_days=260,
)

# NSE equity trading holidays (market closed). Any data for these dates is ignored.
HOLIDAYS = {"2026-01-15", "2026-05-28", "2026-06-26", "2026-09-14", "2026-10-02", "2026-10-20",
            "2026-11-10", "2026-11-24", "2026-12-25"}
CLOSE_FIELD = "last"  # "last" = last traded price (like most charts), "official" = NSE closing price
BASE = Path(__file__).parent
CACHE = BASE / "data" / "bhav"
OUT = BASE / "results"
URL = "https://nsearchives.nseindia.com/products/content/sec_bhavdata_full_{d}.csv"


# ---------------- Data download ----------------
def now_ist():
    t = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    return t.strftime("%d %b %Y, %I:%M %p IST")


def update_cache(days):
    CACHE.mkdir(parents=True, exist_ok=True)
    s = requests.Session()
    s.headers.update({"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)", "Accept": "*/*"})
    try:
        s.get("https://www.nseindia.com", timeout=15)
    except Exception:
        pass
    today, d = date.today(), date.today()
    have = len(list(CACHE.glob("*.csv")))
    limit = today - timedelta(days=int(days * 1.6))
    while d >= limit:
        # Always look at the last 14 days (that is where new files appear).
        # Older days are only fetched until we have enough history.
        if (today - d).days > 14 and have >= days:
            break
        if d.weekday() < 5:
            f, none = CACHE / f"{d}.csv", CACHE / f"{d}.none"
            retry = (today - d).days <= 3  # a recent "missing" note may only mean "not published yet"
            if not f.exists() and (retry or not none.exists()):
                try:
                    r = s.get(URL.format(d=d.strftime("%d%m%Y")), timeout=30)
                    if r.status_code == 200 and len(r.content) > 1000:
                        f.write_bytes(r.content)
                        have += 1
                        print("downloaded", d)
                    elif d < today and r.status_code == 404:
                        none.write_text("holiday or missing")  # do not retry old holidays
                except Exception as e:
                    print("download failed", d, e)
                time.sleep(0.3)
            elif f.exists():
                pass
        d -= timedelta(days=1)


def load_all():
    frames = []
    for f in sorted(CACHE.glob("*.csv")):
        try:
            df = pd.read_csv(f)
        except Exception:
            continue
        df.columns = [c.strip() for c in df.columns]
        df = df[df["SERIES"].astype(str).str.strip() == "EQ"].copy()
        df["date"] = pd.Timestamp(f.stem)
        frames.append(df)
    if not frames:
        sys.exit("No data found. Run without --offline so files can download.")
    d = pd.concat(frames, ignore_index=True)
    d = d.rename(columns={"SYMBOL": "symbol", "OPEN_PRICE": "o", "HIGH_PRICE": "h", "LOW_PRICE": "l",
                          "CLOSE_PRICE": "c_off", "LAST_PRICE": "c_last", "TTL_TRD_QNTY": "v",
                          "TURNOVER_LACS": "tv", "DELIV_PER": "dl"})
    for col in ["o", "h", "l", "c_off", "c_last", "v", "tv", "dl", "PREV_CLOSE"]:
        d[col] = pd.to_numeric(d[col], errors="coerce")
    d["symbol"] = d["symbol"].astype(str).str.strip()
    d["tv"] = d["tv"] / 100.0  # lakh -> crore
    # c_last = last traded price (what most charts show); c_off = NSE official close (VWAP of last 30 min)
    d["c"] = (d["c_last"] if CLOSE_FIELD == "last" else d["c_off"]).fillna(d["c_off"])
    d = d.dropna(subset=["o", "h", "l", "c", "v"]).sort_values(["symbol", "date"]).reset_index(drop=True)
    # Splits/bonuses: NSE's PREV_CLOSE is adjusted on the ex-date. If it differs from yesterday's close by
    # more than 8%, scale all earlier prices (and volumes) so the history is continuous.
    f = d["PREV_CLOSE"] / d.groupby("symbol")["c_off"].shift(1)
    ev = np.where(np.isfinite(f) & (f > 0) & (np.abs(f - 1) > 0.08), f, 1.0)
    d["lf"] = np.log(ev)
    after = d.groupby("symbol")["lf"].transform(lambda x: x[::-1].cumsum()[::-1]) - d["lf"]
    m = np.exp(after)
    for col in ["o", "h", "l", "c", "c_off", "c_last", "PREV_CLOSE"]:
        d[col] = d[col] * m
    d["pc"] = d["PREV_CLOSE"]  # NSE's official previous close (already includes special sessions like Muhurat)
    d["v"] = d["v"] / m
    print(f"Adjusted older prices for {int((ev != 1).sum())} split/bonus events.")
    return d.drop(columns=["lf"])


def load_yahoo():
    """Fallback: Nifty 500 list + Yahoo Finance daily prices. No delivery % (accumulation score is lower)."""
    import io
    try:
        import yfinance as yf
    except ImportError:
        sys.exit("Yahoo needs a package. Run: pip install yfinance")
    syms = []
    try:
        r = requests.get("https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
                         headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
        syms = pd.read_csv(io.StringIO(r.text))["Symbol"].astype(str).str.strip().tolist()
    except Exception as e:
        print("Could not fetch the Nifty 500 list:", e)
    extra = BASE / "symbols.txt"  # optional: add more NSE symbols, one per line
    if extra.exists():
        syms += [x.strip().upper() for x in extra.read_text().split() if x.strip()]
    syms = sorted(set(syms))
    if not syms:
        sys.exit("No symbol list. Create symbols.txt next to scanner.py with one NSE symbol per line.")
    print(f"Yahoo: downloading {len(syms)} stocks (takes a few minutes)...")
    frames = []
    for i in range(0, len(syms), 100):
        batch = [x + ".NS" for x in syms[i:i + 100]]
        try:
            raw = yf.download(batch, period="15mo", interval="1d", group_by="ticker",
                              auto_adjust=False, threads=True, progress=False)
        except Exception as e:
            print("batch failed", i, e); continue
        for t in batch:
            try:
                d = raw[t].dropna(subset=["Close"])
            except KeyError:
                continue
            d = d[d["Volume"] > 0]
            if d.empty:
                continue
            frames.append(pd.DataFrame({
                "symbol": t[:-3], "date": pd.to_datetime(d.index).tz_localize(None).normalize(),
                "o": d["Open"].values, "h": d["High"].values, "l": d["Low"].values,
                "c": d["Close"].values, "v": d["Volume"].values,
                "tv": d["Close"].values * d["Volume"].values / 1e7, "dl": np.nan}))
        time.sleep(1)
    if not frames:
        sys.exit("Yahoo returned no data. Check your internet connection and try again.")
    out = pd.concat(frames, ignore_index=True).sort_values(["symbol", "date"]).reset_index(drop=True)
    out["c_off"] = out["c"]
    out["pc"] = out.groupby("symbol")["c"].shift(1)
    print(f"Yahoo: got data for {out['symbol'].nunique()} stocks")
    return out


def drop_non_trading_days(d):
    """Remove holidays and days where almost every price is unchanged from the day before (stale data)."""
    g = d.sort_values(["symbol", "date"])
    prev = g.groupby("symbol")["c"].shift(1)
    same = (g["c"] == prev).astype(float).where(prev.notna())
    share = same.groupby(g["date"]).mean()
    bad = set(share[share > 0.6].index) | {pd.Timestamp(x) for x in HOLIDAYS}
    gone = sorted(str(x.date()) for x in bad if (d["date"] == x).any())
    if gone:
        print("Ignoring days when the market was closed or data was stale:", ", ".join(gone))
    return d[~d["date"].isin(bad)]


def prep(df):
    out = {}
    for sym, g in df.sort_values("date").groupby("symbol"):
        out[sym] = {
            "date": g["date"].values.astype("datetime64[D]"),
            "o": g["o"].values, "h": g["h"].values, "l": g["l"].values, "c": g["c"].values,
            "v": g["v"].values, "tv": g["tv"].values, "dl": g["dl"].values,
            "pc": g["pc"].values, "c_off": g["c_off"].values,
        }
    return out


# ---------------- Strategy ----------------
def evaluate(sym, s, cfg):
    o, h, l, c, v, tv, dl = (s[k] for k in ["o", "h", "l", "c", "v", "tv", "dl"])
    if c[-1] < cfg["min_price"] or np.nanmean(tv[-20:]) < cfg["min_turnover_cr"]:
        return None
    if any(w in sym for w in cfg["exclude_words"]):
        return None
    if np.mean((h[-20:] - l[-20:]) / c[-20:]) * 100 < cfg["min_avg_range_pct"]:
        return None
    if not (c[-1] > c[-50:].mean() > c[-200:].mean()):  # trend context
        return None
    k = cfg["range_days"]
    hi, lo = h[-k:].max(), l[-k:].min()
    if hi <= lo:
        return None
    rng_pct = (hi - lo) / c[-1] * 100
    pos = (c[-1] - lo) / (hi - lo)
    if rng_pct > cfg["max_range_pct"] or pos < cfg["min_close_pos"]:
        return None

    why = []
    # Tightness (25)
    ranges = h - l
    nr7 = ranges[-1] <= ranges[-7:].min()
    inside = h[-1] <= h[-2] and l[-1] >= l[-2]
    shrink = ranges[-5:].mean() < ranges[-20:].mean()
    tight = 12 * nr7 + 6 * inside + 7 * shrink
    if nr7: why.append("NR7")
    if inside: why.append("inside bar")
    if shrink: why.append("ranges shrinking")

    # Hidden accumulation (25)
    chg = np.diff(c[-k - 1:])
    vol = v[-k:]
    up_v, dn_v = vol[chg > 0].sum(), vol[chg <= 0].sum()
    acc = 0.0
    if up_v > dn_v:
        acc += 10; why.append("up-day volume > down-day volume")
    dl_avg = np.nanmean(dl[-20:]) if np.isfinite(dl[-20:]).any() else np.nan
    if np.isfinite(dl[-1]) and np.isfinite(dl_avg) and dl[-1] > dl_avg:
        acc += 8; why.append(f"delivery {dl[-1]:.0f}% above average")
    vratio = v[-1] / max(v[-21:-1].mean(), 1)
    if vratio >= 1.5:
        acc += 7; why.append(f"volume {vratio:.1f}x average with strong close")

    # Nearness to breakout level (20)
    dist = (hi - c[-1]) / c[-1] * 100
    near = 10 if dist <= 1 else 5 if dist <= 2 else 0
    h52 = h[-250:].max()
    d52 = (h52 - c[-1]) / h52 * 100
    near += 10 if d52 <= 3 else 5 if d52 <= 6 else 0
    if d52 <= 6: why.append(f"{d52:.1f}% below 52-week high")

    parts = {"tightness": float(tight), "accumulation": float(acc), "nearness": float(near),
             "relative_strength": 0.0, "catalyst": 0.0}
    return {
        "symbol": sym, "close": round(float(c[-1]), 2), "trigger": round(float(hi), 2),
        "stop_ref": round(float(lo), 2), "gap_limit": round(float(hi * 1.015), 2),
        "range_pct": round(float(rng_pct), 2), "close_pos": round(float(pos), 2),
        "turnover_cr": round(float(np.nanmean(tv[-20:])), 1),
        "score_parts": parts, "why": why,
    }


def scan(P, asof, cfg=CFG):
    asof = np.datetime64(asof)
    picks, rets, above, adv = [], [], [], []
    for sym, a in P.items():
        i = np.searchsorted(a["date"], asof, side="right")
        if i < 200 or a["date"][i - 1] != asof:
            continue
        s = {k: v[:i] for k, v in a.items()}
        c = s["c"]
        ret20 = (c[-1] / c[-21] - 1) * 100
        rets.append(ret20); above.append(c[-1] > c[-50:].mean()); adv.append(c[-1] > c[-2])
        r = evaluate(sym, s, cfg)
        if r:
            r["ret20"] = round(float(ret20), 2)
            picks.append(r)
    med = float(np.median(rets)) if rets else 0.0
    for p in picks:
        rs = p["ret20"] - med
        p["rs_vs_market"] = round(rs, 2)
        p["score_parts"]["relative_strength"] = round(float(np.clip(rs * 2.5, 0, 20)), 1)
        p["score"] = round(sum(p["score_parts"].values()), 1)
        if rs >= 3: p["why"].append(f"beating market by {rs:.0f}% (20d)")
    picks = [p for p in picks if p["score"] >= cfg["min_score"]]
    picks.sort(key=lambda x: -x["score"])
    pa, ad = (100 * np.mean(above), 100 * np.mean(adv)) if rets else (0, 0)
    mood = "Bullish" if pa >= 55 and ad >= 55 else "Weak" if pa < 40 or ad < 40 else "Neutral"
    market = {"stocks_scanned": len(rets), "pct_above_50dma": round(float(pa), 1),
              "pct_advancing": round(float(ad), 1), "mood": mood}
    return picks[: cfg["top_n"]], market, len(picks)


def outcomes(P, picks, asof, next_date):
    nd = np.datetime64(next_date)
    res = []
    for p in picks:
        a = P.get(p["symbol"])
        if a is None:
            continue
        i = np.searchsorted(a["date"], nd)
        if i >= len(a["date"]) or a["date"][i] != nd:
            continue
        o, h, l, c = a["o"][i], a["h"][i], a["l"][i], a["c"][i]
        hit = bool(h >= p["trigger"])
        res.append({
            "symbol": p["symbol"], "gap_pct": round(float((o / p["close"] - 1) * 100), 2),
            "trigger_hit": hit,
            "high_vs_trigger_pct": round(float((h / p["trigger"] - 1) * 100), 2),
            "close_vs_prev_close_pct": round(float((c / p["close"] - 1) * 100), 2),
            "stop_ref_hit": bool(l <= p["stop_ref"]),
        })
    return res


# ---------------- Runs ----------------
DEFAULT_STATS = {  # from the first 3-year study; `--study` replaces these with fresh numbers
    "list_rates": {"cc>=2": 53.14, "cc>=4": 25.79, "cc>=6": 12.4, "cc>=8": 6.73, "cc>=10": 3.64, "cc>=15": 1.25,
                   "range>=5": 58.53, "range>=8": 23.77, "range>=12": 7.73},
    "list_rates_10": {"cc>=2": 57.49, "cc>=4": 29.79, "cc>=6": 15.36, "cc>=8": 8.63, "cc>=10": 4.81, "cc>=15": 1.56,
                      "range>=5": 67.3, "range>=8": 30.63, "range>=12": 11.26},
    "universe_rates": {"cc>=2": 31.61, "cc>=4": 9.94, "cc>=6": 3.34, "cc>=8": 1.42, "cc>=10": 0.66, "cc>=15": 0.17,
                       "range>=5": 19.37, "range>=8": 4.56, "range>=12": 1.05},
    "gap_table": [
        {"lo": -99, "hi": -3, "n": 8292, "avg_oc": 2.39, "pos_pct": 71},
        {"lo": -3, "hi": -1, "n": 34832, "avg_oc": 0.58, "pos_pct": 62},
        {"lo": -1, "hi": 1, "n": 463348, "avg_oc": -0.19, "pos_pct": None},
        {"lo": 1, "hi": 3, "n": 62996, "avg_oc": -0.44, "pos_pct": 37},
        {"lo": 3, "hi": 99, "n": 10695, "avg_oc": -1.21, "pos_pct": 35}],
    "movers_gap_table": [],
}


def movers_scan(P, asof, cfg=CFG, top=20):
    """Stocks most likely to move a lot tomorrow (direction NOT predicted): big volume/range/move today
    plus high recent volatility, among liquid stocks. Percentile ranks are taken across the liquid universe."""
    asof = np.datetime64(asof)
    rows = []
    for sym, a in P.items():
        if any(w in sym for w in cfg["exclude_words"]):
            continue
        i = np.searchsorted(a["date"], asof, side="right")
        if i < 30 or a["date"][i - 1] != asof:
            continue
        h, l, c, v, tv, dl = (a[k][:i] for k in ("h", "l", "c", "v", "tv", "dl"))
        rng = h - l
        avg = rng[-20:].mean()
        ref = a["pc"][i - 1]
        ref = float(ref) if np.isfinite(ref) and ref > 0 else float(c[-2])
        if c[-1] < cfg["min_price"] or np.nanmean(tv[-20:]) < cfg["min_turnover_cr"] or avg <= 0 \
                or avg / c[-1] * 100 < cfg["min_avg_range_pct"]:
            continue
        prev_dl = dl[-21:-1]
        dlavg = np.nanmean(prev_dl) if np.isfinite(prev_dl).any() else np.nan
        rows.append({"symbol": sym, "close": float(c[-1]), "open": float(a["o"][i - 1]), "high": float(h[-1]),
                     "low": float(l[-1]), "prev_close": float(ref), "official_close": float(a["c_off"][i - 1]),
                     "volume": float(v[-1]), "today_ret": (c[-1] / ref - 1) * 100,
                     "vol_ratio": v[-1] / max(v[-21:-1].mean(), 1), "range_ratio": rng[-1] / avg,
                     "range_today_pct": rng[-1] / c[-1] * 100, "avg_range_pct": avg / c[-1] * 100,
                     "tight6": (h[-6:].max() - l[-6:].min()) / c[-1] * 100,
                     "delivery_ratio": dl[-1] / dlavg if np.isfinite(dl[-1]) and np.isfinite(dlavg) and dlavg > 0 else np.nan})
    if not rows:
        return [], 0
    df = pd.DataFrame(rows)
    mv = df.vol_ratio.rank(pct=True) + df.range_ratio.rank(pct=True) + df.today_ret.abs().rank(pct=True)
    df["score"] = (mv + df.tight6.rank(pct=True)) / 4 * 100
    out = []
    for r in df.sort_values("score", ascending=False).head(top).to_dict("records"):
        out.append({k: (None if (isinstance(v, float) and not np.isfinite(v)) else
                        round(float(v), 2) if k != "symbol" else v) for k, v in r.items() if k != "range_ratio"})
    return out, len(df)


def mover_outcomes(P, movers, next_date):
    nd = np.datetime64(next_date)
    res = {}
    for m in movers:
        a = P.get(m["symbol"])
        if a is None:
            continue
        i = np.searchsorted(a["date"], nd)
        if i >= len(a["date"]) or a["date"][i] != nd:
            continue
        pcn = a["pc"][i]
        pc = float(pcn) if np.isfinite(pcn) and pcn > 0 else (float(a["c"][i - 1]) if i > 0 else m["close"])
        o, h, l, c = (float(a[k][i]) for k in ("o", "h", "l", "c"))
        res[m["symbol"]] = {"gap": round((o / pc - 1) * 100, 2), "oc": round((c / o - 1) * 100, 2),
                            "cc": round((c / pc - 1) * 100, 2), "range_pct": round((h - l) / o * 100, 2),
                            "up": round((h / pc - 1) * 100, 2), "down": round((l / pc - 1) * 100, 2)}
    return res


def prune_history(dates):
    """Delete saved lists for dates that are not real trading days, and redo results that point to a wrong next day."""
    ds = [str(d) for d in dates]
    for pf in sorted((OUT / "history").glob("*.json")):
        old = json.loads(pf.read_text())
        if old.get("asof") not in ds:
            pf.unlink(); print("Removed list for non-trading day", old.get("asof")); continue
        k = ds.index(old["asof"])
        oc = old.get("outcomes") or {}
        nd = oc.get("next_date")
        # redo results that point to a wrong next day, or were measured the old way (not from NSE's previous close)
        if oc and (k + 1 >= len(ds) or nd != ds[k + 1] or oc.get("ref") != "pc"):
            old["outcomes"] = None
            pf.write_text(json.dumps(old, indent=1))


def fill_outcomes(P, dates):
    """Fill next-day results into every saved scan that does not have them yet."""
    ds = [str(d) for d in dates]
    for pf in sorted((OUT / "history").glob("*.json")):
        old = json.loads(pf.read_text())
        if "movers" not in old or old.get("outcomes") or old["asof"] not in ds:
            continue
        k = ds.index(old["asof"])
        if k + 1 >= len(ds):
            continue
        old["outcomes"] = {"next_date": ds[k + 1], "ref": "pc", "rows": mover_outcomes(P, old["movers"], ds[k + 1])}
        pf.write_text(json.dumps(old, indent=1))


def save_scan(P, asof, B):
    _, market, _ = scan(P, asof)
    market.update(breadth_info(B, asof))
    movers, n_liquid = movers_scan(P, asof)
    market["liquid_stocks"] = n_liquid
    payload = {"asof": asof, "generated_at": now_ist(),
               "market": market, "movers": movers, "outcomes": None}
    (OUT / "history" / f"{asof}.json").write_text(json.dumps(payload, indent=1))
    return payload


def build_dashboard(latest, B):
    try:
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.ciphers.aead import AESGCM
        from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    except ImportError:
        print("Dashboard skipped. Run: pip install cryptography   then run the scanner again."); return
    pwf = BASE / "dashboard_password.txt"
    pw = os.environ.get("DASH_PASSWORD") or (pwf.read_text().strip() if pwf.exists() else "")
    if not pw:
        try:
            pw = input("Choose a dashboard password: ").strip()
        except EOFError:
            pw = ""
        if not pw:
            print("Dashboard skipped (no password set)."); return
        pwf.write_text(pw)
    stats = json.loads((OUT / "stats.json").read_text()) if (OUT / "stats.json").exists() else DEFAULT_STATS
    stats.setdefault("list_rates_10", DEFAULT_STATS["list_rates_10"])
    hist = []
    for f in sorted((OUT / "history").glob("*.json"), reverse=True)[:60]:
        d = json.loads(f.read_text())
        if "movers" not in d:
            continue
        res = (d.get("outcomes") or {}).get("rows", {})
        rows = [{"symbol": m["symbol"], "score": m["score"], **res.get(m["symbol"], {})} for m in d["movers"]]
        hist.append({"date": d["asof"], "mood": d["market"].get("mood", "-"), "rows": rows})
    data = {"asof": latest["asof"], "market": latest["market"], "movers": latest["movers"], "stats": stats,
            "history": hist, "breadth": [{"date": str(ix.date()), "above": int(r.above), "below": int(r.below),
                                          "pct": round(float(r.pct), 1)} for ix, r in B.tail(150).iterrows()]}
    salt, iv = os.urandom(16), os.urandom(12)
    key = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=salt, iterations=200000).derive(pw.encode())
    ct = AESGCM(key).encrypt(iv, json.dumps(data).encode(), None)
    blob = {k: base64.b64encode(v).decode() for k, v in (("s", salt), ("i", iv), ("c", ct))}
    blob["g"] = latest["generated_at"]  # not secret: shows how fresh the data is
    (OUT / "data.json").write_text(json.dumps(blob))  # the dashboard fetches this file on open when hosted online
    tpl = BASE / "dashboard_template.html"
    if not tpl.exists():
        print("dashboard_template.html must be in the same folder as scanner.py"); return
    (OUT / "dashboard.html").write_text(tpl.read_text(encoding="utf-8").replace("__DATA__", json.dumps(blob)),
                                        encoding="utf-8")
    print(f"Dashboard saved: {OUT / 'dashboard.html'}  (open it in a browser, password from dashboard_password.txt)")


def nightly(P, dates):
    asof = str(dates[-1])
    OUT.mkdir(exist_ok=True); (OUT / "history").mkdir(exist_ok=True)
    B = breadth_series(P)
    prune_history(dates)
    latest = save_scan(P, asof, B)
    fill_outcomes(P, dates)
    (OUT / "latest.json").write_text(json.dumps(latest, indent=1))
    (OUT / "breadth.json").write_text(json.dumps([
        {"date": str(ix.date()), "above": int(r.above), "below": int(r.below), "pct": round(float(r.pct), 1)}
        for ix, r in B.tail(150).iterrows()]))
    (OUT / "index.json").write_text(json.dumps(sorted((f.stem for f in (OUT / "history").glob("*.json")), reverse=True)))
    m = latest["market"]
    print(f"{asof}: {m['mood']} market. Breadth {m.get('pct_above_20ema')}% above 20 EMA "
          f"({m.get('breadth_5d_change', 0):+} pts in 5 days). {m['liquid_stocks']} liquid stocks ranked.")
    print("Tomorrow's movers (direction not predicted):")
    for x in latest["movers"]:
        print(f"  {x['symbol']:<12} close {x['close']:>9}  today {x['today_ret']:+5.1f}%  vol {x['vol_ratio']:.1f}x  "
              f"usual range {x['avg_range_pct']:.1f}%  score {x['score']:.0f}")
    build_dashboard(latest, B)


def backfill(P, dates, n):
    OUT.mkdir(exist_ok=True); (OUT / "history").mkdir(exist_ok=True)
    B = breadth_series(P)
    prune_history(dates)
    for d in dates[-n - 1:-1]:
        save_scan(P, str(d), B)
    fill_outcomes(P, dates)
    print(f"Saved {n} past days of movers lists with their next-day results. Now run: python scanner.py")


def write_stats(df):
    """Fresh numbers for the dashboard: hit rates of the daily top-20 movers and the opening-gap table."""
    df = df[df.gap_next.abs() < 19.9].copy()
    df["cc"] = ((1 + df.gap_next / 100) * (1 + df.open_to_close_next / 100) - 1) * 100
    g = df.groupby("date")
    mv = g["vol_ratio"].rank(pct=True) + g["range_ratio_today"].rank(pct=True) + df.today_ret.abs().groupby(df["date"]).rank(pct=True)
    df["score"] = mv + g["tight_6d_pct"].rank(pct=True)
    ranked = df.sort_values(["date", "score"], ascending=[True, False])
    top = ranked.groupby("date").head(20)
    top10 = ranked.groupby("date").head(10)

    def rates(x):
        r = {f"cc>={t}": round(float((x.cc.abs() >= t).mean() * 100), 2) for t in (2, 4, 6, 8, 10, 15)}
        r.update({f"range>={t}": round(float((x.range_next_pct >= t).mean() * 100), 2) for t in (5, 8, 12)})
        return r

    edges = [-99, -5, -3, -1.5, -0.5, 0.5, 1.5, 3, 5, 99]

    def gap_table(x):
        out = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            y = x[(x.gap_next >= lo) & (x.gap_next < hi)]
            if len(y) >= 30:
                out.append({"lo": lo, "hi": hi, "n": int(len(y)), "avg_oc": round(float(y.open_to_close_next.mean()), 2),
                            "pos_pct": round(float((y.open_to_close_next > 0).mean() * 100), 1)})
        return out

    stats = {"list_rates": rates(top), "list_rates_10": rates(top10), "universe_rates": rates(df), "gap_table": gap_table(df),
             "movers_gap_table": gap_table(top), "source": f"study of {len(df):,} stock-days"}
    OUT.mkdir(exist_ok=True)
    (OUT / "stats.json").write_text(json.dumps(stats, indent=1))
    print("Saved fresh numbers for the dashboard (results/stats.json).")


def check_symbol(sym, rows=6):
    """Print the raw NSE numbers for one stock so you can compare them with your chart."""
    out = []
    for f in sorted(CACHE.glob("*.csv"))[-rows:]:
        df = pd.read_csv(f)
        df.columns = [c.strip() for c in df.columns]
        df = df[(df["SYMBOL"].astype(str).str.strip() == sym.upper()) & (df["SERIES"].astype(str).str.strip() == "EQ")]
        if len(df):
            r = df.iloc[0]
            hi, lo, last = float(r["HIGH_PRICE"]), float(r["LOW_PRICE"]), float(r["LAST_PRICE"])
            out.append({"date": f.stem, "open": r["OPEN_PRICE"], "high": hi, "low": lo, "last_price": last,
                        "official_close": r["CLOSE_PRICE"], "prev_close": r["PREV_CLOSE"],
                        "range_rs": round(hi - lo, 2), "range_pct_of_close": round((hi - lo) / last * 100, 2),
                        "volume": r["TTL_TRD_QNTY"]})
    if not out:
        print(f"{sym.upper()} not found in the downloaded NSE files (EQ series)."); return
    print(pd.DataFrame(out).to_string(index=False))
    print("Compare with your chart: last_price = last traded price, official_close = NSE closing price (VWAP of last 30 min).")


def breadth_series(P):
    """Market breadth: how many stocks close above their 20-day EMA, for every date in the data."""
    frames = []
    for sym, a in P.items():
        if len(a["c"]) < 40:
            continue
        e = pd.Series(a["c"]).ewm(span=20, adjust=False).mean().values
        frames.append(pd.DataFrame({"date": a["date"][20:], "up": a["c"][20:] > e[20:]}))
    allf = pd.concat(frames)
    b = allf.groupby("date")["up"].agg(above="sum", total="count")
    b["below"] = b["total"] - b["above"]
    b["pct"] = b["above"] / b["total"] * 100
    b.index = pd.to_datetime(b.index)
    return b.sort_index()


def breadth_info(B, asof):
    try:
        i = B.index.get_loc(pd.Timestamp(asof))
    except KeyError:
        return {}
    out = {"pct_above_20ema": round(float(B["pct"].iloc[i]), 1)}
    if i >= 5:
        out["breadth_5d_change"] = round(float(B["pct"].iloc[i] - B["pct"].iloc[i - 5]), 1)
    return out


def simulate(P, p, nd, cost=0.1):
    """Simulate one intraday trade: buy the breakout above the trigger, exit at the close or at the stop.
    Daily candles cannot show which came first, so if the low touched the stop we assume the stop was hit."""
    a = P.get(p["symbol"])
    if a is None:
        return None
    i = np.searchsorted(a["date"], np.datetime64(nd))
    if i >= len(a["date"]) or a["date"][i] != np.datetime64(nd):
        return None
    o, h, l, c = (float(a[k][i]) for k in ("o", "h", "l", "c"))
    trig = p["trigger"]
    if o >= trig:
        entry = o
    elif h >= trig:
        entry = trig
    else:
        return None  # trigger never touched, no trade
    res = {"entry": round(entry, 2), "extended": bool(entry > p["gap_limit"])}
    for name, stop in (("wide", p["stop_ref"]), ("tight", max(p["stop_ref"], entry * 0.985))):
        exit_ = stop if (l <= stop < entry) else c
        res["pnl_" + name] = round((exit_ / entry - 1) * 100 - cost, 3)
    return res


def summarize(label, d, col):
    if d.empty:
        print(f"  {label}: no trades"); return
    w = d[col] > 0
    aw = d.loc[w, col].mean() if w.any() else 0.0
    al = d.loc[~w, col].mean() if (~w).any() else 0.0
    print(f"  {label}: {len(d)} trades, win {w.mean() * 100:.0f}%, avg {d[col].mean():+.2f}% per trade, "
          f"avg win {aw:+.2f}%, avg loss {al:+.2f}%")


def study(P):
    """Research mode: for ALL liquid stock-days, which of today's features relate to tomorrow's
    open-to-close return and tomorrow's range? Shows 5 groups (quintiles) for each feature."""
    frames = []
    for sym, a in P.items():
        if len(a["c"]) < 260 or any(w in sym for w in CFG["exclude_words"]):
            continue
        d = pd.DataFrame({k: a[k] for k in ("o", "h", "l", "c", "v", "tv", "dl")})
        rng = d.h - d.l
        avg_rng = rng.rolling(20).mean()
        f = pd.DataFrame({"symbol": sym, "date": a["date"]})
        f["today_ret"] = d.c.pct_change() * 100
        f["vol_ratio"] = d.v / d.v.rolling(20).mean().shift(1)
        f["delivery_ratio"] = d.dl / d.dl.rolling(20).mean().shift(1)
        f["close_pos_today"] = np.where(rng > 0, (d.c - d.l) / rng.where(rng > 0), np.nan)
        f["range_ratio_today"] = rng / avg_rng
        f["tight_6d_pct"] = (d.h.rolling(6).max() - d.l.rolling(6).min()) / d.c * 100
        h52 = d.h.rolling(250, min_periods=120).max()
        f["dist_52w_high_pct"] = (h52 - d.c) / h52 * 100
        f["ret_20d"] = d.c.pct_change(20) * 100
        f["gap_next"] = (d.o.shift(-1) / d.c - 1) * 100
        f["open_to_close_next"] = (d.c.shift(-1) / d.o.shift(-1) - 1) * 100
        f["range_next_pct"] = (d.h.shift(-1) - d.l.shift(-1)) / d.o.shift(-1) * 100
        f["range_next_ratio"] = (d.h.shift(-1) - d.l.shift(-1)) / avg_rng
        keep = (d.c >= CFG["min_price"]) & (d.tv.rolling(20).mean() >= CFG["min_turnover_cr"]) \
            & ((avg_rng / d.c * 100) >= CFG["min_avg_range_pct"])
        frames.append(f[keep & f["open_to_close_next"].notna()])
    df = pd.concat(frames, ignore_index=True)
    for col in ("gap_next", "open_to_close_next"):
        df[col] = df[col].clip(-20, 20)
    print(f"Study: {len(df):,} stock-days, {df['symbol'].nunique()} stocks, {df['date'].nunique()} days")
    print(f"Baseline (all stocks): next-day open-to-close {df['open_to_close_next'].mean():+.3f}%, "
          f"gap {df['gap_next'].mean():+.3f}%, range {df['range_next_pct'].mean():.2f}% "
          f"(range vs 20d avg: {df['range_next_ratio'].mean():.2f}x)")
    print("Costs for an intraday trade are about 0.1%, so an edge must be clearly above that.")
    feats = ["today_ret", "vol_ratio", "delivery_ratio", "close_pos_today", "range_ratio_today",
             "tight_6d_pct", "dist_52w_high_pct", "ret_20d"]
    rows = []
    for ft in feats:
        x = df.dropna(subset=[ft])
        try:
            x = x.assign(q=pd.qcut(x[ft], 5, labels=False, duplicates="drop"))
        except ValueError:
            continue
        g = x.groupby("q").agg(n=("symbol", "count"), feature_avg=(ft, "mean"),
                               gap=("gap_next", "mean"), open_to_close=("open_to_close_next", "mean"),
                               range_pct=("range_next_pct", "mean"), range_x=("range_next_ratio", "mean")).round(3)
        print(f"\n{ft}  (group 0 = lowest 20%, group 4 = highest 20%)")
        print(g)
        if len(g) >= 2:
            rows.append((ft, g["open_to_close"].iloc[-1] - g["open_to_close"].iloc[0],
                         g["range_x"].iloc[-1] - g["range_x"].iloc[0]))
    print("\nSummary: top group minus bottom group")
    for ft, d1, d2 in rows:
        print(f"  {ft:<20} open-to-close {d1:+.3f}%   range {d2:+.2f}x")
    write_stats(df)


def backtest(P, dates, n):
    rows = []
    B = breadth_series(P)
    for i in range(len(dates) - n - 1, len(dates) - 1):
        d, nd = str(dates[i]), str(dates[i + 1])
        picks, market, _ = scan(P, d)
        bi = breadth_info(B, d)
        for p in picks:
            o = outcomes(P, [p], d, nd)
            if not o:
                continue
            r = o[0]
            r.update({"asof": d, "score": p["score"], "mood": market["mood"],
                      "breadth": bi.get("pct_above_20ema"), "breadth_chg": bi.get("breadth_5d_change")})
            sim = simulate(P, p, nd)
            r.update(sim if sim else {"entry": None, "extended": None, "pnl_wide": None, "pnl_tight": None})
            rows.append(r)
    if not rows:
        print("No backtest rows. Not enough history (need 200+ days before each test day)."); return
    df = pd.DataFrame(rows)
    df["band"] = pd.cut(df["score"], [0, 50, 65, 80, 101], labels=["<50", "50-65", "65-80", "80+"])
    print(f"Picks tested: {len(df)} over {df['asof'].nunique()} days")
    print(f"Trigger hit rate: {df['trigger_hit'].mean() * 100:.1f}%")
    print(f"Avg next-day close vs prev close: {df['close_vs_prev_close_pct'].mean():.2f}%")
    print("\nSimulated intraday trades (buy breakout above trigger, exit at close or stop, 0.1% costs):")
    t = df[df["entry"].notna()]
    for col, nm in (("pnl_wide", "stop = range low"), ("pnl_tight", "stop = 1.5% below entry (or range low)")):
        print(f"Stop rule: {nm}")
        summarize("all trades", t, col)
        summarize("skip gap-ups above limit", t[t["extended"] == False], col)
        summarize("skip gap-ups + skip Weak market days", t[(t["extended"] == False) & (t["mood"] != "Weak")], col)
        summarize("score 50+ only", t[t["score"] >= 50], col)
        summarize("breadth rising (5 days) only", t[t["breadth_chg"] > 0], col)
    print("\nBy score band (tight stop):")
    print(t.groupby("band", observed=True).agg(trades=("symbol", "count"),
          win_rate=("pnl_tight", lambda x: (x > 0).mean()), avg_pnl=("pnl_tight", "mean")).round(2))
    print("\nBy market mood (tight stop):")
    print(t.groupby("mood").agg(trades=("symbol", "count"),
          win_rate=("pnl_tight", lambda x: (x > 0).mean()), avg_pnl=("pnl_tight", "mean")).round(2))
    t = t.assign(breadth_dir=np.where(t["breadth_chg"] > 0, "rising", "falling"))
    print("\nBy breadth direction (tight stop):")
    print(t.groupby("breadth_dir").agg(trades=("symbol", "count"),
          win_rate=("pnl_tight", lambda x: (x > 0).mean()), avg_pnl=("pnl_tight", "mean")).round(2))
    OUT.mkdir(exist_ok=True)
    df.to_csv(OUT / "backtest.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--backtest", type=int, default=0)
    ap.add_argument("--offline", action="store_true")
    ap.add_argument("--check", metavar="SYMBOL", help="show raw NSE numbers for one stock, to compare with your chart")
    ap.add_argument("--close", choices=["last", "official"], default="last",
                    help="which closing price to use: last traded price (default) or NSE official close")
    ap.add_argument("--backfill", type=int, default=0, help="save movers lists for the last N past days (fills History)")
    ap.add_argument("--study", action="store_true", help="research mode: which features predict next-day moves")
    ap.add_argument("--history", type=int, default=CFG["history_days"],
                    help="trading days of NSE data to keep, e.g. 750 for a longer backtest")
    ap.add_argument("--source", choices=["auto", "nse", "yahoo"], default="auto")
    a = ap.parse_args()
    CLOSE_FIELD = a.close
    if a.check:
        check_symbol(a.check); sys.exit()
    if a.source == "yahoo":
        data = load_yahoo()
    else:
        if not a.offline:
            update_cache(a.history)
        n_files = len(list(CACHE.glob("*.csv")))
        if a.source == "auto" and n_files < 210 and not a.offline:
            print(f"NSE gave only {n_files} days of data. Switching to Yahoo Finance.")
            data = load_yahoo()
        else:
            data = load_all()
    data = drop_non_trading_days(data)
    P = prep(data)
    dates = sorted(data["date"].dt.date.unique())
    if a.backfill:
        backfill(P, dates, a.backfill)
    elif a.study:
        study(P)
    elif a.backtest:
        backtest(P, dates, a.backtest)
    else:
        nightly(P, dates)
