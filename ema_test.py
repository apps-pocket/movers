#!/usr/bin/env python3
"""
EMA 9/21 crossover test on 3-minute candles, for the stocks in the scanner lists. For learning only.

Part 1: which stocks crossed, how far price went, fixed targets, time of day, VWAP, trailing %.
Part 2: your stop-loss method (low of the candle BEFORE the crossover candle, or 60-75% down that candle if it is big),
        results in R (R = distance from entry to your stop), target 1R..4R, break-even and "lock profit" rules,
        candle / EMA9 trailing, and indicator filters (daily RSI, 3-minute RSI, ADX, volume, VWAP, gap)
        with a train/test check so we do not fool ourselves.

Data: Yahoo Finance 1-minute candles (about 30 days only) built into 3-minute candles, plus daily candles for the daily RSI.
Usage:   python ema_test.py --top 20
Needs:   pip install pandas numpy yfinance
"""
import argparse, datetime as dt, itertools, json, os
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent
HIST, OUT = BASE / "results" / "history", BASE / "results"
TARGETS, STOPS, TRAILS = (0.5, 1, 1.5, 2, 3, 4), (0.5, 1, 1.5, 2), (0.5, 1, 1.5, 2)

# Trade management rules in R (R = entry-to-stop distance): (name, kind, p1, p2)
STRATS = [(f"t{k}", "t", k, None) for k in (1, 2, 3, 4)] + [("hold", "t", 99, None)]
for _y, _xs in ((1, (0, 0.5)), (1.5, (0, 0.5, 1)), (2, (0, 0.5, 1, 1.5)), (3, (0, 1, 2))):
    STRATS += [(f"bk{_y}_{_x}", "bk", _y, _x) for _x in _xs]  # when price reaches _y R, move the stop to _x R; target 4R
for _y in (1, 1.5, 2):
    STRATS += [(f"cd{_y}", "cd", _y, None), (f"em{_y}", "ema", _y, None)]  # candle trail / EMA9 trail after _y R
FIXED_ONLY = [s for s in STRATS if s[0] in ("t1", "t2", "t3", "t4")]
FIXED3 = [s for s in STRATS if s[0] in ("t1", "t2", "t3")]


def tm(t):
    h, m = t.split(":")
    return int(h) * 60 + int(m)


def load_lists(top):
    lists = [json.loads(f.read_text()) for f in sorted(HIST.glob("*.json"))]
    lists = [x for x in lists if "movers" in x]
    return [(a["asof"], b["asof"], [m["symbol"] for m in a["movers"][:top]]) for a, b in zip(lists[:-1], lists[1:])]


def fetch_day(symbols, day):
    import yfinance as yf
    start = pd.Timestamp(day) - pd.Timedelta(days=5)
    end = pd.Timestamp(day) + pd.Timedelta(days=1)
    tick = [s + ".NS" for s in symbols]
    raw = yf.download(tick, start=start.strftime("%Y-%m-%d"), end=end.strftime("%Y-%m-%d"), interval="1m",
                      group_by="ticker", auto_adjust=False, progress=False, threads=True)
    res = {}
    for s, t in zip(symbols, tick):
        try:
            d = raw[t].dropna(subset=["Close"])
        except Exception:
            continue
        if len(d):
            res[s] = d
    return res


def fetch_daily(symbols):
    import yfinance as yf
    tick = [s + ".NS" for s in symbols]
    raw = yf.download(tick, period="1y", interval="1d", group_by="ticker", auto_adjust=False, progress=False, threads=True)
    res = {}
    for s, t in zip(symbols, tick):
        try:
            c = raw[t]["Close"].dropna()
        except Exception:
            continue
        if len(c) > 20:
            res[s] = c
    return res


def to_3m(d):
    d = d.copy()
    d.index = pd.to_datetime(d.index)
    if d.index.tz is not None:
        d.index = d.index.tz_convert("Asia/Kolkata").tz_localize(None)
    b = d.resample("3min").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
    b = b.dropna(subset=["Close"])
    m = b.index.hour * 60 + b.index.minute
    return b[(m >= 555) & (m < 930)]


def rsi_series(close, n=14):
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def adx_series(h, l, c, n=14):
    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=h.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=h.index)
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    a = 1 / n
    atr = tr.ewm(alpha=a, adjust=False).mean().replace(0, np.nan)
    pdi = 100 * pdm.ewm(alpha=a, adjust=False).mean() / atr
    mdi = 100 * mdm.ewm(alpha=a, adjust=False).mean() / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=a, adjust=False).mean()


def daily_rsi_before(close, day):
    r = rsi_series(close)
    dates = np.array([x.date() for x in pd.to_datetime(r.index)])
    m = dates < day
    return float(r.values[m][-1]) if m.any() else np.nan


def sim(side, entry, H, L, exit_p, T, S, cost):
    for h, l in zip(H, L):
        stop = l <= entry * (1 - S / 100) if side == 1 else h >= entry * (1 + S / 100)
        tgt = h >= entry * (1 + T / 100) if side == 1 else l <= entry * (1 - T / 100)
        if stop:
            return -S - cost
        if tgt:
            return T - cost
    return side * (exit_p / entry - 1) * 100 - cost


def sim_trail(side, entry, H, L, exit_p, init, trail, cost):
    best = entry
    stop = entry * (1 - init / 100) if side == 1 else entry * (1 + init / 100)
    for h, l in zip(H, L):
        if side == 1:
            if l <= stop:
                return (stop / entry - 1) * 100 - cost
            best = max(best, h)
            stop = max(stop, best * (1 - trail / 100))
        else:
            if h >= stop:
                return (1 - stop / entry) * 100 - cost
            best = min(best, l)
            stop = min(stop, best * (1 + trail / 100))
    return side * (exit_p / entry - 1) * 100 - cost


def run_strategy(side, entry, stop, H, L, C, E9, exit_p, cost_pct, kind, p1, p2):
    """Result in R (after costs). Inside one candle we assume the stop is hit before the target (cautious)."""
    R = abs(entry - stop)
    cost_R = cost_pct / 100 * entry / R
    cur, active = stop, False
    reach = (lambda h, l, k: h >= entry + k * R) if side == 1 else (lambda h, l, k: l <= entry - k * R)
    for h, l, c, e in zip(H, L, C, E9):
        if (side == 1 and l <= cur) or (side == -1 and h >= cur):
            return side * (cur - entry) / R - cost_R
        if kind == "t":
            if reach(h, l, p1):
                return p1 - cost_R
        elif kind == "bk":
            if reach(h, l, 4):
                return 4 - cost_R
            if not active and reach(h, l, p1):
                cur, active = entry + side * p2 * R, True
        elif kind == "cd":
            if not active and reach(h, l, p1):
                active = True
            if active:
                cur = max(cur, l) if side == 1 else min(cur, h)
        elif kind == "ema":
            if not active and reach(h, l, p1):
                active = True
            if active and ((side == 1 and c < e) or (side == -1 and c > e)):
                return side * (c - entry) / R - cost_R
    return side * (exit_p - entry) / R - cost_R


def fix_stop(side, entry, stop, floor):
    risk = (entry - stop) * side / entry * 100
    if not np.isfinite(risk) or risk < floor:
        stop, risk = entry * (1 - side * floor / 100), floor
    return stop, risk


def stops_for(side, i, entry, hi, lo, a):
    """Your stop: the candle BEFORE the crossover candle. Big candle: 60-75% of the way down (buy) / up (sell) that candle."""
    Lp, Hp = lo[i - 1], hi[i - 1]
    rng = Hp - Lp
    recent = hi[max(0, i - 21):i - 1] - lo[max(0, i - 21):i - 1]
    big = bool(len(recent) and rng > a.big * np.median(recent))
    if side == 1:
        plain, rule = Lp, (Hp - a.big_pct * rng if big else Lp)
    else:
        plain, rule = Hp, (Lp + a.big_pct * rng if big else Hp)
    s_plain, r_plain = fix_stop(side, entry, plain, a.min_risk)
    s_rule, r_rule = fix_stop(side, entry, rule, a.min_risk)
    return s_plain, r_plain, s_rule, r_rule, big


def find_events(b, sym, ld, day, a, rsid=np.nan):
    if len(b) < 40:
        return []
    b = b.copy()
    b["e9"] = b.Close.ewm(span=9, adjust=False).mean()
    b["e21"] = b.Close.ewm(span=21, adjust=False).mean()
    tp = (b.High + b.Low + b.Close) / 3
    dkey = pd.Series(b.index.date, index=b.index)
    b["vwap"] = (tp * b.Volume).groupby(dkey).cumsum() / b.Volume.groupby(dkey).cumsum().replace(0, np.nan)
    b["rsi3"] = rsi_series(b.Close)
    b["adx3"] = adx_series(b.High, b.Low, b.Close)
    tr_ = pd.concat([b.High - b.Low, (b.High - b.Close.shift()).abs(), (b.Low - b.Close.shift()).abs()], axis=1).max(axis=1)
    b["atr"] = tr_.ewm(alpha=1 / 14, adjust=False).mean()
    up = (b.e9 > b.e21).values
    idx = b.index
    mins = np.asarray(idx.hour * 60 + idx.minute)
    on_day = np.array([x == day for x in idx.date])
    cross = np.zeros(len(b), bool)
    cross[1:] = up[1:] != up[:-1]
    cross &= on_day
    hi, lo, cl, op, vol = b.High.values, b.Low.values, b.Close.values, b.Open.values, b.Volume.values
    e9v, rsi3, adx3, vw, atrv = b.e9.values, b.rsi3.values, b.adx3.values, b.vwap.values, b.atr.values
    day_idx = np.where(on_day & (mins <= tm(a.exit)))[0]
    if len(day_idx) == 0:
        return []
    last_i, first_i = day_idx[-1], np.where(on_day)[0][0]
    gap_pct = (op[first_i] / cl[first_i - 1] - 1) * 100 if first_i > 0 else np.nan
    cross_idx = np.where(cross)[0]
    out, nth = [], 0
    for i in cross_idx:
        if mins[i] < tm(a.start) or mins[i] > tm(a.last_entry) or i >= last_i or i < 2:
            continue
        nth += 1
        side = 1 if up[i] else -1
        nxt = cross_idx[cross_idx > i]
        opp = len(nxt) > 0 and nxt[0] <= last_i
        end = nxt[0] if opp else last_i
        entry, exit_p = cl[i], cl[end]
        H, L = hi[i + 1:end + 1], lo[i + 1:end + 1]
        if side == 1:
            best, worst = (H.max() / entry - 1) * 100, (L.min() / entry - 1) * 100
        else:
            best, worst = (1 - L.min() / entry) * 100, (1 - H.max() / entry) * 100
        row = dict(list_date=ld, test_date=str(day), symbol=sym, time=idx[i].strftime("%H:%M"), side=side,
                   entry=round(entry, 2), exit_time=idx[end].strftime("%H:%M"),
                   exit_reason="opposite cross" if opp else "time", best_pct=round(best, 2), worst_pct=round(worst, 2),
                   pnl_pct=round(side * (exit_p / entry - 1) * 100 - a.cost, 2),
                   pnl_hold_pct=round(side * (cl[last_i] / entry - 1) * 100 - a.cost, 2), nth=nth,
                   above_vwap=bool(cl[i] > vw[i]) if np.isfinite(vw[i]) else False,
                   rsi_d=round(rsid, 1) if np.isfinite(rsid) else np.nan,
                   rsi3=round(float(rsi3[i]), 1) if np.isfinite(rsi3[i]) else np.nan,
                   adx3=round(float(adx3[i]), 1) if np.isfinite(adx3[i]) else np.nan,
                   volx=round(float(vol[i] / max(np.mean(vol[max(0, i - 20):i]), 1)), 2),
                   gap_pct=round(gap_pct, 2) if np.isfinite(gap_pct) else np.nan)
        for T in TARGETS:
            for S in STOPS:
                row[f"T{T}_S{S}"] = round(sim(side, entry, H, L, exit_p, T, S, a.cost), 3)
        for t in TRAILS:
            row[f"TR{t}"] = round(sim_trail(side, entry, H, L, exit_p, 1.0, t, a.cost), 3)
        # Part 2: your stop and R-based management (the trade runs to the stop, the target or the exit time)
        s_plain, r_plain, s_rule, r_rule, big = stops_for(side, i, entry, hi, lo, a)
        row.update(risk_U=round(r_rule, 3), risk_L=round(r_plain, 3), big_candle=big)
        Hm, Lm, Cm, Em = hi[i + 1:last_i + 1], lo[i + 1:last_i + 1], cl[i + 1:last_i + 1], e9v[i + 1:last_i + 1]
        exit_day = cl[last_i]
        for name, kind, p1, p2 in STRATS:
            row[f"U_{name}"] = round(run_strategy(side, entry, s_rule, Hm, Lm, Cm, Em, exit_day, a.cost, kind, p1, p2), 3)
        for name, kind, p1, p2 in FIXED_ONLY:
            row[f"L_{name}"] = round(run_strategy(side, entry, s_plain, Hm, Lm, Cm, Em, exit_day, a.cost, kind, p1, p2), 3)
            f_stop, f_risk = fix_stop(side, entry, entry * (1 - side * 0.01), a.min_risk)
            row[f"F_{name}"] = round(run_strategy(side, entry, f_stop, Hm, Lm, Cm, Em, exit_day, a.cost, kind, p1, p2), 3)
        # more stop sizes: fixed 0.75% / 1.25% and 1.5 x ATR of the 3-minute candles
        alt = {"P075": entry * (1 - side * 0.0075), "P125": entry * (1 - side * 0.0125),
               "A15": entry - side * 1.5 * (atrv[i] if np.isfinite(atrv[i]) else 0)}
        for tag, st in alt.items():
            st, rk = fix_stop(side, entry, st, a.min_risk)
            if tag == "A15":
                row["risk_A15"] = round(rk, 3)
            for name, kind, p1, p2 in FIXED3:
                row[f"{tag}_{name}"] = round(run_strategy(side, entry, st, Hm, Lm, Cm, Em, exit_day, a.cost, kind, p1, p2), 3)
        out.append(row)
    return out


def side_stats(x):
    r = {"events": len(x), "win %": (x.pnl_pct > 0).mean() * 100, "avg P&L % (net)": x.pnl_pct.mean(),
         "median P&L %": x.pnl_pct.median(), "avg best point %": x.best_pct.mean(),
         "avg worst point %": x.worst_pct.mean(), "biggest best %": x.best_pct.max(),
         "biggest worst %": x.worst_pct.min(), "hold to exit time avg %": x.pnl_hold_pct.mean()}
    for t in (0.5, 1, 2, 3):
        r[f"best point >= {t}% (share %)"] = (x.best_pct >= t).mean() * 100
    return pd.Series(r).round(2)


def brief(x):
    if x.empty:
        return pd.Series({"trades": 0})
    return pd.Series({"trades": len(x), "win %": (x.pnl_pct > 0).mean() * 100, "avg P&L %": x.pnl_pct.mean(),
                      "avg best %": x.best_pct.mean(), "avg worst %": x.worst_pct.mean()}).round(2)


def report(ev, a, skipped):
    L = []
    w = L.append
    w(f"EMA 9/21 crossover test, 3-minute candles, top {a.top} stocks of each list, costs {a.cost}% per trade")
    w(f"Entries between {a.start} and {a.last_entry}; PART 1 trades end at the opposite crossover or {a.exit}.")
    if skipped:
        w(f"Skipped {len(skipped)} lists older than 4 weeks (Yahoo has no 1-minute data): {skipped[0]} to {skipped[-1]}")
    if ev.empty:
        w("No crossovers found (or Yahoo returned no data).")
        return "\n".join(L)
    w(f"Newest test day in this report: {ev.test_date.max()}. A list is tested on the NEXT trading day, so the latest list is added "
      f"after the next day's scan has run.")
    w(f"Test days (including days saved by earlier runs): {ev.test_date.nunique()} | stock-days with a crossover: "
      f"{ev.groupby(['test_date', 'symbol']).ngroups} | crossovers: {len(ev)}")
    buy, sell = ev[ev.side == 1], ev[ev.side == -1]
    w("\nPART 1\n1) BUY = 9 crosses above 21 (go long), SELL = 9 crosses below 21 (go short).")
    w(pd.DataFrame({"BUY": side_stats(buy) if len(buy) else np.nan, "SELL": side_stats(sell) if len(sell) else np.nan,
                    "BOTH": side_stats(ev)}).to_string())
    cols = [c for c in ev.columns if c.startswith("T") and "_S" in c]
    m = ev[cols].mean().round(3)
    w("\n2) Fixed % target / stop-loss, average net P&L % per trade (all trades):")
    w(pd.DataFrame([[m[f"T{t}_S{s}"] for s in STOPS] for t in TARGETS], index=[f"target {t}%" for t in TARGETS],
                   columns=[f"stop {s}%" for s in STOPS]).to_string())
    hr = ev.time.str[:2].astype(int) * 60 + ev.time.str[3:].astype(int)
    slots = pd.cut(hr, [0, 629, 689, 749, 809, 869, 929, 1440],
                   labels=["9:30-10:29", "10:30-11:29", "11:30-12:29", "12:30-13:29", "13:30-14:29", "14:30-14:45", "later"])
    w("\n3) Time of the crossover (net, ends at opposite crossover)")
    w(ev.groupby(slots, observed=True).apply(brief).to_string())
    w("\n4) 1st crossover of the day vs later ones")
    w(ev.groupby(ev.nth.clip(upper=4)).apply(brief).rename(index={4: "4 or more"}).to_string())
    vwap_ok = ((ev.side == 1) & ev.above_vwap) | ((ev.side == -1) & ~ev.above_vwap)
    w("\n5) VWAP filter (buy above VWAP, sell below)")
    w(pd.DataFrame({"with VWAP": brief(ev[vwap_ok]), "against VWAP": brief(ev[~vwap_ok])}).to_string())
    w(f"\n6) Last {a.show} test days, stock by stock (B = buy cross, S = sell cross):")
    for d in sorted(ev.test_date.unique())[-a.show:]:
        w(f"  {d}")
        for s, g in ev[ev.test_date == d].groupby("symbol"):
            w(f"    {s}: " + " | ".join(f"{'B' if r.side == 1 else 'S'} {r.time} best {r.best_pct:+.1f}% worst {r.worst_pct:+.1f}% "
                                         f"end {r.pnl_pct:+.1f}%" for r in g.itertuples()))
    return "\n".join(L)


def part2(ev, a):
    L = []
    w = L.append
    if ev.empty or "U_t1" not in ev.columns:
        return "PART 2: no data."
    ev = ev.copy()
    ev["rsi3_al"] = np.where(ev.side == 1, ev.rsi3, 100 - ev.rsi3)
    ev["rsid_al"] = np.where(ev.side == 1, ev.rsi_d, 100 - ev.rsi_d)
    ev["gap_al"] = ev.side * ev.gap_pct
    ev["vwap_ok"] = ((ev.side == 1) & ev.above_vwap) | ((ev.side == -1) & ~ev.above_vwap)
    ev["hm"] = ev.time.str[:2].astype(int) * 60 + ev.time.str[3:].astype(int)
    subs = {"All crossovers": ev, "First crossover of the day": ev[ev.nth == 1],
            "First crossover + VWAP": ev[(ev.nth == 1) & ev.vwap_ok]}
    w("=" * 70)
    w("PART 2: YOUR STOP (candle before the crossover candle) AND R-BASED TRADE MANAGEMENT")
    w("R = distance from entry to your stop. 'Net R' is after costs. A trade that hits its stop is about -1R.")
    w(f"Stop rule: low (buy) / high (sell) of the candle before the crossover candle; if that candle is more than {a.big}x "
      f"the usual candle size, the stop is {a.big_pct * 100:.0f}% of the way down (buy) / up (sell) that candle.")
    w("\n7) How big is the risk? (stop distance in % of price)")
    w(pd.DataFrame({"your rule": ev.risk_U.describe()[["25%", "50%", "75%"]], "plain previous-candle low/high": ev.risk_L.describe()[["25%", "50%", "75%"]]})
      .round(2).T.to_string())
    w(f"   Big candles: {100 * ev.big_candle.mean():.0f}% of crossovers. Typical cost in R: {a.cost / ev.risk_U.median():.2f}R "
      f"(small stops make costs matter more).")
    w("\n8) Fixed target with your stop (win % = trades ending in profit)")
    rows = []
    for nm, x in subs.items():
        for k in (1, 2, 3, 4):
            c = f"U_t{k}"
            rows.append((nm, f"{k}R", len(x), (x[c] > 0).mean() * 100, x[c].mean(), (x[c] * x.risk_U).mean()))
    w(pd.DataFrame(rows, columns=["trades from", "target", "trades", "win %", "avg net R", "avg net %"]).round(2).to_string(index=False))
    cR = a.cost / ev.risk_U.median()
    w(f"   Break-even win rate = (1 + cost in R) / (target + 1). With your stop the typical cost is {cR:.2f}R, so a 1R target needs about "
      f"{100 * (1 + cR) / 2:.0f}% wins, 2R {100 * (1 + cR) / 3:.0f}%, 3R {100 * (1 + cR) / 4:.0f}%, 4R {100 * (1 + cR) / 5:.0f}%. "
      f"With a 1% stop the cost is only {a.cost:.2f}R: 1R needs {100 * (1 + a.cost) / 2:.0f}%, 2R {100 * (1 + a.cost) / 3:.0f}%, 3R {100 * (1 + a.cost) / 4:.0f}%.")
    ucols = [f"U_{s[0]}" for s in STRATS]
    for nm in ("First crossover of the day", "All crossovers"):
        x = subs[nm]
        w(f"\n9) Where to move the stop when price reaches +1R, +2R...? ({nm}, {len(x)} trades, target 4R). Average net R:")
        ys, xs = (1, 1.5, 2, 3), (0, 0.5, 1, 1.5, 2)
        M = pd.DataFrame(index=[f"reach {y}R" for y in ys], columns=[f"stop to {xx}R" for xx in xs], dtype=float)
        for y in ys:
            for xx in xs:
                c = f"U_bk{y}_{xx}"
                if c in x:
                    M.loc[f"reach {y}R", f"stop to {xx}R"] = x[c].mean()
        w(M.round(2).to_string(na_rep="-"))
        w("   (stop to 0R = break-even; compare with 'target 4R, stop never moved' and 'hold to 3:15' below)")
        def pf(v):
            lose = -v[v < 0].sum()
            return v[v > 0].sum() / lose if lose > 0 else np.nan
        tab = pd.DataFrame({"win %": [(x[c] > 0).mean() * 100 for c in ucols],
                            "avg net R (each trade capped at 10R)": [x[c].clip(upper=10).mean() for c in ucols],
                            "median net R": [x[c].median() for c in ucols], "profit factor": [pf(x[c]) for c in ucols],
                            "avg net %": [(x[c] * x.risk_U).mean() for c in ucols],
                            "reached ~4R %": [(x[c] > 3.4).mean() * 100 for c in ucols]},
                           index=[s[0] for s in STRATS]).round(2)
        tab = tab.rename(columns={"avg net R (each trade capped at 10R)": "avg R (cap 10)"})
        w(f"   All rules ranked by average R (a few huge winners can distort uncapped averages, so each trade is capped at 10R). t = fixed target, bk = move stop, cd = trail under each candle after +yR, "
          f"em = exit when price closes beyond EMA9 after +yR, hold = stop only until {a.exit}):")
        w(tab.sort_values("avg R (cap 10)", ascending=False).head(10).to_string())
        w("   Reference: " + "; ".join(f"{n} {tab.loc[n, 'avg R (cap 10)']:+.2f}R (win {tab.loc[n, 'win %']:.0f}%)" for n in ("t1", "t2", "t3", "t4", "hold")))
    w("\n10) Which stop size works best? Same entries, fixed targets. Each cell: average net R / win %")
    stop_cols = [("your rule (candle)", "U"), ("plain candle low/high", "L"), ("fixed 0.75%", "P075"), ("fixed 1%", "F"),
                 ("fixed 1.25%", "P125"), ("1.5 x ATR", "A15")]
    best = (ev.nth == 1) & (ev.adx3 >= 20) & (ev.rsi3_al >= 55)
    subs10 = {"All crossovers": ev, "First crossover": ev[ev.nth == 1], "First + VWAP": subs["First crossover + VWAP"],
              "First + ADX>=20 + RSI3>=55": ev[best]}
    rows = []
    for nm, x in subs10.items():
        for lab, pfx in stop_cols:
            r_ = [f"{nm} ({len(x)})", lab]
            for k in (1, 2, 3):
                c = f"{pfx}_t{k}"
                r_.append(f"{x[c].mean():+.2f} / {100 * (x[c] > 0).mean():.0f}%" if c in x and len(x) else "-")
            rows.append(r_)
    w(pd.DataFrame(rows, columns=["trades from", "stop", "1R target", "2R target", "3R target"]).to_string(index=False))
    if "risk_A15" in ev:
        w(f"   Typical stop size: your rule {ev.risk_U.median():.2f}%, plain candle {ev.risk_L.median():.2f}%, 1.5 x ATR {ev.risk_A15.median():.2f}%")

    def bucket(series, bins, labels):
        g = ev.groupby(pd.cut(series, bins, labels=labels), observed=True)
        return pd.DataFrame({"trades": g.size(), "win % at 1R": g.apply(lambda x: (x.U_t1 > 0).mean() * 100),
                             "win % at 2R": g.apply(lambda x: (x.U_t2 > 0).mean() * 100),
                             "avg net R at 2R": g.apply(lambda x: x.U_t2.mean())}).round(2)
    w("\n11) Indicator filters (all crossovers, your stop). 'aligned' = for sells the RSI is flipped, so high = momentum with the trade")
    w("Daily RSI of the stock (previous day's close), aligned:")
    w(bucket(ev.rsid_al, [-1, 40, 55, 70, 101], ["below 40", "40-55", "55-70", "above 70"]).to_string())
    w("3-minute RSI at the crossover, aligned:")
    w(bucket(ev.rsi3_al, [-1, 45, 55, 65, 101], ["below 45", "45-55", "55-65", "above 65"]).to_string())
    w("ADX (trend strength) on 3-minute candles:")
    w(bucket(ev.adx3, [-1, 15, 20, 25, 1000], ["below 15", "15-20", "20-25", "above 25"]).to_string())
    w("Volume of the crossover candle vs the previous 20 candles:")
    w(bucket(ev.volx, [-1, 1, 1.5, 2.5, 1e9], ["below 1x", "1-1.5x", "1.5-2.5x", "above 2.5x"]).to_string())
    w("Opening gap in the direction of the trade (%):")
    w(bucket(ev.gap_al, [-1000, -0.5, 0.5, 1000], ["against", "flat", "with the trade"]).to_string())

    flags = {"first": ev.nth == 1, "VWAP": ev.vwap_ok, "before 10:30": ev.hm < 630, "ADX>=20": ev.adx3 >= 20,
             "RSI3>=55": ev.rsi3_al >= 55, "dailyRSI>=55": ev.rsid_al >= 55, "volume>=1.5x": ev.volx >= 1.5, "gap with": ev.gap_al > 0}
    days = sorted(ev.test_date.unique())
    cut = int(len(days) * 0.6)
    train = ev.test_date.isin(days[:cut]).values
    test = ~train
    res = []
    for r in (1, 2, 3):
        for combo in itertools.combinations(flags, r):
            m = np.ones(len(ev), bool)
            for c in combo:
                m &= flags[c].fillna(False).values
            tr, te = m & train, m & test
            if tr.sum() < 40:
                continue
            f = lambda mask, col: (ev[col][mask] > 0).mean() * 100 if mask.sum() else np.nan
            res.append(dict(rule=" + ".join(combo), train_trades=int(tr.sum()), train_win1=f(tr, "U_t1"), train_win2=f(tr, "U_t2"),
                            train_R2=ev.U_t2[tr].mean(), test_trades=int(te.sum()), test_win1=f(te, "U_t1"),
                            test_win2=f(te, "U_t2"), test_R2=ev.U_t2[te].mean() if te.sum() else np.nan))
    w(f"\n12) Combining filters, with an honest check. Rules were chosen using the first {cut} days (train) and then checked on the "
      f"remaining {len(days) - cut} days (test) that were NOT used for choosing. {len(res)} rules tried (need 40+ train trades).")
    base = dict(rule="no filter", train_trades=int(train.sum()), train_win1=(ev.U_t1[train] > 0).mean() * 100,
                train_win2=(ev.U_t2[train] > 0).mean() * 100, train_R2=ev.U_t2[train].mean(), test_trades=int(test.sum()),
                test_win1=(ev.U_t1[test] > 0).mean() * 100, test_win2=(ev.U_t2[test] > 0).mean() * 100, test_R2=ev.U_t2[test].mean())
    if res:
        R = pd.DataFrame(res)
        w("Best by average R at a 2R target (train), and what happened on the test days:")
        w(pd.concat([pd.DataFrame([base]), R.sort_values("train_R2", ascending=False).head(6)]).round(2).to_string(index=False))
        w("Highest win rate at a 1R target (train):")
        w(R.sort_values("train_win1", ascending=False).head(5).round(2).to_string(index=False))
        w("   If a rule is good on train but weak on test, it was luck. Only trust rules that hold on BOTH and keep holding on new days.")
    return "\n".join(L)


def merge_events(new):
    f = OUT / "ema_test_events.csv"
    if f.exists():
        try:
            old = pd.read_csv(f)
            need = ["nth", "above_vwap", "TR0.5", "U_t1", "rsi3", "risk_U", "A15_t1", "P075_t1"]
            if all(c in old.columns for c in need):
                old = old.dropna(subset=["nth", "above_vwap", "U_t1", "risk_U", "A15_t1"])
                old["above_vwap"] = old["above_vwap"].astype(str).str.lower().eq("true")
                old["big_candle"] = old["big_candle"].astype(str).str.lower().eq("true")
                new = pd.concat([old, new], ignore_index=True) if len(new) else old
        except Exception as e:
            print("Could not read the earlier events file:", e)
    if len(new):
        new = new.drop_duplicates(subset=["test_date", "symbol", "time", "side"], keep="last").reset_index(drop=True)
    return new


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--cost", type=float, default=0.1, help="round-trip cost in %% per trade")
    ap.add_argument("--start", default="09:30")
    ap.add_argument("--last-entry", default="14:45")
    ap.add_argument("--exit", default="15:15")
    ap.add_argument("--show", type=int, default=2)
    ap.add_argument("--big", type=float, default=1.5, help="a candle is 'big' if it is this many times the usual candle size")
    ap.add_argument("--big-pct", type=float, default=0.65, help="for big candles the stop sits this share down (buy) / up (sell) the candle (0.6-0.75)")
    ap.add_argument("--min-risk", type=float, default=0.2, help="smallest stop distance in %% of price")
    a = ap.parse_args()
    plan = []
    for ld, td, syms in load_lists(a.top):
        if (dt.date.today() - dt.date.fromisoformat(td)).days > 28:
            plan.append((ld, td, syms, True))
        else:
            plan.append((ld, td, syms, False))
    skipped = [p[1] for p in plan if p[3]]
    todo = [p for p in plan if not p[3]]
    allsyms = sorted({s for p in todo for s in p[2]})
    daily = {}
    if allsyms:
        try:
            daily = fetch_daily(allsyms)
        except Exception as e:
            print("Daily candles not available, daily RSI will be empty:", e)
    rows = []
    for ld, td, syms, _ in todo:
        day = dt.date.fromisoformat(td)
        data = fetch_day(syms, day)
        print(f"{td}: {len(data)} of {len(syms)} stocks had data")
        for s, d in data.items():
            rsid = daily_rsi_before(daily[s], day) if s in daily else np.nan
            rows += find_events(to_3m(d), s, ld, day, a, rsid)
    ev = merge_events(pd.DataFrame(rows))
    txt = report(ev, a, skipped) + "\n\n" + part2(ev, a)
    print("\n" + txt)
    OUT.mkdir(exist_ok=True)
    (OUT / "ema_test_report.txt").write_text(txt, encoding="utf-8")
    if len(ev):
        ev.to_csv(OUT / "ema_test_events.csv", index=False)
    summ = os.environ.get("GITHUB_STEP_SUMMARY")
    if summ:
        with open(summ, "a", encoding="utf-8") as f:
            f.write("```\n" + txt + "\n```\n")


if __name__ == "__main__":
    main()
