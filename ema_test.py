#!/usr/bin/env python3
"""
EMA 9/21 crossover test on 3-minute candles, for the stocks in the scanner lists. For learning only.

For every saved list (results/history/*.json) it takes the top stocks and looks at the NEXT trading day:
which stocks had a 9/21 EMA crossover, how far the price moved afterwards (best and worst point),
and which target / stop-loss levels would have worked best. Buy side and sell side are tested.

Data: Yahoo Finance 1-minute candles, built into 3-minute candles. Yahoo keeps only about 30 days of
1-minute data, so only the last 4 weeks of lists can be tested.

Usage:   python ema_test.py                      (top 10 stocks of each list)
         python ema_test.py --top 20 --cost 0.1
Needs:   pip install pandas numpy yfinance
"""
import argparse, datetime as dt, json, os
from pathlib import Path

import numpy as np
import pandas as pd

BASE = Path(__file__).parent
HIST, OUT = BASE / "results" / "history", BASE / "results"
TARGETS, STOPS, TRAILS = (0.5, 1, 1.5, 2, 3, 4), (0.5, 1, 1.5, 2), (0.5, 1, 1.5, 2)


def tm(t):
    h, m = t.split(":")
    return int(h) * 60 + int(m)


def load_lists(top):
    lists = [json.loads(f.read_text()) for f in sorted(HIST.glob("*.json"))]
    lists = [x for x in lists if "movers" in x]
    return [(a["asof"], b["asof"], [m["symbol"] for m in a["movers"][:top]]) for a, b in zip(lists[:-1], lists[1:])]


def fetch_day(symbols, day):
    """1-minute candles for the test day plus a few days before it (so the EMAs are warmed up)."""
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


def to_3m(d):
    d = d.copy()
    d.index = pd.to_datetime(d.index)
    if d.index.tz is not None:
        d.index = d.index.tz_convert("Asia/Kolkata").tz_localize(None)
    b = d.resample("3min").agg({"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"})
    b = b.dropna(subset=["Close"])
    m = b.index.hour * 60 + b.index.minute
    return b[(m >= 555) & (m < 930)]  # 9:15 to 15:30


def sim(side, entry, H, L, exit_p, T, S, cost):
    """Exit at target T% or stop S%. If both fall in one candle we assume the stop came first (cautious)."""
    for h, l in zip(H, L):
        stop = l <= entry * (1 - S / 100) if side == 1 else h >= entry * (1 + S / 100)
        tgt = h >= entry * (1 + T / 100) if side == 1 else l <= entry * (1 - T / 100)
        if stop:
            return -S - cost
        if tgt:
            return T - cost
    return side * (exit_p / entry - 1) * 100 - cost


def sim_trail(side, entry, H, L, exit_p, init, trail, cost):
    """Start with an initial stop; once price moves in our favour the stop follows `trail`% behind the best price."""
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


def find_events(b, sym, ld, day, a):
    if len(b) < 30:
        return []
    b = b.copy()
    b["e9"] = b.Close.ewm(span=9, adjust=False).mean()
    b["e21"] = b.Close.ewm(span=21, adjust=False).mean()
    tp = (b.High + b.Low + b.Close) / 3
    dkey = pd.Series(b.index.date, index=b.index)
    b["vwap"] = (tp * b.Volume).groupby(dkey).cumsum() / b.Volume.groupby(dkey).cumsum().replace(0, np.nan)
    up = (b.e9 > b.e21).values
    idx = b.index
    mins = np.asarray(idx.hour * 60 + idx.minute)
    on_day = np.array([x == day for x in idx.date])
    cross = np.zeros(len(b), bool)
    cross[1:] = up[1:] != up[:-1]
    cross &= on_day
    hi, lo, cl = b.High.values, b.Low.values, b.Close.values
    day_idx = np.where(on_day & (mins <= tm(a.exit)))[0]
    if len(day_idx) == 0:
        return []
    last_i = day_idx[-1]
    cross_idx = np.where(cross)[0]
    out, nth = [], 0
    for i in cross_idx:
        if mins[i] < tm(a.start) or mins[i] > tm(a.last_entry) or i >= last_i:
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
                   pnl_hold_pct=round(side * (cl[last_i] / entry - 1) * 100 - a.cost, 2),
                   nth=nth, above_vwap=bool(cl[i] > b.vwap.values[i]) if np.isfinite(b.vwap.values[i]) else False)
        for t in TRAILS:
            row[f"TR{t}"] = round(sim_trail(side, entry, H, L, exit_p, 1.0, t, a.cost), 3)
        for T in TARGETS:
            for S in STOPS:
                row[f"T{T}_S{S}"] = round(sim(side, entry, H, L, exit_p, T, S, a.cost), 3)
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


def report(ev, a, skipped):
    L = []
    w = L.append
    w(f"EMA 9/21 crossover test, 3-minute candles, top {a.top} stocks of each list, costs {a.cost}% per trade")
    w(f"Entries between {a.start} and {a.last_entry}; every trade ends at the opposite crossover or {a.exit}.")
    if skipped:
        w(f"Skipped {len(skipped)} lists older than 4 weeks (Yahoo has no 1-minute data): {skipped[0]} to {skipped[-1]}")
    if ev.empty:
        w("No crossovers found (or Yahoo returned no data).")
        return "\n".join(L)
    w(f"Test days: {ev.test_date.nunique()} | stock-days with a crossover: {ev.groupby(['test_date', 'symbol']).ngroups} | crossovers: {len(ev)}")
    buy, sell = ev[ev.side == 1], ev[ev.side == -1]
    w("\n1) BUY = 9 crosses above 21 (go long), SELL = 9 crosses below 21 (go short). Best/worst point = best/worst price reached after entry.")
    w(pd.DataFrame({"BUY": side_stats(buy) if len(buy) else np.nan, "SELL": side_stats(sell) if len(sell) else np.nan,
                    "BOTH": side_stats(ev)}).to_string())
    w("\n2) How far did price go in our favour (best point %) and against us (worst point %)?")
    q = pd.DataFrame({"best % BOTH": ev.best_pct.quantile([.25, .5, .75, .9]).round(2).values,
                      "worst % BOTH": ev.worst_pct.quantile([.75, .5, .25, .1]).round(2).values},
                     index=["typical small", "median", "good", "very good / very bad"])
    w(q.to_string())
    cols = [c for c in ev.columns if c.startswith("T")]
    for name, x in (("BOTH sides", ev), ("BUY only", buy), ("SELL only", sell)):
        if x.empty:
            continue
        m = x[cols].mean().round(3)
        mat = pd.DataFrame([[m[f"T{t}_S{s}"] for s in STOPS] for t in TARGETS],
                           index=[f"target {t}%" for t in TARGETS], columns=[f"stop {s}%" for s in STOPS])
        w(f"\n3) Average net P&L % per trade for each target and stop-loss, {name} ({len(x)} trades):")
        w(mat.to_string())
        top = m.sort_values(ascending=False).head(3)
        w("   best combinations: " + "; ".join(f"{k.replace('_', ' / ').replace('T', 'target ', 1).replace('S', 'stop ')}: {v:+.3f}% "
          f"(win {100 * (x[k] > 0).mean():.0f}%)" for k, v in top.items()))
    def brief(x):
        if x.empty:
            return pd.Series({"trades": 0})
        return pd.Series({"trades": len(x), "win %": (x.pnl_pct > 0).mean() * 100, "avg P&L %": x.pnl_pct.mean(),
                          "avg best %": x.best_pct.mean(), "avg worst %": x.worst_pct.mean()}).round(2)
    hr = ev.time.str[:2].astype(int) * 60 + ev.time.str[3:].astype(int)
    slots = pd.cut(hr, [0, 629, 689, 749, 809, 869, 929, 1440],
                   labels=["9:30-10:29", "10:30-11:29", "11:30-12:29", "12:30-13:29", "13:30-14:29", "14:30-14:45", "later"])
    w("\n4) Does the time of the crossover matter? (all trades, net)")
    w(ev.groupby(slots, observed=True).apply(brief).to_string())
    w("\n5) Is the 1st crossover of the day better than later ones? (nth crossover of that stock that day)")
    w(ev.groupby(ev.nth.clip(upper=4)).apply(brief).rename(index={4: "4 or more"}).to_string())
    cnt = ev.groupby(["test_date", "symbol"]).symbol.transform("count")
    w("\n6) Choppy days (6 or more crossovers in one stock) vs calmer days (5 or fewer). Note: you only know this at the end of the day, so it explains the result but cannot be used as a rule.")
    w(pd.DataFrame({"6+ crossovers": brief(ev[cnt >= 6]), "5 or fewer": brief(ev[cnt < 6])}).to_string())
    with_vwap = ev[((ev.side == 1) & ev.above_vwap) | ((ev.side == -1) & ~ev.above_vwap)]
    against = ev.drop(with_vwap.index)
    w("\n7) VWAP filter: buy only above VWAP, sell only below VWAP (trading with the day's average price)")
    w(pd.DataFrame({"with VWAP": brief(with_vwap), "against VWAP": brief(against)}).to_string())
    first = ev[ev.nth == 1]
    w("\n8) Only the first crossover of each stock each day, and the same with the VWAP filter")
    w(pd.DataFrame({"first cross, ends at opposite cross": brief(first),
                    "first cross, with VWAP": brief(first.loc[first.index.intersection(with_vwap.index)])}).to_string())
    tcols = [c for c in ev.columns if c.startswith("TR")]
    w("\n9) Trailing stop instead of a fixed target (initial stop 1%, then the stop follows the best price). Average net P&L %:")
    w(pd.DataFrame({"BOTH": ev[tcols].mean(), "BUY": buy[tcols].mean() if len(buy) else np.nan,
                    "SELL": sell[tcols].mean() if len(sell) else np.nan}).round(3)
      .rename(index={f"TR{t}": f"follows {t}% behind best price" for t in TRAILS}).to_string())
    w("\n10) Five best and five worst crossovers (time, side, final net P&L %):")
    cs = ["test_date", "symbol", "time", "side", "entry", "best_pct", "worst_pct", "pnl_pct"]
    w(ev.sort_values("pnl_pct", ascending=False)[cs].head(5).to_string(index=False))
    w(ev.sort_values("pnl_pct")[cs].head(5).to_string(index=False))
    w(f"\n11) Last {a.show} test days, stock by stock (B = buy cross, S = sell cross):")
    for d in sorted(ev.test_date.unique())[-a.show:]:
        w(f"  {d}")
        for s, g in ev[ev.test_date == d].groupby("symbol"):
            w(f"    {s}: " + " | ".join(f"{'B' if r.side == 1 else 'S'} {r.time} best {r.best_pct:+.1f}% worst {r.worst_pct:+.1f}% "
                                         f"end {r.pnl_pct:+.1f}%" for r in g.itertuples()))
    return "\n".join(L)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=10)
    ap.add_argument("--cost", type=float, default=0.1, help="round-trip cost in %% per trade")
    ap.add_argument("--start", default="09:30", help="earliest entry time")
    ap.add_argument("--last-entry", default="14:45")
    ap.add_argument("--exit", default="15:15", help="time when open trades are closed")
    ap.add_argument("--show", type=int, default=3, help="how many recent days to list stock by stock")
    a = ap.parse_args()
    rows, skipped = [], []
    for ld, td, syms in load_lists(a.top):
        day = dt.date.fromisoformat(td)
        if (dt.date.today() - day).days > 28:
            skipped.append(td)
            continue
        data = fetch_day(syms, day)
        print(f"{td}: {len(data)} of {len(syms)} stocks had data")
        for s, d in data.items():
            rows += find_events(to_3m(d), s, ld, day, a)
    ev = pd.DataFrame(rows)
    txt = report(ev, a, skipped)
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
