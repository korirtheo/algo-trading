#!/usr/bin/env python3
"""Historical dynamic top-gainer lifecycle builder for Alpaca SIP.

Goal
----
Reconstruct a causal top-gainer leaderboard every 5 minutes from 04:00 ET through
16:00 ET, then retain 1-minute SIP bars for every symbol that ever enters the
leaderboard. Each entrant's lifecycle is D-1 AH (16:00-19:59 ET) through D AH
(16:00-19:59 ET), allowing AH->PM->RTH research without lookahead.

Architecture (speed-first)
--------------------------
1) Pull 1Day bars once for the historical asset master to create prior-close and
   RTH-high references.
2) Pull 30Min PM bars across the broad universe as a cheap candidate screen.
3) Union generous PM candidates with RTH daily-high candidates.
4) Pull 5Min bars only for candidate symbol-days and rank causally at completed
   5-minute boundaries. 09:30 is handled separately using the exact opening
   print from a 1Min bar so it is compatible with the legacy 09:30 universe.
5) Pull 1Min D-1 AH -> D AH only for symbols that actually enter top N.

Every stage is resumable. Canonical outputs are CSV.GZ; SQLite is only an
internal index/cache and can be rebuilt.

Credentials are read from ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import heapq
import json
import os
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Iterable
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
UTC = timezone.utc
DATA = "https://data.alpaca.markets"
TRADING = "https://paper-api.alpaca.markets"
EXCHANGES = {"NASDAQ", "NYSE", "AMEX", "ARCA", "NYSEARCA", "BATS"}
BAD_NAME = re.compile(r"\b(warrant|warrants|unit|units|right|rights)\b", re.I)
BAD_SYMBOL = re.compile(r"(?:\.W|\.WS|\.U|\.R|/WS|/U|/R)$", re.I)
STD_SYMBOL = re.compile(r"[A-Z][A-Z.]{0,5}$")

MEMBERSHIP_FIELDS = [
    "event_date", "snapshot_et", "session", "rank", "symbol", "last_price",
    "prev_close", "gain_pct", "pm_volume_to_snapshot", "first_seen_today"
]
ENTRANT_FIELDS = ["event_date", "symbol", "first_seen_et", "first_rank", "first_session"]
LIFECYCLE_FIELDS = [
    "timestamp_utc", "timestamp_et", "symbol", "open", "high", "low", "close",
    "volume", "trade_count", "vwap", "event_date", "bar_date_et", "session", "source"
]


def atomic_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()


def parse_ts(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", "+00:00"))


def iso_et(d: date, hh: int, mm: int = 0) -> str:
    return datetime(d.year, d.month, d.day, hh, mm, tzinfo=ET).astimezone(UTC).isoformat()


def chunked(xs: list[str], n: int) -> Iterable[list[str]]:
    for i in range(0, len(xs), n):
        yield xs[i:i+n]


def session_of(ts_et: datetime) -> str:
    m = ts_et.hour * 60 + ts_et.minute
    if 4 * 60 <= m < 9 * 60 + 30:
        return "PM"
    if 9 * 60 + 30 <= m < 16 * 60:
        return "RTH"
    if 16 * 60 <= m < 20 * 60:
        return "AH"
    return "OFF"


@dataclass
class RequestStats:
    requests: int = 0
    pages: int = 0
    retries: int = 0
    rate_429: int = 0
    server_5xx: int = 0
    bytes_read: int = 0


class AdaptiveRateLimiter:
    def __init__(self, configured_rpm: int = 0, safety: float = 0.92):
        self.configured_rpm = configured_rpm
        self.detected_rpm = None
        self.safety = safety
        self.calls = deque()
        self.lock = threading.Lock()

    @property
    def rpm(self) -> int:
        raw = self.configured_rpm or self.detected_rpm or 180
        return max(1, int(raw * self.safety))

    def observe(self, hdrs) -> None:
        try:
            v = hdrs.get("X-RateLimit-Limit") or hdrs.get("x-ratelimit-limit")
            if v:
                with self.lock:
                    self.detected_rpm = int(float(v))
        except Exception:
            pass

    def acquire(self) -> None:
        while True:
            now = time.monotonic()
            with self.lock:
                while self.calls and now - self.calls[0] >= 60:
                    self.calls.popleft()
                if len(self.calls) < self.rpm:
                    self.calls.append(now)
                    return
                delay = max(0.02, 60 - (now - self.calls[0]))
            time.sleep(min(delay, 1.0))


class AlpacaClient:
    def __init__(self, limiter: AdaptiveRateLimiter, stats: RequestStats):
        key = os.environ.get("ALPACA_API_KEY_ID")
        sec = os.environ.get("ALPACA_API_SECRET_KEY")
        if not key or not sec:
            raise SystemExit("Set ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY")
        self.hdrs = {
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": sec,
            "Accept": "application/json",
            "User-Agent": "dynamic-gainer-builder/1.0",
        }
        self.limiter = limiter
        self.stats = stats
        self.lock = threading.Lock()

    def _bump(self, **kw):
        with self.lock:
            for k, v in kw.items():
                setattr(self.stats, k, getattr(self.stats, k) + v)

    def get_json(self, url: str, tries: int = 12):
        delay = 0.5
        last_error = None
        for attempt in range(tries):
            self.limiter.acquire()
            req = urllib.request.Request(url, headers=self.hdrs)
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    raw = r.read()
                    self.limiter.observe(r.headers)
                    self._bump(requests=1, bytes_read=len(raw))
                    return json.loads(raw)
            except urllib.error.HTTPError as e:
                self._bump(requests=1)
                if e.code == 429:
                    self._bump(rate_429=1, retries=1)
                    retry = e.headers.get("Retry-After")
                    time.sleep(float(retry) if retry else delay)
                elif 500 <= e.code < 600:
                    self._bump(server_5xx=1, retries=1)
                    time.sleep(delay)
                else:
                    body = e.read().decode("utf-8", "ignore")[:500]
                    raise RuntimeError(f"HTTP {e.code}: {body}") from e
            except Exception as e:
                last_error = f"{type(e).__name__}: {e}"
                self._bump(retries=1)
                time.sleep(delay)
            delay = min(delay * 1.8, 20)
        raise RuntimeError(f"request failed after retries: {url}; last_error={last_error}")

    def bars(self, symbols: list[str], timeframe: str, start: str, end: str, *, asof: str | None = None):
        out = defaultdict(list)
        token = None
        while True:
            params = {
                "symbols": ",".join(symbols), "timeframe": timeframe,
                "start": start, "end": end, "limit": 10000,
                "adjustment": "raw", "feed": "sip", "sort": "asc",
            }
            if asof:
                params["asof"] = asof
            if token:
                params["page_token"] = token
            p = self.get_json(DATA + "/v2/stocks/bars?" + urllib.parse.urlencode(params))
            self._bump(pages=1)
            for sym, rows in (p.get("bars") or {}).items():
                out[sym].extend(rows)
            token = p.get("next_page_token")
            if not token:
                return dict(out)


def get_asset_master(client: AlpacaClient, outdir: Path) -> list[str]:
    path = outdir / "asset_master.json"
    if path.exists():
        return json.loads(path.read_text())["eligible_symbols"]
    assets = []
    for status in ("active", "inactive"):
        q = urllib.parse.urlencode({"asset_class": "us_equity", "status": status})
        p = client.get_json(TRADING + "/v2/assets?" + q)
        if isinstance(p, list):
            assets.extend(p)
    eligible, rejected = [], []
    for a in assets:
        sym = (a.get("symbol") or "").strip()
        name = (a.get("name") or "").strip()
        exch = (a.get("exchange") or "").upper()
        why = None
        if not sym: why = "blank"
        elif exch not in EXCHANGES: why = f"exchange:{exch}"
        elif not STD_SYMBOL.fullmatch(sym): why = "nonstandard_symbol"
        elif BAD_NAME.search(name): why = "warrant_unit_right_name"
        elif BAD_SYMBOL.search(sym): why = "warrant_unit_right_symbol"
        if why: rejected.append({"symbol":sym,"name":name,"exchange":exch,"reason":why})
        else: eligible.append(sym)
    eligible = sorted(set(eligible))
    atomic_json(path, {"eligible_count":len(eligible),"rejected_count":len(rejected),
                       "eligible_symbols":eligible,"rejected":rejected})
    return eligible


def get_calendar(client: AlpacaClient, year: int, outdir: Path) -> list[date]:
    p = outdir / "calendar.json"
    # Exclude the current ET date so the final session's after-hours bars are
    # complete. This also keeps future calendar sessions out of a year-to-date run.
    through = min(date(year, 12, 31), datetime.now(ET).date() - timedelta(days=1))
    q = urllib.parse.urlencode({"start":f"{year}-01-01", "end":through.isoformat()})
    rows = client.get_json(TRADING + "/v2/calendar?" + q)
    ds = [date.fromisoformat(r["date"]) for r in rows]
    atomic_json(p, {"dates":[d.isoformat() for d in ds]})
    return ds


def prev_map(calendar: list[date]) -> dict[date, date]:
    return {calendar[i]: calendar[i-1] for i in range(1, len(calendar))}


def valid_gzip(path: Path) -> bool:
    """A completed shard must decompress through its end-of-stream marker."""
    try:
        with gzip.open(path, "rb") as f:
            while f.read(1024 * 1024):
                pass
        return True
    except (OSError, EOFError):
        return False


def init_db(path: Path):
    con = sqlite3.connect(path)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA synchronous=NORMAL")
    con.execute("PRAGMA temp_store=MEMORY")
    con.execute("""CREATE TABLE IF NOT EXISTS daily_ref(
        event_date TEXT NOT NULL, symbol TEXT NOT NULL, prev_close REAL NOT NULL,
        rth_high REAL, rth_close REAL, rth_high_gain REAL,
        PRIMARY KEY(event_date,symbol)) WITHOUT ROWID""")
    con.execute("CREATE INDEX IF NOT EXISTS daily_ref_date ON daily_ref(event_date)")
    con.commit()
    return con


def stage_daily_reference(client, symbols, cal, year, outdir, workers, batch_size):
    stage = outdir / "stage1_daily_reference"; stage.mkdir(parents=True, exist_ok=True)
    done = stage / "COMPLETE.json"; dbp = stage / "reference.sqlite"
    if done.exists() and dbp.exists(): return dbp
    shards = stage / "shards"; shards.mkdir(exist_ok=True)
    start = (cal[0] - timedelta(days=10)).isoformat() + "T00:00:00Z"
    end = (cal[-1] + timedelta(days=1)).isoformat() + "T00:00:00Z"
    batches = list(enumerate(chunked(symbols, batch_size)))

    def work(item):
        idx, syms = item; dest = shards / f"batch_{idx:04d}.csv.gz"
        if dest.exists() and dest.stat().st_size > 50 and valid_gzip(dest): return idx, "skip"
        if dest.exists(): dest.unlink()
        bars = client.bars(list(syms), "1Day", start, end, asof=f"{year}-12-31")
        tmp = dest.with_name(dest.name + ".part")
        with gzip.open(tmp, "wt", newline="") as f:
            w = csv.writer(f); w.writerow(["event_date","symbol","prev_close","rth_high","rth_close","rth_high_gain"])
            for sym, rr in bars.items():
                rr = sorted(rr, key=lambda x:x["t"])
                prev = None
                for b in rr:
                    d = parse_ts(b["t"]).date()
                    close = float(b["c"]); high = float(b["h"])
                    if d.year == year and prev and prev > 0:
                        w.writerow([d.isoformat(), sym, prev, high, close, (high/prev-1)*100])
                    prev = close
        tmp.replace(dest)
        return idx, "ok"

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(work, x) for x in batches]
        for n,f in enumerate(as_completed(futs),1):
            idx, st = f.result()
            print(f"daily_ref {n}/{len(futs)} batch={idx:04d} {st}", flush=True)

    con = init_db(dbp)
    con.execute("DELETE FROM daily_ref")
    sql = "INSERT OR REPLACE INTO daily_ref VALUES (?,?,?,?,?,?)"
    for p in sorted(shards.glob("batch_*.csv.gz")):
        with gzip.open(p,"rt",newline="") as f:
            r=csv.DictReader(f); buf=[]
            for x in r:
                buf.append((x["event_date"],x["symbol"],float(x["prev_close"]),float(x["rth_high"]),float(x["rth_close"]),float(x["rth_high_gain"])))
                if len(buf)>=5000:
                    con.executemany(sql,buf); buf.clear()
            if buf: con.executemany(sql,buf)
    con.commit(); rows=con.execute("SELECT COUNT(*) FROM daily_ref").fetchone()[0]; con.close()
    atomic_json(done,{"year":year,"rows":rows,"sha256":sha256(dbp)})
    return dbp


def heap_push_top(heap, item, k):
    # item=(score,symbol); retain k largest scores
    if len(heap)<k: heapq.heappush(heap,item)
    elif item[0]>heap[0][0]: heapq.heapreplace(heap,item)


def stage_candidate_days(client, symbols, cal, year, outdir, dbp, workers, batch_size,
                         pool_size, gain_floor):
    stage=outdir/"stage2_candidates"; stage.mkdir(parents=True,exist_ok=True)
    days=stage/"days"; days.mkdir(exist_ok=True)
    con=sqlite3.connect(dbp)
    # Build per-date refs once into compact dicts; ~one trading day at a time in workers isn't sqlite-thread friendly.
    refs={}
    for d in cal:
        rows=con.execute("SELECT symbol,prev_close,rth_high_gain FROM daily_ref WHERE event_date=?",(d.isoformat(),)).fetchall()
        refs[d.isoformat()]={s:(pc,hg) for s,pc,hg in rows}
    con.close()

    def work(d):
        ds=d.isoformat(); dest=days/f"{ds}.csv.gz"
        if dest.exists() and dest.stat().st_size>50: return ds,"skip",0
        ref=refs.get(ds,{})
        syms=[s for s in symbols if s in ref]
        # RTH: daily high is a safe coarse envelope for any intraday 5m close.
        rth_heap=[]
        rth_floor=set()
        for s,(pc,hg) in ref.items():
            heap_push_top(rth_heap,(hg,s),pool_size)
            if hg>=gain_floor: rth_floor.add(s)
        candidate=set(s for _,s in rth_heap)|rth_floor
        # PM: 30m high is a safe envelope for any 5m close in that interval.
        bucket_heaps=defaultdict(list); floor_syms=set()
        for grp in chunked(syms,batch_size):
            data=client.bars(list(grp),"30Min",iso_et(d,4),iso_et(d,9,30),asof=ds)
            for s,rr in data.items():
                pc=ref[s][0]
                for b in rr:
                    ts=parse_ts(b["t"]).astimezone(ET)
                    if not (4<=ts.hour<10) or (ts.hour==9 and ts.minute>=30): continue
                    g=(float(b["h"])/pc-1)*100
                    key=ts.strftime("%H:%M")
                    heap_push_top(bucket_heaps[key],(g,s),pool_size)
                    if g>=gain_floor: floor_syms.add(s)
        candidate |= floor_syms
        for hp in bucket_heaps.values(): candidate |= {s for _,s in hp}
        with gzip.open(dest,"wt",newline="") as f:
            w=csv.writer(f); w.writerow(["event_date","symbol","prev_close"])
            for s in sorted(candidate): w.writerow([ds,s,ref[s][0]])
        return ds,"ok",len(candidate)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs=[ex.submit(work,d) for d in cal]
        for n,f in enumerate(as_completed(futs),1):
            ds,st,c=f.result(); print(f"candidates {n}/{len(futs)} {ds} {st} n={c}",flush=True)
    return days


def completed_snapshot_time(bar_start_et: datetime) -> datetime:
    return bar_start_et + timedelta(minutes=5)


def stage_membership(client, cal, year, outdir, candidate_days, top_n, batch_size, workers):
    stage=outdir/"stage3_membership"; stage.mkdir(parents=True,exist_ok=True)
    days=stage/"days"; days.mkdir(exist_ok=True)

    def work(d):
        ds=d.isoformat(); dest=days/f"{ds}.csv.gz"; ent=days/f"{ds}_entrants.csv.gz"
        if dest.exists() and ent.exists() and valid_gzip(dest) and valid_gzip(ent): return ds,"skip",0
        if dest.exists(): dest.unlink()
        if ent.exists(): ent.unlink()
        cp=candidate_days/f"{ds}.csv.gz"
        if not cp.exists(): return ds,"no_candidates",0
        with gzip.open(cp,"rt",newline="") as f:
            rows=list(csv.DictReader(f))
        prev={r["symbol"]:float(r["prev_close"]) for r in rows}; syms=list(prev)
        # updates[snapshot][symbol]=(price, incremental_volume). 5m bar close becomes known at bar_start+5m.
        updates=defaultdict(dict); pm_vol=defaultdict(float)
        for grp in chunked(syms,batch_size):
            data=client.bars(list(grp),"5Min",iso_et(d,4),iso_et(d,16),asof=ds)
            for s,rr in data.items():
                for b in rr:
                    st=parse_ts(b["t"]).astimezone(ET); snap=completed_snapshot_time(st)
                    if snap>datetime(d.year,d.month,d.day,16,0,tzinfo=ET): continue
                    updates[snap][s]=(float(b["c"]),float(b["v"]),st)
        # exact opening print; at 09:30 use 09:30 1m open, not a future-containing 5m close.
        open_px={}
        for grp in chunked(syms,batch_size):
            data=client.bars(list(grp),"1Min",iso_et(d,9,30),iso_et(d,9,31),asof=ds)
            for s,rr in data.items():
                if rr: open_px[s]=float(rr[0]["o"])

        # Build snapshot schedule: completed PM 5m bars, exact 09:30 open, completed RTH 5m bars.
        snaps=[]
        t=datetime(d.year,d.month,d.day,4,5,tzinfo=ET)
        while t<=datetime(d.year,d.month,d.day,9,25,tzinfo=ET): snaps.append(t); t+=timedelta(minutes=5)
        snaps.append(datetime(d.year,d.month,d.day,9,30,tzinfo=ET))
        t=datetime(d.year,d.month,d.day,9,35,tzinfo=ET)
        while t<=datetime(d.year,d.month,d.day,16,0,tzinfo=ET): snaps.append(t); t+=timedelta(minutes=5)

        last={}; first_seen={}; out=[]
        # accumulate PM volume from 5m updates as they become known
        for snap in snaps:
            if snap.hour==9 and snap.minute==30:
                for s,p in open_px.items(): last[s]=p
            else:
                for s,(p,v,st) in updates.get(snap,{}).items():
                    last[s]=p
                    if st.hour<9 or (st.hour==9 and st.minute<30): pm_vol[s]+=v
            ranked=[]
            for s,p in last.items():
                pc=prev.get(s)
                if pc and pc>0: ranked.append(((p/pc-1)*100,s,p))
            ranked.sort(reverse=True)
            sess="PM" if snap.time()<datetime(d.year,d.month,d.day,9,30,tzinfo=ET).time() else "RTH"
            for rank,(g,s,p) in enumerate(ranked[:top_n],1):
                if s not in first_seen: first_seen[s]=(snap,rank,sess)
                out.append([ds,snap.isoformat(),sess,rank,s,p,prev[s],g,pm_vol.get(s,0.0),first_seen[s][0].isoformat()])
        with gzip.open(dest,"wt",newline="") as f:
            w=csv.writer(f); w.writerow(MEMBERSHIP_FIELDS); w.writerows(out)
        with gzip.open(ent,"wt",newline="") as f:
            w=csv.writer(f); w.writerow(ENTRANT_FIELDS)
            for s,(snap,rank,sess) in sorted(first_seen.items(),key=lambda x:x[1][0]):
                w.writerow([ds,s,snap.isoformat(),rank,sess])
        return ds,"ok",len(first_seen)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs=[ex.submit(work,d) for d in cal]
        for n,f in enumerate(as_completed(futs),1):
            ds,st,c=f.result(); print(f"membership {n}/{len(futs)} {ds} {st} entrants={c}",flush=True)

    mem=stage/f"membership_{year}_top{top_n}.csv.gz"; entrants=stage/f"entrants_{year}_top{top_n}.csv.gz"
    with gzip.open(mem,"wt",newline="") as fo:
        w=csv.writer(fo); w.writerow(MEMBERSHIP_FIELDS)
        for p in sorted(days.glob("20??-??-??.csv.gz")):
            with gzip.open(p,"rt",newline="") as fi:
                r=csv.reader(fi); next(r,None); w.writerows(r)
    with gzip.open(entrants,"wt",newline="") as fo:
        w=csv.writer(fo); w.writerow(ENTRANT_FIELDS)
        for p in sorted(days.glob("*_entrants.csv.gz")):
            with gzip.open(p,"rt",newline="") as fi:
                r=csv.reader(fi); next(r,None); w.writerows(r)
    return mem,entrants


def load_entrants(path: Path):
    by=defaultdict(list)
    with gzip.open(path,"rt",newline="") as f:
        for r in csv.DictReader(f): by[date.fromisoformat(r["event_date"])].append(r)
    return by


def stage_lifecycle(client, cal, year, outdir, entrants_path, batch_size, workers, months=None):
    stage=outdir/"stage4_lifecycle"; stage.mkdir(parents=True,exist_ok=True)
    days=stage/"days"; days.mkdir(exist_ok=True)
    by=load_entrants(entrants_path); pm=prev_map(cal)
    # The first trading day needs its prior-year trading session for D-1 AH.
    if cal:
        first=cal[0]
        q=urllib.parse.urlencode({"start":(first-timedelta(days=10)).isoformat(),"end":(first-timedelta(days=1)).isoformat()})
        prior=client.get_json(TRADING + "/v2/calendar?" + q)
        if prior:
            pm[first]=date.fromisoformat(prior[-1]["date"])

    def work(d):
        ds=d.isoformat(); dest=days/f"{ds}.csv.gz"
        if dest.exists() and dest.stat().st_size>50 and valid_gzip(dest): return ds,"skip",0
        if dest.exists(): dest.unlink()
        rows=by.get(d,[]); syms=sorted({r["symbol"] for r in rows})
        if not syms or d not in pm: return ds,"empty",0
        pd=pm[d]; allrows=[]
        for grp in chunked(syms,batch_size):
            data=client.bars(list(grp),"1Min",iso_et(pd,16),iso_et(d,20),asof=ds)
            for s,rr in data.items():
                for b in rr:
                    tu=parse_ts(b["t"]); te=tu.astimezone(ET); sess=session_of(te)
                    if sess=="OFF": continue
                    # Only previous-day AH plus event-day PM/RTH/AH.
                    if te.date()==pd and sess!="AH": continue
                    if te.date() not in (pd,d): continue
                    allrows.append([tu.isoformat(),te.isoformat(),s,b["o"],b["h"],b["l"],b["c"],b["v"],b.get("n"),b.get("vw"),ds,te.date().isoformat(),sess,"alpaca_sip"])
        allrows.sort(key=lambda x:(x[0],x[2]))
        tmp=dest.with_name(dest.name + ".part")
        with gzip.open(tmp,"wt",newline="") as f:
            w=csv.writer(f); w.writerow(LIFECYCLE_FIELDS); w.writerows(allrows)
        tmp.replace(dest)
        return ds,"ok",len(allrows)

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs=[ex.submit(work,d) for d in cal if d in by and (months is None or d.month in months)]
        for n,f in enumerate(as_completed(futs),1):
            ds,st,c=f.result(); print(f"lifecycle {n}/{len(futs)} {ds} {st} bars={c}",flush=True)

    # Monthly canonical partitions keep files manageable and portable.
    outputs=[]
    for m in range(1,13):
        if months is not None and m not in months: continue
        ps=sorted(days.glob(f"{year}-{m:02d}-??.csv.gz"))
        if not ps: continue
        out=stage/f"lifecycle_{year}_{m:02d}.csv.gz"
        with gzip.open(out,"wt",newline="") as fo:
            w=csv.writer(fo); w.writerow(LIFECYCLE_FIELDS)
            for p in ps:
                with gzip.open(p,"rt",newline="") as fi:
                    r=csv.reader(fi); next(r,None); w.writerows(r)
        outputs.append(out)
    return outputs


def validate_outputs(year, outdir, membership, entrants, lifecycle_parts, top_n):
    report={"year":year,"top_n":top_n,"status":"PASS","checks":{}}
    # Membership structural checks.
    nmem=0; bad_rank=0; snapshots=set(); syms=set()
    with gzip.open(membership,"rt",newline="") as f:
        for r in csv.DictReader(f):
            nmem+=1; snapshots.add((r["event_date"],r["snapshot_et"])); syms.add((r["event_date"],r["symbol"]))
            if not (1<=int(r["rank"])<=top_n): bad_rank+=1
    nent=0; entrant_keys=set()
    with gzip.open(entrants,"rt",newline="") as f:
        for r in csv.DictReader(f): nent+=1; entrant_keys.add((r["event_date"],r["symbol"]))
    lifecycle_keys=set(); life_rows=0; d1ah=set()
    for p in lifecycle_parts:
        with gzip.open(p,"rt",newline="") as f:
            for r in csv.DictReader(f):
                life_rows+=1; lifecycle_keys.add((r["event_date"],r["symbol"]))
                if r["session"]=="AH" and r["bar_date_et"]<r["event_date"]: d1ah.add((r["event_date"],r["symbol"]))
    missing_life=entrant_keys-lifecycle_keys
    report["checks"]={
        "membership_rows":nmem,"snapshot_count":len(snapshots),"distinct_event_symbols":len(syms),
        "bad_rank_rows":bad_rank,"entrant_rows":nent,"lifecycle_rows":life_rows,
        "entrants_with_any_lifecycle":len(entrant_keys-missing_life),
        "entrants_with_observed_d1_ah_trade_bars":len(d1ah),
        "missing_any_lifecycle_count":len(missing_life),
    }
    if bad_rank or missing_life: report["status"]="FAIL"
    rp=outdir/f"VALIDATION_{year}.json"; atomic_json(rp,report)
    return rp,report


def write_manifest(outdir, year, args, stats, outputs):
    obj={
        "year":year,"created_utc":datetime.now(UTC).isoformat(),
        "method":"dynamic_top_gainers_5m_causal",
        "snapshot_semantics":{
            "PM":"completed 5-minute SIP bar; snapshot t uses information through t",
            "09:30":"exact 09:30 1-minute opening print",
            "RTH":"completed 5-minute SIP bar; snapshot t uses information through t",
        },
        "lifecycle":"previous trading day AH 16:00-19:59 ET through event day AH 19:59 ET",
        "parameters":vars(args),"request_stats":asdict(stats),
        "outputs":[{"path":str(p),"bytes":p.stat().st_size,"sha256":sha256(p)} for p in outputs if p.exists()],
    }
    p=outdir/f"MANIFEST_{year}.json"; atomic_json(p,obj); return p


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--year",type=int,required=True)
    ap.add_argument("--output",required=True)
    ap.add_argument("--top-n",type=int,default=200)
    ap.add_argument("--candidate-pool",type=int,default=1500,
                    help="coarse top-high names retained per 30m PM bucket and RTH day")
    ap.add_argument("--candidate-gain-floor",type=float,default=0.5,
                    help="also retain every name whose coarse high gain reaches this percent")
    ap.add_argument("--workers",type=int,default=8)
    ap.add_argument("--symbol-batch",type=int,default=180)
    ap.add_argument("--rpm",type=int,default=0,help="0=auto-detect; conservative 180 until detected")
    ap.add_argument("--stop-after",choices=["daily","candidates","membership","lifecycle"])
    ap.add_argument("--lifecycle-month",type=int,choices=range(1,13),
                    help="build one lifecycle month for durable checkpointing")
    args=ap.parse_args()

    outdir=Path(args.output); outdir.mkdir(parents=True,exist_ok=True)
    stats=RequestStats(); limiter=AdaptiveRateLimiter(args.rpm); client=AlpacaClient(limiter,stats)
    syms=get_asset_master(client,outdir); cal=get_calendar(client,args.year,outdir)
    print(f"year={args.year} eligible_symbols={len(syms)} trading_days={len(cal)}",flush=True)

    dbp=stage_daily_reference(client,syms,cal,args.year,outdir,args.workers,args.symbol_batch)
    if args.stop_after=="daily": return
    cand=stage_candidate_days(client,syms,cal,args.year,outdir,dbp,args.workers,args.symbol_batch,args.candidate_pool,args.candidate_gain_floor)
    if args.stop_after=="candidates": return
    mem,ent=stage_membership(client,cal,args.year,outdir,cand,args.top_n,args.symbol_batch,args.workers)
    if args.stop_after=="membership": return
    months={args.lifecycle_month} if args.lifecycle_month else None
    life=stage_lifecycle(client,cal,args.year,outdir,ent,args.symbol_batch,args.workers,months)
    if args.stop_after=="lifecycle": return
    rp,report=validate_outputs(args.year,outdir,mem,ent,life,args.top_n)
    outputs=[mem,ent,*life,rp]
    mp=write_manifest(outdir,args.year,args,stats,outputs)
    print(json.dumps({"validation":report,"manifest":str(mp),"request_stats":asdict(stats)},indent=2),flush=True)

if __name__=="__main__":
    main()
