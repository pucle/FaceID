"""Client for the club portal (https://portal.o365hust.club). The portal's API is NOT public, so every URL / field name is
configurable in config.json -> "portal" (see README section 'Hop dong API'). Until an API exists, use the CSV fallback.

Error policy for attendance events:
  2xx or 409 (already recorded)  -> synced            5xx / timeout / connection error / 429 -> retry later (backoff)
  400/404/409x/422 (bad event)   -> permanent failure  401/403 -> auth problem (stop syncing, alert), events stay queued
"""
import csv
import datetime as dt
import hashlib
import hmac
import json
import os
import re
import time
from pathlib import Path

try:
    import requests
except ImportError:                                           # pragma: no cover
    requests = None


class PortalAuthError(RuntimeError):
    pass


class PortalError(RuntimeError):
    def __init__(self, msg, status=None, permanent=False):
        super().__init__(msg); self.status, self.permanent = status, permanent


def _split_endpoint(spec):
    m, _, path = spec.strip().partition(" ")
    return m.upper(), path.strip()


def _first(d, *keys, default=None):
    for k in keys:
        if isinstance(d, dict) and d.get(k) not in (None, ""):
            return d[k]
    return default


class PortalClient:
    def __init__(self, pcfg, session=None):
        if requests is None:
            raise RuntimeError("Can `pip install requests`")
        self.cfg = pcfg
        self.base = pcfg["base_url"].rstrip("/")
        self.http = session or requests.Session()
        self.timeout = float(pcfg.get("timeout_s", 10)); self.verify = bool(pcfg.get("verify_tls", True))
        self.fields = pcfg["fields"]

    # ---------- plumbing ----------
    def _headers(self, body=None):
        a = self.cfg.get("auth", {}); h = {"Accept": "application/json", "User-Agent": "club-face-id/2"}
        typ = a.get("type", "bearer")
        if typ != "none":
            tok = os.environ.get(a.get("token_env", "PORTAL_TOKEN"), "")
            if not tok:
                raise PortalAuthError(f"Chua dat bien moi truong {a.get('token_env', 'PORTAL_TOKEN')} (token cua tai khoan dich vu portal)")
            h[a.get("header", "Authorization")] = (f"{a.get('scheme', 'Bearer')} {tok}").strip() if typ == "bearer" else tok
        sec_env = self.cfg.get("hmac_secret_env")
        if sec_env and body is not None and os.environ.get(sec_env):
            ts = str(int(time.time()))
            sig = hmac.new(os.environ[sec_env].encode(), ts.encode() + b"." + body, hashlib.sha256).hexdigest()
            h["X-Timestamp"], h["X-Signature"] = ts, sig
        return h

    def _call(self, name, body=None, **fmt):
        method, path = _split_endpoint(self.cfg["endpoints"][name])
        path = path.format(**{k: (v if v is not None else "") for k, v in fmt.items()}) if fmt else path
        url = path if path.startswith("http") else self.base + path
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        h = self._headers(data)
        if data is not None:
            h["Content-Type"] = "application/json; charset=utf-8"
        if body is not None and isinstance(body, dict) and body.get("idempotency_key"):
            h["Idempotency-Key"] = body["idempotency_key"]
        try:
            r = self.http.request(method, url, data=data, headers=h, timeout=self.timeout, verify=self.verify)
        except requests.RequestException as e:
            raise PortalError(f"network: {type(e).__name__}: {e}", None, False)
        if r.status_code in (401, 403):
            raise PortalAuthError(f"{r.status_code} {name}: token sai/het han/khong du quyen")
        return r

    @staticmethod
    def _items(js):
        if isinstance(js, list):
            return js
        for k in ("results", "items", "data", "members", "sessions"):
            if isinstance(js, dict) and isinstance(js.get(k), list):
                return js[k]
        return []

    # ---------- reads ----------
    def health(self):
        r = self._call("health")
        return r.status_code < 300

    def fetch_members(self, since=None, max_pages=200):
        """Yields normalised dicts {member_id, name, ban, status, portal_id}. Accepts list / {results|items|data:[..]}; pages by {page}."""
        f = self.fields; page = 1; seen = 0
        while page <= max_pages:
            r = self._call("members", since=since or "", page=page)
            if r.status_code >= 400:
                raise PortalError(f"members {r.status_code}: {r.text[:200]}", r.status_code, r.status_code in (400, 404))
            items = self._items(r.json())
            if not items:
                break
            for it in items:
                mid = _first(it, f["member_id"], "mssv", "student_id")
                if not mid:
                    continue
                st = str(_first(it, f["status"], default="active")).lower()
                yield dict(member_id=str(mid), name=_first(it, f["name"], "full_name", "name", default=""), ban=_first(it, f["ban"], "department"),
                           status="inactive" if st in ("inactive", "left", "disabled", "0", "false", "nghi", "da_nghi") else "active",
                           portal_id=str(_first(it, f["portal_id"], "id", default="")) or None)
            seen += len(items)
            if "{page}" not in self.cfg["endpoints"]["members"]:
                break
            page += 1

    def fetch_sessions(self):
        r = self._call("sessions")
        if r.status_code >= 400:
            raise PortalError(f"sessions {r.status_code}: {r.text[:200]}", r.status_code)
        out = []
        for s in self._items(r.json()):
            out.append(dict(session_id=str(_first(s, "id", "session_id", "uuid")), title=_first(s, "title", "name", default=""),
                            start=_first(s, "start", "start_time", "starts_at"), end=_first(s, "end", "end_time", "ends_at"), raw=s))
        return out

    @staticmethod
    def pick_session(sessions, now=None):
        """The open session whose [start,end] contains now (ISO strings); falls back to the only open session."""
        now = now or dt.datetime.now().astimezone()
        def p(x):
            try:
                return dt.datetime.fromisoformat(str(x).replace("Z", "+00:00"))
            except Exception:
                return None
        inside = []
        for s in sessions:
            a, b = p(s["start"]), p(s["end"])
            if a and b and a.tzinfo and a <= now <= b:
                inside.append(s)
        if inside:
            return inside[0]
        return sessions[0] if len(sessions) == 1 else None

    # ---------- writes ----------
    @staticmethod
    def event_payload(ev, model_tag=""):
        return dict(idempotency_key=ev["idem_key"], session_id=ev.get("session_id"), member_id=ev["member_id"], timestamp=ev["ts"],
                    confidence=round(float(ev.get("score") or 0), 4), method="face", device_id=ev.get("device_id"), frames=ev.get("frames"), model=model_tag)

    def post_event(self, ev, model_tag=""):
        """Returns ('ok'|'duplicate'|'retry'|'permanent', detail)."""
        try:
            r = self._call("attendance", body=self.event_payload(ev, model_tag))
        except PortalAuthError:
            raise
        except PortalError as e:
            return "retry", str(e)
        c = r.status_code
        if c < 300:
            return "ok", r.text[:120]
        if c == 409:
            return "duplicate", r.text[:120]
        if c == 429 or c >= 500:
            return "retry", f"{c} {r.text[:120]}"
        return "permanent", f"{c} {r.text[:200]}"

    def push_face_status(self, member_id, status):
        r = self._call("face_status", body=status, member_id=member_id)
        return r.status_code < 300


def sync_events(store, client, model_tag="", batch=50, log=print):
    """Send queued events. Stops at the first auth error / network outage (events stay queued). Returns counts."""
    n = dict(ok=0, duplicate=0, retry=0, permanent=0)
    for ev in store.pending_events(batch):
        try:
            res, detail = client.post_event(ev, model_tag)
        except PortalAuthError as e:
            log(f"[portal] AUTH: {e}"); n["auth_error"] = 1; break
        if res in ("ok", "duplicate"):
            store.mark_synced([ev["event_id"]]); n[res] += 1
        elif res == "retry":
            store.mark_failed([ev["event_id"]], detail); n["retry"] += 1
            break                                           # outage: do not hammer; try again next cycle
        else:
            store.mark_failed([ev["event_id"]], detail, permanent=True); n["permanent"] += 1
            log(f"[portal] su kien {ev['event_id']} ({ev['member_id']}) bi tu choi: {detail}")
    return n


def export_csv(store, out_dir, session_id=None, since_iso=None):
    """Fallback when the portal has no API: CSV (UTF-8 BOM, opens in Excel) of events; import it manually / with the portal's Excel import."""
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    rows = [e for e in store.recent_events(100000) if (session_id is None or e["session_id"] == session_id) and (since_iso is None or e["ts"] >= since_iso)]
    rows.sort(key=lambda e: e["ts"])
    mem = {m["member_id"]: m for m in store.list_members()}
    tag = re.sub(r"[^\w\-]+", "_", str(session_id or "all"))
    p = Path(out_dir) / f"diemdanh_{tag}_{dt.date.today().isoformat()}.csv"
    with open(p, "w", newline="", encoding="utf-8-sig") as fh:
        w = csv.writer(fh); w.writerow(["thoi_gian", "mssv", "ho_ten", "ban", "do_tin_cay", "phien", "thiet_bi", "da_dong_bo"])
        for e in rows:
            m = mem.get(e["member_id"], {})
            w.writerow([e["ts"], e["member_id"], m.get("name", ""), m.get("ban", ""), f"{e['score']:.3f}" if e["score"] is not None else "", e["session_id"] or "", e["device_id"] or "", e["synced"]])
    return p, len(rows)


def sync_members_into_store(store, client, since=None, deactivate_missing=False, log=print):
    """Portal is the source of truth for NAME / BAN / STATUS. Existing members are updated, new ones are created WITHOUT a face
    (they appear in `manage.py list --no-face` until enrolled). Face samples are never touched."""
    seen, added, updated = set(), 0, 0
    for m in client.fetch_members(since):
        seen.add(m["member_id"])
        cur = store.get_member(m["member_id"])
        if cur is None:
            store.add_member(m["member_id"], m["name"] or m["member_id"], m["ban"], m["portal_id"], m["status"]); added += 1
        else:
            ch = {k: m[k] for k in ("name", "ban", "status", "portal_id") if m.get(k) and m[k] != cur.get(k)}
            if ch:
                store.update_member(m["member_id"], **ch); updated += 1
    gone = 0
    if deactivate_missing and since is None:
        for cur in store.list_members("active"):
            if cur["member_id"] not in seen:
                store.update_member(cur["member_id"], status="inactive"); gone += 1
    log(f"[portal] dong bo thanh vien: +{added} moi, {updated} cap nhat, {gone} ngung hoat dong")
    return dict(added=added, updated=updated, deactivated=gone)
