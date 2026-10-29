"""Portal gia lap de thu tich hop khi portal that chua co API.   python tools/mock_portal.py [--port 8765]
Token: bien moi truong MOCK_TOKEN (mac dinh SECRET). Phia service: set PORTAL_TOKEN=SECRET, portal.enabled=true, portal.base_url=http://127.0.0.1:8765
Hanh vi: /health, /members (phan trang), /attendance/sessions, POST /attendance/face-events (Idempotency-Key -> 409 neu trung;
member 'BAD' -> 422; `--flaky N` tra 503 N request dau de thu retry)."""
import argparse, datetime as dt, http.server, json, os, urllib.parse

STATE = dict(events={}, fail_next=0)
MEMBERS = [dict(id=1, mssv="20250007", full_name="Nguyen Van A", ban="CM", status="active"),
           dict(id=2, mssv="20259001", full_name="Le Thi B", ban="TC", status="active"),
           dict(id=3, mssv="20240001", full_name="Tran C (da nghi)", ban="SK", status="inactive")]
TOKEN = os.environ.get("MOCK_TOKEN", "SECRET")


class H(http.server.BaseHTTPRequestHandler):
    def log_message(self, fmt, *a):
        print("[mock]", self.command, self.path, "->", getattr(self, "_code", "?"))

    def _send(self, code, obj=None):
        self._code = code; b = json.dumps(obj or {}).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def _auth(self):
        if self.headers.get("Authorization") != f"Bearer {TOKEN}":
            self._send(401, {"detail": "bad token"}); return False
        return True

    def do_GET(self):
        if not self._auth(): return
        u = urllib.parse.urlparse(self.path); q = urllib.parse.parse_qs(u.query)
        if u.path.endswith("/health"): return self._send(200, {"ok": True, "time": dt.datetime.now().astimezone().isoformat()})
        if u.path.endswith("/members"): return self._send(200, {"results": MEMBERS if q.get("page", ["1"])[0] == "1" else []})
        if u.path.endswith("/sessions"):
            now = dt.datetime.now().astimezone()
            return self._send(200, {"items": [{"id": "EV-DEMO", "title": "Sinh hoat (demo)", "start": (now - dt.timedelta(hours=1)).isoformat(), "end": (now + dt.timedelta(hours=2)).isoformat()}]})
        self._send(404)

    def do_POST(self):
        if not self._auth(): return
        body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
        if STATE["fail_next"] > 0:
            STATE["fail_next"] -= 1; return self._send(503, {"detail": "maintenance"})
        if body.get("member_id") == "BAD": return self._send(422, {"code": "unknown_member"})
        k = self.headers.get("Idempotency-Key") or body.get("idempotency_key")
        if k in STATE["events"]: return self._send(409, {"detail": "duplicate"})
        STATE["events"][k] = body; print("[mock] DIEM DANH", body["member_id"], body["timestamp"], body["confidence"]); self._send(201, {"id": len(STATE["events"]), "status": "present"})

    def do_PUT(self):
        if not self._auth(): return
        self.rfile.read(int(self.headers.get("Content-Length", 0))); self._send(200, {})


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--port", type=int, default=8765); ap.add_argument("--flaky", type=int, default=0)
    a = ap.parse_args(); STATE["fail_next"] = a.flaky
    print(f"mock portal http://127.0.0.1:{a.port}  token={TOKEN}"); http.server.ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
