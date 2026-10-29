"""FaceStore: SQLite database of members, face samples (embeddings), attendance events and the sync outbox.

Why a database and not only gallery.npz?  gallery.npz is a *compiled* artefact (centroids / selected templates). To add,
remove or re-weight faces later you need the individual samples. Member info (name, ban, status) lives in `members`
and can be edited without touching any embedding; the gallery is rebuilt from the samples in well under a second.
"""
import datetime as dt
import hashlib
import json
import sqlite3
import time
from pathlib import Path

import cv2
import numpy as np

from face_id_runtime import Gallery, _l2

SCHEMA_VERSION = 1
SCHEMA = """
CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE IF NOT EXISTS members(
  member_id TEXT PRIMARY KEY,           -- MSSV (stable key; also the key used by the portal)
  name TEXT NOT NULL, ban TEXT, portal_id TEXT,
  status TEXT NOT NULL DEFAULT 'active',  -- active | inactive (inactive = not in gallery, data kept)
  note TEXT, created_at TEXT, updated_at TEXT);
CREATE TABLE IF NOT EXISTS samples(
  sample_id INTEGER PRIMARY KEY AUTOINCREMENT,
  member_id TEXT NOT NULL REFERENCES members(member_id) ON DELETE CASCADE ON UPDATE CASCADE,
  emb BLOB NOT NULL,                      -- float32[512], L2-normalised
  kind TEXT NOT NULL DEFAULT 'real',      -- real | aug (simulated webcam) | auto
  source TEXT, pose TEXT, det_score REAL, face_side REAL, blur REAL, yaw REAL, pitch REAL,
  crop_path TEXT, enabled INTEGER NOT NULL DEFAULT 1, created_at TEXT);
CREATE INDEX IF NOT EXISTS ix_samples_member ON samples(member_id);
CREATE TABLE IF NOT EXISTS events(
  event_id INTEGER PRIMARY KEY AUTOINCREMENT,
  idem_key TEXT UNIQUE, ts TEXT NOT NULL, member_id TEXT NOT NULL, score REAL, score2 REAL, frames INTEGER,
  session_id TEXT, device_id TEXT, synced INTEGER NOT NULL DEFAULT 0, attempts INTEGER NOT NULL DEFAULT 0,
  last_error TEXT, synced_at TEXT);
CREATE INDEX IF NOT EXISTS ix_events_pending ON events(synced, event_id);
"""


def now_iso():
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def emb_to_blob(e):
    return np.asarray(e, np.float32).reshape(512).tobytes()


def blob_to_emb(b):
    return np.frombuffer(b, np.float32).copy()


class FaceStore:
    def __init__(self, db_path, crops_dir=None):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.crops_dir = Path(crops_dir) if crops_dir else self.path.parent / "crops"
        self.db = sqlite3.connect(str(self.path), timeout=30, isolation_level=None)   # autocommit; explicit BEGIN where needed
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL"); self.db.execute("PRAGMA foreign_keys=ON")
        self.db.executescript(SCHEMA)
        if self.get_meta("schema_version") is None:
            self.set_meta("schema_version", str(SCHEMA_VERSION))

    def close(self):
        self.db.close()

    # ---------------- meta / model guard ----------------
    def get_meta(self, k, default=None):
        r = self.db.execute("SELECT value FROM meta WHERE key=?", (k,)).fetchone()
        return r["value"] if r else default

    def set_meta(self, k, v):
        self.db.execute("INSERT INTO meta(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value", (k, str(v)))

    def check_models(self, shas, adopt=True):
        """Embeddings are only comparable when produced by the SAME recognizer. Returns (ok, message)."""
        have = self.get_meta("recognizer_sha256")
        if have is None:
            if adopt:
                self.set_meta("recognizer_sha256", shas["recognizer_sha256"]); self.set_meta("detector_sha256", shas["detector_sha256"])
            return True, "model adopted"
        if have != shas["recognizer_sha256"]:
            return False, ("Model nhan dien KHAC voi model da dung de tao embedding trong DB. "
                           "Chay `python manage.py reembed` (can crops) hoac dung dung model cu.")
        return True, "ok"

    def fingerprint(self):
        """Cheap signature of everything that influences the gallery (members + samples); used for hot reload."""
        a = self.db.execute("SELECT COUNT(*), COALESCE(MAX(updated_at),''), COALESCE(SUM(status='active'),0) FROM members").fetchone()
        b = self.db.execute("SELECT COUNT(*), COALESCE(MAX(sample_id),0), COALESCE(SUM(enabled),0) FROM samples").fetchone()
        return tuple(a) + tuple(b)

    # ---------------- members ----------------
    def add_member(self, member_id, name, ban=None, portal_id=None, status="active", note=None, upsert=False):
        member_id = str(member_id).strip()
        if not member_id or not name:
            raise ValueError("member_id (MSSV) va name la bat buoc")
        t = now_iso()
        exists = self.get_member(member_id) is not None
        if exists and not upsert:
            raise ValueError(f"Thanh vien {member_id} da ton tai (dung `update-member` de sua)")
        if exists:
            self.update_member(member_id, name=name, **{k: v for k, v in dict(ban=ban, portal_id=portal_id, note=note).items() if v is not None})
            return member_id
        self.db.execute("INSERT INTO members(member_id,name,ban,portal_id,status,note,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?)",
                        (member_id, name, ban, portal_id, status, note, t, t))
        return member_id

    def get_member(self, member_id):
        r = self.db.execute("SELECT * FROM members WHERE member_id=?", (str(member_id),)).fetchone()
        return dict(r) if r else None

    def update_member(self, member_id, **fields):
        allowed = {"name", "ban", "portal_id", "status", "note"}
        bad = set(fields) - allowed
        if bad:
            raise ValueError(f"Truong khong hop le: {sorted(bad)}")
        if "status" in fields and fields["status"] not in ("active", "inactive"):
            raise ValueError("status phai la active hoac inactive")
        if not fields:
            return
        sets = ", ".join(f"{k}=?" for k in fields)
        cur = self.db.execute(f"UPDATE members SET {sets}, updated_at=? WHERE member_id=?", (*fields.values(), now_iso(), str(member_id)))
        if cur.rowcount == 0:
            raise KeyError(f"Khong co thanh vien {member_id}")

    def rename_member_id(self, old, new):
        """Fix a wrong MSSV: samples follow (ON UPDATE CASCADE); crop folder is renamed too."""
        if self.get_member(new):
            raise ValueError(f"MSSV moi {new} da ton tai")
        self.db.execute("UPDATE members SET member_id=?, updated_at=? WHERE member_id=?", (str(new), now_iso(), str(old)))
        self.db.execute("UPDATE events SET member_id=? WHERE member_id=?", (str(new), str(old)))
        o, n = self.crops_dir / str(old), self.crops_dir / str(new)
        if o.exists():
            o.rename(n)
            self.db.execute("UPDATE samples SET crop_path=replace(crop_path,?,?) WHERE member_id=?", (f"{old}/", f"{new}/", str(new)))

    def delete_member(self, member_id, purge_events=False):
        """Hard delete (right to erasure): member, all embeddings and crop files. Events are kept unless purge_events."""
        member_id = str(member_id)
        n = self.db.execute("SELECT COUNT(*) c FROM samples WHERE member_id=?", (member_id,)).fetchone()["c"]
        cur = self.db.execute("DELETE FROM members WHERE member_id=?", (member_id,))
        if cur.rowcount == 0:
            raise KeyError(f"Khong co thanh vien {member_id}")
        d = self.crops_dir / member_id
        if d.exists():
            for f in d.glob("*"):
                f.unlink()
            d.rmdir()
        if purge_events:
            self.db.execute("DELETE FROM events WHERE member_id=?", (member_id,))
        return n

    def list_members(self, status=None):
        q = ("SELECT m.*, COALESCE(SUM(s.kind='real' AND s.enabled=1),0) AS n_real, COALESCE(SUM(s.kind='aug' AND s.enabled=1),0) AS n_aug, "
             "COALESCE(SUM(s.kind='auto' AND s.enabled=1),0) AS n_auto FROM members m LEFT JOIN samples s ON s.member_id=m.member_id ")
        args = ()
        if status:
            q += "WHERE m.status=? "; args = (status,)
        q += "GROUP BY m.member_id ORDER BY m.member_id"
        return [dict(r) for r in self.db.execute(q, args).fetchall()]

    # ---------------- samples ----------------
    def add_sample(self, member_id, emb, kind="real", source=None, pose=None, quality=None, crop=None):
        quality = quality or {}
        crop_path = None
        sid = self.db.execute(
            "INSERT INTO samples(member_id,emb,kind,source,pose,det_score,face_side,blur,yaw,pitch,crop_path,enabled,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,1,?)",
            (str(member_id), emb_to_blob(emb), kind, source, pose, quality.get("det_score"), quality.get("face_side"), quality.get("blur"),
             quality.get("yaw_proxy"), quality.get("pitch_proxy"), None, now_iso())).lastrowid
        if crop is not None:
            d = self.crops_dir / str(member_id); d.mkdir(parents=True, exist_ok=True)
            p = d / f"{sid}.jpg"
            cv2.imwrite(str(p), crop, [cv2.IMWRITE_JPEG_QUALITY, 95])
            crop_path = f"{member_id}/{sid}.jpg"
            self.db.execute("UPDATE samples SET crop_path=? WHERE sample_id=?", (crop_path, sid))
        return sid

    def samples(self, member_id=None, kinds=None, enabled_only=True, with_emb=True):
        q, args = "SELECT * FROM samples WHERE 1=1", []
        if member_id is not None:
            q += " AND member_id=?"; args.append(str(member_id))
        if kinds:
            q += f" AND kind IN ({','.join('?' * len(kinds))})"; args += list(kinds)
        if enabled_only:
            q += " AND enabled=1"
        q += " ORDER BY sample_id"
        out = []
        for r in self.db.execute(q, args).fetchall():
            d = dict(r)
            d["emb"] = blob_to_emb(d["emb"]) if with_emb else None
            out.append(d)
        return out

    def set_sample_enabled(self, sample_id, enabled):
        self.db.execute("UPDATE samples SET enabled=? WHERE sample_id=?", (1 if enabled else 0, int(sample_id)))

    def delete_samples(self, member_id, kinds=None, sample_ids=None):
        rows = self.samples(member_id, kinds, enabled_only=False, with_emb=False)
        if sample_ids is not None:
            keep = set(int(i) for i in sample_ids); rows = [r for r in rows if r["sample_id"] in keep]
        for r in rows:
            if r["crop_path"]:
                (self.crops_dir / r["crop_path"]).unlink(missing_ok=True)
            self.db.execute("DELETE FROM samples WHERE sample_id=?", (r["sample_id"],))
        return len(rows)

    def replace_embeddings(self, mapping, shas=None):
        """mapping {sample_id: emb}; used by reembed after a model upgrade."""
        self.db.execute("BEGIN")
        try:
            for sid, e in mapping.items():
                self.db.execute("UPDATE samples SET emb=? WHERE sample_id=?", (emb_to_blob(e), int(sid)))
            if shas:
                self.set_meta("recognizer_sha256", shas["recognizer_sha256"]); self.set_meta("detector_sha256", shas["detector_sha256"])
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK"); raise

    # ---------------- gallery ----------------
    @staticmethod
    def _farthest(X, k, first_idx=None):
        if len(X) <= k:
            return list(range(len(X)))
        c = _l2(X.mean(0)); sel = [int(np.argmax(X @ c)) if first_idx is None else first_idx]
        d = 1 - X @ X[sel[0]]
        while len(sel) < k:
            j = int(np.argmax(d)); sel.append(j); d = np.minimum(d, 1 - X @ X[j])
        return sel

    def compile_gallery(self, method="hybrid", threshold=0.55, margin=0.0, topk=2, max_templates=12, aug_weight=0.5, status="active"):
        """Build a Gallery from enabled samples of `status` members. Returns (gallery, info).
        centroid = weighted mean of ALL enabled samples (aug weighted `aug_weight`); templates = up to max_templates diverse samples
        (real samples first, farthest-point selection)."""
        members = {m["member_id"]: m for m in self.list_members(status)}
        ids, E, L, C, skipped = [], [], [], [], []
        by = {}
        for s in self.samples(kinds=None):
            if s["member_id"] in members:
                by.setdefault(s["member_id"], []).append(s)
        for mid in sorted(members):
            ss = by.get(mid, [])
            if not ss:
                skipped.append(mid); continue
            X = np.stack([s["emb"] for s in ss]); w = np.array([aug_weight if s["kind"] == "aug" else 1.0 for s in ss], np.float32)
            cent = _l2((X * w[:, None]).sum(0))
            real = [i for i, s in enumerate(ss) if s["kind"] != "aug"] or list(range(len(ss)))
            aug = [i for i in range(len(ss)) if i not in set(real)]
            if len(real) >= max_templates:
                pick = [real[i] for i in self._farthest(X[real], max_templates)]
            else:
                rest = max_templates - len(real)
                pick = list(real) + ([aug[i] for i in self._farthest(X[aug], rest)] if aug and rest > 0 else [])
            k = len(ids); ids.append(mid); C.append(cent)
            for i in pick:
                E.append(X[i]); L.append(k)
        if not ids:
            return Gallery([], np.zeros((0, 512), np.float32), np.zeros(0, np.int64), method, threshold, margin, topk), dict(members=0, templates=0, skipped=skipped)
        g = Gallery(ids, np.stack(E), np.array(L), method, threshold, margin, topk, centroids=np.stack(C))
        return g, dict(members=len(ids), templates=len(E), skipped_no_samples=skipped)

    def save_gallery(self, path, **kw):
        g, info = self.compile_gallery(**kw)
        tmp = Path(str(path) + ".tmp.npz")
        g.save(tmp)
        tmp.replace(path)                                   # atomic: the service never sees a half-written file
        return g, info

    # ---------------- attendance events / outbox ----------------
    def add_event(self, member_id, score, score2=None, frames=None, session_id=None, device_id=None, ts=None, idem_key=None):
        ts = ts or now_iso()
        key = idem_key or hashlib.sha1(f"{session_id}|{member_id}|{device_id}|{ts}".encode()).hexdigest()[:32]
        try:
            return self.db.execute("INSERT INTO events(idem_key,ts,member_id,score,score2,frames,session_id,device_id) VALUES(?,?,?,?,?,?,?,?)",
                                   (key, ts, str(member_id), score, score2, frames, session_id, device_id)).lastrowid
        except sqlite3.IntegrityError:
            return None

    def pending_events(self, limit=100, max_attempts=50):
        return [dict(r) for r in self.db.execute("SELECT * FROM events WHERE synced=0 AND attempts<? ORDER BY event_id LIMIT ?", (max_attempts, limit))]

    def mark_synced(self, event_ids):
        for i in event_ids:
            self.db.execute("UPDATE events SET synced=1, synced_at=?, last_error=NULL WHERE event_id=?", (now_iso(), i))

    def mark_failed(self, event_ids, err, permanent=False):
        for i in event_ids:
            self.db.execute("UPDATE events SET attempts=attempts+?, last_error=? WHERE event_id=?", (999 if permanent else 1, str(err)[:300], i))

    def recent_events(self, n=20):
        return [dict(r) for r in self.db.execute("SELECT * FROM events ORDER BY event_id DESC LIMIT ?", (n,))]

    # ---------------- backup ----------------
    def backup(self, dest):
        dest = Path(dest); dest.parent.mkdir(parents=True, exist_ok=True)
        out = sqlite3.connect(str(dest)); self.db.backup(out); out.close()
        return dest
