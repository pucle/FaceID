"""Headless attendance service: camera -> detect all faces -> quality gate -> track + fuse frames -> match -> event -> outbox -> portal.

  python attendance_service.py                       # camera 0, config.json
  python attendance_service.py --show                # preview window (q = quit)
  python attendance_service.py --session-id EV123    # force a session (otherwise taken from the portal / left empty)
  python attendance_service.py --source DIR_OR_VIDEO # replay images/video instead of a camera (testing)
"""
import argparse
import csv
import datetime as dt
import os
import signal
import sys
import threading
import time
from pathlib import Path

import cv2
import numpy as np

from common import data_dir, load_config, load_engine, read_image, setup_logging, strip_accents
from facestore import FaceStore
from face_id_runtime import FaceTracker, UNKNOWN, quality_gate


class FrameSource:
    """camera index | video file | folder of images (each image repeated `repeat` times to emulate consecutive frames)."""

    def __init__(self, src, cam_cfg, repeat=6):
        self.kind, self.items, self.cap, self.i, self.repeat = None, [], None, 0, repeat
        if src is None or str(src).isdigit():
            self.kind = "camera"; self.cap = self._open(int(src) if src is not None else cam_cfg["index"], cam_cfg)
        elif Path(src).is_dir():
            self.kind = "images"
            self.items = [read_image(p, 1280) for p in sorted(q for q in Path(src).rglob("*") if q.suffix.lower() in (".jpg", ".jpeg", ".png", ".bmp"))]
            self.items = [x for x in self.items if x is not None]
        else:
            self.kind = "video"; self.cap = cv2.VideoCapture(str(src))
        if self.kind != "images" and (self.cap is None or not self.cap.isOpened()):
            raise RuntimeError(f"Khong mo duoc nguon hinh: {src if src is not None else 'camera ' + str(cam_cfg['index'])}. "
                               "Dong ung dung khac dang dung camera / thu --camera 1 / kiem tra quyen camera Windows.")

    @staticmethod
    def _open(idx, c):
        backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_ANY] if sys.platform.startswith("win") else [cv2.CAP_ANY]
        for be in backends:
            cap = cv2.VideoCapture(idx, be)
            if cap.isOpened():
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, c["width"]); cap.set(cv2.CAP_PROP_FRAME_HEIGHT, c["height"])
                ok, _ = cap.read()
                if ok:
                    return cap
            cap.release()
        return None

    def read(self):
        if self.kind == "images":
            if self.i >= len(self.items) * self.repeat:
                return False, None
            fr = self.items[self.i // self.repeat]; self.i += 1
            return True, fr.copy()
        return self.cap.read()

    def grab(self):
        if self.cap is not None:
            self.cap.grab()

    def close(self):
        if self.cap is not None:
            self.cap.release()


class Service:
    def __init__(self, cfg, args, log):
        self.cfg, self.args, self.log = cfg, args, log
        self.store = FaceStore(data_dir(cfg) / "faceid.db", data_dir(cfg) / "crops")
        self.engine, shas = load_engine(cfg)
        ok, msg = self.store.check_models(shas)
        if not ok:
            raise SystemExit(msg)
        self.mcfg = cfg["matching"]
        self.members = {}
        self.gallery = None
        self._dv = None
        self.stop = threading.Event()
        self.session_id = args.session_id or cfg["service"].get("session_id")
        self.stats = dict(frames=0, faces=0, skipped={}, events=0, started=time.time(), lat=[])
        self.pilot = Path(cfg["service"]["pilot_dir"]) if cfg["service"].get("pilot_dir") else None
        self.client = None
        if cfg["portal"].get("enabled") and not args.offline:
            from portal_client import PortalClient
            self.client = PortalClient(cfg["portal"])
        self.reload_gallery(force=True)
        t = cfg["tracking"]
        self.tracker = FaceTracker(self.gallery, t["confirm_frames"], t["window"], t["min_vote_ratio"], t["iou_thr"], t["ttl_s"], t["cooldown_s"])
        self.model_tag = f"{self.mcfg['method']}@{self.mcfg['threshold']:.2f}"
        self.unknown_saved = {}

    # ---- gallery hot reload: picks up enrolments done by `manage.py` while the service is running ----
    def reload_gallery(self, force=False):
        dv = self.store.db.execute("PRAGMA data_version").fetchone()[0]
        if not force and dv == self._dv:
            return
        self._dv = dv
        fp = self.store.fingerprint()
        if not force and fp == getattr(self, "_fp", None):      # commits from the sync thread (events) do not change the gallery
            return
        self._fp = fp
        m = self.mcfg
        g, info = self.store.compile_gallery(m["method"], m["threshold"], m["margin"], m["topk"], m["max_templates_per_id"], self.cfg["enroll"]["aug_weight"])
        self.members = {x["member_id"]: x for x in self.store.list_members()}
        if getattr(self, "tracker", None) is not None:
            self.tracker.gallery = g
        self.gallery = g
        self.log.info(f"gallery: {info['members']} thanh vien, {info['templates']} template, method={m['method']} tau={m['threshold']:.2f} margin={m['margin']:.2f}"
                      + (f" | khong co anh: {len(info['skipped_no_samples'])}" if info.get("skipped_no_samples") else ""))

    def background(self):
        """Portal housekeeping off the camera thread: session refresh, outbox sync, heartbeat."""
        from portal_client import PortalAuthError, PortalError, sync_events
        store = FaceStore(data_dir(self.cfg) / "faceid.db", data_dir(self.cfg) / "crops")      # own connection: sqlite objects are per-thread
        interval = float(self.cfg["service"]["sync_interval_s"]); hb = data_dir(self.cfg) / self.cfg["service"]["heartbeat_file"]
        last_sess = 0.0
        while not self.stop.is_set():
            try:
                if self.client:
                    if not self.args.session_id and not self.cfg["service"].get("session_id") and time.time() - last_sess > 300:
                        last_sess = time.time()
                        s = self.client.pick_session(self.client.fetch_sessions())
                        if s and s["session_id"] != self.session_id:
                            self.session_id = s["session_id"]; self.log.info(f"[portal] phien hien tai: {s['session_id']} {s['title']}")
                    n = sync_events(store, self.client, self.model_tag, log=self.log.warning)
                    if any(n.get(k) for k in ("ok", "retry", "permanent", "auth_error")):
                        self.log.info(f"[portal] dong bo: {n}")
            except PortalAuthError as e:
                self.log.error(f"[portal] AUTH: {e}")
            except (PortalError, Exception) as e:          # never kill the service because of the network
                self.log.warning(f"[portal] loi tam thoi: {e}")
            try:
                st = self.stats; up = time.time() - st["started"]
                hb.write_text(f"{dt.datetime.now().astimezone().isoformat(timespec='seconds')} frames={st['frames']} events={st['events']} "
                              f"fps={st['frames'] / max(up, 1):.1f} lat_ms={np.mean(st['lat'][-50:]) if st['lat'] else 0:.0f} session={self.session_id}\n", encoding="utf-8")
            except Exception:
                pass
            self.stop.wait(interval)
        store.close()

    def save_pilot(self, frame, bbox, folder, tag):
        if not self.pilot:
            return
        x1, y1, x2, y2 = [int(v) for v in bbox]; h, w = frame.shape[:2]; pw, ph = int(0.4 * (x2 - x1)), int(0.4 * (y2 - y1))
        crop = frame[max(0, y1 - ph):min(h, y2 + ph), max(0, x1 - pw):min(w, x2 + pw)]
        d = self.pilot / folder; d.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(d / f"{tag}.jpg"), crop, [cv2.IMWRITE_JPEG_QUALITY, 92])

    def process(self, frame, now=None):
        """One frame. Returns (events, display)."""
        t0 = time.perf_counter()
        t = self.cfg["tracking"]; q = self.cfg["live_quality"]
        faces = self.engine.analyse_all(frame, max_faces=t["max_faces"], min_side=min(q["min_face_px"], 60))
        good = []
        for f in faces:
            ok, why = quality_gate(f, **q)
            if ok:
                good.append(f)
            else:
                self.stats["skipped"][why] = self.stats["skipped"].get(why, 0) + 1
        self.stats["faces"] += len(good); self.stats["frames"] += 1
        events, display = self.tracker.update(good, now)
        self.stats["lat"].append((time.perf_counter() - t0) * 1000); self.stats["lat"] = self.stats["lat"][-200:]
        for ev in events:
            m = self.members.get(ev["identity"], {})
            ts = dt.datetime.now().astimezone().isoformat(timespec="seconds")
            eid = self.store.add_event(ev["identity"], ev["score"], ev["score2"], ev["frames"], self.session_id, self.cfg["service"]["device_id"], ts)
            self.stats["events"] += 1
            self.log.info(f"DIEM DANH: {m.get('name', ev['identity'])} ({ev['identity']}, {m.get('ban', '')}) score={ev['score']:.3f} 2nd={ev['score2']:.3f} "
                          f"frames={ev['frames']} votes={ev['votes']} face={ev['face_side']:.0f}px session={self.session_id} event#{eid}")
            self.save_pilot(frame, ev["bbox"], ev["identity"], dt.datetime.now().strftime("%Y%m%d_%H%M%S"))
        for d in display:                                   # keep a few stranger crops for pilot calibration (1 per track)
            if self.pilot and d["identity"] == UNKNOWN and d["frames"] >= t["confirm_frames"] and d["track_id"] not in self.unknown_saved:
                self.unknown_saved[d["track_id"]] = 1
                self.save_pilot(frame, d["bbox"], "UNKNOWN", f"{dt.datetime.now().strftime('%Y%m%d_%H%M%S')}_t{d['track_id']}_{d['score']:.2f}")
        return events, display

    def draw(self, frame, display):
        for d in display:
            x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
            col = (60, 200, 60) if d["confirmed"] else ((0, 200, 230) if d["identity"] != UNKNOWN else (60, 60, 230))
            cv2.rectangle(frame, (x1, y1), (x2, y2), col, 2)
            nm = self.members.get(d["identity"], {}).get("name", d["identity"]) if d["identity"] != UNKNOWN else "UNKNOWN"
            cv2.putText(frame, f"{strip_accents(str(nm))[:22]} {d['score']:.2f} [{d['frames']}f]", (x1, max(14, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        return frame

    def run(self, source, show=False, max_frames=None):
        src = FrameSource(source if source is not None else self.args.camera, self.cfg["camera"])
        th = threading.Thread(target=self.background, daemon=True); th.start()
        interval = 1.0 / max(self.cfg["tracking"]["fps"], 0.5); last = 0.0; fails = 0; reload_t = 0.0
        self.log.info(f"service chay: nguon={src.kind} fps_xu_ly={self.cfg['tracking']['fps']} threads={self.cfg['models']['threads']} "
                      f"session={self.session_id} portal={'bat' if self.client else 'tat (chi luu DB/CSV)'}")
        try:
            while not self.stop.is_set():
                now = time.time()
                if src.kind == "camera" and now - last < interval and not show:
                    src.grab(); time.sleep(0.005); continue
                ok, frame = src.read()
                if not ok:
                    if src.kind in ("images", "video"):
                        break
                    fails += 1
                    if fails > 100:
                        raise RuntimeError("Camera ngung tra khung hinh")
                    time.sleep(0.05); continue
                fails = 0
                if src.kind != "camera" or now - last >= interval:
                    last = now
                    if self.cfg["service"]["reload_gallery"] and now - reload_t > 3:
                        reload_t = now; self.reload_gallery()
                    sim_now = now if src.kind == "camera" else self.stats["frames"] * interval + 1e6
                    _, display = self.process(frame, sim_now)
                    if show:
                        cv2.imshow("Club Face ID", self.draw(frame, display))
                        if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                            break
                if max_frames and self.stats["frames"] >= max_frames:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self.stop.set(); src.close(); cv2.destroyAllWindows()
            th.join(timeout=3)
            from portal_client import sync_events
            if self.client:
                try:
                    sync_events(self.store, self.client, self.model_tag, log=self.log.warning)
                except Exception:
                    pass
            st = self.stats
            self.log.info(f"ket thuc: {st['frames']} khung, {st['faces']} mat dat chat luong, {st['events']} diem danh, bo qua theo ly do {st['skipped']}, "
                          f"tre TB {np.mean(st['lat']) if st['lat'] else 0:.0f} ms/khung")
        return self.stats


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None); ap.add_argument("--camera", type=int, default=None)
    ap.add_argument("--source", default=None, help="thu muc anh / file video (kiem thu) thay cho camera")
    ap.add_argument("--session-id", default=None); ap.add_argument("--show", action="store_true")
    ap.add_argument("--offline", action="store_true", help="khong goi portal (chi ghi DB)"); ap.add_argument("--max-frames", type=int, default=None)
    ap.add_argument("--pilot-dir", default=None, help="luu anh khuon mat xac nhan/UNKNOWN de hieu chinh bang `manage.py calibrate probes`")
    a = ap.parse_args(argv)
    cfg = load_config(a.config)
    if a.camera is not None:
        cfg["camera"]["index"] = a.camera
    if a.pilot_dir:
        cfg["service"]["pilot_dir"] = a.pilot_dir
    log = setup_logging(cfg)
    svc = Service(cfg, a, log)
    signal.signal(signal.SIGTERM, lambda *_: svc.stop.set())
    svc.run(a.source, a.show, a.max_frames)


if __name__ == "__main__":
    main()
