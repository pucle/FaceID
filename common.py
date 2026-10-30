"""Shared helpers: configuration, logging, engine/gallery loading, image IO."""
import copy
import hashlib
import json
import logging
import os
import logging.handlers
import sys
import unicodedata
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

DEFAULT_CONFIG = {
    "paths": {"models_dir": "models", "data_dir": "data"},
    "models": {"detector": "det_500m.onnx", "recognizer": "w600k_mbf.onnx", "det_size": [640, 640], "det_thresh": 0.5,
               "threads": 2, "tta_flip": False, "pad_retry": True,
               "provider": "cpu", "gpu_device_id": 0},   # provider: cpu | cuda | directml | auto (xem README muc 2.1)
    # decision rule; overwritten by `manage.py calibrate --apply` or by the notebook bundle (import-seed)
    "matching": {"method": "hybrid", "threshold": 0.55, "margin": 0.0, "topk": 2, "max_templates_per_id": 12},
    # faces below these gates are SKIPPED in live mode (never guessed)
    "live_quality": {"min_det_score": 0.6, "min_face_px": 70, "min_blur_var": 0.0, "max_abs_yaw": 0.6,
                     "min_pitch": 0.2, "max_pitch": 1.0},
    "tracking": {"confirm_frames": 3, "window": 8, "min_vote_ratio": 0.6, "iou_thr": 0.25, "ttl_s": 1.5,
                 "cooldown_s": 300, "fps": 5, "max_faces": 4},
    "enroll": {"min_det_score": 0.7, "min_face_px": 120, "max_second_face_ratio": 0.5, "min_blur_var": 0.0,
               "min_self_cos": 0.35, "near_dup_cos": 0.97, "conflict_cos": 0.5, "augment": True, "aug_per_sample": 3,
               "store_crops": True, "max_side": 1600, "aug_weight": 0.5,
               "pose_yaw_turn": 0.2, "pose_yaw_front": 0.12, "pose_pitch_up": 0.82, "pose_pitch_down": 1.2,
               "frames_per_pose": 6},
    "camera": {"index": 0, "width": 640, "height": 480},
    "service": {"session_id": None, "device_id": "kiosk-1", "pilot_dir": None, "heartbeat_file": "heartbeat.txt",
                "reload_gallery": True, "sync_interval_s": 20, "log_file": "service.log"},
    "portal": {"enabled": False, "base_url": "https://portal.o365hust.club", "auth": {"type": "bearer", "token_env": "PORTAL_TOKEN",
               "header": "Authorization", "scheme": "Bearer"}, "timeout_s": 10, "verify_tls": True, "hmac_secret_env": None,
               "endpoints": {"health": "GET /api/v1/health", "members": "GET /api/v1/members?updated_since={since}&page={page}",
                             "sessions": "GET /api/v1/attendance/sessions?status=open", "attendance": "POST /api/v1/attendance/face-events",
                             "attendance_batch": "POST /api/v1/attendance/face-events/batch",
                             "face_status": "PUT /api/v1/members/{member_id}/face-status"},
               "fields": {"member_id": "mssv", "name": "full_name", "ban": "ban", "status": "status", "portal_id": "id"},
               "csv_fallback_dir": "exports"},
}


def deep_update(base, new):
    for k, v in (new or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


def load_config(path=None):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    p = Path(path) if path else HERE / "config.json"
    if p.exists():
        deep_update(cfg, json.loads(p.read_text(encoding="utf-8")))
    cfg["_path"] = str(p)
    base = p.parent
    for k in ("models_dir", "data_dir"):
        q = Path(cfg["paths"][k])
        cfg["paths"][k] = str(q if q.is_absolute() else (base / q))
    return cfg


def save_config(cfg, path=None):
    p = Path(path or cfg.get("_path") or HERE / "config.json")
    out = {k: v for k, v in cfg.items() if not k.startswith("_")}
    for k in ("models_dir", "data_dir"):          # keep relative paths relative
        try:
            out["paths"][k] = str(Path(out["paths"][k]).relative_to(p.parent))
        except Exception:
            pass
    p.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


def data_dir(cfg):
    d = Path(cfg["paths"]["data_dir"]); d.mkdir(parents=True, exist_ok=True)
    return d


def setup_logging(cfg, name="faceid", console=True, file=None):
    lg = logging.getLogger(name)
    if lg.handlers:
        return lg
    lg.setLevel(logging.INFO); lg.propagate = False
    fmt = logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s", "%Y-%m-%d %H:%M:%S")
    if console:
        sh = logging.StreamHandler(sys.stdout); sh.setFormatter(fmt); lg.addHandler(sh)
    f = file or cfg["service"].get("log_file")
    if f:
        fp = Path(f) if Path(f).is_absolute() else data_dir(cfg) / f
        fh = logging.handlers.RotatingFileHandler(fp, maxBytes=2_000_000, backupCount=5, encoding="utf-8")
        fh.setFormatter(fmt); lg.addHandler(fh)
    return lg


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def load_engine(cfg, threads=None, provider=None, log=None):
    """Returns (engine, shas). Provider: argument > env FACEID_PROVIDER > config models.provider. GPU unavailable -> CPU + warning."""
    from face_id_runtime import FaceEngine, resolve_providers
    m, md = cfg["models"], Path(cfg["paths"]["models_dir"])
    det, rec = md / m["detector"], md / m["recognizer"]
    for p in (det, rec):
        if not p.exists():
            raise FileNotFoundError(f"Thieu model {p}. Chay: python manage.py fetch-models  (hoac copy tu goi notebook)")
    spec = provider or os.environ.get("FACEID_PROVIDER") or m.get("provider", "cpu")
    provs, note = resolve_providers(spec, m.get("gpu_device_id", 0))
    eng = FaceEngine(det, rec, tuple(m["det_size"]), m["det_thresh"], tuple(provs),
                     threads if threads is not None else m["threads"], 40, m.get("pad_retry", True), tta_flip=m.get("tta_flip", False))
    eng.provider_spec, eng.provider_note = spec, note
    want_gpu = str(spec).lower() not in ("cpu",) and "KHONG kha dung" not in note and "khong co GPU" not in note
    if want_gpu and not eng.on_gpu:
        note += " | CANH BAO: ONNX Runtime da quay ve CPU (thieu CUDA/cuDNN? xem README 2.1)"
    eng.provider_note = note
    (log.info if log else print)(f"[engine] provider={spec}: {note} | dang dung: {eng.active_providers}") if str(spec).lower() != "cpu" else None
    return eng, dict(detector_sha256=sha256_file(det), recognizer_sha256=sha256_file(rec))


def read_image(path, max_side=1600):
    """Unicode-safe image read (Windows paths with Vietnamese accents) + bounded downscale."""
    try:
        img = cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)
    except (OSError, ValueError):
        return None
    if img is None:
        return None
    m = max(img.shape[:2])
    if max_side and m > max_side:
        s = max_side / m
        img = cv2.resize(img, (int(round(img.shape[1] * s)), int(round(img.shape[0] * s))), interpolation=cv2.INTER_AREA)
    return img


def strip_accents(s):
    s = unicodedata.normalize("NFD", s)
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return s.replace("đ", "d").replace("Đ", "D")


def nfc(s):
    return unicodedata.normalize("NFC", str(s))
