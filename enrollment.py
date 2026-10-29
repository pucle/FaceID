"""Enrollment logic: quality-gated, self-consistent, conflict-checked, augmented. Used by manage.py (photos / folder / webcam / seed)."""
import re
import time
from pathlib import Path

import cv2
import numpy as np

from common import nfc, read_image
from face_id_runtime import UNKNOWN, _l2, degrade_crop

MSSV_RE = re.compile(r"(?<!\d)(\d{8,10})(?!\d)")
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".jfif", ".tif", ".tiff"}


def parse_label(text):
    """'Bui Huu Tung_202514742_CM' | 'La Huy Gia Huy - 202412885' | '202514742' -> (name or None, mssv or None, ban or None)"""
    t = nfc(text).strip()
    m = MSSV_RE.search(t)
    if not m:
        return t or None, None, None
    name = re.sub(r"[_\-\.]+", " ", t[:m.start()]).strip()
    name = " ".join(w[:1].upper() + w[1:].lower() for w in name.split())
    tail = re.search(r"(?<![A-Za-z])(CM|TC|SK|TT)(?![A-Za-z])", t[m.end():], re.I)
    return (name or None), m.group(1), (tail.group(1).upper() if tail else None)


def list_images(folder):
    return sorted(p for p in Path(folder).rglob("*") if p.suffix.lower() in IMG_EXTS)


def screen_face(face, ecfg):
    """Reject reasons for ONE enrollment face record (None = acceptable)."""
    if face.get("status") == "no_face":
        return "no_face"
    if face.get("face_side", 0) < ecfg["min_face_px"]:
        return f"face_too_small({face.get('face_side', 0):.0f}px<{ecfg['min_face_px']})"
    if face.get("det_score", 0) < ecfg["min_det_score"]:
        return f"low_det_score({face.get('det_score', 0):.2f})"
    if face.get("second_face_ratio", 0) >= ecfg["max_second_face_ratio"]:
        return "multiple_faces(khong ro mat nao la cua thanh vien)"
    if ecfg.get("min_blur_var", 0) > 0 and face.get("blur", 1e9) < ecfg["min_blur_var"]:
        return "blurry"
    return None


def _consistency_filter(new_embs, ref_embs, thr):
    """Drop candidates that disagree with the member's own other photos (wrong person / bystander).
    Returns (keep_mask, cos_to_others). Needs >= 2 other embeddings; otherwise everything is kept (cannot judge)."""
    keep = np.ones(len(new_embs), bool)
    cos = np.full(len(new_embs), np.nan, np.float32)
    for _ in range(len(new_embs)):
        pool = [e for e in ref_embs] + [new_embs[i] for i in range(len(new_embs)) if keep[i]]
        if len(pool) < 3:
            break
        worst, worst_c = None, 9.0
        for i in np.where(keep)[0]:
            others = [e for j, e in enumerate(ref_embs)] + [new_embs[j] for j in np.where(keep)[0] if j != i]
            if len(others) < 2:
                continue
            c = float(new_embs[i] @ _l2(np.mean(others, 0)))
            cos[i] = c
            if c < worst_c:
                worst, worst_c = i, c
        if worst is not None and worst_c < thr:
            keep[worst] = False
        else:
            break
    return keep, cos


def make_aug_embeddings(engine, crop, n, rng, level="mixed"):
    return engine.embed_crops([degrade_crop(crop, rng, level) for _ in range(n)], tta=engine.tta_flip) if n > 0 else np.zeros((0, 512), np.float32)


def enroll_images(store, engine, cfg, member_id, images, source="photo", pose=None, augment=None, force=False, dry_run=False, log=print, seed=0):
    """images: list of (name, bgr_image). Returns a report dict. Never raises on a bad photo."""
    ecfg = cfg["enroll"]; member_id = str(member_id)
    if store.get_member(member_id) is None:
        raise KeyError(f"Chua co thanh vien {member_id}. Tao truoc bang `manage.py add-member`.")
    augment = ecfg["augment"] if augment is None else augment
    rep = dict(member_id=member_id, accepted=[], rejected=[], warnings=[])
    cands = []
    for name, img in images:
        if img is None:
            rep["rejected"].append((name, "khong_doc_duoc_anh")); continue
        f = engine.analyse(img, keep_crop=True)
        why = screen_face(f, ecfg)
        if why:
            rep["rejected"].append((name, why)); continue
        cands.append(dict(name=name, face=f))
    existing = store.samples(member_id, kinds=["real", "auto"])
    ex_embs = [s["emb"] for s in existing]
    # near-duplicates (same frame twice) are useless: keep the first
    kept, kept_embs = [], list(ex_embs)
    for c in cands:
        e = c["face"]["emb"]
        if kept_embs and float(np.max(np.stack(kept_embs) @ e)) >= ecfg["near_dup_cos"]:
            rep["rejected"].append((c["name"], "trung_anh(giong anh da co)")); continue
        kept.append(c); kept_embs.append(e)
    cands = kept
    # self-consistency with the member's other photos
    if cands:
        keep, cos = _consistency_filter([c["face"]["emb"] for c in cands], ex_embs, ecfg["min_self_cos"])
        for c, k, v in zip(cands, keep, cos):
            c["self_cos"] = None if np.isnan(v) else float(v)
        for c in [c for c, k in zip(cands, keep) if not k]:
            rep["rejected"].append((c["name"], f"khong_khop_voi_cac_anh_khac(cos={c['self_cos']:.2f}<{ecfg['min_self_cos']}): co the sai nguoi/co nguoi khac trong anh"))
        cands = [c for c, k in zip(cands, keep) if k]
    # conflict with ANOTHER member
    if cands and not force:
        g, _ = store.compile_gallery(method="multi_max", max_templates=50)
        keep2 = []
        for c in cands:
            if len(g.ids):
                sc = g.identity_scores(c["face"]["emb"])
                order = np.argsort(sc)[::-1]
                top = [i for i in order[:3] if g.ids[int(i)] != member_id]
                if top and float(sc[top[0]]) >= ecfg["conflict_cos"]:
                    o = g.ids[int(top[0])]
                    rep["rejected"].append((c["name"], f"giong_thanh_vien_khac({o} cos={float(sc[top[0]]):.2f}): trung nguoi hoac gan nhan sai; dung --force neu chac chan"))
                    continue
            keep2.append(c)
        cands = keep2
    rng = np.random.RandomState(seed)
    for c in cands:
        f = c["face"]
        rep["accepted"].append((c["name"], f"side={f['face_side']:.0f}px det={f['det_score']:.2f} yaw={f['yaw_proxy']:+.2f}"))
        if dry_run:
            continue
        crop = f["crop"] if ecfg["store_crops"] else None
        store.add_sample(member_id, f["emb"], "real", f"{source}:{c['name']}"[:200], pose, f, crop)
        if augment:
            for e in make_aug_embeddings(engine, f["crop"], ecfg["aug_per_sample"], rng):
                store.add_sample(member_id, e, "aug", f"aug:{c['name']}"[:200], pose, f, None)
    n_real = len(store.samples(member_id, kinds=["real", "auto"]))
    if n_real < 5:
        rep["warnings"].append(f"{member_id} moi co {n_real} anh that: nen co >= 8 anh (thang, +-45 do, ngua, cui) de on dinh")
    return rep


def enroll_folder(store, engine, cfg, folder, create_missing=True, **kw):
    """folder/<label>/<images>, label like 'Ho Ten_MSSV_Ban' or just MSSV. Creates members when missing (name from folder)."""
    reports = []
    for sub in sorted(p for p in Path(folder).iterdir() if p.is_dir()):
        name, mssv, ban = parse_label(sub.name)
        if not mssv:
            reports.append(dict(member_id=sub.name, accepted=[], rejected=[(sub.name, "ten_thu_muc_khong_co_MSSV")], warnings=[])); continue
        if store.get_member(mssv) is None:
            if not create_missing or not name:
                reports.append(dict(member_id=mssv, accepted=[], rejected=[(sub.name, "chua_co_thanh_vien_va_thieu_ten")], warnings=[])); continue
            store.add_member(mssv, name, ban)
        imgs = [(p.name, read_image(p, cfg["enroll"]["max_side"])) for p in list_images(sub)]
        reports.append(enroll_images(store, engine, cfg, mssv, imgs, source="folder", **kw))
    return reports


# ---------------- webcam capture (pose guided) ----------------
POSES = [("thang", "Nhin THANG vao camera"), ("quay1", "Quay dau sang MOT ben (~45 do)"), ("quay2", "Quay sang BEN KIA (~45 do)"),
         ("ngua", "Ngua dau nhe"), ("cui", "Cui dau nhe")]


def pose_matches(name, yaw, pitch, ecfg, ref_pitch=0.55, first_turn_sign=None):
    """True when the head pose (proxies from 5 landmarks, NOT degrees) fits the requested pose."""
    if name == "thang":
        return abs(yaw) <= ecfg["pose_yaw_front"] and abs(pitch - ref_pitch) <= 0.12
    if name == "quay1":
        return abs(yaw) >= ecfg["pose_yaw_turn"] and abs(yaw) <= 0.65
    if name == "quay2":
        return abs(yaw) >= ecfg["pose_yaw_turn"] and abs(yaw) <= 0.65 and (first_turn_sign is None or np.sign(yaw) == -first_turn_sign)
    if name == "ngua":
        return abs(yaw) <= 0.2 and pitch <= ref_pitch * ecfg["pose_pitch_up"]
    if name == "cui":
        return abs(yaw) <= 0.2 and pitch >= ref_pitch * ecfg["pose_pitch_down"]
    return False


def capture_frames(engine, cfg, frame_iter, log=print, show=None, max_seconds_per_pose=25.0, clock=time.time):
    """Drives the pose script. frame_iter yields BGR frames (camera or test images). show(frame, text, color) optional UI hook.
    Returns list of (name, frame, pose). Frames are processed at <= ~5 fps and must differ from the previous capture."""
    ecfg = cfg["enroll"]
    out, ref_pitch, first_sign = [], 0.55, None
    last_emb = {}
    frames = iter(frame_iter)
    for pose, prompt in POSES:
        got, t_start, t_last = 0, clock(), 0.0
        while got < ecfg["frames_per_pose"]:
            if clock() - t_start > max_seconds_per_pose:
                log(f"  [bo qua] tu the '{pose}' het thoi gian ({got}/{ecfg['frames_per_pose']})"); break
            try:
                frame = next(frames)
            except StopIteration:
                return out
            if clock() - t_last < 0.2:
                continue
            t_last = clock()
            f = engine.analyse(frame)
            msg, col = prompt, (0, 200, 230)
            if f.get("status") == "ok" and screen_face(f, dict(ecfg, min_face_px=ecfg["min_face_px"] * 0.8)) is None:
                if pose == "thang" and got == 0:
                    ref_pitch = f["pitch_proxy"]
                if pose_matches(pose, f["yaw_proxy"], f["pitch_proxy"], ecfg, ref_pitch, first_sign):
                    prev = last_emb.get(pose)
                    if prev is None or float(prev @ f["emb"]) < 0.97:
                        last_emb[pose] = f["emb"]; got += 1
                        out.append((f"cam_{pose}_{got}", frame.copy(), pose))
                        if pose == "quay1" and first_sign is None:
                            first_sign = float(np.sign(f["yaw_proxy"]))
                        msg, col = f"{prompt}  OK {got}/{ecfg['frames_per_pose']}", (60, 200, 60)
                else:
                    msg = f"{prompt}  (yaw={f['yaw_proxy']:+.2f} pitch={f['pitch_proxy']:.2f})"
            else:
                msg, col = "Dua mat vao gan hon / du sang", (60, 60, 230)
            if show:
                if show(frame, msg, col):
                    return out
        log(f"  tu the {pose}: {got} khung")
    return out
