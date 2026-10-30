"""Club Face ID runtime v2 (CPU): SCRFD detector + ArcFace-style ONNX embedder + Gallery + trackers.

v2 adds (all backward compatible with v1 names): multi-face analysis, flip-TTA, hybrid matching, FaceTracker
(temporal fusion over several frames), quality gates, webcam-domain degradation (degrade_crop) for augmentation/calibration.
The training notebook writes THIS file to export/ and imports it, so what is evaluated is what is deployed.
Dependencies: numpy, opencv-python, onnxruntime.
"""
import json
import math
import time
from pathlib import Path

# ---- SCRFD face detector (ONNX Runtime) ----
import cv2
import numpy as np
import onnxruntime as ort

# 5-point ArcFace template for a 112x112 crop (left eye, right eye, nose, left mouth, right mouth)
ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
UNKNOWN = "UNKNOWN"


def _prov_name(p):
    return p[0] if isinstance(p, (tuple, list)) else p


def resolve_providers(spec="cpu", device_id=0):
    """Turn a user choice into an ONNX Runtime provider list (CPU is ALWAYS appended as fallback). Returns (providers, note).
    spec: 'cpu' | 'cuda' | 'directml' | 'auto' (best available: CUDA > DirectML > CPU) | an explicit provider name | list of names."""
    avail = ort.get_available_providers()
    if isinstance(spec, (list, tuple)):
        want = [p for p in spec if _prov_name(p) in avail]
        return (want + ["CPUExecutionProvider"] if "CPUExecutionProvider" not in map(_prov_name, want) else want), "explicit"
    sp = str(spec or "cpu").strip().lower()
    table = {"cuda": "CUDAExecutionProvider", "gpu": "CUDAExecutionProvider", "directml": "DmlExecutionProvider", "dml": "DmlExecutionProvider",
             "cpu": "CPUExecutionProvider"}
    if sp == "auto":
        for name in ("CUDAExecutionProvider", "DmlExecutionProvider"):
            if name in avail:
                sp = {v: k for k, v in table.items()}[name]; break
        else:
            return ["CPUExecutionProvider"], "auto: khong co GPU provider -> CPU"
    name = table.get(sp, spec)
    if name == "CPUExecutionProvider":
        return ["CPUExecutionProvider"], "cpu"
    if name not in avail:
        hint = {"CUDAExecutionProvider": "pip uninstall -y onnxruntime && pip install onnxruntime-gpu   (NVIDIA, CUDA 12 + cuDNN 9)",
                "DmlExecutionProvider": "pip uninstall -y onnxruntime && pip install onnxruntime-directml   (Windows, GPU DirectX 12)"}.get(name, "")
        return ["CPUExecutionProvider"], f"'{spec}' KHONG kha dung (co: {avail}) -> dung CPU. {hint}"
    first = (name, {"device_id": int(device_id)}) if name in ("CUDAExecutionProvider", "DmlExecutionProvider") else name
    return [first, "CPUExecutionProvider"], f"{name} (device {device_id}) + CPU du phong"


_PRELOADED = [False]


def make_session(path, providers=("CPUExecutionProvider",), threads=0):
    so = ort.SessionOptions()
    if threads and threads > 0:
        so.intra_op_num_threads = int(threads)
        so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    so.log_severity_level = 3          # errors only (the detector's declared output shapes are fixed at 640 -> harmless warnings at other sizes)
    try:
        # do not let idle ONNX Runtime threads busy-wait: the PC is also used for office work
        so.add_session_config_entry("session.intra_op.allow_spinning", "0")
    except Exception:
        pass
    avail = ort.get_available_providers()
    prov = [p for p in providers if _prov_name(p) in avail] or ["CPUExecutionProvider"]
    if any(_prov_name(p) == "CUDAExecutionProvider" for p in prov) and not _PRELOADED[0]:
        _PRELOADED[0] = True
        try:                            # onnxruntime-gpu >= 1.21: find CUDA/cuDNN installed by pip (nvidia-*) or by PyTorch
            ort.preload_dlls()
        except Exception:
            pass
    return ort.InferenceSession(str(path), sess_options=so, providers=prov)


def _distance2bbox(points, distance):
    return np.stack([points[:, 0] - distance[:, 0], points[:, 1] - distance[:, 1],
                     points[:, 0] + distance[:, 2], points[:, 1] + distance[:, 3]], axis=-1)


def _distance2kps(points, distance):
    out = []
    for i in range(0, distance.shape[1], 2):
        out.append(points[:, 0] + distance[:, i])
        out.append(points[:, 1] + distance[:, i + 1])
    return np.stack(out, axis=-1)


def _nms(dets, thresh):
    x1, y1, x2, y2, sc = dets[:, 0], dets[:, 1], dets[:, 2], dets[:, 3], dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = sc.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0.0, xx2 - xx1 + 1) * np.maximum(0.0, yy2 - yy1 + 1)
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[np.where(ovr <= thresh)[0] + 1]
    return keep


class SCRFD:
    """SCRFD face detector (ONNX, 9-output variant with 5-point landmarks), e.g. det_500m / det_10g."""

    def __init__(self, path, det_size=(640, 640), det_thresh=0.5, nms_thresh=0.4,
                 providers=("CPUExecutionProvider",), threads=0):
        self.session = make_session(path, providers, threads)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        if len(self.output_names) != 9:
            raise RuntimeError(f"Expected a 9-output SCRFD model with keypoints, got {len(self.output_names)} outputs")
        self.fmc, self.strides, self.num_anchors = 3, (8, 16, 32), 2
        self.det_size, self.det_thresh, self.nms_thresh = tuple(det_size), det_thresh, nms_thresh
        self._anchor_cache = {}

    def _anchors(self, h, w, stride):
        key = (h, w, stride)
        if key not in self._anchor_cache:
            ac = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32)
            ac = (ac * stride).reshape(-1, 2)
            ac = np.stack([ac] * self.num_anchors, axis=1).reshape(-1, 2)
            self._anchor_cache[key] = ac
        return self._anchor_cache[key]

    def detect(self, img_bgr, max_num=0):
        """Returns (dets[n,5] = x1,y1,x2,y2,score ; kps[n,5,2]) in original image coordinates."""
        in_w, in_h = self.det_size
        ih, iw = img_bgr.shape[:2]
        im_ratio, model_ratio = ih / iw, in_h / in_w
        if im_ratio > model_ratio:
            new_h, new_w = in_h, int(in_h / im_ratio)
        else:
            new_w, new_h = in_w, int(in_w * im_ratio)
        det_scale = new_h / ih
        resized = cv2.resize(img_bgr, (new_w, new_h))
        canvas = np.zeros((in_h, in_w, 3), dtype=np.uint8)
        canvas[:new_h, :new_w, :] = resized
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / 128, (in_w, in_h), (127.5, 127.5, 127.5), swapRB=True)
        outs = self.session.run(self.output_names, {self.input_name: blob})
        scores_l, boxes_l, kps_l = [], [], []
        for idx, stride in enumerate(self.strides):
            scores = outs[idx]
            bbox_preds = outs[idx + self.fmc] * stride
            kps_preds = outs[idx + self.fmc * 2] * stride
            h, w = in_h // stride, in_w // stride
            anchors = self._anchors(h, w, stride)
            pos = np.where(scores >= self.det_thresh)[0]
            if pos.size == 0:
                continue
            boxes_l.append(_distance2bbox(anchors, bbox_preds)[pos])
            kps_l.append(_distance2kps(anchors, kps_preds)[pos].reshape(-1, 5, 2))
            scores_l.append(scores[pos])
        if not scores_l:
            return np.zeros((0, 5), np.float32), np.zeros((0, 5, 2), np.float32)
        scores = np.vstack(scores_l).ravel()
        boxes = np.vstack(boxes_l) / det_scale
        kps = np.vstack(kps_l) / det_scale
        order = scores.argsort()[::-1]
        dets = np.hstack([boxes, scores[:, None]]).astype(np.float32)[order]
        kps = kps[order]
        keep = _nms(dets, self.nms_thresh)
        dets, kps = dets[keep], kps[keep]
        if max_num > 0 and len(dets) > max_num:
            dets, kps = dets[:max_num], kps[:max_num]
        return dets, kps.astype(np.float32)


# ---- 5-point alignment, ArcFace embedder, quality measures, FaceEngine (detect -> align -> embed) ----
def umeyama_similarity(src, dst):
    """Least-squares similarity transform (rotation+uniform scale+translation), 2x3 matrix mapping src->dst."""
    src, dst = np.asarray(src, np.float64), np.asarray(dst, np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    cov = dc.T @ sc / len(src)
    U, S, Vt = np.linalg.svd(cov)
    d = np.ones(2)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        d[-1] = -1
    R = U @ np.diag(d) @ Vt
    var_s = (sc ** 2).sum() / len(src)
    scale = (S * d).sum() / var_s
    t = mu_d - scale * R @ mu_s
    return np.hstack([scale * R, t[:, None]]).astype(np.float32)


def align_face(img_bgr, kps5, size=112):
    M = umeyama_similarity(kps5, ARCFACE_DST * (size / 112.0))
    return cv2.warpAffine(img_bgr, M, (size, size), borderValue=0.0)


class ArcFaceEmbedder:
    """ArcFace-style recognition model (ONNX): 112x112 aligned BGR crop -> L2-normalised 512-d embedding."""

    def __init__(self, path, providers=("CPUExecutionProvider",), threads=0):
        self.session = make_session(path, providers, threads)
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]

    def embed(self, aligned_list):
        if len(aligned_list) == 0:
            return np.zeros((0, 512), np.float32)
        blob = cv2.dnn.blobFromImages(aligned_list, 1.0 / 127.5, (112, 112), (127.5, 127.5, 127.5), swapRB=True)
        out = self.session.run(self.output_names, {self.input_name: blob})[0].astype(np.float32)
        return out / np.maximum(np.linalg.norm(out, axis=1, keepdims=True), 1e-12)


def yaw_proxy(kps5):
    """Rough horizontal head-turn proxy from 5 landmarks: nose offset from eye-midpoint / inter-eye distance."""
    le, re, nose = kps5[0], kps5[1], kps5[2]
    ied = max(float(np.linalg.norm(re - le)), 1e-6)
    return float((nose[0] - (le[0] + re[0]) / 2.0) / ied)


def blur_score(img_bgr, bbox):
    x1, y1, x2, y2 = [int(round(v)) for v in bbox]
    h, w = img_bgr.shape[:2]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w, x2), min(h, y2)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return 0.0
    g = cv2.cvtColor(img_bgr[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    g = cv2.resize(g, (112, 112), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def pitch_proxy(kps5):
    """Vertical head-tilt proxy: (nose - eye line) / (mouth - eye line). ~0.5 frontal; smaller = looking up, larger = looking down."""
    le, re, nose, lm, rm = kps5
    ey = (le[1] + re[1]) / 2.0
    my = (lm[1] + rm[1]) / 2.0
    return float((nose[1] - ey) / max(my - ey, 1e-6))


class FaceEngine:
    """Detector + aligner + embedder. v2: multi-face analysis (analyse_all) and optional flip test-time augmentation."""

    def __init__(self, det_path, rec_path, det_size=(640, 640), det_thresh=0.5, providers=("CPUExecutionProvider",),
                 threads=0, min_face_px=40, pad_retry=True, tta_flip=False):
        t0 = time.perf_counter()
        self.det = SCRFD(det_path, det_size, det_thresh, providers=providers, threads=threads)
        self.rec = ArcFaceEmbedder(rec_path, providers, threads)
        self.load_seconds = time.perf_counter() - t0
        self.min_face_px, self.pad_retry, self.tta_flip = min_face_px, pad_retry, bool(tta_flip)

    @property
    def active_providers(self):
        """Providers ONNX Runtime really attached to each model (a requested GPU can silently fall back to CPU)."""
        return dict(detector=self.det.session.get_providers(), recognizer=self.rec.session.get_providers())

    @property
    def on_gpu(self):
        return all(p[0] != "CPUExecutionProvider" for p in self.active_providers.values())

    # ---- helpers ----
    def embed_crops(self, crops, tta=None):
        """List of aligned 112x112 BGR crops -> [n,512] L2-normalised embeddings (flip-TTA averaged when enabled)."""
        tta = self.tta_flip if tta is None else tta
        if len(crops) == 0:
            return np.zeros((0, 512), np.float32)
        if not tta:
            return self.rec.embed(list(crops))
        both = self.rec.embed(list(crops) + [c[:, ::-1].copy() for c in crops])
        n = len(crops)
        m = both[:n] + both[n:]
        return (m / np.maximum(np.linalg.norm(m, axis=1, keepdims=True), 1e-12)).astype(np.float32)

    def _detect(self, img_bgr, timings=None):
        tm = {} if timings is None else timings
        t0 = time.perf_counter()
        dets, kps = self.det.detect(img_bgr)
        used_pad = False
        if len(dets) == 0 and self.pad_retry:
            h, w = img_bgr.shape[:2]
            p = int(0.35 * max(h, w))
            work = cv2.copyMakeBorder(img_bgr, p, p, p, p, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            dets, kps = self.det.detect(work)
            if len(dets):
                dets = dets.copy(); kps = kps.copy()
                dets[:, [0, 2]] -= p; dets[:, [1, 3]] -= p; kps[:, :, 0] -= p; kps[:, :, 1] -= p
                used_pad = True
        tm["detect"] = tm.get("detect", 0.0) + time.perf_counter() - t0
        return dets, kps, used_pad

    def _record(self, img_bgr, det, k, emb, used_pad, n_faces, keep_crop, crop):
        x1, y1, x2, y2, sc = [float(v) for v in det]
        ih, iw = img_bgr.shape[:2]
        side = min(x2 - x1, y2 - y1)
        area = (x2 - x1) * (y2 - y1)
        r = dict(n_faces=int(n_faces), status="ok" if side >= self.min_face_px else "too_small", img_w=iw, img_h=ih, used_pad=used_pad,
                 det_score=sc, bbox=[x1, y1, x2, y2], kps=k.reshape(-1).tolist(), face_area_ratio=float(area / (iw * ih)),
                 face_side=float(side), yaw_proxy=yaw_proxy(k), pitch_proxy=pitch_proxy(k),
                 blur=blur_score(img_bgr, (x1, y1, x2, y2)), emb=emb)
        if keep_crop:
            r["crop"] = crop
        return r

    # ---- single best face (v1 behaviour, unchanged results when tta_flip=False) ----
    def analyse(self, img_bgr, timings=None, keep_crop=False):
        tm = {} if timings is None else timings
        dets, kps, used_pad = self._detect(img_bgr, tm)
        ih, iw = img_bgr.shape[:2]
        res = {"n_faces": int(len(dets)), "status": "no_face", "img_w": iw, "img_h": ih, "used_pad": used_pad}
        if len(dets) == 0:
            return res
        areas = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
        best = int(np.argmax(areas * dets[:, 4]))
        t1 = time.perf_counter()
        aligned = align_face(img_bgr, kps[best])
        tm["align"] = tm.get("align", 0.0) + time.perf_counter() - t1
        t2 = time.perf_counter()
        emb = self.embed_crops([aligned])[0]
        tm["embed"] = tm.get("embed", 0.0) + time.perf_counter() - t2
        res = self._record(img_bgr, dets[best], kps[best], emb, used_pad, len(dets), keep_crop, aligned)
        if len(dets) > 1:
            res["second_face_ratio"] = float(np.sort(areas)[-2] / areas[best])
        return res

    # ---- every face in the frame (attendance kiosk) ----
    def analyse_all(self, img_bgr, max_faces=5, min_side=None, timings=None, keep_crop=False):
        """Returns a list of face records (largest/most confident first), each with an embedding. Faces smaller than
        `min_side` px are skipped BEFORE embedding (saves CPU)."""
        tm = {} if timings is None else timings
        dets, kps, used_pad = self._detect(img_bgr, tm)
        if len(dets) == 0:
            return []
        min_side = self.min_face_px if min_side is None else min_side
        areas = (dets[:, 2] - dets[:, 0]) * (dets[:, 3] - dets[:, 1])
        side = np.minimum(dets[:, 2] - dets[:, 0], dets[:, 3] - dets[:, 1])
        order = [int(i) for i in np.argsort(-(areas * dets[:, 4])) if side[i] >= min_side][:max_faces]
        if not order:
            return []
        t1 = time.perf_counter()
        crops = [align_face(img_bgr, kps[i]) for i in order]
        tm["align"] = tm.get("align", 0.0) + time.perf_counter() - t1
        t2 = time.perf_counter()
        embs = self.embed_crops(crops)
        tm["embed"] = tm.get("embed", 0.0) + time.perf_counter() - t2
        out = []
        biggest = float(areas[order[0]])
        for j, i in enumerate(order):
            r = self._record(img_bgr, dets[i], kps[i], embs[j], used_pad, len(dets), keep_crop, crops[j])
            r["area_rel"] = float(areas[i] / max(biggest, 1e-6))
            out.append(r)
        return out


# ---- Gallery (templates + decision rule), trackers, recognise_image, loader for the exported artifacts ----
def _l2(x):
    x = np.asarray(x, np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), 1e-12)


class Gallery:
    """Member templates + decision rule (cosine similarity, threshold, ambiguity margin).

    methods: centroid | multi_max | multi_topk | hybrid  (hybrid = mean of centroid similarity and best single-template similarity)
    `embeddings`/`labels` = per-sample templates; `centroids` = optional [n_ids, D] (computed from the templates when absent)."""

    METHODS = ("centroid", "multi_max", "multi_topk", "hybrid")

    def __init__(self, ids, embeddings, labels, method="multi_max", threshold=0.4, margin=0.0, topk=2, adapter=None, centroids=None):
        self.ids = list(ids)
        self.emb = np.asarray(embeddings, np.float32).reshape(-1, 512) if len(embeddings) else np.zeros((0, 512), np.float32)
        self.lab = np.asarray(labels, np.int64)
        self.method, self.threshold, self.margin, self.topk = method, float(threshold), float(margin), int(topk)
        self.adapter = None if adapter is None else np.asarray(adapter, np.float32)
        self._idx = [np.where(self.lab == i)[0] for i in range(len(self.ids))]
        if centroids is None:
            centroids = np.zeros((len(self.ids), 512), np.float32)
            for i, ix in enumerate(self._idx):
                if ix.size:
                    centroids[i] = _l2(self.emb[ix].mean(0))
        self.cent = np.asarray(centroids, np.float32).reshape(-1, 512)

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        cfg = json.loads(str(z["config_json"]))
        ad = z["adapter"] if "adapter" in z.files else None
        ce = z["centroids"] if "centroids" in z.files else None
        return cls(json.loads(str(z["ids_json"])), z["embeddings"], z["labels"], cfg["method"], cfg["threshold"],
                   cfg.get("margin", 0.0), cfg.get("topk", 2), ad, ce)

    def save(self, path):
        cfg = dict(method=self.method, threshold=self.threshold, margin=self.margin, topk=self.topk)
        extra = {} if self.adapter is None else {"adapter": self.adapter}
        np.savez_compressed(path, embeddings=self.emb, labels=self.lab, ids_json=json.dumps(self.ids, ensure_ascii=False),
                            config_json=json.dumps(cfg), centroids=self.cent, **extra)

    def transform(self, q):
        if self.adapter is None:
            return q
        x = q @ self.adapter
        return (x / max(float(np.linalg.norm(x)), 1e-12)).astype(np.float32)

    def identity_scores(self, q):
        out = np.full(len(self.ids), -1.0, np.float32)
        if not len(self.ids):
            return out
        sims = self.emb @ q if len(self.emb) else np.zeros(0, np.float32)
        csim = self.cent @ q
        for i, ix in enumerate(self._idx):
            if ix.size == 0:
                continue
            if self.method == "centroid":
                out[i] = csim[i]
                continue
            s = sims[ix]
            if self.method == "multi_topk":
                out[i] = np.sort(s)[::-1][: self.topk].mean()
            elif self.method == "hybrid":
                out[i] = 0.5 * (float(csim[i]) + float(s.max()))
            else:
                out[i] = s.max()
        return out

    def top2(self, q):
        """(id1, s1, id2, s2) ignoring the threshold."""
        sc = self.identity_scores(self.transform(q))
        if not len(sc):
            return None, -1.0, None, -1.0
        order = np.argsort(sc)[::-1]
        i1 = int(order[0]); i2 = int(order[1]) if len(order) > 1 else None
        return self.ids[i1], float(sc[i1]), (self.ids[i2] if i2 is not None else None), (float(sc[i2]) if i2 is not None else -1.0)

    def match(self, q):
        if not len(self.ids):
            return UNKNOWN, -1.0, -1.0, "empty_gallery"
        sc = self.identity_scores(self.transform(q))
        order = np.argsort(sc)[::-1]
        s1 = float(sc[order[0]]); s2 = float(sc[order[1]]) if len(order) > 1 else -1.0
        if s1 < self.threshold:
            return UNKNOWN, s1, s2, "low_score"
        if (s1 - s2) < self.margin:
            return UNKNOWN, s1, s2, "ambiguous"
        return self.ids[int(order[0])], s1, s2, "ok"


class AttendanceTracker:
    """Requires `confirm_frames` consecutive frames with the same identity, then enforces a cooldown per identity."""

    def __init__(self, cooldown_s=300.0, confirm_frames=3, streak_timeout_s=1.5):
        self.cooldown_s, self.confirm_frames, self.streak_timeout_s = cooldown_s, confirm_frames, streak_timeout_s
        self.last_logged, self.streak, self.last_seen = {}, {}, {}

    def update(self, identities, now=None):
        now = time.time() if now is None else now
        seen = set(i for i in identities if i != UNKNOWN)
        for i in list(self.streak):
            if i not in seen and now - self.last_seen.get(i, 0) > self.streak_timeout_s:
                self.streak.pop(i, None)
        events = []
        for i in seen:
            self.streak[i] = self.streak.get(i, 0) + 1
            self.last_seen[i] = now
            if self.streak[i] >= self.confirm_frames and now - self.last_logged.get(i, -1e18) >= self.cooldown_s:
                self.last_logged[i] = now
                events.append(i)
        return events


def recognise_image(engine, gallery, img_bgr, timings=None, min_det_score=0.0):
    r = engine.analyse(img_bgr, timings)
    if r["status"] == "ok" and r.get("det_score", 1.0) < min_det_score:
        r["status"] = "low_det_score"          # same quality gate as the evaluation (CFG["min_det_score"])
    if r["status"] != "ok":
        return {"identity": UNKNOWN, "reason": r["status"], "score": None, "score2": None, "analysis": r}
    t0 = time.perf_counter()
    ident, s1, s2, reason = gallery.match(r["emb"])
    if timings is not None:
        timings["match"] = timings.get("match", 0.0) + time.perf_counter() - t0
    return {"identity": ident, "reason": reason, "score": s1, "score2": s2, "analysis": r}


def load_engine_from_dir(art_dir, threads=0, providers=("CPUExecutionProvider",)):
    art_dir = Path(art_dir)
    cfg = json.loads((art_dir / "model_config.json").read_text(encoding="utf-8"))
    eng = FaceEngine(art_dir / cfg["detector_file"], art_dir / cfg["recognizer_file"], tuple(cfg["det_size"]),
                     cfg["det_thresh"], providers, threads, cfg.get("min_face_px", 40), cfg.get("pad_retry", True),
                     tta_flip=cfg.get("tta_flip", False))
    return eng, Gallery.load(art_dir / "gallery.npz"), cfg

def split_label(label):
    """'Le Van A_202512345_CM' -> ('Le Van A', '202512345', 'CM')"""
    parts = str(label).rsplit("_", 2)
    return (parts[0], parts[1], parts[2]) if len(parts) == 3 else (str(label), "", "")


# ---- quality gate, webcam-domain degradation, multi-frame tracker (v2) ----
def quality_gate(face, min_det_score=0.6, min_face_px=70, min_blur_var=0.0, max_abs_yaw=0.6, min_pitch=0.2, max_pitch=1.0):
    """Returns (ok, reason) for a face record produced by FaceEngine. Poor faces are SKIPPED instead of guessed."""
    if face.get("status") != "ok":
        return False, face.get("status", "no_face")
    if face.get("det_score", 1.0) < min_det_score:
        return False, "low_det_score"
    if face.get("face_side", 1e9) < min_face_px:
        return False, "too_small"
    if min_blur_var > 0 and face.get("blur", 1e9) < min_blur_var:
        return False, "blurry"
    if abs(face.get("yaw_proxy", 0.0)) > max_abs_yaw:
        return False, "head_turned"
    p = face.get("pitch_proxy", 0.5)
    if not (min_pitch <= p <= max_pitch):
        return False, "head_tilted"
    return True, "ok"


def degrade_crop(crop_bgr, rng, level="mixed"):
    """Simulate a webcam capture of an aligned 112x112 BGR crop: landmark jitter, low resolution, blur, exposure, noise, JPEG.
    level: 'mild' | 'hard' | 'mixed' (random per call). Deterministic for a given numpy RandomState."""
    lv = level if level != "mixed" else ("mild" if rng.rand() < 0.5 else "hard")
    img = crop_bgr
    ang, sc = rng.uniform(-4, 4), rng.uniform(0.95, 1.05)
    tx, ty = rng.uniform(-2.5, 2.5), rng.uniform(-2.5, 2.5)
    M = cv2.getRotationMatrix2D((56, 56), ang, sc); M[:, 2] += (tx, ty)
    img = cv2.warpAffine(img, M, (112, 112), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    s = int(rng.uniform(70, 100) if lv == "mild" else rng.uniform(44, 76))      # effective face side in px
    img = cv2.resize(cv2.resize(img, (s, s), interpolation=cv2.INTER_AREA), (112, 112), interpolation=cv2.INTER_LINEAR)
    if rng.rand() < (0.4 if lv == "mild" else 0.7):
        sg = rng.uniform(0.4, 0.9) if lv == "mild" else rng.uniform(0.7, 1.4)
        img = cv2.GaussianBlur(img, (0, 0), sg)
    f = img.astype(np.float32) / 255.0
    f = np.power(np.clip(f, 0, 1), rng.uniform(0.7, 1.4) if lv == "mild" else rng.uniform(0.55, 1.7))
    f = (f - 0.5) * (rng.uniform(0.85, 1.15) if lv == "mild" else rng.uniform(0.65, 1.2)) + 0.5
    f = f + rng.uniform(-0.05, 0.05)
    if rng.rand() < 0.6:
        f = f + rng.normal(0, (2.0 if lv == "mild" else 5.0) / 255.0, f.shape)
    img = (np.clip(f, 0, 1) * 255).astype(np.uint8)
    if rng.rand() < 0.6:
        q = int(rng.uniform(55, 90) if lv == "mild" else rng.uniform(30, 70))
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok:
            img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return img


def _iou(a, b):
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


class FaceTracker:
    """Temporal fusion for attendance: faces are tracked across frames (IoU), their embeddings are averaged
    (quality-weighted) and the *fused* embedding is matched. A member is confirmed only when
      (1) the track has >= confirm_frames good frames, (2) the fused match passes threshold + margin, and
      (3) >= min_vote_ratio of the recent per-frame top-1 votes agree with the fused identity.
    A per-member cooldown then prevents duplicate events. UNKNOWN is never confirmed."""

    def __init__(self, gallery, confirm_frames=3, window=8, min_vote_ratio=0.6, iou_thr=0.25, ttl_s=1.5,
                 cooldown_s=300.0, max_tracks=16):
        self.gallery, self.confirm_frames, self.window = gallery, int(confirm_frames), int(window)
        self.min_vote_ratio, self.iou_thr, self.ttl_s = float(min_vote_ratio), float(iou_thr), float(ttl_s)
        self.cooldown_s, self.max_tracks = float(cooldown_s), int(max_tracks)
        self.tracks, self.last_logged, self._next = {}, {}, 1

    def reset_cooldown(self, ident=None):
        if ident is None:
            self.last_logged.clear()
        else:
            self.last_logged.pop(ident, None)

    def update(self, faces, now=None):
        """faces: list of face records (bbox, emb, det_score, face_side). Returns (events, display).
        event = dict(identity, score, score2, frames, votes, track_id, face_side, bbox)."""
        now = time.time() if now is None else now
        for tid in [t for t, tr in self.tracks.items() if now - tr["last"] > self.ttl_s]:
            self.tracks.pop(tid)
        pairs = []
        for fi, f in enumerate(faces):
            for tid, tr in self.tracks.items():
                v = _iou(f["bbox"], tr["bbox"])
                if v >= self.iou_thr:
                    pairs.append((v, fi, tid))
        pairs.sort(reverse=True)
        used_f, used_t, assign = set(), set(), {}
        for v, fi, tid in pairs:
            if fi in used_f or tid in used_t:
                continue
            used_f.add(fi); used_t.add(tid); assign[fi] = tid
        events, display = [], []
        for fi, f in enumerate(faces):
            tid = assign.get(fi)
            if tid is None:
                if len(self.tracks) >= self.max_tracks:
                    continue
                tid = self._next; self._next += 1
                self.tracks[tid] = dict(embs=[], w=[], top1=[], bbox=f["bbox"], last=now, born=now)
            tr = self.tracks[tid]
            tr["bbox"], tr["last"] = f["bbox"], now
            w = float(f.get("det_score", 1.0)) * min(1.0, float(f.get("face_side", 100.0)) / 100.0)
            i1, s1, _, _ = self.gallery.top2(f["emb"])
            tr["embs"].append(np.asarray(f["emb"], np.float32)); tr["w"].append(max(w, 1e-3)); tr["top1"].append(i1)
            for k in ("embs", "w", "top1"):
                tr[k] = tr[k][-self.window:]
            n = len(tr["embs"])
            fused = _l2((np.stack(tr["embs"]) * np.array(tr["w"])[:, None]).sum(0))
            ident, sc, sc2, reason = self.gallery.match(fused)
            votes = sum(1 for t in tr["top1"] if t == ident) if ident != UNKNOWN else 0
            need = max(self.confirm_frames, int(math.ceil(self.min_vote_ratio * n)))
            confirmed = ident != UNKNOWN and n >= self.confirm_frames and votes >= need
            display.append(dict(track_id=tid, bbox=f["bbox"], identity=ident if confirmed or ident != UNKNOWN else UNKNOWN,
                                score=sc, score2=sc2, reason=reason if ident == UNKNOWN else ("ok" if confirmed else "collecting"),
                                frames=n, votes=votes, confirmed=confirmed))
            if confirmed and now - self.last_logged.get(ident, -1e18) >= self.cooldown_s:
                self.last_logged[ident] = now
                events.append(dict(identity=ident, score=sc, score2=sc2, frames=n, votes=votes, track_id=tid,
                                   face_side=float(f.get("face_side", 0.0)), bbox=list(f["bbox"])))
        return events, display
