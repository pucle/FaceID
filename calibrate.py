"""Calibration / evaluation tools.

loo     : leave-one-photo-out on the samples stored in the database (+ optional webcam-degraded queries built from stored crops).
          Impostors = leave-one-identity-out (a member's photo scored against a gallery that does not contain him/her).
probes  : labelled photos taken with the REAL camera (folder per member; any other folder = stranger) -> the number that matters.
Both end in `choose` which picks (method, threshold, margin) maximising correct-accept subject to FAR <= target.
"""
import math
from pathlib import Path

import numpy as np

from common import nfc, read_image
from enrollment import list_images, parse_label
from face_id_runtime import Gallery, UNKNOWN, _l2, degrade_crop, quality_gate

TAUS = np.round(np.arange(0.20, 0.951, 0.01), 3)
MARGINS = (0.0, 0.02, 0.05)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def _parent(src):
    s = str(src or "")
    return s.split(":", 1)[1] if ":" in s else s


def _parts(samples, aug_weight, max_templates, farthest):
    X = np.stack([s["emb"] for s in samples]); w = np.array([aug_weight if s["kind"] == "aug" else 1.0 for s in samples], np.float32)
    cent = _l2((X * w[:, None]).sum(0))
    real = [i for i, s in enumerate(samples) if s["kind"] != "aug"] or list(range(len(samples)))
    aug = [i for i in range(len(samples)) if i not in set(real)]
    if len(real) >= max_templates:
        pick = [real[i] for i in farthest(X[real], max_templates)]
    else:
        rest = max_templates - len(real)
        pick = list(real) + ([aug[i] for i in farthest(X[aug], rest)] if aug and rest > 0 else [])
    return cent, X[pick]


def collect_loo(store, engine, cfg, methods=("centroid", "multi_max", "hybrid", "multi_topk"), degrade=True, n_degrade=2, seed=0, use_aug_in_gallery=True, log=print):
    """Returns dict[domain][method] -> dict(gen_s1, gen_s2, gen_ok, imp_s1, imp_s2, n_members)."""
    mcfg = cfg["matching"]; aug_w = cfg["enroll"]["aug_weight"]; K = mcfg["max_templates_per_id"]
    rows = store.samples(kinds=["real", "aug"] if use_aug_in_gallery else ["real"])
    by = {}
    for r in rows:
        if store.get_member(r["member_id"])["status"] == "active":
            by.setdefault(r["member_id"], []).append(r)
    ids = sorted(m for m, v in by.items() if any(s["kind"] == "real" for s in v))
    if len(ids) < 3:
        raise RuntimeError("Can it nhat 3 thanh vien co anh de chay LOO")
    idx = {m: i for i, m in enumerate(ids)}
    farthest = store._farthest
    full = [_parts(by[m], aug_w, K, farthest) for m in ids]
    all_T = [t for _, T in full for t in T]; lab = np.concatenate([[i] * len(T) for i, (_, T) in enumerate(full)])
    queries = []                                      # (member_idx, parent, emb, domain)
    rng = np.random.RandomState(seed)
    for m in ids:
        for s in by[m]:
            if s["kind"] != "real":
                continue
            queries.append((idx[m], _parent(s["source"]), s["emb"], "clean"))
            if degrade and engine is not None and s["crop_path"]:
                crop = __import__("cv2").imread(str(store.crops_dir / s["crop_path"]))
                if crop is not None:
                    es = engine.embed_crops([degrade_crop(crop, rng, "mixed") for _ in range(n_degrade)], tta=engine.tta_flip)
                    queries += [(idx[m], _parent(s["source"]), e, "webcam") for e in es]
    log(f"LOO: {len(ids)} thanh vien, {sum(1 for q in queries if q[3] == 'clean')} anh that, {sum(1 for q in queries if q[3] == 'webcam')} anh gia lap webcam")
    out = {}
    for method in methods:
        g = Gallery(ids, np.stack(all_T), lab, method, 0.0, 0.0, mcfg["topk"], centroids=np.stack([c for c, _ in full]))
        for mi, par, q, dom in queries:
            own = ids[mi]
            rest = [s for s in by[own] if _parent(s["source"]) != par]
            sc = g.identity_scores(q)
            if rest and any(s["kind"] == "real" for s in rest):
                c, T = _parts(rest, aug_w, K, farthest)
                sc[mi] = Gallery([own], T, np.zeros(len(T), np.int64), method, 0.0, 0.0, mcfg["topk"], centroids=c[None]).identity_scores(q)[0]
            else:
                sc[mi] = -1.0                          # member has no other photo: cannot be a fair genuine trial
                continue
            order = np.argsort(-sc); i1, s1, s2 = int(order[0]), float(sc[order[0]]), float(sc[order[1]])
            d = out.setdefault(dom, {}).setdefault(method, dict(gen_s1=[], gen_s2=[], gen_ok=[], imp_s1=[], imp_s2=[], n_members=len(ids)))
            d["gen_s1"].append(s1); d["gen_s2"].append(s2); d["gen_ok"].append(i1 == mi)
            sc2 = sc.copy(); sc2[mi] = -np.inf; o2 = np.argsort(-sc2)
            d["imp_s1"].append(float(sc2[o2[0]])); d["imp_s2"].append(float(sc2[o2[1]]))
    for dom in out:
        for m in out[dom]:
            for k in ("gen_s1", "gen_s2", "gen_ok", "imp_s1", "imp_s2"):
                out[dom][m][k] = np.array(out[dom][m][k])
    return out


def rates(d, tau, margin):
    g1, g2, ok = d["gen_s1"], d["gen_s2"], d["gen_ok"]
    acc = (g1 >= tau) & ((g1 - g2) >= margin)
    tpir = float((acc & ok).mean()) if len(g1) else float("nan")
    misid = float((acc & ~ok).mean()) if len(g1) else float("nan")
    i1, i2 = d["imp_s1"], d["imp_s2"]
    far = float(((i1 >= tau) & ((i1 - i2) >= margin)).mean()) if len(i1) else float("nan")
    return tpir, misid, far


def choose(pools, far_target=0.01, log=print, floor=0.45, ceil=0.75):
    """pools: list of per-domain dicts {method: d}. Pooled over domains, per method: among (tau, margin) with FAR<=target take those within
    0.5 pt of the best correct-accept rate and pick the HIGHEST tau (a mis-identification is worse than asking someone to retry; with few
    impostor trials a low tau is a bet on strangers that look unlike anyone enrolled). tau stays within [floor, ceil].
    Across methods: within 1 pt of the best -> simpler method."""
    methods = sorted({m for p in pools for m in p})
    simp = {"centroid": 0, "multi_max": 1, "hybrid": 2, "multi_topk": 3}
    table = []
    taus_all = TAUS[(TAUS >= floor) & (TAUS <= ceil)]
    for m in methods:
        ds = [p[m] for p in pools if m in p]
        cat = {k: np.concatenate([d[k] for d in ds]) for k in ("gen_s1", "gen_s2", "gen_ok", "imp_s1", "imp_s2")}
        feas = []
        for mg in MARGINS:
            for t in taus_all:
                tp, mi, fa = rates(cat, t, mg)
                if fa <= far_target:
                    feas.append((t, mg, tp, mi, fa))
        if not feas:
            lo = min(((t, mg, *rates(cat, t, mg)) for mg in MARGINS for t in taus_all), key=lambda r: (r[4], -r[2]))
            table.append(dict(method=m, tau=float(lo[0]), margin=float(lo[1]), tpir=lo[2], misid=lo[3], far=lo[4], met=False, n_imp=len(cat["imp_s1"]))); continue
        bt = max(r[2] for r in feas); cand = [r for r in feas if r[2] >= bt - 0.005]
        mm = min(r[1] for r in cand); tau = max(r[0] for r in cand if r[1] == mm)
        r = next(r for r in cand if r[0] == tau and r[1] == mm)
        table.append(dict(method=m, tau=float(tau), margin=float(mm), tpir=r[2], misid=r[3], far=r[4], met=True, n_imp=len(cat["imp_s1"])))
    ok = [t for t in table if t["met"]] or table
    top = max(t["tpir"] for t in ok)
    near = sorted([t for t in ok if t["tpir"] >= top - 0.01], key=lambda t: (simp.get(t["method"], 9), -t["tpir"]))
    return near[0], table


def run_loo(store, engine, cfg, far_target=0.01, degrade=True, log=print, **kw):
    res = collect_loo(store, engine, cfg, degrade=degrade, log=log, **kw)
    pools = [res[d] for d in res]
    best, table = choose(pools, far_target)
    log(f"\nKet qua LOO (gop {'+'.join(res)}), muc tieu FAR <= {far_target:.1%}:")
    log(f"{'method':<11}{'tau':>6}{'margin':>8}{'dung':>8}{'nham':>8}{'FAR':>8}  dat_FAR")
    for t in table:
        log(f"{t['method']:<11}{t['tau']:>6.2f}{t['margin']:>8.2f}{t['tpir']:>8.1%}{t['misid']:>8.1%}{t['far']:>8.1%}  {'co' if t['met'] else 'KHONG'}")
    for dom in res:
        d = res[dom][best["method"]]; tp, mi, fa = rates(d, best["tau"], best["margin"])
        k = int(round(fa * len(d["imp_s1"])))
        lo, hi = wilson(k, len(d["imp_s1"]))
        log(f"  [{dom}] n_gen={len(d['gen_s1'])} n_imp={len(d['imp_s1'])}: dung {tp:.1%} | nham {mi:.1%} | FAR {fa:.2%} (95% {lo:.2%}-{hi:.2%})")
    log(f"=> De xuat: method={best['method']} threshold={best['tau']:.2f} margin={best['margin']:.2f}")
    best["reliable"] = best.get("n_imp", 0) >= 200 and len(res[next(iter(res))][best["method"]]["gen_s1"]) >= 50
    if best.get("n_imp", 0) < 200:
        log(f"  ! chi co {best.get('n_imp', 0)} luot thu nguoi la: FAR <1% chua do duoc chac chan; gallery cang dong thi nguong can cang cao. "
            "Chay lai sau khi dang ky them va CHAC CHAN do bang `calibrate probes` voi camera that.")
    return best, table, res


# ---------------- real-camera probes ----------------
def run_probes(store, engine, cfg, folder, far_target=0.01, log=print, max_files=None):
    """folder/<MSSV or 'Ho Ten_MSSV_Ban' or anything else>/<photos taken by the real camera>.
    Folder whose MSSV is an active member = genuine; any other folder = stranger (expected UNKNOWN)."""
    mcfg = cfg["matching"]; q = cfg["live_quality"]
    g, _ = store.compile_gallery(mcfg["method"], 0.0, 0.0, mcfg["topk"], mcfg["max_templates_per_id"], cfg["enroll"]["aug_weight"])
    known = set(g.ids)
    gen, imp, skipped = [], [], {}
    n = 0
    for sub in sorted(p for p in Path(folder).iterdir() if p.is_dir()):
        _, mssv, _ = parse_label(sub.name)
        truth = mssv if mssv in known else None
        for p in list_images(sub):
            if max_files and n >= max_files:
                break
            n += 1
            img = read_image(p, 1600)
            f = engine.analyse(img) if img is not None else dict(status="no_face")
            ok, why = quality_gate(f, **q) if f.get("status") == "ok" else (False, f.get("status", "no_face"))
            if not ok:
                skipped[why] = skipped.get(why, 0) + 1; continue
            i1, s1, i2, s2 = g.top2(f["emb"])
            if truth:
                gen.append((s1, s2, i1 == truth))
            else:
                imp.append((s1, s2))
    d = dict(gen_s1=np.array([a for a, _, _ in gen]), gen_s2=np.array([b for _, b, _ in gen]), gen_ok=np.array([c for _, _, c in gen], bool),
             imp_s1=np.array([a for a, _ in imp]), imp_s2=np.array([b for _, b in imp]))
    log(f"Probes: {len(gen)} anh thanh vien, {len(imp)} anh nguoi la, bo qua {sum(skipped.values())} anh do chat luong {skipped}")
    if not len(gen):
        raise RuntimeError("Khong co anh thanh vien hop le (ten thu muc phai chua MSSV cua thanh vien trong DB)")
    best, table = choose([{mcfg["method"]: d}], far_target) if len(imp) else (dict(method=mcfg["method"], tau=mcfg["threshold"], margin=mcfg["margin"]), [])
    best["reliable"] = len(gen) >= 30 and len(imp) >= 50
    if not best["reliable"]:
        log(f"  ! Chua du du lieu de de xuat nguong dang tin (can >= 30 anh thanh vien va >= 50 anh nguoi la; hien {len(gen)}/{len(imp)}). Dang giu nguong cu.")
        best = dict(method=mcfg["method"], tau=mcfg["threshold"], margin=mcfg["margin"], reliable=False)
    for tau in (mcfg["threshold"], best["tau"]):
        tp, mi, fa = rates(d, tau, mcfg["margin"] if tau == mcfg["threshold"] else best["margin"]) if len(imp) else (*rates(dict(d, imp_s1=np.array([]), imp_s2=np.array([])), tau, mcfg["margin"])[:2], float("nan"))
        log(f"  tau={tau:.2f}: dung {tp:.1%} | nham nguoi {mi:.1%} | khong nhan ra {1 - tp - mi:.1%} | nguoi la lot {fa:.1%}" if len(imp) else f"  tau={tau:.2f}: dung {tp:.1%} | nham {mi:.1%} (khong co anh nguoi la de do FAR)")
    if len(imp) < 50:
        log(f"  ! chi co {len(imp)} anh nguoi la: FAR chua du do phan giai, nen them anh nguoi la (khach, nguoi ngoai CLB).")
    return best, d
