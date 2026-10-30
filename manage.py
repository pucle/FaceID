#!/usr/bin/env python
"""Club Face ID - quan tri (them/sua/xoa mat, hieu chinh, dong bo portal).  `python manage.py <lenh> -h` de xem chi tiet.

  add-member  update-member  list  show  delete-member
  enroll (anh)  enroll-folder (nhieu nguoi)  capture (webcam)  remove-samples
  import-seed (tu notebook)  reembed  calibrate loo|probes  rebuild-gallery
  sync-members  sync-events  export-csv  backup  doctor  bench  fetch-models
  (them --provider cuda|directml|auto truoc ten lenh de dung GPU, vd: manage.py --provider cuda doctor)
"""
import argparse
import csv
import json
import os
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import cv2
import numpy as np

from common import HERE, data_dir, load_config, load_engine, read_image, save_config, setup_logging, sha256_file
from enrollment import capture_frames, enroll_folder, enroll_images, list_images, make_aug_embeddings, parse_label
from facestore import FaceStore

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

KNOWN_SHA = {"det_500m.onnx": "5e4447f50245bbd7966bd6c0fa52938c61474a04ec7def48753668a9d8b4ea3a",
             "w600k_mbf.onnx": "9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f"}
PACK_URL = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_sc.zip"


def open_store(cfg):
    d = data_dir(cfg)
    return FaceStore(d / "faceid.db", d / "crops")


def need_engine(cfg, store):
    eng, shas = load_engine(cfg)
    ok, msg = store.check_models(shas)
    if not ok:
        sys.exit(msg)
    return eng


def print_report(rep, verbose=True):
    print(f"\n== {rep['member_id']}: nhan {len(rep['accepted'])} anh, loai {len(rep['rejected'])} ==")
    if verbose:
        for n, d in rep["accepted"]:
            print(f"  + {n}  ({d})")
    for n, why in rep["rejected"]:
        print(f"  - {n}: {why}")
    for w in rep["warnings"]:
        print(f"  ! {w}")


# ------------------------------------------------------------------ commands
def cmd_fetch_models(a, cfg):
    import zipfile
    md = Path(cfg["paths"]["models_dir"]); md.mkdir(parents=True, exist_ok=True)
    z = md / "buffalo_sc.zip"
    print("tai", PACK_URL); urllib.request.urlretrieve(PACK_URL, z)
    with zipfile.ZipFile(z) as zf:
        for fn in KNOWN_SHA:
            (md / fn).write_bytes(zf.read(fn))
    z.unlink()
    for fn, want in KNOWN_SHA.items():
        got = sha256_file(md / fn); print(f"{fn}: {'OK' if got == want else 'SHA KHAC (' + got[:12] + ')'}")
    print("Luu y giay phep: model InsightFace pretrained ghi 'non-commercial research'. Kiem tra dieu khoan truoc khi dung thuc te.")


def cmd_add_member(a, cfg):
    st = open_store(cfg)
    st.add_member(a.id, a.name, a.ban, a.portal_id, upsert=a.upsert, note=a.note)
    print(f"da them {a.id} - {a.name}")


def cmd_update_member(a, cfg):
    st = open_store(cfg)
    f = {k: v for k, v in dict(name=a.name, ban=a.ban, status=a.status, note=a.note, portal_id=a.portal_id).items() if v is not None}
    if f:
        st.update_member(a.id, **f)
    if a.new_id:
        st.rename_member_id(a.id, a.new_id); print(f"da doi MSSV {a.id} -> {a.new_id}")
    print("da cap nhat (khong can tao lai embedding; service tu nap lai gallery trong vai giay)")


def cmd_list(a, cfg):
    st = open_store(cfg)
    rows = st.list_members(a.status)
    if a.no_face:
        rows = [r for r in rows if r["n_real"] + r["n_auto"] == 0]
    if a.weak:
        rows = [r for r in rows if 0 < r["n_real"] + r["n_auto"] < a.weak]
    print(f"{'MSSV':<12}{'Ho ten':<30}{'Ban':<5}{'TT':<9}{'that':>5}{'aug':>5}{'auto':>5}")
    for r in rows:
        print(f"{r['member_id']:<12}{(r['name'] or '')[:28]:<30}{(r['ban'] or ''):<5}{r['status']:<9}{r['n_real']:>5}{r['n_aug']:>5}{r['n_auto']:>5}")
    print(f"-- {len(rows)} thanh vien")


def cmd_show(a, cfg):
    st = open_store(cfg); m = st.get_member(a.id)
    if not m:
        sys.exit("khong co thanh vien nay")
    print(json.dumps(m, ensure_ascii=False, indent=2))
    for s in st.samples(a.id, enabled_only=False, with_emb=False):
        print(f"  #{s['sample_id']:<5}{s['kind']:<5} en={s['enabled']} side={s['face_side'] or 0:>5.0f} det={s['det_score'] or 0:.2f} yaw={s['yaw'] or 0:+.2f} pose={s['pose'] or '-':<6} {s['source']}")


def cmd_delete_member(a, cfg):
    st = open_store(cfg)
    if not a.yes:
        sys.exit("Xoa VINH VIEN thanh vien + toan bo embedding/anh crop. Them --yes de xac nhan.")
    n = st.delete_member(a.id, a.purge_events)
    print(f"da xoa {a.id} ({n} mau). Nho xoa ca anh goc o noi khac va cac ban backup cu.")


def cmd_enroll(a, cfg):
    st = open_store(cfg); eng = need_engine(cfg, st)
    paths = []
    for p in a.photos:
        pp = Path(p)
        paths += list_images(pp) if pp.is_dir() else [pp]
    imgs = [(p.name, read_image(p, cfg["enroll"]["max_side"])) for p in paths]
    rep = enroll_images(st, eng, cfg, a.id, imgs, source="photo", pose=a.pose, augment=not a.no_augment, force=a.force, dry_run=a.dry_run)
    print_report(rep)
    if a.dry_run:
        print("(dry-run: chua ghi gi vao DB)")


def cmd_enroll_folder(a, cfg):
    st = open_store(cfg); eng = need_engine(cfg, st)
    for rep in enroll_folder(st, eng, cfg, a.folder, create_missing=not a.no_create, augment=not a.no_augment, force=a.force, dry_run=a.dry_run):
        print_report(rep, verbose=a.verbose)


def cmd_capture(a, cfg):
    st = open_store(cfg); eng = need_engine(cfg, st)
    if st.get_member(a.id) is None:
        sys.exit("Tao thanh vien truoc: add-member")
    from attendance_service import FrameSource
    src = FrameSource(a.camera if a.camera is not None else cfg["camera"]["index"], cfg["camera"])
    def gen():
        while True:
            ok, fr = src.read()
            if not ok:
                return
            yield fr
    def show(frame, msg, col):
        v = frame.copy(); cv2.putText(v, msg[:70], (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2)
        cv2.imshow("Dang ky khuon mat - q: thoat", v)
        return (cv2.waitKey(1) & 0xFF) in (ord("q"), 27)
    print("Lam theo huong dan tren cua so. Giu khuon mat ro, du sang, khong deo khau trang/kinh toi mau.")
    try:
        got = capture_frames(eng, cfg, gen(), show=show)
    finally:
        src.close(); cv2.destroyAllWindows()
    rep = enroll_images(st, eng, cfg, a.id, [(n, f) for n, f, _ in got], source="webcam", pose=None, augment=not a.no_augment, force=a.force, dry_run=a.dry_run)
    print_report(rep, verbose=False)


def cmd_remove_samples(a, cfg):
    st = open_store(cfg)
    kinds = None if a.kind == "all" else [a.kind]
    n = st.delete_samples(a.id, kinds, a.sample_ids)
    print(f"da xoa {n} mau")


def cmd_import_seed(a, cfg):
    st = open_store(cfg); d = Path(a.bundle)
    mc = json.loads((d / "model_config.json").read_text(encoding="utf-8"))
    md = Path(cfg["paths"]["models_dir"]); md.mkdir(parents=True, exist_ok=True)
    for fn in (mc["detector_file"], mc["recognizer_file"]):
        if (d / fn).exists() and not (md / fn).exists():
            shutil.copy2(d / fn, md / fn)
    cfg["models"]["detector"], cfg["models"]["recognizer"] = mc["detector_file"], mc["recognizer_file"]
    eng, shas = load_engine(cfg)
    if shas["recognizer_sha256"] != mc["recognizer_sha256"]:
        sys.exit("SHA-256 recognizer khac voi luc export -> embedding khong dung duoc voi model nay")
    ok, msg = st.check_models(shas)
    if not ok:
        sys.exit(msg)
    z = np.load(d / "seed_embeddings.npz", allow_pickle=False); E = z["emb"]
    rows = list(csv.DictReader(open(d / "seed_samples.csv", encoding="utf-8-sig")))
    assert len(rows) == len(E), "seed_samples.csv va seed_embeddings.npz lech so dong"
    rng = np.random.RandomState(0); added = dict(members=0, real=0, aug=0)
    print(f"Nhap {len(rows)} anh (moi anh sinh them {cfg['enroll']['aug_per_sample'] if cfg['enroll']['augment'] and not a.no_augment else 0} mau gia lap webcam) - co the mat vai phut tren CPU...")
    t0 = time.time(); st.db.execute("BEGIN")                    # ONE transaction: thousands of autocommits are very slow on HDD
    for n_, (r, e) in enumerate(zip(rows, E), 1):
        if n_ % 50 == 0:
            print(f"  {n_}/{len(rows)} anh  ({time.time() - t0:.0f}s)", flush=True)
        mid = r["mssv"]
        if st.get_member(mid) is None:
            st.add_member(mid, r["name"], r["ban"] or None); added["members"] += 1
        crop = read_image(d / r["crop"], 0) if r.get("crop") and (d / r["crop"]).exists() else None
        q = dict(det_score=float(r["det_score"] or 0), face_side=float(r["face_side"] or 0), blur=float(r["blur"] or 0),
                 yaw_proxy=float(r["yaw"] or 0), pitch_proxy=float(r["pitch"] or 0))
        st.add_sample(mid, e, "real", f"notebook:{r['source']}", None, q, crop if cfg["enroll"]["store_crops"] else None); added["real"] += 1
        if crop is not None and cfg["enroll"]["augment"] and not a.no_augment:
            for ae in make_aug_embeddings(eng, crop, cfg["enroll"]["aug_per_sample"], rng):
                st.add_sample(mid, ae, "aug", f"aug:notebook:{r['source']}", None, q, None); added["aug"] += 1
    st.db.execute("COMMIT")
    if not a.keep_config:
        for k in ("method", "threshold", "margin", "topk"):
            if k in mc:
                cfg["matching"][k] = mc[k]
        cfg["models"]["tta_flip"] = bool(mc.get("tta_flip", False))
        for k in ("min_det_score", "min_face_px"):
            if f"live_{k}" in mc:
                cfg["live_quality"][k] = mc[f"live_{k}"]
        save_config(cfg)
    print(f"da nhap: {added}. Cau hinh khop: {cfg['matching']}, tta_flip={cfg['models']['tta_flip']}")


def cmd_reembed(a, cfg):
    st = open_store(cfg); eng, shas = load_engine(cfg)
    rows = [s for s in st.samples(kinds=["real", "auto"], enabled_only=False, with_emb=False)]
    miss = [s for s in rows if not s["crop_path"] or not (st.crops_dir / s["crop_path"]).exists()]
    if miss:
        sys.exit(f"{len(miss)} mau that khong co crop -> khong the tinh lai embedding. Dang ky lai nhung nguoi nay.")
    mapping, crops_by = {}, {}
    for s in rows:
        crops_by[s["sample_id"]] = cv2.imread(str(st.crops_dir / s["crop_path"]))
    ids = list(crops_by)
    for i in range(0, len(ids), 64):
        part = ids[i:i + 64]
        for sid, e in zip(part, eng.embed_crops([crops_by[j] for j in part])):
            mapping[sid] = e
    for m in st.list_members():
        st.delete_samples(m["member_id"], ["aug"])
    st.replace_embeddings(mapping, shas)
    rng = np.random.RandomState(0)
    for s in rows:
        for e in make_aug_embeddings(eng, crops_by[s["sample_id"]], cfg["enroll"]["aug_per_sample"], rng):
            st.add_sample(s["member_id"], e, "aug", f"aug:{s['source']}", s["pose"], {}, None)
    print(f"da tinh lai {len(mapping)} embedding bang model hien tai; nho chay lai `calibrate loo --apply`.")


def cmd_calibrate(a, cfg):
    import calibrate as cal
    st = open_store(cfg); eng = need_engine(cfg, st)
    if a.mode == "loo":
        best, table, _ = cal.run_loo(st, eng, cfg, a.far, degrade=not a.no_degrade)
    else:
        best, _ = cal.run_probes(st, eng, cfg, a.folder, a.far)
    if a.apply and not best.get("reliable", True) and not a.force_apply:
        print("KHONG ghi: du lieu chua du tin cay (them anh/nguoi la, hoac dung --force-apply neu chap nhan rui ro).")
    elif a.apply:
        cfg["matching"].update(method=best["method"], threshold=round(float(best["tau"]), 3), margin=round(float(best["margin"]), 3))
        save_config(cfg); print(f"da ghi vao config.json: {cfg['matching']}")
    else:
        print("(them --apply de ghi de xuat vao config.json)")


def cmd_rebuild_gallery(a, cfg):
    st = open_store(cfg); m = cfg["matching"]
    out = Path(a.out) if a.out else data_dir(cfg) / "gallery.npz"
    g, info = st.save_gallery(out, method=m["method"], threshold=m["threshold"], margin=m["margin"], topk=m["topk"],
                              max_templates=m["max_templates_per_id"], aug_weight=cfg["enroll"]["aug_weight"])
    print(f"gallery -> {out}: {info}")


def _client(cfg):
    from portal_client import PortalClient
    return PortalClient(cfg["portal"])


def cmd_sync_members(a, cfg):
    from portal_client import sync_members_into_store
    st = open_store(cfg)
    sync_members_into_store(st, _client(cfg), since=a.since, deactivate_missing=a.deactivate_missing)


def cmd_sync_events(a, cfg):
    from portal_client import sync_events
    st = open_store(cfg); m = cfg["matching"]
    print(sync_events(st, _client(cfg), f"{m['method']}@{m['threshold']:.2f}", batch=a.batch))


def cmd_export_csv(a, cfg):
    from portal_client import export_csv
    st = open_store(cfg)
    p, n = export_csv(st, a.out or cfg["portal"]["csv_fallback_dir"], a.session)
    print(f"{n} dong -> {p}")


def cmd_backup(a, cfg):
    st = open_store(cfg)
    dest = Path(a.out) / f"faceid_{time.strftime('%Y%m%d_%H%M%S')}.db"
    print("backup ->", st.backup(dest), "(chua ma hoa: hay luu o o dia BitLocker/ma hoa)")


def cmd_doctor(a, cfg):
    ok = True
    def chk(name, good, msg=""):
        nonlocal ok
        print(f"[{'OK' if good else 'LOI'}] {name} {msg}"); ok &= bool(good)
    md = Path(cfg["paths"]["models_dir"])
    for fn in (cfg["models"]["detector"], cfg["models"]["recognizer"]):
        p = md / fn; chk(f"model {fn}", p.exists() and (fn not in KNOWN_SHA or sha256_file(p) == KNOWN_SHA[fn]), "" if p.exists() else "(chua co: manage.py fetch-models)")
    try:
        st = open_store(cfg); mem = st.list_members("active")
        chk("co so du lieu", True, f"{len(mem)} thanh vien active, {sum(1 for m in mem if m['n_real'] + m['n_auto'] == 0)} chua co mat")
        eng, shas = load_engine(cfg); good, msg = st.check_models(shas, adopt=False); chk("model khop DB", good, "" if st.get_meta("recognizer_sha256") else "(DB chua co embedding)")
        t0 = time.perf_counter(); img = np.zeros((480, 640, 3), np.uint8)
        for _ in range(5):
            eng.analyse_all(img)
        import onnxruntime as _ort
        chk("provider", eng.provider_spec in ("cpu", "auto") or eng.on_gpu, f"yeu cau={eng.provider_spec} | dang dung={eng.active_providers['detector'][0]} | kha dung={_ort.get_available_providers()}"
            + (f" | {eng.provider_note}" if eng.provider_spec != "cpu" else ""))
        chk("toc do detect (khung den)", True, f"{(time.perf_counter() - t0) / 5 * 1000:.0f} ms/khung")
    except Exception as e:
        chk("engine/DB", False, f"{type(e).__name__}: {e}")
    if a.camera:
        c = cv2.VideoCapture(cfg["camera"]["index"]); chk("camera", c.isOpened() and c.read()[0]); c.release()
    if cfg["portal"]["enabled"]:
        try:
            chk("portal /health", _client(cfg).health(), cfg["portal"]["base_url"])
        except Exception as e:
            chk("portal", False, f"{type(e).__name__}: {e}")
    else:
        print("[--] portal chua bat (config.json -> portal.enabled)")
    sys.exit(0 if ok else 1)


def cmd_bench(a, cfg):
    """Do toc do that tren may nay: CPU vs GPU. Dung --image (khung camera that) de so sanh cong bang."""
    import statistics
    st = open_store(cfg)
    img = read_image(a.image, 1280) if a.image else None
    note = ""
    if img is None:                                          # fallback: a stored face crop pasted on a 640x480 canvas
        for s in st.samples(kinds=["real"], enabled_only=False, with_emb=False):
            if s["crop_path"] and (st.crops_dir / s["crop_path"]).exists():
                crop = cv2.resize(cv2.imread(str(st.crops_dir / s["crop_path"])), (220, 220))
                img = np.full((480, 640, 3), 128, np.uint8); img[130:350, 210:430] = crop; note = " (khung tong hop tu 1 crop)"; break
    if img is None:
        sys.exit("Can --image <khung camera .jpg> (hoac da co crop trong DB)")
    print(f"Anh do: {img.shape[1]}x{img.shape[0]}{note}; {a.n} lan/provider (sau 3 lan khoi dong)")
    print(f"{'provider':<10}{'dang dung':<26}{'mat':>4}{'TB ms':>9}{'p95 ms':>9}{'fps':>7}")
    for spec in [x.strip() for x in a.providers.split(",") if x.strip()]:
        eng, _ = load_engine(cfg, provider=spec)
        for _ in range(3):
            eng.analyse_all(img)
        ts, nf = [], 0
        for _ in range(a.n):
            t0 = time.perf_counter(); nf = len(eng.analyse_all(img, max_faces=4)); ts.append((time.perf_counter() - t0) * 1000)
        used = eng.active_providers["recognizer"][0].replace("ExecutionProvider", "")
        tag = used + ("" if (spec.lower() == "cpu") == (used == "CPU") or spec.lower() == "auto" else "  <- quay ve CPU!")
        print(f"{spec:<10}{tag:<26}{nf:>4}{statistics.mean(ts):>9.1f}{sorted(ts)[int(0.95 * (len(ts) - 1))]:>9.1f}{1000 / statistics.mean(ts):>7.1f}")
    print("GPU thuong chi nhanh hon ro khi nhieu mat/khung hoac do phan giai det_size lon; model nho nen CPU van du dung. Quyet dinh theo so do tren camera that.")



def build_parser():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--provider", default=None, help="cpu | cuda | directml | auto  (mac dinh: config models.provider)")
    sp = ap.add_subparsers(dest="cmd", required=True)
    def P(name, fn, **kw):
        p = sp.add_parser(name, **kw); p.set_defaults(fn=fn); return p
    P("fetch-models", cmd_fetch_models)
    p = P("add-member", cmd_add_member); p.add_argument("--id", required=True, help="MSSV"); p.add_argument("--name", required=True)
    p.add_argument("--ban"); p.add_argument("--portal-id"); p.add_argument("--note"); p.add_argument("--upsert", action="store_true")
    p = P("update-member", cmd_update_member); p.add_argument("--id", required=True)
    for k in ("name", "ban", "note", "portal-id", "new-id"):
        p.add_argument(f"--{k}")
    p.add_argument("--status", choices=["active", "inactive"])
    p = P("list", cmd_list); p.add_argument("--status", choices=["active", "inactive"]); p.add_argument("--no-face", action="store_true")
    p.add_argument("--weak", type=int, help="chi hien nguoi co it hon N anh that")
    p = P("show", cmd_show); p.add_argument("--id", required=True)
    p = P("delete-member", cmd_delete_member); p.add_argument("--id", required=True); p.add_argument("--yes", action="store_true"); p.add_argument("--purge-events", action="store_true")
    p = P("enroll", cmd_enroll, help="them anh cho 1 thanh vien"); p.add_argument("--id", required=True); p.add_argument("photos", nargs="+")
    p.add_argument("--pose"); p.add_argument("--force", action="store_true"); p.add_argument("--dry-run", action="store_true"); p.add_argument("--no-augment", action="store_true")
    p = P("enroll-folder", cmd_enroll_folder, help="thu muc/<Ho Ten_MSSV_Ban>/anh"); p.add_argument("folder"); p.add_argument("--no-create", action="store_true")
    p.add_argument("--force", action="store_true"); p.add_argument("--dry-run", action="store_true"); p.add_argument("--no-augment", action="store_true"); p.add_argument("--verbose", action="store_true")
    p = P("capture", cmd_capture, help="dang ky bang webcam (huong dan tu the)"); p.add_argument("--id", required=True); p.add_argument("--camera", type=int)
    p.add_argument("--force", action="store_true"); p.add_argument("--dry-run", action="store_true"); p.add_argument("--no-augment", action="store_true")
    p = P("remove-samples", cmd_remove_samples); p.add_argument("--id", required=True); p.add_argument("--kind", choices=["real", "aug", "auto", "all"], default="all"); p.add_argument("--sample-ids", type=int, nargs="*")
    p = P("import-seed", cmd_import_seed, help="nhap goi seed tu notebook"); p.add_argument("bundle"); p.add_argument("--no-augment", action="store_true"); p.add_argument("--keep-config", action="store_true")
    P("reembed", cmd_reembed, help="tinh lai embedding sau khi doi model (can crops)")
    p = P("calibrate", cmd_calibrate); p.add_argument("mode", choices=["loo", "probes"]); p.add_argument("folder", nargs="?")
    p.add_argument("--far", type=float, default=0.01); p.add_argument("--apply", action="store_true"); p.add_argument("--force-apply", action="store_true"); p.add_argument("--no-degrade", action="store_true")
    p = P("rebuild-gallery", cmd_rebuild_gallery); p.add_argument("--out")
    p = P("sync-members", cmd_sync_members); p.add_argument("--since"); p.add_argument("--deactivate-missing", action="store_true")
    p = P("sync-events", cmd_sync_events); p.add_argument("--batch", type=int, default=200)
    p = P("export-csv", cmd_export_csv); p.add_argument("--out"); p.add_argument("--session")
    p = P("backup", cmd_backup); p.add_argument("--out", default="backups")
    p = P("doctor", cmd_doctor); p.add_argument("--camera", action="store_true")
    p = P("bench", cmd_bench, help="do toc do CPU vs GPU"); p.add_argument("--image"); p.add_argument("-n", type=int, default=30)
    p.add_argument("--providers", default="cpu,cuda", help="vd: cpu,cuda,directml")
    return ap


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.provider:
        os.environ["FACEID_PROVIDER"] = a.provider
    cfg = load_config(a.config)
    if a.cmd == "calibrate" and a.mode == "probes" and not a.folder:
        sys.exit("calibrate probes can THU_MUC anh thu bang camera that")
    return a.fn(a, cfg)


if __name__ == "__main__":
    main()
