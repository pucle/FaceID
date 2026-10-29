"""python -m unittest tests.test_core -v     (test engine can models/*.onnx; neu thieu se bo qua phan do)"""
import os, sys, tempfile, threading, unittest, http.server
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT)); sys.path.insert(0, str(ROOT / "tools"))
from common import load_config
from face_id_runtime import FaceTracker, Gallery, UNKNOWN, degrade_crop, quality_gate
from facestore import FaceStore


def unit(rng, d=512):
    v = rng.normal(size=d).astype(np.float32); return v / np.linalg.norm(v)


def person(rng, n, noise=0.35):
    c = unit(rng); out = []
    for _ in range(n):
        v = c + noise * unit(rng); out.append(v / np.linalg.norm(v))
    return np.stack(out)


class StoreTests(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.TemporaryDirectory(); self.st = FaceStore(Path(self.d.name) / "t.db"); self.rng = np.random.RandomState(0)

    def tearDown(self):
        self.st.close(); self.d.cleanup()

    def fill(self, n_people=6, n=6):
        P = {}
        for i in range(n_people):
            mid = f"2025{i:04d}"; self.st.add_member(mid, f"Nguoi {i}", "CM"); P[mid] = person(self.rng, n)
            for e in P[mid]: self.st.add_sample(mid, e)
        return P

    def test_gallery_match_and_info_edit_does_not_touch_embeddings(self):
        P = self.fill(); g, info = self.st.compile_gallery("hybrid", 0.5)
        self.assertEqual(info["members"], 6)
        mid = "20250002"; q = P[mid].mean(0); q /= np.linalg.norm(q)
        self.assertEqual(g.match(q)[0], mid)
        self.st.update_member(mid, name="Ten Moi", ban="TC")
        g2, _ = self.st.compile_gallery("hybrid", 0.5)
        self.assertEqual(g2.match(q)[0], mid); self.assertEqual(self.st.get_member(mid)["name"], "Ten Moi")

    def test_unknown_person_rejected(self):
        self.fill(); g, _ = self.st.compile_gallery("hybrid", 0.55)
        self.assertEqual(g.match(unit(self.rng))[0], UNKNOWN)

    def test_inactive_member_leaves_gallery_and_returns(self):
        P = self.fill(); mid = "20250001"; q = P[mid].mean(0)
        self.st.update_member(mid, status="inactive"); g, _ = self.st.compile_gallery("hybrid", 0.5)
        self.assertNotIn(mid, g.ids); self.st.update_member(mid, status="active"); g, _ = self.st.compile_gallery("hybrid", 0.5)
        self.assertIn(mid, g.ids)

    def test_rename_and_delete(self):
        self.fill(); self.st.rename_member_id("20250003", "20259999")
        self.assertIsNone(self.st.get_member("20250003")); self.assertEqual(len(self.st.samples("20259999")), 6)
        self.assertEqual(self.st.delete_member("20259999"), 6); self.assertEqual(len(self.st.samples("20259999")), 0)

    def test_event_idempotency(self):
        self.fill(); a = self.st.add_event("20250001", .9, .1, 3, "S", "d", ts="2026-01-01T10:00:00+07:00"); b = self.st.add_event("20250001", .9, .1, 3, "S", "d", ts="2026-01-01T10:00:00+07:00")
        self.assertIsNotNone(a); self.assertIsNone(b)


class TrackerTests(unittest.TestCase):
    def test_needs_consecutive_agreement_and_cooldown(self):
        rng = np.random.RandomState(1); P = [person(rng, 5) for _ in range(4)]
        ids = [f"m{i}" for i in range(4)]; E = np.vstack(P); L = np.repeat(np.arange(4), 5)
        g = Gallery(ids, E, L, "hybrid", 0.5); tr = FaceTracker(g, confirm_frames=3, cooldown_s=60)
        mk = lambda e, x=0: dict(bbox=[10 + x, 10, 110 + x, 110], emb=e, det_score=.9, face_side=100.)
        probe = P[2].mean(0); probe /= np.linalg.norm(probe)
        ev = [tr.update([mk(probe, t)], now=t * .2)[0] for t in range(6)]
        self.assertEqual([len(e) for e in ev], [0, 0, 1, 0, 0, 0]); self.assertEqual(ev[2][0]["identity"], "m2")

    def test_flip_flopping_identity_never_confirms(self):
        rng = np.random.RandomState(2); P = [person(rng, 5) for _ in range(2)]
        g = Gallery(["a", "b"], np.vstack(P), np.repeat([0, 1], 5), "multi_max", 0.3); tr = FaceTracker(g, confirm_frames=4, min_vote_ratio=0.75)
        a, b = P[0].mean(0), P[1].mean(0)
        evs = []
        for t in range(8):
            e = a if t % 2 == 0 else b; evs += tr.update([dict(bbox=[0, 0, 100, 100], emb=e / np.linalg.norm(e), det_score=.9, face_side=100.)], now=t * .2)[0]
        self.assertEqual(evs, [])

    def test_unknown_never_logged(self):
        rng = np.random.RandomState(3); P = person(rng, 5); g = Gallery(["a"], P, np.zeros(5, int), "multi_max", 0.6); tr = FaceTracker(g)
        for t in range(10):
            self.assertEqual(tr.update([dict(bbox=[0, 0, 100, 100], emb=unit(rng), det_score=.9, face_side=100.)], now=t * .2)[0], [])


class MiscTests(unittest.TestCase):
    def test_quality_gate(self):
        f = dict(status="ok", det_score=.9, face_side=120, yaw_proxy=.1, pitch_proxy=.5, blur=100)
        self.assertTrue(quality_gate(f)[0]); self.assertEqual(quality_gate(dict(f, face_side=40))[1], "too_small"); self.assertEqual(quality_gate(dict(f, yaw_proxy=.9))[1], "head_turned")

    def test_degrade_is_deterministic_and_changes_image(self):
        img = np.random.RandomState(0).randint(0, 255, (112, 112, 3)).astype(np.uint8)
        a = degrade_crop(img, np.random.RandomState(5)); b = degrade_crop(img, np.random.RandomState(5))
        self.assertTrue((a == b).all()); self.assertFalse((a == img).all()); self.assertEqual(a.shape, img.shape)


class PortalTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import mock_portal
        os.environ["PORTAL_TOKEN"] = "SECRET"; mock_portal.TOKEN = "SECRET"; mock_portal.STATE.update(events={}, fail_next=0)
        cls.mp = mock_portal; cls.srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), mock_portal.H); threading.Thread(target=cls.srv.serve_forever, daemon=True).start()
        cfg = load_config(None); cfg["portal"]["base_url"] = f"http://127.0.0.1:{cls.srv.server_address[1]}"; cls.cfg = cfg

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()

    def test_sync_flow(self):
        from portal_client import PortalClient, sync_events, sync_members_into_store
        d = tempfile.TemporaryDirectory(); st = FaceStore(Path(d.name) / "t.db"); c = PortalClient(self.cfg["portal"])
        self.assertTrue(c.health()); r = sync_members_into_store(st, c); self.assertEqual(r["added"], 3)
        self.assertEqual(st.get_member("20240001")["status"], "inactive")
        st.add_event("20250007", .9, .1, 3, "EV", "k1", ts="2026-01-01T10:00:00+07:00"); st.add_event("BAD", .9, .1, 3, "EV", "k1", ts="2026-01-01T10:00:01+07:00")
        self.mp.STATE["fail_next"] = 1
        self.assertEqual(sync_events(st, c)["retry"], 1)                       # outage: stays queued
        n = sync_events(st, c); self.assertEqual((n["ok"], n["permanent"]), (1, 1))
        st.db.execute("UPDATE events SET synced=0, attempts=0 WHERE member_id='20250007'"); self.assertEqual(sync_events(st, c)["duplicate"], 1)   # 409 == already recorded
        os.environ["PORTAL_TOKEN"] = "WRONG"; st.add_event("20250007", .9, .1, 3, "EV", "k1", ts="2026-01-01T11:00:00+07:00")
        self.assertEqual(sync_events(st, c).get("auth_error"), 1); self.assertEqual(len(st.pending_events()), 1); os.environ["PORTAL_TOKEN"] = "SECRET"; st.close(); d.cleanup()


@unittest.skipUnless((ROOT / "models" / "w600k_mbf.onnx").exists(), "models/*.onnx chua co")
class EngineTests(unittest.TestCase):
    def test_engine_loads_and_embeds_crop(self):
        from common import load_engine
        eng, shas = load_engine(load_config(None)); e = eng.embed_crops([np.zeros((112, 112, 3), np.uint8)])
        self.assertEqual(e.shape, (1, 512)); self.assertAlmostEqual(float(np.linalg.norm(e[0])), 1.0, places=3)


if __name__ == "__main__":
    unittest.main(verbosity=2)
