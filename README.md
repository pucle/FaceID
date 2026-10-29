# Club Face ID — gói triển khai điểm danh bằng khuôn mặt (CLB o365 – HUST)

Gói này nối **notebook `o365face_v2.ipynb`** (đo đạc, chọn cấu hình, xuất dữ liệu gốc) với **vận hành thực tế**:
đăng ký thêm mặt, sửa thông tin, hiệu chỉnh độ chính xác, chạy điểm danh bằng camera và gửi kết quả về **portal CLB** (`https://portal.o365hust.club`).
Không có giao diện đồ hoạ: mọi thứ là dòng lệnh + một dịch vụ chạy nền.

> **Đã kiểm chứng gì?** Toàn bộ code đã được chạy thử trong sandbox: notebook v2 chạy hết (Mục 1–19) trên bộ ảnh mặt công khai gồm 9 người; `import-seed`, đăng ký, chặn trùng người, hiệu chỉnh, service phát lại ảnh, đồng bộ với portal giả (retry/409/422/401/mất mạng), và 12 unit test (`python -m unittest tests.test_core -v`).
> **Chưa kiểm chứng:** camera thật, đăng ký bằng webcam có hướng dẫn tư thế (`capture`) trên phần cứng, **API portal thật** (không công khai — xem §7), và độ chính xác trên dữ liệu thật của CLB. Ngưỡng/độ chính xác phải đo lại bằng camera điểm danh thật (§5.3).

---
## 1. Tổng quan

```
 ảnh đăng ký / webcam ──► enroll (lọc chất lượng + chặn nhầm người + thêm mẫu giả lập webcam) ──┐
                                                                                                ▼
 portal ──members──► FaceStore (SQLite: thành viên + từng embedding + crop 112×112) ──► Gallery (tự biên dịch, tự nạp lại)
                                                                                                │
 camera ─► phát hiện TẤT CẢ mặt ─► cổng chất lượng ─► FaceTracker (gộp nhiều khung) ─► khớp ───┘
                                                         │ xác nhận
                                                         ▼
                                          events (hàng đợi bền) ──► portal /face-events  (hoặc CSV)
```

| File | Việc |
|---|---|
| `manage.py` | **Mọi thao tác quản trị**: thêm/sửa/xoá thành viên, đăng ký ảnh/webcam, hiệu chỉnh, đồng bộ portal, sao lưu, `doctor` |
| `attendance_service.py` | Dịch vụ điểm danh chạy nền (camera → portal) |
| `face_id_runtime.py` | Lõi: SCRFD + MobileFaceNet (ONNX), Gallery, FaceTracker, giả lập webcam. **Chính file này được notebook dùng để đo** |
| `facestore.py` | Cơ sở dữ liệu `data/faceid.db` |
| `enrollment.py` | Logic đăng ký + chống nhầm người |
| `calibrate.py` | Đo & đề xuất ngưỡng (`loo`, `probes`) |
| `portal_client.py` | Client portal (cấu hình được, hàng đợi, CSV dự phòng) |
| `config.json` | Cấu hình (tạo từ `config.example.json`; `import-seed` tự ghi phần khớp/ngưỡng) |
| `tools/mock_portal.py` | Portal giả để thử tích hợp |
| `docs/PORTAL_API.md` | **Hợp đồng API đề xuất gửi BĐH/dev portal** |
| `scripts/` | Khởi động tự động (Windows `.bat`, Linux systemd) |

### Dữ liệu nằm ở đâu — vì sao đổi cách lưu
Trước đây chỉ có `gallery.npz` (gallery đã biên dịch — với phương pháp *centroid* mỗi người chỉ còn một vector): **không thể thêm/bớt một ảnh** nếu không có ảnh gốc. Nay `data/faceid.db` giữ **từng embedding**
(`real` = ảnh thật, `aug` = bản giả lập webcam sinh từ ảnh thật, `auto` = dành cho thêm sau), cộng crop 112×112 (`data/crops/<MSSV>/`) để **tính lại embedding khi đổi model** mà không cần ảnh gốc.
Gallery luôn được biên dịch lại từ DB (≈ vài chục ms) — service đang chạy **tự nạp lại trong ≤ 3 giây** sau mỗi thay đổi, không cần khởi động lại.
Thông tin hiển thị (tên, ban, trạng thái) nằm ở bảng `members` — sửa tên/ban **không đụng tới embedding**.

---
## 2. Cài đặt

```bash
# Python 3.10+  (Windows: tạo venv cho sạch)
python -m venv .venv && .venv\Scripts\activate        # Linux: source .venv/bin/activate
pip install -r requirements.txt                       # máy chủ không màn hình: dùng opencv-python-headless

# (A) Có gói seed từ notebook (khuyên dùng — kèm model + cấu hình + ngưỡng đã chọn)
unzip faceid_seed_bundle.zip                           # tạo thư mục faceid_seed_bundle/
python manage.py import-seed faceid_seed_bundle

# (B) Chưa có seed: tải model rồi tạo từng thành viên (§3)
python manage.py fetch-models

python manage.py doctor --camera                       # kiểm tra model, DB, camera, (portal)
```
`import-seed` làm: tạo thành viên (khoá = **MSSV** lấy từ nhãn `Họ Tên_MSSV_Ban`), nạp từng embedding, sinh mẫu giả lập webcam từ crop, kiểm tra **SHA-256 model khớp**, rồi ghi `method/threshold/margin/tta_flip` do notebook chọn vào `config.json`.
Nếu có 2 nhãn trùng MSSV, chúng bị **gộp** (notebook đã in cảnh báo) — hãy sửa tên thư mục từ gốc.

Máy i3-9100F dùng chung văn phòng: mặc định `models.threads=2`, xử lý 5 khung/giây, ≤ 4 mặt/khung. `doctor` in thời gian detect trên máy bạn; nếu > 150 ms/khung, hạ `tracking.fps` hoặc `det_size` (320×320) — nhưng đo lại độ chính xác.

---
## 3. Quy trình hằng ngày

> Khoá thành viên là **MSSV** (8–10 chữ số). Tên/ban chỉ là thông tin hiển thị.

### 3.1 Đăng ký thành viên mới
```bash
python manage.py add-member --id 20252345 --name "Nguyễn Văn A" --ban CM
```
Rồi **một trong ba cách** thêm mặt:

**(a) Từ ảnh** (khuyên dùng ≥ 8 ảnh: thẳng, ±45° trái/phải, ngửa nhẹ, cúi nhẹ; ánh sáng/ngày khác nhau; mặt ≥ 120 px)
```bash
python manage.py enroll --id 20252345 "D:\anh\A" --dry-run     # xem trước: ảnh nào nhận/loại và vì sao
python manage.py enroll --id 20252345 "D:\anh\A"
```
**(b) Nhiều người cùng lúc** — thư mục `GỐC\<Họ Tên_MSSV_Ban>\ảnh…` (đúng định dạng dataset notebook; tự tạo thành viên nếu thiếu)
```bash
python manage.py enroll-folder "D:\dangky_moi" --verbose
```
**(c) Webcam có hướng dẫn tư thế** (5 tư thế × 6 khung: thẳng → quay một bên → quay bên kia → ngửa → cúi)
```bash
python manage.py capture --id 20252345 --camera 0
```
Cửa sổ hiện hướng dẫn và giá trị `yaw/pitch`. Tư thế quá khó (hết 25 s) tự bỏ qua. *Chưa thử với camera thật — nếu ngưỡng tư thế không hợp camera của bạn, chỉnh `enroll.pose_*` trong config hoặc dùng cách (a).*

**Mỗi ảnh đi qua các lớp bảo vệ** (đây là phần làm tăng độ chính xác *từ gốc*, vì mẫu sai/bẩn sinh ra nhầm người mãi mãi):

| Lớp | Loại ảnh khi… | Cấu hình |
|---|---|---|
| Có mặt / đủ lớn / đủ tin cậy | không thấy mặt; mặt < `enroll.min_face_px`; `det_score` thấp | `enroll.min_*` |
| Nhiều người trong ảnh | mặt thứ hai ≥ 50 % diện tích mặt chính (không biết mặt nào của thành viên) | `max_second_face_ratio` |
| Ảnh trùng | giống một ảnh đã có (cos ≥ 0.97) | `near_dup_cos` |
| **Nhất quán với chính người đó** | cos với các ảnh còn lại < 0.35 → *nhiều khả năng sai người/người khác trong ảnh* | `min_self_cos` |
| **Trùng người khác** | giống một thành viên *khác* (cos ≥ 0.5): cùng người đăng ký 2 MSSV hoặc gắn nhầm nhãn | `conflict_cos`; bỏ qua bằng `--force` |
| Mẫu giả lập webcam | *không loại*, mà **thêm** 3 bản/ảnh (hạ độ phân giải, mờ, nhiễu, đổi sáng, JPEG, lệch landmark) | `augment`, `aug_per_sample` |

> Tập 2–3 ảnh/người là **không đủ** cho thực tế (cảnh báo `<5 ảnh` sẽ hiện). `python manage.py list --weak 8` liệt kê ai còn thiếu.

### 3.2 Thêm ảnh cho người đã có (nhận diện kém / đổi kiểu tóc / đeo kính)
Đúng lệnh `enroll`/`capture` ở trên — **cộng dồn** vào mẫu cũ, không mất gì. Ảnh gần giống mẫu cũ bị bỏ qua tự động.
Muốn làm lại từ đầu: `manage.py remove-samples --id 20252345 --kind real` rồi đăng ký lại. `--kind aug` chỉ xoá bản giả lập (sẽ sinh lại bằng `reembed`).
Xem chi tiết từng mẫu: `manage.py show --id 20252345` (mỗi mẫu có `sample_id`, kích thước mặt, yaw…); xoá mẫu xấu: `remove-samples --id … --sample-ids 12 15`.

### 3.3 Sửa thông tin
```bash
python manage.py update-member --id 20252345 --name "Nguyễn Văn An" --ban TC
python manage.py update-member --id 20252345 --new-id 20252346        # sửa nhầm MSSV: mẫu + crop + lịch sử điểm danh đi theo
python manage.py update-member --id 20252345 --status inactive         # nghỉ CLB: khỏi gallery, GIỮ dữ liệu; --status active để quay lại
```
Không cần tính lại gì. Nếu bật đồng bộ portal, **portal là nguồn sự thật** cho tên/ban/trạng thái: `manage.py sync-members` sẽ ghi đè tên/ban/trạng thái theo portal (không đụng khuôn mặt).

### 3.4 Xoá (quyền được xoá dữ liệu)
```bash
python manage.py delete-member --id 20252345 --yes [--purge-events]
```
Xoá vĩnh viễn thành viên + mọi embedding + crop (lịch sử điểm danh giữ lại trừ khi `--purge-events`). Nhớ xoá cả ảnh gốc ở nơi khác và các bản sao lưu cũ.

### 3.5 Hiệu chỉnh ("calibration") — khi nào, làm gì

| Tình huống | Lệnh |
|---|---|
| Vừa nhập seed / thêm nhiều người / đổi cấu hình | `python manage.py calibrate loo` — leave-one-photo-out ngay trên DB, thêm cả ảnh giả lập webcam; in method + ngưỡng đề xuất |
| **Trước khi dùng thật & mỗi khi đổi camera/vị trí/ánh sáng** | `python manage.py calibrate probes <thư_mục_ảnh_camera_thật> --apply` |
| Đổi model nhận diện | `python manage.py reembed` (cần crop) → `calibrate loo` |
| Có người hay bị nhận nhầm | thêm ảnh cho **cả hai** người (§3.2) + tăng ngưỡng/`margin` |

**`calibrate loo`**: mỗi ảnh thật lần lượt bị rút khỏi gallery (cùng cả các bản augment sinh từ nó — không rò rỉ) rồi nhận diện lại; “người lạ” = chính thành viên đó khi gallery *không có* họ.
Chọn `(method, threshold, margin)` thoả **FAR ≤ 1 %**, rồi lấy ngưỡng **cao nhất** trong khoảng cách tối đa 0.5 điểm % so với tốt nhất (nhầm người nguy hiểm hơn phải thử lại), kẹp trong [0.45, 0.75].
Nó in cảnh báo khi số lượt thử người lạ < 200 — **gallery càng đông, ngưỡng cần càng cao** (nhiều so sánh hơn → dễ có người “giống”), nên chạy lại sau mỗi đợt đăng ký lớn.

**`calibrate probes DIR`** — con số đáng tin nhất. Cấu trúc:
```
DIR/20252345/*.jpg          ← ảnh của thành viên (tên thư mục chứa MSSV, chụp/trích từ CHÍNH camera điểm danh)
DIR/20251111_Tran_B/*.jpg   ← thành viên khác
DIR/KHACH/*.jpg             ← bất kỳ thư mục nào không phải MSSV trong DB = người lạ (khách, người ngoài CLB)
```
Cần ≥ 30 ảnh thành viên và ≥ 50 ảnh người lạ để `--apply` được ghi (dưới đó nó từ chối, `--force-apply` để bỏ qua). Ảnh trượt cổng chất lượng được đếm riêng (mặt quá nhỏ/quay…).

**Cách thu ảnh thật mà không tốn công:** chạy service với `--pilot-dir pilot` (hoặc `service.pilot_dir`): mỗi lần xác nhận lưu mặt vào `pilot/<MSSV>/`, mỗi người lạ lưu vào `pilot/UNKNOWN/` (tên file ghi điểm). Sau 1–2 buổi, **BĐH duyệt**: chuyển ảnh nhận nhầm sang đúng thư mục MSSV (hoặc `UNKNOWN`), rồi `calibrate probes pilot`.
*Lưu ý: ảnh trong `pilot/` là dữ liệu sinh trắc — thư mục này cần cùng mức bảo vệ như `data/`, và nên xoá sau khi hiệu chỉnh.*

---
## 4. Chạy điểm danh

```bash
python attendance_service.py --show                     # thử, có cửa sổ xem trước (q để thoát)
python attendance_service.py --offline --pilot-dir pilot  # ghi DB, chưa gửi portal (giai đoạn thử)
python attendance_service.py                            # chạy thật (gửi portal nếu portal.enabled)
python attendance_service.py --source D:\thu_anh        # phát lại thư mục ảnh/video để kiểm thử
python attendance_service.py --session-id EV-2026-10-30 # ép phiên (nếu portal không tự chọn được)
```
Quyết định điểm danh **không dựa vào một khung hình**. Với mỗi khuôn mặt đang được theo dõi (IoU giữa các khung):
1. **Cổng chất lượng** (`live_quality`): bỏ qua khung có mặt < 70 px, `det_score` < 0.6, quay quá (yaw > 0.6), cúi/ngửa quá. Bỏ qua chứ **không đoán**; lý do được thống kê trong log cuối buổi.
2. **Gộp embedding** nhiều khung (trọng số theo độ tin cậy × kích thước mặt) → khớp với gallery.
3. Xác nhận khi: ≥ `confirm_frames` (3) khung tốt, điểm gộp ≥ ngưỡng, hơn người thứ hai ≥ `margin`, và ≥ 60 % khung gần nhất đồng ý cùng người đó (chống nhảy qua lại giữa hai người giống nhau).
4. **Cooldown** `cooldown_s` (300 s) cho mỗi người → không ghi trùng khi đứng lâu trước camera.
5. Ghi vào bảng `events` **trước**, rồi mới gửi portal (luồng nền) → mất mạng/portal bảo trì không mất điểm danh.

Nhiều người đi qua cùng lúc được xử lý riêng từng track (bản cũ chỉ nhìn mặt lớn nhất).
`data/heartbeat.txt` cập nhật mỗi chu kỳ (số khung, fps, độ trễ, phiên) — dùng cho giám sát; log ở `data/service.log` (xoay vòng).

**Đặt camera (tác động độ chính xác hơn mọi tham số):** ngang tầm mắt, cách ~0.6–1.5 m, mặt chiếm ≥ 100 px trong khung 640×480; **ánh sáng chiếu vào mặt** (không để cửa sổ/đèn *sau lưng* người đứng); tránh camera quá cao nhìn xuống; xếp hàng để từng người dừng ~1 s.

---
## 5. Độ chính xác: đã làm gì, đo thế nào, giới hạn

### 5.1 Các cải tiến so với baseline notebook gốc
1. **Khoảng cách miền (domain gap)** — ảnh đăng ký thường là ảnh chất lượng cao (mặt 200–500 px) còn webcam ~70–150 px, mờ, nhiễu. Bổ sung mẫu **giả lập webcam** khi đăng ký (`aug`, trọng số 0.5 khi tính centroid) và *đo* trong notebook Mục 17 bằng probe giả lập webcam trên crop (ảnh probe và ảnh tạo mẫu dùng hai dòng ngẫu nhiên khác nhau).
2. **Matching `hybrid`/`multi_max`** (trung bình điểm centroid và mẫu gần nhất) bên cạnh `centroid`: giữ được nhiều tư thế thay vì “trung bình hoá” khuôn mặt. Notebook chọn cái tốt nhất bằng số liệu, không đoán.
3. **Gộp nhiều khung + bỏ phiếu + cooldown** thay vì quyết định theo khung đơn lẻ: nhiễu từng khung bị trung bình đi; kẻ giống người khác phải “lừa” nhiều khung liên tiếp.
4. **Cổng chất lượng** ở cả đăng ký và điểm danh; **bỏ qua** khung kém thay vì đoán.
5. **Nhiều mặt/khung**, mỗi track tính riêng.
6. **Chống mẫu bẩn** khi đăng ký (nhất quán, trùng người khác, nhiều mặt) — nguồn nhầm người dai dẳng nhất trong thực tế.
7. **Ngưỡng thận trọng + có chặn dưới/trên** và từ chối `--apply` khi dữ liệu hiệu chỉnh quá ít.
8. Tuỳ chọn **flip-TTA** (`models.tta_flip`, tốn thêm ~1 lần embed/mặt): notebook đo có đáng không rồi mới đề xuất.

### 5.2 Đọc kết quả notebook Mục 17 đúng cách
* Bảng so sánh các cấu hình trên **test giả lập webcam**: số liệu `đúng / nhầm người / người lạ lọt`. Ngưỡng chọn trên validation, không nhìn test.
* Toàn bộ là **mô phỏng** trên crop đã căn chỉnh — nó phản ánh độ phân giải/mờ/nhiễu/ánh sáng/lệch landmark nhưng *không* phản ánh: góc camera, ánh sáng thật, bị che, người đi lại, lỗi phát hiện mặt ở độ phân giải thấp. Vì vậy §5.3 là bắt buộc.
* Nếu mọi cấu hình đều 100 % → tập test quá dễ/quá nhỏ; đừng kết luận gì. Nhìn cột *webcam* và khoảng tin cậy.

### 5.3 Quy trình “lên sóng” khuyến nghị
1. `import-seed` → `doctor --camera`.
2. Chạy thử 1–2 buổi: `attendance_service.py --offline --pilot-dir pilot` (ghi nhận nhưng chưa gửi portal).
3. BĐH duyệt `pilot/` → `calibrate probes pilot --apply`. Cần cả ảnh **người lạ thật** (khách, người ngoài CLB).
4. Bật portal. Tuần đầu: có người ngồi kiểm tra chéo; mọi ca nhầm → thêm ảnh cho cả hai người + ghi lại.
5. Sau mỗi đợt đăng ký ≥ 10 người: `calibrate loo`.

### 5.4 Giới hạn cần biết
* **Không có chống giả mạo (liveness)**: ảnh in/màn hình điện thoại có thể qua mặt. Điểm danh có giám sát hoặc thêm bước (QR/thủ công) cho trường hợp quan trọng.
* Khẩu trang, kính râm, mũ che trán làm giảm điểm mạnh; anh em sinh đôi/rất giống nhau cần `margin` cao hơn.
* Model InsightFace pretrained: điều khoản ghi *nghiên cứu phi thương mại* — kiểm tra trước khi dùng ngoài phạm vi CLB.
* Mục tiêu FAR ≤ 1 % đo trên ít lượt thử không phải bảo đảm; đo lại bằng ảnh thật.

---
## 6. Cấu hình (`config.json`)
Chỉ cần ghi các khoá muốn đổi (phần còn lại lấy mặc định, xem `config.example.json`).

| Khoá | Ý nghĩa / mặc định |
|---|---|
| `models.threads` / `det_thresh` / `tta_flip` | luồng ONNX (2) / ngưỡng phát hiện (0.5) / flip-TTA (false) |
| `matching.method` `threshold` `margin` `max_templates_per_id` | **do notebook/`calibrate` ghi**; `hybrid`, 12 mẫu/người |
| `live_quality.*` | `min_face_px` 70, `min_det_score` 0.6, `max_abs_yaw` 0.6 … |
| `tracking.*` | `confirm_frames` 3, `min_vote_ratio` 0.6, `cooldown_s` 300, `fps` 5, `max_faces` 4 |
| `enroll.*` | các lớp bảo vệ ở §3.1, `store_crops` (true: cần để `reembed`), `aug_weight` 0.5 |
| `service.*` | `device_id`, `session_id` (cố định), `pilot_dir`, `sync_interval_s` |
| `portal.*` | `enabled`, `base_url`, `auth.token_env`, `endpoints`, `fields`, `csv_fallback_dir` |

Token portal **không** ghi trong file: đặt biến môi trường `PORTAL_TOKEN` (Windows: `setx PORTAL_TOKEN "…"`; Linux: `/etc/faceid.env`).

---
## 7. Tích hợp portal CLB

### 7.1 Hiện trạng (kiểm tra ngày 29/10/2026)
Truy cập `https://portal.o365hust.club/` chỉ thấy trang đăng nhập: *“Đăng ký trực, sự kiện và điểm danh”*; đăng nhập bằng **MSSV** (hoặc email cho thầy cô) + mật khẩu + **“Xác thực bảo mật”**; mọi trang khác cần đăng nhập. **Không tìm thấy tài liệu API công khai.**
Vì vậy: (1) **đừng** viết robot đăng nhập bằng form (có xác thực bảo mật; mong manh, dễ vi phạm điều khoản); (2) cần BĐH/dev portal cung cấp một **tài khoản dịch vụ + token** và ít nhất 3 endpoint. Mọi yêu cầu nằm trong **`docs/PORTAL_API.md`** — gửi file đó cho người phụ trách portal.

### 7.2 Portal cần có gì để phối hợp tốt (tóm tắt)
| # | Cần | Để làm gì |
|---|---|---|
| 1 | Tài khoản dịch vụ + token quyền tối thiểu, thu hồi được, (tuỳ chọn) giới hạn IP, HTTPS | không dùng mật khẩu người thật, không dính captcha |
| 2 | `GET members` (MSSV, họ tên, ban, **trạng thái**, `updated_since`) | đồng bộ ai được vào gallery; người nghỉ tự rời; thành viên mới hiện trong `list --no-face` |
| 3 | `GET sessions?status=open` (trực / sự kiện / họp, giờ bắt đầu–kết thúc, cửa sổ check-in) | biết đang điểm danh cho phiên nào |
| 4 | `POST face-events` **idempotent** (`Idempotency-Key`), trả `201/409/422` rõ nghĩa | không mất/không trùng khi mạng chập chờn |
| 5 | Quy tắc *có mặt/muộn/trùng* **do portal quyết** | đổi luật không sửa AI |
| 6 | Cờ “cần duyệt” khi `confidence` sát ngưỡng + màn hình duyệt cho BĐH | cái bẫy lớn nhất: nhầm người im lặng |
| 7 | (Nên) `PUT members/{mssv}/face-status` | portal hiện “đã đăng ký mặt” |
| 8 | (Nên) phản hồi `verified=false` cho từng bản ghi | nhãn *nhầm người* thật để hiệu chỉnh |
| 9 | Trường đồng ý xử lý dữ liệu sinh trắc + thời điểm | tuân thủ pháp luật (§9) |
| 10 | Đồng hồ chuẩn (`health.time`), múi giờ `+07:00` | tránh lệch giờ điểm danh |

### 7.3 Cấu hình & thử
```jsonc
// config.json
"portal": { "enabled": true, "base_url": "https://portal.o365hust.club",
            "endpoints": { "members": "GET /api/v1/members?updated_since={since}&page={page}", ... },   // đổi theo portal thật
            "fields":    { "member_id": "mssv", "name": "full_name", "ban": "ban", "status": "status" } }
```
```bash
python tools/mock_portal.py --port 8765 --flaky 2        # terminal 1: portal giả (503 hai lần đầu để thử retry)
set PORTAL_TOKEN=SECRET                                   # terminal 2 (Linux: export)
# đặt portal.base_url = http://127.0.0.1:8765, portal.enabled = true
python manage.py doctor ; python manage.py sync-members ; python attendance_service.py --source D:\thu_anh
```
Lệnh liên quan: `sync-members [--deactivate-missing]`, `sync-events` (đẩy hàng đợi thủ công), `export-csv [--session ID]` (UTF-8 BOM, mở bằng Excel).

### 7.4 Hành vi khi lỗi
`2xx`/`409` → đã đồng bộ · `5xx`/`429`/timeout → giữ hàng đợi, thử lại mỗi `sync_interval_s` · `4xx` (422…) → loại, ghi `last_error` (xem bằng `export-csv`, cột `da_dong_bo=0`) · `401/403` → dừng, log lỗi, hàng đợi giữ nguyên.
Không có API → dùng `export-csv` cuối buổi và nhập vào portal.

---
## 8. Vận hành
* **Tự khởi động (Windows):** Task Scheduler → *At startup*, chạy `scripts\start_service.bat` (tự khởi động lại sau 10 s nếu dừng; chạy dưới tài khoản có quyền camera). Hoặc dùng NSSM bọc `python attendance_service.py`.
* **Linux:** `scripts/faceid.service` (systemd, `Restart=always`).
* **Sao lưu:** `python manage.py backup --out D:\backup` (DB nhất quán, kể cả đang chạy). Sao lưu cả `data/crops/`. Bản sao lưu là dữ liệu sinh trắc → mã hoá ổ đĩa (BitLocker).
* **Cập nhật gallery khi service chạy:** cứ `enroll`/`update-member`… — service tự nạp lại ≤ 3 s (log: `gallery: N thanh vien …`).
* **Đổi model:** thay file ONNX trong `models/`, sửa tên trong `config.json`, `python manage.py reembed`, rồi `calibrate loo --apply`. DB từ chối chạy nếu model lệch (kiểm SHA-256) — chống so sánh embedding khác không gian.
* **Giám sát:** `data/heartbeat.txt` (cũ hơn vài phút = service treo), `data/service.log`.
* **Kiểm thử tự động:** `python -m unittest tests.test_core -v`.

---
## 9. Bảo mật, quyền riêng tư, pháp lý
* Embedding, crop và ảnh `pilot/` là **dữ liệu sinh trắc học**. Theo Luật Bảo vệ dữ liệu cá nhân số 91/2025/QH15 (hiệu lực 01/01/2026) và Nghị định 356/2025/NĐ-CP, dữ liệu sinh trắc học thuộc danh mục **dữ liệu cá nhân nhạy cảm**; khi xin đồng ý phải nói rõ đây là dữ liệu nhạy cảm, và có quy định riêng về việc ghi hình/sinh trắc (Điều 31–32 Luật). *Đây là thông tin tham khảo, không phải tư vấn pháp lý — hãy hỏi người có chuyên môn/nhà trường về mẫu đồng ý, thời hạn lưu trữ và thông báo ghi hình tại nơi đặt camera.*
* Thực hành tối thiểu: chỉ lưu những gì cần (tắt `enroll.store_crops` nếu không cần `reembed`; xoá `pilot/` sau hiệu chỉnh), ổ đĩa mã hoá, hạn chế người truy cập `data/`, **không gửi ảnh/embedding lên portal**, có đường xoá (`delete-member`), biển thông báo khu vực điểm danh, ghi nhận việc đồng ý (trường portal §7.2 #9).
* Token portal lưu trong biến môi trường; xoay vòng định kỳ; tài khoản dịch vụ quyền tối thiểu.

---
## 10. Xử lý sự cố

| Triệu chứng | Nguyên nhân thường gặp → cách xử lý |
|---|---|
| `Model nhan dien KHAC…` | DB tạo bằng model khác → dùng đúng model cũ, hoặc `reembed` |
| Không mở được camera | app khác đang dùng; thử `--camera 1`; quyền camera Windows; (Windows dùng DirectShow/MSMF tự động) |
| Ai đó không bao giờ được nhận | xem `show --id`; ít ảnh/ảnh toàn thẳng → `capture`; kiểm tra `status=active`; xem log “bo qua theo ly do” (mặt nhỏ? quay?) |
| Nhận nhầm A thành B | thêm ảnh cho cả A và B; tăng `margin` (vd 0.05) hoặc `threshold`; kiểm tra A,B có thật là 2 người khác nhau (trùng đăng ký?) |
| Quá nhiều “UNKNOWN” | ánh sáng/khoảng cách; `live_quality.min_face_px` hạ xuống 60 (sau khi `calibrate probes`); thêm mẫu webcam thật |
| Điểm danh trùng | `cooldown_s` quá nhỏ; hoặc portal chưa trả 409 cho trùng |
| Portal `AUTH` | token hết hạn/sai quyền; hàng đợi vẫn giữ, sửa token rồi service tự gửi lại |
| `enroll` loại hết ảnh | `--dry-run` để xem lý do; ảnh quá nhỏ → hạ `enroll.min_face_px`, nhưng đừng dưới ~90 |
| CPU cao khi dùng văn phòng | `models.threads=1`, `tracking.fps=3`, `max_faces=2` |

---
## 11. Bảng lệnh nhanh
```
manage.py add-member | update-member | list [--no-face|--weak N] | show | delete-member | remove-samples
manage.py enroll | enroll-folder | capture            (các lệnh đăng ký; --dry-run, --force, --no-augment)
manage.py import-seed | reembed | rebuild-gallery
manage.py calibrate loo|probes DIR [--far 0.01] [--apply]
manage.py sync-members | sync-events | export-csv | backup | doctor | fetch-models
attendance_service.py [--show] [--offline] [--source X] [--session-id ID] [--pilot-dir D] [--camera N]
```
