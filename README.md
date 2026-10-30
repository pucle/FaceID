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
| `pilot/` *(tạo khi chạy)* | ảnh mặt thu bằng camera thật để hiệu chỉnh — xem **§3.6** |

> Muốn viết app xung quanh hệ thống: xem **§12** (bản đồ file, cấu trúc DB, API Python, hành vi cần biết) và **§13** (danh sách triển khai).

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

### 2.1 Dùng GPU (tùy chọn)

Mặc định hệ thống chạy **CPU** (`models.provider = "cpu"`). Có thể chuyển sang GPU mà **không đổi dữ liệu** (DB, gallery, ngưỡng giữ nguyên): chỉ đổi nơi chạy hai model ONNX (phát hiện + nhận diện).

**Có đáng dùng không?** Hai model rất nhỏ nên CPU thường đã đủ cho 1 camera (trong thử nghiệm của mình, một nhân CPU ảo xử lý khoảng 35 ms/khung có 1 mặt). GPU hữu ích khi: nhiều mặt mỗi khung, `det_size` lớn, nhiều camera, hoặc muốn **giải phóng CPU** cho máy văn phòng. Với khung nhỏ, chi phí chép dữ liệu CPU↔GPU có thể làm GPU **không nhanh hơn**. Đừng đoán — đo bằng `bench` (bên dưới).

| Loại GPU | Cài gói | Giá trị `provider` | Ghi chú |
|---|---|---|---|
| NVIDIA | `onnxruntime-gpu` | `cuda` | cần driver mới + **CUDA 12.x và cuDNN 9.x** (bản PyPI hiện tại) |
| Windows, mọi GPU DirectX 12 (AMD/Intel/NVIDIA) | `onnxruntime-directml` | `directml` | *chưa thử trong quá trình phát triển* |
| tự chọn tốt nhất có sẵn | — | `auto` | CUDA > DirectML > CPU |

**Cài (NVIDIA):** `onnxruntime` và `onnxruntime-gpu` **không được cài cùng nhau** (cùng tên module):
```bash
pip uninstall -y onnxruntime onnxruntime-gpu
pip install "onnxruntime-gpu[cuda,cudnn]"      # kéo theo CUDA/cuDNN dạng pip (ORT >= 1.21); hoặc tự cài CUDA 12 + cuDNN 9 hệ thống
```
Code gọi `onnxruntime.preload_dlls()` giúp tìm CUDA/cuDNN do pip hoặc PyTorch cài sẵn. DirectML: `pip uninstall -y onnxruntime && pip install onnxruntime-directml`.

**Bật** (ưu tiên: tham số dòng lệnh > biến môi trường `FACEID_PROVIDER` > `config.json`):
```bash
python manage.py --provider cuda doctor                 # tham số đặt TRƯỚC tên lệnh
python attendance_service.py --provider cuda --show
```
hoặc ghi vào `config.json`: `"models": {"provider": "cuda", "gpu_device_id": 0}` (`gpu_device_id` = số thứ tự GPU khi có nhiều card). Với `scripts/start_service.bat`, thêm `--provider cuda` vào dòng `python attendance_service.py …`.

**Kiểm tra GPU thật sự được dùng** (quan trọng — ONNX Runtime có thể *nhìn thấy* CUDA nhưng vẫn lặng lẽ quay về CPU nếu thiếu cuDNN):
```bash
python manage.py --provider cuda doctor      # dòng [OK] provider … dang dung=CUDAExecutionProvider ; nếu hiện CPUExecutionProvider -> [LOI]
python manage.py bench --image khung_camera.jpg --providers cpu,cuda -n 30     # so sánh ms/khung, cột "dang dung"
```
`bench` in `<- quay ve CPU!` khi GPU được yêu cầu nhưng không dùng được. Log service cũng ghi `provider=CUDA` hoặc `provider=CPU` ở dòng khởi động.

**An toàn:** nếu GPU không có / thiếu thư viện, hệ thống **tự dùng CPU và in cảnh báo kèm cách cài** (không dừng điểm danh). Kết quả embedding GPU và CPU chỉ khác nhau ở sai số dấu phẩy động rất nhỏ nên **không cần tính lại embedding**; để chắc chắn sau khi đổi, chạy lại `calibrate loo` (hoặc `probes`) một lần.

**Notebook/Kaggle:** bước trích embedding dùng GPU nếu `onnxruntime-gpu` có sẵn (`use_cuda_for_extraction=True` mặc định). Trên Kaggle, ô đầu tiên: `!pip uninstall -y onnxruntime && pip install -q onnxruntime-gpu` rồi **Restart session**; Mục 2 sẽ ghi `CUDAExecutionProvider` vào log. Phần đo tốc độ trong notebook vẫn nhằm CPU (mục tiêu triển khai).

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

### 3.6 Thư mục `pilot/` — ảnh thu bằng camera thật (chi tiết)

**Là gì:** một chức năng *thu thập ảnh* của service. Nó **không ảnh hưởng đến việc nhận diện hay điểm danh**; chỉ lưu thêm ảnh mặt ra đĩa để bạn dùng hiệu chỉnh.
Mặc định **tắt**. Bật bằng `attendance_service.py --pilot-dir pilot` hoặc `config.json → service.pilot_dir`. (`scripts/start_service.bat` đã truyền sẵn `--pilot-dir pilot`; muốn tắt thì xoá tham số đó.)

**Ghi gì, ở đâu** (code: hàm `Service.save_pilot` trong `attendance_service.py`):

| Sự kiện | Đường dẫn file | Ghi chú |
|---|---|---|
| Điểm danh **thành công** | `pilot/<MSSV>/<YYYYmmdd_HHMMSS>.jpg` | vùng mặt từ khung camera, chừa viền 40 % mỗi phía, JPEG q92 |
| **Người lạ** (đủ chất lượng nhưng không khớp ai, theo dõi ≥ `confirm_frames` khung) | `pilot/UNKNOWN/<YYYYmmdd_HHMMSS>_t<mã_track>_<điểm>.jpg` | **một ảnh cho mỗi lần xuất hiện** (mỗi track); điểm là điểm cao nhất với một thành viên, vd `_0.28` |

Đường dẫn tương đối (`pilot`) được hiểu **theo thư mục đang chạy lệnh** — hãy chạy từ thư mục `club_face_id/` (file `.bat` đã tự `cd` vào đó), hoặc ghi đường dẫn tuyệt đối (`D:\faceid\pilot`).

**Để làm gì:** đó là ảnh chụp bằng **chính camera và ánh sáng điểm danh của bạn**, loại dữ liệu mà bộ ảnh đăng ký không có. Bạn duyệt lại:

* ảnh **sai người** (nhận nhầm A thành B) thì chuyển sang đúng thư mục MSSV;
* ảnh của **người ngoài CLB** thì để ở `UNKNOWN`.

Sau đó chạy `python manage.py calibrate probes pilot --apply`. Lệnh này đo tỉ lệ đúng / nhầm người / người lạ lọt trên ảnh thật rồi ghi ngưỡng mới vào `config.json`
(cần ≥ 30 ảnh thành viên và ≥ 50 ảnh người lạ, nếu không nó từ chối ghi). Tên thư mục chỉ cần **chứa MSSV** của một thành viên đang có trong DB để được tính là ảnh thành viên; mọi thư mục khác = người lạ.

**Cách dùng đúng, từng bước:**
1. Chạy service với `--offline --pilot-dir pilot` 1–2 buổi sinh hoạt.
2. Mở `pilot/`, xem từng thư mục MSSV: ảnh nào là người khác → kéo sang thư mục MSSV đúng (tạo thư mục mới nếu cần). Xem `pilot/UNKNOWN/`: nếu có *thành viên thật* bị coi là lạ → kéo vào thư mục MSSV của họ.
3. `python manage.py calibrate probes pilot` (xem số liệu) rồi thêm `--apply` để ghi.
4. Với người bị nhận nhầm/không nhận ra nhiều: thêm ảnh bằng `enroll` (§3.2) — ảnh `pilot/` cũng dùng được: `python manage.py enroll --id MSSV pilot\MSSV`.
5. **Xoá `pilot/` sau khi xong**; khi chạy lâu dài bỏ `--pilot-dir` (thư mục này lớn dần và chứa dữ liệu sinh trắc — cần bảo vệ như `data/`).

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
| `models.provider` / `gpu_device_id` | `cpu` (mặc định) \| `cuda` \| `directml` \| `auto`; số thứ tự GPU (0) — xem §2.1 |
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
| CPU cao khi dùng văn phòng | `models.threads=1`, `tracking.fps=3`, `max_faces=2`; hoặc chuyển GPU (§2.1) |
| `--provider cuda` nhưng `doctor` báo `[LOI] provider … dang dung=CPU` | chưa cài `onnxruntime-gpu` (còn `onnxruntime`), hoặc thiếu CUDA 12/cuDNN 9 (`libcudnn…so.9 cannot open`), hoặc driver cũ; gỡ cả hai gói ONNX Runtime rồi cài lại một gói GPU |

---
## 11. Bảng lệnh nhanh
```
manage.py add-member | update-member | list [--no-face|--weak N] | show | delete-member | remove-samples
manage.py enroll | enroll-folder | capture            (các lệnh đăng ký; --dry-run, --force, --no-augment)
manage.py import-seed | reembed | rebuild-gallery
manage.py calibrate loo|probes DIR [--far 0.01] [--apply]
manage.py sync-members | sync-events | export-csv | backup | doctor | bench | fetch-models
manage.py --provider cuda|directml|auto <lệnh>          (chạy bằng GPU, xem §2.1)
attendance_service.py [--show] [--offline] [--source X] [--session-id ID] [--pilot-dir D] [--camera N] [--provider cuda]
```

---
## 12. Tài liệu tham chiếu cho người viết ứng dụng

Phần này dành cho việc **viết app/dashboard/bot xung quanh hệ thống** (xem điểm danh, đăng ký thành viên, nút bấm quản trị…). Có 3 cách tích hợp, từ đơn giản đến sâu:

| Cách | Khi nào | Cách làm |
|---|---|---|
| **A. Gọi CLI** | app muốn thêm/sửa/xoá, hiệu chỉnh | chạy `python manage.py …` bằng `subprocess` (đọc stdout; **mã thoát 0 = ổn**, ≠ 0 = lỗi và thông báo ở stderr/stdout) |
| **B. Đọc SQLite (chỉ đọc)** | dashboard xem điểm danh, danh sách thành viên | mở `data/faceid.db` ở chế độ `mode=ro`; DB dùng WAL nên đọc song song với service **không bị chặn** |
| **C. Import Python** | app Python muốn đăng ký/khớp mặt trực tiếp | `from facestore import FaceStore`, `from enrollment import …` (ví dụ §12.4) |

> Service **không mở cổng mạng nào** (không có HTTP server). Nó chỉ *gọi ra* portal (`portal.base_url`). Nếu cần API cho app web, hãy viết một lớp mỏng (Flask/FastAPI) gọi cách A/B/C ở trên, đừng cho nhiều tiến trình tự ghi tay vào các bảng.

### 12.1 Bản đồ file & thư mục khi chạy
(đường dẫn tính từ thư mục `club_face_id/`; `data_dir` = `paths.data_dir`, mặc định `data/`)

| Đường dẫn | Ai ghi | Nội dung |
|---|---|---|
| `config.json` | người dùng / `import-seed` / `calibrate --apply` | cấu hình (xem §6). Tạo từ `config.example.json` |
| `models/det_500m.onnx`, `models/w600k_mbf.onnx` | `fetch-models` / `import-seed` | model phát hiện + nhận diện (SHA-256 được kiểm) |
| `data/faceid.db` (+ `-wal`, `-shm`) | `FaceStore` | **toàn bộ dữ liệu**: thành viên, mẫu mặt, sự kiện điểm danh (§12.2) |
| `data/crops/<MSSV>/<sample_id>.jpg` | đăng ký | crop 112×112 đã căn chỉnh của mẫu `real` (để `reembed`). Mẫu `aug` không có file |
| `data/service.log` | service | log xoay vòng (2 MB × 5) |
| `data/heartbeat.txt` | service (mỗi `sync_interval_s`) | 1 dòng: `<ISO time> frames=… events=… fps=… lat_ms=… session=…` — dùng giám sát sống/chết |
| `data/restart.log` | `start_service.bat` | mỗi lần tự khởi động lại |
| `data/gallery.npz` | `manage.py rebuild-gallery` | **chỉ để xuất/kiểm tra** — service không đọc file này, nó tự biên dịch từ DB |
| `pilot/…` | service (nếu bật) | ảnh thu để hiệu chỉnh (§3.6) |
| `exports/diemdanh_<phiên>_<ngày>.csv` | `export-csv` | điểm danh dạng Excel (UTF-8 BOM) |
| `backups/faceid_<ngày_giờ>.db` | `backup` | bản sao DB nhất quán |

Địa chỉ mạng liên quan: portal `https://portal.o365hust.club` (cấu hình `portal.base_url`); portal giả `http://127.0.0.1:8765`; biến môi trường `PORTAL_TOKEN` (tên đổi được ở `portal.auth.token_env`); chữ ký HMAC tuỳ chọn ở biến tên trong `portal.hmac_secret_env`.

### 12.2 Cơ sở dữ liệu `data/faceid.db` (SQLite)
**`members`** — một dòng mỗi thành viên

| Cột | Ý nghĩa |
|---|---|
| `member_id` (PK) | **MSSV**, chuỗi. Khoá nối với portal và với thư mục `crops/` |
| `name`, `ban`, `note` | hiển thị; `ban` thường `CM/TC/SK/TT` |
| `portal_id` | id nội bộ của portal (nếu có) |
| `status` | `active` (vào gallery) \| `inactive` (giữ dữ liệu, không nhận diện) |
| `created_at`, `updated_at` | ISO có múi giờ |

**`samples`** — một dòng mỗi embedding

| Cột | Ý nghĩa |
|---|---|
| `sample_id` (PK, tự tăng), `member_id` (FK, xoá thành viên → xoá mẫu) | |
| `emb` | BLOB `float32[512]` đã chuẩn hoá L2 (2048 byte) |
| `kind` | `real` ảnh thật · `aug` bản giả lập webcam sinh từ ảnh thật · `auto` (dành cho thêm sau) |
| `source` | nguồn: `photo:<tên file>`, `folder:…`, `webcam:cam_thang_1`, `notebook:…`, `aug:<nguồn gốc>` |
| `pose`, `det_score`, `face_side`, `blur`, `yaw`, `pitch` | chất lượng lúc đăng ký |
| `crop_path` | đường dẫn tương đối trong `data/crops/` |
| `enabled` | 0 = tạm tắt (không vào gallery) |

**`events`** — điểm danh = đồng thời là **hàng đợi gửi portal** (outbox)

| Cột | Ý nghĩa |
|---|---|
| `event_id` (PK), `idem_key` (UNIQUE) | khoá chống trùng = `sha1(phiên\|MSSV\|thiết bị\|thời điểm)[:32]`; cũng gửi lên portal làm `Idempotency-Key` |
| `ts` | thời điểm ISO có offset múi giờ của máy (`2026-10-30T19:03:12+07:00`) |
| `member_id`, `score`, `score2`, `frames` | ai, điểm khớp, điểm người thứ hai, số khung đã gộp |
| `session_id`, `device_id` | phiên (có thể `NULL`), thiết bị (`service.device_id`) |
| `synced` (0/1), `attempts`, `last_error`, `synced_at` | trạng thái gửi portal. `attempts ≥ 50` hoặc lỗi 4xx (đặt 999) → dừng thử lại |

**`meta`** — `schema_version`, `recognizer_sha256`, `detector_sha256` (chống trộn embedding của hai model khác nhau).

Truy vấn mẫu (chỉ đọc):
```sql
-- điểm danh hôm nay
SELECT e.ts, e.member_id, m.name, m.ban, e.score, e.synced FROM events e JOIN members m USING(member_id)
WHERE date(e.ts) = date('now','localtime') ORDER BY e.ts;
-- ai chưa có khuôn mặt
SELECT member_id, name FROM members WHERE status='active'
  AND member_id NOT IN (SELECT member_id FROM samples WHERE kind='real' AND enabled=1);
-- sự kiện chưa gửi được portal
SELECT event_id, member_id, attempts, last_error FROM events WHERE synced=0;
```
**Quy tắc an toàn:** chỉ **đọc** trực tiếp bằng SQL; mọi thao tác **ghi** đi qua `manage.py` hoặc lớp `FaceStore` (nó giữ ràng buộc, xoá file crop, kiểm tra model…). Mỗi luồng/tiến trình dùng **kết nối riêng** (đối tượng SQLite không dùng chung giữa luồng).

### 12.3 Hành vi cần biết khi thiết kế app
* **Cooldown nằm trong bộ nhớ** của service (`tracking.cooldown_s`, 300 s/người). Khởi động lại service ⇒ cooldown mất ⇒ cùng một người có thể được ghi lại. Vì `idem_key` chứa thời điểm nên DB *không* tự loại → **portal/app nên loại trùng theo (MSSV, phiên)**: lần đầu thắng.
* **Thời điểm** lấy theo đồng hồ + múi giờ máy chủ: giữ máy đồng bộ giờ (NTP).
* **Thay đổi DB (đăng ký/sửa/xoá) có hiệu lực trong ≤ 3 giây** với service đang chạy; nó chỉ biên dịch lại gallery khi nội dung thành viên/mẫu thật sự đổi.
* Thành viên `inactive`, hoặc `active` nhưng **chưa có mẫu**, không nằm trong gallery (không bao giờ được nhận diện). `manage.py list --no-face` liệt kê nhóm sau.
* Hàng đợi gửi portal chạy trong luồng nền; khi tắt service có một lần gửi cuối. Sự kiện bị portal từ chối vĩnh viễn (422…) nằm lại với `last_error` — app nên hiển thị cho BĐH.
* **Quyết định điểm danh** (xem §4) là bất đồng bộ qua nhiều khung: app đừng kỳ vọng kết quả tức thì ở khung đầu; thường mất ~0.6–1 s (3 khung ở 5 khung/s).

### 12.4 API Python (cách C)
Chạy từ thư mục `club_face_id/` (hoặc thêm nó vào `sys.path`).
```python
from common import load_config, data_dir, load_engine, read_image
from facestore import FaceStore
import enrollment

cfg   = load_config("config.json")
store = FaceStore(data_dir(cfg) / "faceid.db", data_dir(cfg) / "crops")      # mỗi luồng một đối tượng
eng, shas = load_engine(cfg)                                                  # nạp model
ok, msg = store.check_models(shas)                                            # PHẢI True trước khi thêm/khớp embedding

# --- thành viên ---
store.add_member("20252345", "Nguyễn Văn A", "CM")                            # ValueError nếu đã có (upsert=True để ghi đè thông tin)
store.update_member("20252345", name="Nguyễn Văn An", status="inactive")      # trường: name, ban, portal_id, status, note
store.rename_member_id("20252345", "20252346"); store.delete_member("20252346", purge_events=False)
store.list_members(status=None)          # [{member_id,name,ban,status,n_real,n_aug,n_auto,...}]

# --- đăng ký ảnh (qua mọi lớp bảo vệ §3.1) ---
imgs = [("a.jpg", read_image("a.jpg", cfg["enroll"]["max_side"]))]            # list[(tên, ảnh BGR)] — read_image trả None nếu không đọc được
rep = enrollment.enroll_images(store, eng, cfg, "20252345", imgs, source="app", dry_run=False, force=False)
rep["accepted"]   # [(tên, mô tả)]       rep["rejected"]  # [(tên, lý do)]       rep["warnings"]  # [str]
store.samples("20252345", kinds=["real"], enabled_only=False, with_emb=False)  # danh sách mẫu; delete_samples(mid, kinds, sample_ids)

# --- nhận diện một ảnh ---
f = eng.analyse(read_image("probe.jpg"))              # mặt lớn nhất; hoặc eng.analyse_all(img, max_faces=5) -> list
g, info = store.compile_gallery(**{k: cfg["matching"][k] for k in ("method","threshold","margin","topk")},
                                max_templates=cfg["matching"]["max_templates_per_id"], aug_weight=cfg["enroll"]["aug_weight"])
ident, s1, s2, why = g.match(f["emb"])               # ident = MSSV hoặc "UNKNOWN"; why ∈ ok|low_score|ambiguous|empty_gallery

# --- điểm danh / portal ---
store.add_event("20252345", score=0.87, score2=0.2, frames=3, session_id="EV1", device_id="app")   # trả None nếu trùng idem_key
store.recent_events(20); store.pending_events(100)
from portal_client import PortalClient, sync_events, sync_members_into_store, export_csv
```
Bản ghi mặt `f` (từ `analyse`/`analyse_all`) có: `status` (`ok|too_small|no_face`), `bbox[x1,y1,x2,y2]`, `kps` (10 số), `det_score`, `face_side` (px), `yaw_proxy`, `pitch_proxy`, `blur`, `emb` (float32[512]), `second_face_ratio` (nếu ≥ 2 mặt), thêm `crop` khi gọi `keep_crop=True`.
Ngưỡng/`method` luôn lấy từ `cfg["matching"]` để app và service **khớp cùng một luật**. Muốn phiên bản đa khung như service, dùng `face_id_runtime.FaceTracker(gallery).update(faces, now)` → `(events, display)`; mỗi `event` có `identity, score, score2, frames, votes, track_id, face_side, bbox`.

### 12.5 Mã thoát & lỗi thường gặp của CLI
`manage.py` trả `0` khi thành công; thông báo lỗi dạng `sys.exit("…")` (mã `1`); `doctor` trả `1` nếu có mục `[LOI]` (tiện cho giám sát). Lỗi cần app xử lý riêng: *thiếu model* (`fetch-models`), *model lệch DB* (`reembed`), *khoá đã tồn tại* (`add-member`), *không có thành viên* (`KeyError`).

---
## 13. Triển khai — danh sách kiểm tra

| ☐ | Việc | Lệnh / nơi |
|---|---|---|
| ☐ | Python ≥ 3.10, venv, `pip install -r requirements.txt` | §2 |
| ☐ | Nạp seed từ notebook, model đúng SHA | `manage.py import-seed …` |
| ☐ | `doctor --camera` toàn `[OK]` | |
| ☐ | Đồng hồ máy đồng bộ NTP, múi giờ Việt Nam | Windows: *Date & time → Sync now* |
| ☐ | Camera đặt đúng (§4), ánh sáng vào mặt | |
| ☐ | Chạy thử `--offline --pilot-dir pilot` 1–2 buổi | §3.6 |
| ☐ | Duyệt `pilot/` → `calibrate probes pilot --apply` | |
| ☐ | Có token portal → `setx PORTAL_TOKEN …` → `portal.enabled=true` → `doctor` thấy `portal /health OK` | §7 |
| ☐ | `sync-members` chạy được; `list --no-face` ra danh sách người cần đăng ký mặt | |
| ☐ | Task Scheduler / systemd tự chạy `scripts/start_service.bat` | §8 |
| ☐ | Lịch `backup` (hằng tuần) + ổ đĩa mã hoá | §8–§9 |
| ☐ | Biển thông báo ghi hình + mẫu đồng ý dữ liệu sinh trắc | §9 |
| ☐ | Xoá `pilot/` sau hiệu chỉnh; bỏ `--pilot-dir` khi chạy lâu dài | §3.6 |
