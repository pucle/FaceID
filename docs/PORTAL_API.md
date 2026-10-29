# Hợp đồng API portal ⇄ hệ thống điểm danh khuôn mặt

> Tài liệu này dành cho người phát triển **portal.o365hust.club**. Portal hiện chỉ có trang đăng nhập công khai (MSSV/email + mật khẩu + "Xác thực bảo mật"); **không tìm thấy tài liệu API công khai**, nên đây là *đề xuất tối thiểu* để hai hệ thống phối hợp ổn định.
> Client phía khuôn mặt (`portal_client.py`) đã viết sẵn theo hợp đồng này; mọi URL / tên trường chỉnh được trong `config.json → portal` nên **không cần sửa code** nếu portal đặt tên khác.
> Dùng `python tools/mock_portal.py` để chạy thử phía khuôn mặt với một portal giả đúng hợp đồng này.

## 0. Nguyên tắc
| Nguyên tắc | Lý do |
|---|---|
| **Portal là nguồn sự thật** cho hồ sơ (MSSV, họ tên, ban, trạng thái), phiên điểm danh, và kết luận *có mặt/đi muộn*. | Tránh hai nơi lệch dữ liệu. |
| **Hệ thống khuôn mặt chỉ là một "thiết bị điểm danh"**: gửi *sự kiện thô* (ai, lúc nào, độ tin cậy). Portal tự áp quy tắc (cửa sổ check-in, muộn, trùng…). | Đổi luật điểm danh không phải sửa AI. |
| **Khóa nối = MSSV** (chuỗi). | MSSV vừa là tên đăng nhập portal, vừa nằm trong nhãn dataset. |
| **Không đăng nhập bằng form** (có captcha/"xác thực bảo mật"). Dùng **tài khoản dịch vụ + token**. | Tự động hóa form đăng nhập mong manh và dễ vi phạm điều khoản. |
| **Idempotent**: gửi lại cùng một sự kiện không tạo bản ghi thứ hai. | Mạng chập chờn → client sẽ gửi lại. |

## 1. Xác thực
`Authorization: Bearer <token>` — token của **tài khoản dịch vụ** (vai trò `kiosk`), thu hồi/xoay vòng được, quyền tối thiểu:
`members:read`, `sessions:read`, `attendance:write`, `face_status:write`. Khuyến nghị: giới hạn IP của máy chủ CLB, HTTPS bắt buộc, rate-limit, ghi audit log.
(Tuỳ chọn) chữ ký HMAC: header `X-Timestamp` + `X-Signature = HMAC_SHA256(secret, timestamp + "." + body)`.
Mã lỗi: `401/403` → client dừng gửi và giữ hàng đợi (báo lỗi rõ cho người vận hành).

## 2. Endpoint (đường dẫn mặc định, đổi được trong config)
### 2.1 `GET /api/v1/health`
`200 {"ok": true, "time": "2026-10-29T19:00:00+07:00"}` — `time` giúp phát hiện **đồng hồ máy chủ khuôn mặt lệch**.

### 2.2 `GET /api/v1/members?updated_since=<ISO>&page=<n>`
```json
{"results":[{"id":17,"mssv":"20252345","full_name":"Nguyễn Văn A","ban":"CM","status":"active"}],"next":null}
```
`status`: `active` | `inactive` (đã nghỉ/bị khoá). Trả `[]` ở trang rỗng để client dừng. Có thể thêm `face_consent` (bool) + `face_consent_at` nếu portal quản lý đồng ý xử lý dữ liệu sinh trắc.
Client: thành viên mới → tạo **chưa có mặt** (xem `manage.py list --no-face`); `inactive` → loại khỏi gallery nhưng giữ dữ liệu; tên/ban đổi → cập nhật, **không** tính lại embedding.

### 2.3 `GET /api/v1/attendance/sessions?status=open`
```json
{"items":[{"id":"EV-2026-10-30","title":"Sinh hoạt tháng 10","type":"event","start":"2026-10-30T19:00:00+07:00","end":"2026-10-30T21:00:00+07:00","checkin_opens":"2026-10-30T18:30:00+07:00","checkin_closes":"2026-10-30T20:00:00+07:00"}]}
```
Cần: phiên **đang mở** gồm *trực*, *sự kiện*, *họp* (đúng các mục portal nêu: "Đăng ký trực, sự kiện và điểm danh"). Thời gian có múi giờ. Client chọn phiên có `start ≤ now ≤ end`; nếu chỉ có 1 phiên mở thì dùng phiên đó; nếu mơ hồ → gửi `session_id=null` và portal tự gán theo thời gian/thiết bị (hoặc đặt `service.session_id` cố định).

### 2.4 `POST /api/v1/attendance/face-events`  (**quan trọng nhất**)
Headers: `Idempotency-Key: <key>`  ·  Body:
```json
{"idempotency_key":"3f9a…","session_id":"EV-2026-10-30","member_id":"20252345",
 "timestamp":"2026-10-30T19:03:12+07:00","confidence":0.873,"method":"face",
 "device_id":"kiosk-1","frames":4,"model":"multi_max@0.48"}
```
| Mã | Ý nghĩa | Client làm gì |
|---|---|---|
| `201`/`200` `{"id":…,"status":"present\|late"}` | ghi nhận | đánh dấu đã đồng bộ |
| `409` | đã có (cùng key hoặc đã điểm danh phiên này) | coi như thành công |
| `422` `{"code":"not_registered\|session_closed\|unknown_member"}` | sự kiện không hợp lệ | **không gửi lại**, ghi `last_error` để người vận hành xem |
| `429`, `5xx`, timeout | tạm thời | giữ hàng đợi, thử lại sau |
| `401/403` | sai/hết hạn token | dừng, báo lỗi |

Quy tắc phía portal (đề xuất): điểm danh **lần đầu thắng**; `confidence` thấp hơn ngưỡng cảnh báo (vd < τ+0.05) → đánh dấu *cần duyệt* thay vì từ chối.
Tuỳ chọn `POST …/face-events/batch` (≤100 sự kiện) cho đồng bộ sau mất mạng.

### 2.5 `PUT /api/v1/members/{mssv}/face-status`  (tuỳ chọn, rất nên có)
`{"enrolled":true,"samples":8,"updated_at":"…","model":"w600k_mbf"}` → portal hiển thị "đã đăng ký khuôn mặt" để BĐH biết ai còn thiếu.

### 2.6 Phản hồi đúng/sai (tuỳ chọn, để **hiệu chỉnh bằng sự thật thực tế**)
`PATCH /api/v1/attendance/face-events/{id} {"verified":false,"reviewer":"…"}` và `GET …/face-events?verified=false&since=…`.
Mỗi lần BĐH sửa "người này không phải người kia" là một nhãn *nhầm người* quý giá cho `calibrate`. (Chưa được client sử dụng — cần thêm khi portal có endpoint.)

## 3. Phi chức năng cần thống nhất
* **Múi giờ**: ISO-8601 có offset (`+07:00`); portal lưu UTC. Máy chủ khuôn mặt đồng bộ NTP.
* **Độ trễ**: ghi nhận < 2 s; client có hàng đợi bền (SQLite) nên portal có thể bảo trì mà không mất điểm danh.
* **Quyền riêng tư**: *không* gửi ảnh/embedding lên portal. Chỉ gửi MSSV + thời gian + độ tin cậy. Nếu muốn ảnh bằng chứng, phải là tính năng bật riêng, có thời hạn xoá.
* **Đồng ý xử lý dữ liệu sinh trắc**: nên có trường lưu việc đồng ý + thời điểm (xem README §9).
* **Không có API?** Dùng `python manage.py export-csv` (UTF-8 BOM, mở được bằng Excel) để nhập tay/nhập hàng loạt vào portal; hoặc nhờ BĐH thêm 3 endpoint tối thiểu: `members`, `sessions`, `face-events`.
