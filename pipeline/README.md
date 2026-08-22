# HydraPlus-Net & UPAR_UNIFIED: Pedestrian Attribute Recognition (PAR)

Dự án AI/Deep Learning phục vụ bài toán **Nhận diện thuộc tính người đi bộ (Pedestrian Attribute Recognition - PAR)**. Hệ thống tích hợp bộ dữ liệu hợp nhất **UPAR_UNIFIED** (gồm 145,656 ảnh từ 3 tập dữ liệu Market-1501, PA-100K và PETA) và huấn luyện trên mô hình **HydraPlus-Net (ResNet50 + Spatial Attention)** để dự đoán đồng thời **40 thuộc tính nhị phân** của người đi bộ.

Dự án đã được refactor thành cấu trúc mô-đun chuẩn hóa (Modular Architecture), sẵn sàng cho việc đóng gói, phát triển mở rộng và tích hợp CI/CD.

---

## 1. Giới thiệu

### Bài toán Computer Vision
**Pedestrian Attribute Recognition (PAR)** là một bài toán phân loại đa nhãn (Multi-Label Binary Classification) trong thị giác máy tính. Cho đầu vào là một ảnh cắt người đi bộ (pedestrian crop image), mô hình có nhiệm vụ tự động dự đoán danh sách các thuộc tính ngoại hình và phụ kiện của người đó.

### Ý nghĩa thực tế
* **Hệ thống giám sát an ninh (CCTV/Surveillance)**: Cho phép tìm kiếm người dựa trên mô tả văn bản (ví dụ: *"tìm người nam, mặc áo xanh, mặc quần đen, đeo ba lô"*).
* **Định danh lại người (Person Re-Identification - ReID)**: Bổ sung thông tin ngữ nghĩa (semantic attributes) giúp tăng độ chính xác khi theo vết mục tiêu qua nhiều camera.
* **Phân tích hành vi & quản lý đám đông**: Hỗ trợ thu thập thống kê mật độ, độ tuổi, giới tính trong các khu vực công cộng.

### Input & Output của Mô hình
* **Đầu vào (Input)**: Ảnh người đi bộ có kích thước $256 \times 128$ (3 kênh màu RGB).
* **Đầu ra (Output)**: Vector 40 chiều đại diện cho xác suất của **40 UPAR Attributes** chuẩn hóa từ `00 Age-Young` đến `39 Accessory-Hat`.

---

## 2. Cấu trúc Project

```text
D:\AI DATASET/
├── README.md                           # Tài liệu hướng dẫn sử dụng dự án
├── requirements.txt                    # Danh sách thư viện phụ thuộc Python
├── .gitignore                          # Cấu hình bỏ qua tệp tạm, dữ liệu nặng và checkpoints
│
├── UPAR_UNIFIED/                       # Thư mục bộ dữ liệu hợp nhất UPAR
│   └── annotations/                    # Các tệp nhãn pickle chuẩn hóa (unified_annotations.pkl, train.pkl, val.pkl, test.pkl)
│
├── configs/                            # Cấu hình tham số hệ thống
│   └── upar.yaml                       # File cấu hình YAML (Dataset, Model, Training Hyperparameters)
│
├── data/                               # Thư mục lưu trữ hướng dẫn dữ liệu local
│   └── README.md                       # Hướng dẫn đặt bộ ảnh gốc Market-1501, PA-100K, PETA
│
├── datasets/                           # Quản lý nạp và chuẩn hóa dữ liệu
│   └── upar/
│       ├── __init__.py
│       ├── loader.py                   # PyTorch Dataset Loader & ImageNet Transformations
│       ├── mapping/                    # Tệp JSON ánh xạ thuộc tính (Market1501, PA100K, PETA)
│       └── scripts/                    # Scripts gộp & kiểm tra nhãn (merge_datasets.py, validate_*.py)
│
├── models/                             # Định nghĩa kiến trúc mạng Neural Networks
│   └── hydraplus/
│       ├── __init__.py
│       ├── backbone.py                 # Feature Extractor ResNet50 (Pretrained & Custom)
│       └── par_model.py                # UnifiedPARModel & Spatial Attention Module (MDA)
│
├── training/                           # Huấn luyện, Loss function & Evaluation
│   ├── __init__.py
│   ├── train.py                        # Pipeline huấn luyện chính (AMP, Checkpointing, Reports)
│   ├── loss.py                         # MaskedBCEWithLogitsLoss (Khử nhãn chưa xác định = 2)
│   └── evaluate.py                     # Hàm tính toán chỉ số mA, Accuracy, Precision, Recall, F1
│
├── inference/                          # Suy luận & Trực quan hóa kết quả
│   ├── __init__.py
│   └── predict_image.py                # Script nhận diện ảnh đơn lẻ & xuất đồ họa đối chiếu side-by-side
│
├── tests/                              # Bộ kiểm thử tự động (Unit Tests)
│   ├── test_dataset.py                 # Test kiểm tra nạp dữ liệu UPAR
│   ├── test_model.py                   # Test kiểm tra Model Forward Pass
│   └── test_training.py                # Test kiểm tra Backward Gradient & Masked Loss
│
├── checkpoints/                        # Nơi lưu tệp trọng số mô hình (.pth)
│   └── .gitkeep
│
├── reports/                            # Nơi xuất báo cáo kết quả huấn luyện
│   └── .gitkeep
│
└── docs/                               # Tài liệu thiết kế kỹ thuật
    └── architecture.md                 # Giải thích chi tiết kiến trúc HydraPlus-Net & Loss Masking
```

### Chi tiết các tệp quan trọng

| Thành phần / File               | Chức năng                                                         | Input / Output                                                                            | Có thể chỉnh sửa                   |
| :------------------------------ | :---------------------------------------------------------------- | :---------------------------------------------------------------------------------------- | :--------------------------------- |
| `UPAR_UNIFIED/annotations/`     | Thư mục chứa 4 tệp pickle nhãn (`unified_annotations`, `train`, `val`, `test`) | Input: Data annotation pickle<br>Output: Ma trận nhãn `[N, 40]`                           | Tạo lại qua script `merge_datasets`|
| `datasets/upar/loader.py`       | Nạp ảnh & pickle nhãn, thực hiện tiền xử lý `[3, 256, 128]`       | Input: Pickle annotation<br>Output: `(image_tensor, label_tensor, dataset_id, rel_path)` | Thêm/sửa Data Augmentation         |
| `models/hydraplus/backbone.py`  | Tạo backbone ResNet50 trích xuất đặc trưng                        | Input: `[B, 3, 256, 128]`<br>Output: Feature map `[B, 2048, 8, 4]`                        | Đổi backbone (ResNet34/DenseNet...) |
| `models/hydraplus/par_model.py` | Định nghĩa kiến trúc HydraPlus-Net & Spatial Attention            | Input: Feature map<br>Output: Logits `[B, 40]`                                            | Thay đổi số lượng thuộc tính       |
| `training/loss.py`              | Hàm loss `MaskedBCEWithLogitsLoss` khử giá trị nhãn `2`           | Input: `logits [B, 40]`, `targets [B, 40]`<br>Output: Scalar Loss value                   | Điều chỉnh `pos_weight`            |
| `training/evaluate.py`          | Tính mA (mean Accuracy), Accuracy, Precision, Recall, F1          | Input: `logits`, `targets`<br>Output: Dictionary chỉ số                                   | Thay đổi ngưỡng threshold          |
| `training/train.py`             | Điều phối huấn luyện, validation, checkpointing & export report   | Input: CLI arguments / `configs/upar.yaml`<br>Output: Model `.pth` & Report files        | Thay đổi `batch_size`, `epochs`    |
| `inference/predict_image.py`    | Chạy dự đoán trên ảnh bất kỳ, vẽ biểu đồ đối chiếu 2 bên          | Input: Đường dẫn ảnh / tên ảnh<br>Output: Terminal output & ảnh `result_<name>.png`       | Thay đổi ngưỡng threshold          |

---

## 3. Dataset

Project sử dụng bộ dữ liệu **UPAR_UNIFIED**, được hợp nhất từ 3 bộ dữ liệu chuẩn quốc tế (loại bỏ RAP2):

| Dataset gốc    | Mục đích              | Số lượng ảnh | Dataset ID | Tỷ lệ trong UPAR_UNIFIED |
| :------------- | :-------------------- | -----------: | :--------: | -----------------------: |
| **Market-1501** | Training / Val / Test |       29,382 |     0      |                   20.17% |
| **PA-100K**     | Training / Val / Test |       98,909 |     1      |                   67.90% |
| **PETA**       | Training / Val / Test |       17,365 |     2      |                   11.93% |
| **Tổng cộng**  | **UPAR_UNIFIED**      |  **145,656** |    **-**   |              **100.00%** |

### Phân chia Dataset (Dataset Split)
* **Training Set**: 100,593 ảnh (70%)
* **Validation Set**: 15,021 ảnh (10%)
* **Test Set**: 30,042 ảnh (20%)

### Danh sách 40 Thuộc tính UPAR (40 UPAR Attributes)

|  ID  | Tên thuộc tính (Attribute Name) |  ID  | Tên thuộc tính (Attribute Name) |
| :--: | :------------------------------ | :--: | :------------------------------ |
| **00** | Age-Young                      | **20** | LowerBody-Length-Short          |
| **01** | Age-Adult                      | **21** | LowerBody-Color-Black           |
| **02** | Age-Old                        | **22** | LowerBody-Color-Blue            |
| **03** | Gender-Female                  | **23** | LowerBody-Color-Brown           |
| **04** | Hair-Length-Short              | **24** | LowerBody-Color-Green           |
| **05** | Hair-Length-Long               | **25** | LowerBody-Color-Grey            |
| **06** | Hair-Length-Bald               | **26** | LowerBody-Color-Orange          |
| **07** | UpperBody-Length-Short         | **27** | LowerBody-Color-Pink            |
| **08** | UpperBody-Color-Black          | **28** | LowerBody-Color-Purple          |
| **09** | UpperBody-Color-Blue           | **29** | LowerBody-Color-Red             |
| **10** | UpperBody-Color-Brown          | **30** | LowerBody-Color-White           |
| **11** | UpperBody-Color-Green          | **31** | LowerBody-Color-Yellow          |
| **12** | UpperBody-Color-Grey           | **32** | LowerBody-Color-Other           |
| **13** | UpperBody-Color-Orange         | **33** | LowerBody-Type-Trousers&Shorts  |
| **14** | UpperBody-Color-Pink           | **34** | LowerBody-Type-Skirt&Dress      |
| **15** | UpperBody-Color-Purple         | **35** | Accessory-Backpack              |
| **16** | UpperBody-Color-Red            | **36** | Accessory-Bag                   |
| **17** | UpperBody-Color-White          | **37** | Accessory-Glasses-Normal        |
| **18** | UpperBody-Color-Yellow         | **38** | Accessory-Glasses-Sun           |
| **19** | UpperBody-Color-Other          | **39** | Accessory-Hat                   |

### Định dạng Annotation & Quy tắc Nhãn Chưa Xác Định (Unknown Label Handling)
* Trong tệp pickle nhãn (`unified_annotations.pkl`), ma trận nhãn có kích thước $[145656 \times 40]$.
* Giá trị nhãn gồm:
  * `1.0`: Có thuộc tính (Positive).
  * `0.0`: Không có thuộc tính (Negative).
  * `2.0`: Nhãn chưa xác định/thiếu thông tin (Unknown/Unannotated).
* **Quy tắc quan trọng**: Tuyệt đối **KHÔNG** chuyển nhãn `2.0` thành `0.0`. Hàm `MaskedBCEWithLogitsLoss` tự động tạo mặt nạ masking (`targets != 2.0`) để loại bỏ hoàn toàn các vị trí nhãn `2.0` khỏi việc tính Loss và Gradient, giữ nguyên bản chất dữ liệu gốc.

---

## 4. Model Architecture

Project áp dụng mô hình **UnifiedPARModel** dựa trên kiến trúc **HydraPlus-Net**:

```text
Input Pedestrian Image (256 x 128 x 3)
                 ↓
      ImageNet Preprocessing (Normalize)
                 ↓
        Backbone (ResNet50)
                 ↓
     Feature Map (2048 x 8 x 4)
                 ↓
   Spatial Attention Module (MDA)
                 ↓
  Adaptive Average Pooling (2048 -> 512)
                 ↓
       LayerNorm + Linear Classifier
                 ↓
      Logits Output Tensor [B, 40]
                 ↓
          Sigmoid Activation
                 ↓
Attribute Probabilities [0.0 - 1.0]
```

---

## 5. Training

### Cấu hình mặc định
* **Optimizer**: AdamW (`lr = 0.0003`, `weight_decay = 1e-4`)
* **Batch Size**: 64 (hoặc 32 tùy dung lượng GPU)
* **Epochs**: 10 Epochs
* **Mixed Precision (AMP)**: Bật (`torch.cuda.amp.autocast`) giúp tăng tốc độ huấn luyện trên GPU
* **Image Size**: $256 \times 128$

### Câu lệnh chạy Huấn luyện (Training Command)

Chạy trực tiếp từ thư mục gốc project `D:\AI DATASET`:

```powershell
python training/train.py --epochs 10 --batch_size 64
```

Nếu chạy qua Virtual Environment `.venv`:
```powershell
& "D:\AI_Project\.venv\Scripts\python.exe" "training/train.py" --epochs 10 --batch_size 64
```

---

## 6. Evaluation / Testing

Sau mỗi epoch và khi kết thúc huấn luyện, mô hình được đánh giá tự động trên **Test Set (30,042 ảnh)**.

### Kết quả kiểm thử thực tế (10 Epochs Baseline)

```text
========================================
10-EPOCH TRAINING REPORT
========================================
Best Validation mA   : 73.09%  (Best Epoch: 3)
Test mA              : 61.78%
Test Accuracy        : 83.39%
Test Precision       : 48.81%
Test Recall          : 36.60%
Test F1-score        : 36.28%
========================================
```

Các tệp báo cáo chi tiết tự động lưu tại `D:\AI DATASET\reports\`:
* [training_report.txt](file:///d:/AI%20DATASET/reports/training_report.txt)
* [metrics.csv](file:///d:/AI%20DATASET/reports/metrics.csv)
* [per_attribute_metrics.csv](file:///d:/AI%20DATASET/reports/per_attribute_metrics.csv)

---

## 7. Inference trên ảnh mới

Project cung cấp tệp script **`inference/predict_image.py`** hỗ trợ nhận diện thuộc tính trên một ảnh bất kỳ và tự động xuất **biểu đồ đối chiếu trực quan 2 bên (Side-by-side Visual Report)**.

### Câu lệnh chạy Inference

```powershell
python inference/predict_image.py --image "TÊN_ẢNH_HOẶC_ĐƯỜNG_DẪN"
```

#### Ví dụ 1: Nhận diện ảnh trong dataset PETA (Script tự tìm vị trí):
```powershell
python inference/predict_image.py --image "0007_002.jpg"
```

#### Ví dụ 2: Nhận diện ảnh phân biệt rõ thư mục con (ví dụ: `i-LID` vs `CAVIAR4REID`):
```powershell
python inference/predict_image.py --image "i-LID/0007_002.jpg"
```

#### Ví dụ 3: Nhận diện ảnh bất kỳ từ máy tính:
```powershell
python inference/predict_image.py --image "C:\Users\ADMIN\Desktop\pedestrian.jpg"
```

---

## 8. Unit Testing (Bộ kiểm thử tự động)

Dự án cung cấp bộ test tự động trong thư mục `tests/` để xác nhận tính toàn vẹn của mô hình và dữ liệu:

### 1. Kiểm tra DataLoader
```powershell
python tests/test_dataset.py
```

### 2. Kiểm tra Forward Pass của Mô hình
```powershell
python tests/test_model.py
```

### 3. Kiểm tra Forward + Loss + Backward Pass (Gradient)
```powershell
python tests/test_training.py
```

---

## 9. Installation

### Bước 1 — Clone / Chuyển vào thư mục Project
```powershell
cd "D:\AI DATASET"
```

### Bước 2 — Tạo & Kích hoạt Virtual Environment
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Bước 3 — Cài đặt Dependencies
```powershell
pip install -r requirements.txt
```

---

## 10. System Requirements

| Thành phần           | Yêu cầu tối thiểu     | Yêu cầu khuyến nghị               |
| :------------------- | :-------------------- | :-------------------------------- |
| **Hệ điều hành (OS)**| Windows 10/11 64-bit  | Windows 11 / Linux Ubuntu 22.04   |
| **Python**           | 3.10+                 | Python 3.11                       |
| **PyTorch**          | 2.0.0+                | PyTorch 2.4+ (CUDA 12.1)          |
| **GPU**              | NVIDIA GTX 1660 (6GB) | NVIDIA RTX 3060 / RTX 4070 (8GB+) |
| **RAM**              | 16 GB                 | 32 GB                             |
| **Dung lượng đĩa**   | 15 GB free space      | SSD 30 GB free space              |

---

## 11. Pretrained Weights / Checkpoints

Các tệp Checkpoints được lưu tự động tại thư mục `D:\AI DATASET\checkpoints\`:

* **Best Checkpoint**: `checkpoints/hydraplus_upar_best.pth` (Mô hình đạt mA tốt nhất trên tập Validation).
* **Epoch Checkpoints**: `checkpoints/hydraplus_upar_epoch_001.pth` đến `010.pth`.

---

## 12. Configuration Parameters

Các tham số có thể tùy chỉnh trực tiếp qua CLI trong file `training/train.py` hoặc qua file cấu hình `configs/upar.yaml`:

| Parameter        | Ý nghĩa                          | Mặc định                    | Ví dụ tùy chỉnh                               |
| :--------------- | :------------------------------- | :-------------------------: | :-------------------------------------------- |
| `--unified_root` | Thư mục chứa dữ liệu UPAR_UNIFIED| `D:\AI DATASET\UPAR_UNIFIED`| `--unified_root "D:\AI DATASET\UPAR_UNIFIED"` |
| `--epochs`       | Số lượt huấn luyện               | `5`                         | `--epochs 10`                                 |
| `--batch_size`   | Kích thước batch ảnh             | `32`                        | `--batch_size 64`                             |
| `--lr`           | Learning rate ban đầu            | `0.0003`                    | `--lr 0.0001`                                 |
| `--backbone`     | Kiến trúc Backbone               | `resnet50`                  | `--backbone resnet50`                         |
| `--img_height`   | Chiều cao ảnh đầu vào            | `256`                       | `--img_height 256`                            |
| `--img_width`    | Chiều rộng ảnh đầu vào           | `128`                       | `--img_width 128`                             |

---

## 13. Troubleshooting

### Lỗi 1: `ModuleNotFoundError: No module named 'matplotlib'` khi chạy terminal
* **Nguyên nhân**: Bạn đang chạy lệnh bằng Python mặc định của hệ thống thay vì Python của Virtual Environment `.venv`.
* **Cách xử lý**: Kích hoạt `.venv` bằng `.\.venv\Scripts\Activate.ps1` hoặc gọi trực tiếp:
  ```powershell
  D:\AI_Project\.venv\Scripts\python.exe inference/predict_image.py --image "0007_002.jpg"
  ```

### Lỗi 2: `CUDA out of memory`
* **Nguyên nhân**: GPU không đủ VRAM khi đặt Batch Size quá lớn.
* **Cách xử lý**: Giảm batch size xuống 32 hoặc 16:
  ```powershell
  python training/train.py --batch_size 32
  ```

### Lỗi 3: `FileNotFoundError: Missing annotation file`
* **Nguyên nhân**: Chưa tạo tệp `unified_annotations.pkl`.
* **Cách xử lý**: Chạy script tạo dữ liệu hợp nhất:
  ```powershell
  python datasets/upar/scripts/merge_datasets.py
  ```

---

## 14. Component Overview

```text
UPAR_UNIFIED Dataset (145,656 ảnh)
   ↓
datasets/upar/loader.py (Tải ảnh & Pickle nhãn từ UPAR_UNIFIED/annotations/)
   ↓
Preprocessing & Normalization (Resize 256x128, ImageNet Mean/Std)
   ↓
models/hydraplus/backbone.py (ResNet50 Feature Map 2048x8x4)
   ↓
models/hydraplus/par_model.py (Spatial Attention Module + Classifier)
   ↓
training/loss.py (MaskedBCEWithLogitsLoss - Triệt tiêu nhãn 2.0)
   ↓
training/evaluate.py (Tính mA, Precision, Recall, F1 trên 40 Attributes)
   ↓
inference/predict_image.py (Xuất bảng Terminal & Biểu đồ đối chiếu PNG)
```

---

## 15. Technology Stack

* **Ngôn ngữ**: Python 3.11
* **Deep Learning Framework**: PyTorch, torchvision
* **Xử lý ảnh & Dữ liệu**: PIL (Pillow), OpenCV, NumPy, EasyDict, PyYAML
* **Trực quan hóa & Báo cáo**: Matplotlib
* **Tăng tốc phần cứng**: NVIDIA CUDA 12.1, Mixed Precision (AMP)

---

## 16. References

1. **UPAR Challenge**: [UPAR: Unified Pedestrian Attribute Recognition Dataset](https://upar.gi.de/)
2. **HydraPlus-Net Paper**: Liu et al., *"HydraPlus-Net: Attentive Deep Features for Pedestrian Analysis"*, ICCV 2017.
3. **Market-1501 Dataset**: Zheng et al., *"Scalable Person Re-identification: A Benchmark"*, ICCV 2015.
4. **PA-100K Dataset**: Liu et al., *"HydraPlus-Net: Attentive Deep Features for Pedestrian Analysis"*, ICCV 2017.
5. **PETA Dataset**: Deng et al., *"Pedestrian Attribute Recognition At Far Distance"*, ACM MM 2014.

---

## 18. Conclusion

Dự án đã được refactor hoàn tất sang mô hình cấu trúc mô-đun tiêu chuẩn:
* Tách biệt hoàn toàn các nhiệm vụ: `datasets/`, `models/`, `training/`, `inference/`, `tests/`, `configs/`, `checkpoints/`, `reports/`, `docs/`.
* Giữ nguyên 100% logic thuật toán, mô hình HydraPlus-Net và cơ chế **Loss Masking**.
* Sẵn sàng kiểm thử tự động với bộ 3 unit test trong thư mục `tests/`.
