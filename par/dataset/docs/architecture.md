# Kiến trúc Mô hình HydraPlus-Net & Loss Masking

## 1. Tổng quan Kiến trúc

Mô hình **UnifiedPARModel** kế thừa ý tưởng từ kiến trúc **HydraPlus-Net**, kết hợp một Feature Extractor chuẩn (ResNet50) và module **Spatial Attention (MDA)** để chú ý vào từng vùng cơ thể (đầu, thân trên, thân dưới, chân, phụ kiện).

```text
Input Pedestrian Crop (256 x 128 x 3)
                ↓
    ImageNet Normalization
                ↓
       ResNet50 Backbone
                ↓
  Feature Map (2048 x 8 x 4)
                ↓
 Spatial Attention Module (MDA)
                ↓
 Adaptive Avg Pooling (2048 -> 512)
                ↓
 LayerNorm + Linear Classifier
                ↓
 Logits Output Vector [B, 40]
```

---

## 2. Xử lý Nhãn Chưa Xác Định (Masked BCE Loss)

Trong bài toán UPAR_UNIFIED, giá trị nhãn `2.0` thể hiện thông tin thuộc tính chưa được gán nhãn (Unknown / Unannotated).

Hàm tổn thất `MaskedBCEWithLogitsLoss` áp dụng công thức:

$$\mathcal{L} = \frac{\sum_{i,j} M_{i,j} \cdot \text{BCE}(x_{i,j}, y_{i,j})}{\sum_{i,j} M_{i,j} + \epsilon}$$

Trong đó:
* $M_{i,j} = \mathbb{I}(y_{i,j} \neq 2.0)$ là ma trận mặt nạ nhị phân.
* $\text{BCE}$ là hàm Binary Cross-Entropy tiêu chuẩn.
* Cơ chế này loại bỏ hoàn toàn gradient tại các vị trí nhãn $2.0$, tránh làm lệch phân phối của nhãn $0.0$ và $1.0$.
