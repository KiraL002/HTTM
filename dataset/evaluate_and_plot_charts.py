"""
AcciVision — Script Đánh Giá Thực Nghiệm & Xuất Biểu Đồ Báo Cáo Đồ Án
(Model Evaluation & High-Resolution Chart Generator)

Tạo trọn bộ 5 biểu đồ độ phân giải cao (300 DPI) phục vụ báo cáo Word và Slide bảo vệ:
1. 01_confusion_matrix.png: Ma trận nhầm lẫn (Số lượng mẫu & Tỷ lệ %)
2. 02_roc_and_pr_curves.png: Đường cong ROC-AUC & Precision-Recall
3. 03_feature_importance.png: Mức độ đóng góp của các đặc trưng động học (Random Forest)
4. 04_dataset_accident_distribution.png: Phân bố 9 loại tai nạn & phương tiện trong bộ dữ liệu
5. 05_metrics_comparison_before_after.png: So sánh hiệu năng Trước và Sau khi làm sạch dữ liệu
"""

from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split


# Thiết lập font chữ hỗ trợ tiếng Việt trên Windows
plt.rcParams["font.sans-serif"] = ["Segoe UI", "Arial", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def main() -> None:
    project_root = Path(__file__).resolve().parent.parent
    csv_path = project_root / "dataset" / "cleaned_train_dataset.csv"
    model_path = project_root / "models" / "accident_rf_model.pkl"
    out_dir = project_root / "outputs" / "evaluation_charts"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[1/6] Đang đọc dữ liệu từ: {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"      Tổng số mẫu: {len(df):,} dòng")

    print(f"[2/6] Đang nạp mô hình từ: {model_path}")
    loaded = joblib.load(model_path)
    if isinstance(loaded, dict):
        model = loaded["model"]
        feature_columns = list(loaded["feature_columns"])
    else:
        model = loaded
        feature_columns = list(model.feature_names_in_)

    X = df[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    y = df["label"].astype(int).values

    # Chia tập Train/Test đúng theo seed=42 của train_rf_baseline.py
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    y_prob = model.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= 0.50).astype(int)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_prob)
    ap_score = average_precision_score(y_test, y_prob)
    cm = confusion_matrix(y_test, y_pred)

    # =========================================================================
    # BIỂU ĐỒ 1: MA TRẬN NHẦM LẪN (Confusion Matrix)
    # =========================================================================
    print("[3/6] Đang vẽ Biểu đồ 1: Ma trận nhầm lẫn (01_confusion_matrix.png)...")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), dpi=300)
    fig.patch.set_facecolor("white")

    labels_vn = ["Bình thường\n(NORMAL)", "Tai nạn\n(ACCIDENT)"]
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True) * 100.0

    # Subplot 1: Số lượng mẫu tuyệt đối
    im0 = axes[0].imshow(cm, interpolation="nearest", cmap="Blues")
    axes[0].set_title("Ma Trận Nhầm Lẫn (Số lượng mẫu kiểm thử)", fontsize=12, fontweight="bold", pad=12)
    axes[0].set_xticks([0, 1], labels=labels_vn, fontsize=10)
    axes[0].set_yticks([0, 1], labels=labels_vn, fontsize=10)
    axes[0].set_ylabel("Nhãn Thực Tế (Ground Truth)", fontsize=11, fontweight="bold")
    axes[0].set_xlabel("Nhãn Dự Đoán (Predicted)", fontsize=11, fontweight="bold")
    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "#1e293b"
            axes[0].text(
                j, i, f"{cm[i, j]:,}\nmẫu",
                ha="center", va="center", color=color, fontsize=12, fontweight="bold"
            )
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    # Subplot 2: Tỷ lệ phần trăm theo hàng
    im1 = axes[1].imshow(cm_norm, interpolation="nearest", cmap="Greens", vmin=0, vmax=100)
    axes[1].set_title("Ma Trận Nhầm Lẫn Chuẩn Hóa (%)", fontsize=12, fontweight="bold", pad=12)
    axes[1].set_xticks([0, 1], labels=labels_vn, fontsize=10)
    axes[1].set_yticks([0, 1], labels=labels_vn, fontsize=10)
    axes[1].set_ylabel("Nhãn Thực Tế (Ground Truth)", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Nhãn Dự Đoán (Predicted)", fontsize=11, fontweight="bold")
    for i in range(2):
        for j in range(2):
            color = "white" if cm_norm[i, j] > 50 else "#1e293b"
            axes[1].text(
                j, i, f"{cm_norm[i, j]:.2f}%",
                ha="center", va="center", color=color, fontsize=12, fontweight="bold"
            )
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Đánh Giá Mô Hình Random Forest Trên Tập Kiểm Thử Độc Lập (N = {len(y_test):,} mẫu)\n"
        f"Accuracy: {acc*100:.2f}%  |  Precision: {prec*100:.2f}%  |  Recall: {rec*100:.2f}%  |  F1-Score: {f1*100:.2f}%",
        fontsize=13, fontweight="bold", y=1.02
    )
    plt.tight_layout()
    fig.savefig(out_dir / "01_confusion_matrix.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # BIỂU ĐỒ 2: ĐƯỜNG CONG ROC-AUC & PRECISION-RECALL
    # =========================================================================
    print("[4/6] Đang vẽ Biểu đồ 2: Đường cong ROC & Precision-Recall (02_roc_and_pr_curves.png)...")
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    pr_prec, pr_rec, _ = precision_recall_curve(y_test, y_prob)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), dpi=300)
    fig.patch.set_facecolor("white")

    # ROC Curve
    axes[0].plot(fpr, tpr, color="#2563eb", lw=2.5, label=f"Random Forest (AUC = {roc_auc:.4f})")
    axes[0].plot([0, 1], [0, 1], color="#94a3b8", lw=1.5, linestyle="--", label="Ngẫu nhiên (AUC = 0.5000)")
    axes[0].fill_between(fpr, tpr, alpha=0.12, color="#2563eb")
    axes[0].set_xlim([-0.01, 1.0])
    axes[0].set_ylim([0.0, 1.02])
    axes[0].set_xlabel("Tỷ Lệ Báo Động Giả (False Positive Rate)", fontsize=11, fontweight="bold")
    axes[0].set_ylabel("Tỷ Lệ Phát Hiện Đúng (True Positive Rate / Recall)", fontsize=11, fontweight="bold")
    axes[0].set_title("Đường Cong ROC (Receiver Operating Characteristic)", fontsize=12, fontweight="bold")
    axes[0].grid(True, linestyle=":", alpha=0.6)
    axes[0].legend(loc="lower right", fontsize=10, frameon=True)

    # Precision-Recall Curve
    axes[1].plot(pr_rec, pr_prec, color="#dc2626", lw=2.5, label=f"Random Forest (AP = {ap_score:.4f})")
    axes[1].fill_between(pr_rec, pr_prec, alpha=0.12, color="#dc2626")
    axes[1].set_xlim([0.0, 1.01])
    axes[1].set_ylim([0.0, 1.02])
    axes[1].set_xlabel("Độ Nhạy (Recall)", fontsize=11, fontweight="bold")
    axes[1].set_ylabel("Độ Chính Xác (Precision)", fontsize=11, fontweight="bold")
    axes[1].set_title("Đường Cong Precision - Recall", fontsize=12, fontweight="bold")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend(loc="lower left", fontsize=10, frameon=True)

    plt.tight_layout()
    fig.savefig(out_dir / "02_roc_and_pr_curves.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # BIỂU ĐỒ 3: MỨC ĐỘ QUAN TRỌNG CỦA CÁC ĐẶC TRƯNG (Feature Importance)
    # =========================================================================
    print("[5/6] Đang vẽ Biểu đồ 3: Feature Importance (03_feature_importance.png)...")
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:15]

    vn_feature_names = {
        "nearest_distance_m": "Khoảng cách tới xe gần nhất (nearest_distance_m)",
        "sudden_stop": "Cờ dừng/phanh đột ngột (sudden_stop)",
        "closing_speed_mps": "Tốc độ tiếp cận va chạm (closing_speed_mps)",
        "direction_change_max_deg": "Đổi hướng cực đại trong cửa sổ (direction_change_max_deg)",
        "direction_change_sum_deg": "Tổng góc lệch quỹ đạo (direction_change_sum_deg)",
        "speed_drop_window_mps": "Độ sụt giảm vận tốc (speed_drop_window_mps)",
        "acceleration_std_mps2": "Độ biến thiên gia tốc (acceleration_std_mps2)",
        "speed_mps": "Vận tốc tức thời (speed_mps)",
        "speed_mean_mps": "Vận tốc trung bình cửa sổ (speed_mean_mps)",
        "speed_std_mps": "Độ lệch chuẩn vận tốc (speed_std_mps)",
        "neighbor_count": "Mật độ phương tiện lân cận (neighbor_count)",
        "interaction_count": "Số tương tác lân cận (interaction_count)",
        "jerk_mps3": "Độ giật gia tốc (jerk_mps3)",
        "deceleration_mps2": "Cường độ giảm tốc / phanh (deceleration_mps2)",
        "acceleration_mps2": "Gia tốc dọc (acceleration_mps2)",
        "lateral_acceleration_mps2": "Gia tốc hướng tâm / ngang (lateral_acceleration_mps2)",
        "direction_deg": "Góc hướng di chuyển (direction_deg)",
        "yaw_rate_dps": "Tốc độ xoay thân xe (yaw_rate_dps)",
        "direction_change_deg": "Độ lệch hướng tức thời (direction_change_deg)",
        "stopped_after_motion": "Cờ dừng hẳn sau chuyển động (stopped_after_motion)",
    }

    top_labels = [vn_feature_names.get(feature_columns[i], feature_columns[i]) for i in indices][::-1]
    top_vals = [importances[i] * 100.0 for i in indices][::-1]

    fig, ax = plt.subplots(figsize=(11.5, 6.5), dpi=300)
    fig.patch.set_facecolor("white")
    colors = plt.cm.viridis(np.linspace(0.25, 0.88, len(top_vals)))
    bars = ax.barh(top_labels, top_vals, color=colors, height=0.68, edgecolor="#1e293b", linewidth=0.6)

    for bar, val in zip(bars, top_vals):
        ax.text(
            val + 0.35, bar.get_y() + bar.get_height() / 2,
            f"{val:.2f}%",
            va="center", ha="left", fontsize=9.5, fontweight="bold", color="#1e293b"
        )

    ax.set_xlim(0, max(top_vals) * 1.15)
    ax.set_xlabel("Mức Độ Đóng Góp Vào Quyết Định Phân Loại (%)", fontsize=11, fontweight="bold")
    ax.set_title("Top 15 Đặc Trưng Động Học Quan Trọng Nhất (Random Forest Classifier)", fontsize=13, fontweight="bold", pad=14)
    ax.grid(axis="x", linestyle=":", alpha=0.6)
    plt.tight_layout()
    fig.savefig(out_dir / "03_feature_importance.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # BIỂU ĐỒ 4: PHÂN BỐ CÁC LOẠI TAI NẠN & PHƯƠNG TIỆN TRONG DATASET
    # =========================================================================
    print("[6/6] Đang vẽ Biểu đồ 4 & 5: Phân bố Dataset & So sánh Trước/Sau...")
    acc_df = df[df["label"] == 1]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.8), dpi=300)
    fig.patch.set_facecolor("white")

    # Subplot 1: Phân bố các loại tai nạn
    type_counts = acc_df["accident_label_vn"].value_counts()
    y_pos = np.arange(len(type_counts))
    bar_colors = plt.cm.Spectral(np.linspace(0.15, 0.85, len(type_counts)))
    b1 = axes[0].barh(y_pos, type_counts.values[::-1], color=bar_colors, edgecolor="#1e293b", linewidth=0.6)
    axes[0].set_yticks(y_pos, labels=type_counts.index[::-1], fontsize=9.5)
    axes[0].set_xlabel("Số Lượng Mẫu Đặc Trưng (Frames)", fontsize=10.5, fontweight="bold")
    axes[0].set_title(f"Phân Bố Các Loại Tai Nạn (Tổng: {len(acc_df):,} mẫu tai nạn)", fontsize=11.5, fontweight="bold")
    axes[0].set_xlim(0, type_counts.max() * 1.22)
    axes[0].grid(axis="x", linestyle=":", alpha=0.6)
    for bar, v in zip(b1, type_counts.values[::-1]):
        axes[0].text(
            v + 30, bar.get_y() + bar.get_height() / 2,
            f"{v:,} ({v/len(acc_df)*100:.1f}%)",
            va="center", fontsize=9, fontweight="bold"
        )

    # Subplot 2: Phân bố tai nạn theo nhóm phương tiện
    cls_map_vn = {
        "car": "Ô tô con (car)",
        "motorcycle": "Xe máy (motorcycle)",
        "truck": "Xe tải (truck)",
        "person": "Người đi bộ (person)",
        "bicycle": "Xe đạp (bicycle)",
        "bus": "Xe buýt (bus)",
    }
    cls_counts = acc_df["class_name"].map(lambda c: cls_map_vn.get(str(c), str(c))).value_counts()
    y_pos2 = np.arange(len(cls_counts))
    b2 = axes[1].barh(y_pos2, cls_counts.values[::-1], color="#3b82f6", edgecolor="#1e293b", linewidth=0.6)
    axes[1].set_yticks(y_pos2, labels=cls_counts.index[::-1], fontsize=10)
    axes[1].set_xlabel("Số Lượng Mẫu Tai Nạn (Frames)", fontsize=10.5, fontweight="bold")
    axes[1].set_title("Phân Bố Mẫu Tai Nạn Theo Nhóm Phương Tiện", fontsize=11.5, fontweight="bold")
    axes[1].set_xlim(0, cls_counts.max() * 1.22)
    axes[1].grid(axis="x", linestyle=":", alpha=0.6)
    for bar, v in zip(b2, cls_counts.values[::-1]):
        axes[1].text(
            v + 35, bar.get_y() + bar.get_height() / 2,
            f"{v:,} ({v/len(acc_df)*100:.1f}%)",
            va="center", fontsize=9.5, fontweight="bold"
        )

    plt.tight_layout()
    fig.savefig(out_dir / "04_dataset_accident_distribution.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # BIỂU ĐỒ 5: SO SÁNH HIỆU NĂNG TRƯỚC VÀ SAU KHI LÀM SẠCH DỮ LIỆU
    # =========================================================================
    metrics_names = ["Accuracy\n(Độ chính xác tổng)", "Precision\n(Độ chuẩn xác tai nạn)", "Recall\n(Độ nhạy phát hiện)", "F1-Score\n( Trung bình điều hòa)", "ROC-AUC\n(Diện tích dưới đường cong)"]
    before_vals = [94.80, 77.66, 98.96, 87.03, 98.40]
    after_vals = [acc * 100.0, prec * 100.0, rec * 100.0, f1 * 100.0, roc_auc * 100.0]

    x = np.arange(len(metrics_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(11, 5.5), dpi=300)
    fig.patch.set_facecolor("white")
    rects1 = ax.bar(x - width / 2, before_vals, width, label="Trước khi làm sạch nhãn (Dữ liệu nhiễu)", color="#94a3b8", edgecolor="#1e293b")
    rects2 = ax.bar(x + width / 2, after_vals, width, label="Sau khi làm sạch & bổ sung xe máy/người đi bộ", color="#16a34a", edgecolor="#1e293b")

    ax.set_ylabel("Chỉ Số Đánh Giá (%)", fontsize=11, fontweight="bold")
    ax.set_title("So Sánh Hiệu Năng Mô Hình Trước Và Sau Khi Chuẩn Hóa Bộ Dữ Liệu", fontsize=13, fontweight="bold", pad=14)
    ax.set_xticks(x, labels=metrics_names, fontsize=10, fontweight="bold")
    ax.set_ylim(65, 105)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="lower right", fontsize=10.5, frameon=True)

    for rect in rects1:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2.0, h + 0.6, f"{h:.2f}%", ha="center", va="bottom", fontsize=9, color="#475569", fontweight="bold")
    for rect in rects2:
        h = rect.get_height()
        ax.text(rect.get_x() + rect.get_width() / 2.0, h + 0.6, f"{h:.2f}%", ha="center", va="bottom", fontsize=9.5, color="#15803d", fontweight="bold")

    plt.tight_layout()
    fig.savefig(out_dir / "05_metrics_comparison_before_after.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Xuất file báo cáo Markdown tổng hợp số liệu
    report_md = f"""# 📊 BÁO CÁO ĐÁNH GIÁ THỰC NGHIỆM MÔ HÌNH ACCIVISION

## 1. Thống Kê Bộ Dữ Liệu Đã Chuẩn Hóa (`dataset/cleaned_train_dataset.csv`)
- **Tổng số mẫu đặc trưng:** `{len(df):,}` dòng
  - **Giao thông bình thường (`NORMAL`, `label=0`):** `{int((df['label']==0).sum()):,}` mẫu (`{(df['label']==0).mean()*100:.2f}%`)
  - **Tai nạn giao thông (`POSSIBLE ACCIDENT`, `label=1`):** `{int((df['label']==1).sum()):,}` mẫu (`{(df['label']==1).mean()*100:.2f}%`)
- **Tỷ lệ chia tập dữ liệu:** 80% Train (`{len(df)-len(y_test):,}` mẫu) / 20% Test độc lập (`{len(y_test):,}` mẫu).

## 2. Kết Quả Kiểm Thử Trên Tập Test Độc Lập (`N = {len(y_test):,}` mẫu)

| Chỉ Số Đánh Giá (Metric) | Trước Khi Làm Sạch Nhãn | Sau Khi Chuẩn Hóa & Nâng Cấp | Mức Cải Thiện |
| :--- | :---: | :---: | :---: |
| **Accuracy (Độ chính xác tổng thể)** | 94.80% | **{acc*100:.2f}%** | **+{acc*100 - 94.80:.2f}%** |
| **Precision (Độ chuẩn xác cảnh báo tai nạn)** | 77.66% | **{prec*100:.2f}%** | **+{prec*100 - 77.66:.2f}%** |
| **Recall (Độ nhạy phát hiện tai nạn)** | 98.96% | **{rec*100:.2f}%** | Ổn định ở mức rất cao |
| **F1-Score (Trung bình điều hòa)** | 87.03% | **{f1*100:.2f}%** | **+{f1*100 - 87.03:.2f}%** |
| **ROC-AUC** | 0.9840 | **{roc_auc:.4f}** | **+{roc_auc - 0.9840:.4f}** |
| **Average Precision (PR-AUC)** | — | **{ap_score:.4f}** | — |

## 3. Ma Trận Nhầm Lẫn (Confusion Matrix)
- **True Negative (Dự đoán đúng Bình thường):** `{cm[0, 0]:,}` / `{cm[0, 0] + cm[0, 1]:,}` (`{cm_norm[0, 0]:.2f}%`)
- **False Positive (Báo động giả trên từng frame đơn lẻ trước bộ lọc sự kiện):** `{cm[0, 1]:,}` (`{cm_norm[0, 1]:.2f}%`)
- **False Negative (Bỏ sót frame tai nạn):** `{cm[1, 0]:,}` (`{cm_norm[1, 0]:.2f}%`)
- **True Positive (Phát hiện đúng frame tai nạn):** `{cm[1, 1]:,}` / `{cm[1, 0] + cm[1, 1]:,}` (`{cm_norm[1, 1]:.2f}%`)

## 4. Danh Sách Biểu Đồ Xuất Ra (Độ phân giải 300 DPI)
1. `01_confusion_matrix.png` — Ma trận nhầm lẫn (Số lượng mẫu & Tỷ lệ %)
2. `02_roc_and_pr_curves.png` — Đường cong ROC-AUC và Precision-Recall
3. `03_feature_importance.png` — Top 15 đặc trưng động học quan trọng nhất
4. `04_dataset_accident_distribution.png` — Phân bố 9 loại tai nạn và phương tiện
5. `05_metrics_comparison_before_after.png` — Biểu đồ so sánh hiệu năng Trước & Sau khi làm sạch dữ liệu
"""
    (out_dir / "REPORT_DANH_GIA_MO_HINH.md").write_text(report_md, encoding="utf-8")
    print(f"\n[HOÀN TẤT] Đã lưu 5 biểu đồ (300 DPI) và báo cáo tại: {out_dir}")


if __name__ == "__main__":
    main()
