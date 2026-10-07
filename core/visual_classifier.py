"""
AcciVision — Module Thị Giác Xác Thực Tai Nạn (Visual Accident Verifier)
Sử dụng MobileNetV3 để xác thực kép hình ảnh hiện trường va chạm từ frame video.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms


class VisualAccidentVerifier:
    """
    Xác thực hình ảnh va chạm bằng mô hình CNN đã huấn luyện.
    """

    def __init__(
        self,
        model_path: str = "models/visual_accident_classifier.pth",
        threshold: float = 0.40,
        device: str = "cpu",
    ) -> None:
        self.model_path = Path(model_path)
        self.threshold = threshold
        self.device = torch.device(device)
        self.model: Optional[nn.Module] = None
        self.enabled = False

        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        if self.model_path.exists():
            try:
                self._load_model()
                self.enabled = True
            except Exception as e:
                print(f"[WARN] Cannot load visual classifier: {e}")
        else:
            print(f"[WARN] Visual classifier model not found at {self.model_path}")

    def _load_model(self) -> None:
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        model = models.mobilenet_v3_small(weights=None)
        in_features = model.classifier[0].in_features
        model.classifier = nn.Sequential(
            nn.Linear(in_features, 256),
            nn.Hardswish(),
            nn.Dropout(p=0.2),
            nn.Linear(256, 2),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(self.device)
        model.eval()
        self.model = model
        self.acc_index = checkpoint.get("acc_index", 0)

    @torch.no_grad()
    def verify_crop(self, frame: np.ndarray, bbox: Optional[np.ndarray] = None) -> float:
        """
        Dự đoán xác suất tai nạn từ ảnh crop của xe hoặc toàn cảnh va chạm.
        Trả về: float từ 0.0 đến 1.0 (Xác suất hình ảnh có va chạm / biến dạng).
        """
        if not self.enabled or self.model is None or frame is None or frame.size == 0:
            return 0.5  # Neutral default

        h, w = frame.shape[:2]

        if bbox is not None and len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            # Mở rộng vùng crop 20% để lấy thêm bối cảnh xung quanh xe
            pad_w = (x2 - x1) * 0.20
            pad_h = (y2 - y1) * 0.20
            cx1 = max(0, int(x1 - pad_w))
            cy1 = max(0, int(y1 - pad_h))
            cx2 = min(w, int(x2 + pad_w))
            cy2 = min(h, int(y2 + pad_h))

            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0 or (cx2 - cx1 < 10) or (cy2 - cy1 < 10):
                crop = frame
        else:
            crop = frame

        # BGR (OpenCV) -> RGB
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        tensor = self.transform(crop_rgb).unsqueeze(0).to(self.device)

        logits = self.model(tensor)
        probs = torch.softmax(logits, dim=1)
        accident_prob = float(probs[0, self.acc_index].item())
        return accident_prob

    def is_visual_accident(self, frame: np.ndarray, bbox: Optional[np.ndarray] = None) -> Tuple[bool, float]:
        prob = self.verify_crop(frame, bbox)
        return (prob >= self.threshold), prob
