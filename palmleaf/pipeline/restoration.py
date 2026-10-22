import cv2
import torch
import torch.nn as nn
import numpy as np

class MiniUNetRestorer(nn.Module):
    def __init__(self):
        super(MiniUNetRestorer, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))

class ImageRestorationPipeline:
    def __init__(self):
        self.model = MiniUNetRestorer()
        self.model.eval()

    def restore(self, preprocessed_img: np.ndarray, output_path: str) -> str:
        img_norm = preprocessed_img.astype(np.float32) / 255.0
        h, w = img_norm.shape
        target_h, target_w = (h // 16) * 16, (w // 16) * 16
        resized = cv2.resize(img_norm, (target_w, target_h))
        tensor_in = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0)
        with torch.no_grad():
            tensor_out = self.model(tensor_in)
            restored_np = tensor_out.squeeze().numpy()
        restored_scaled = (cv2.resize(restored_np, (w, h)) * 255).astype(np.uint8)
        final_thresh = cv2.adaptiveThreshold(
            restored_scaled, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, 8
        )
        cv2.imwrite(output_path, final_thresh)
        return output_path
