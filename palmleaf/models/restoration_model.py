"""
restoration_model.py
Lightweight U-Net style Denoising Autoencoder for manuscript restoration.
Works on single-channel (grayscale) 256x256 patches.
"""
import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNetRestorer(nn.Module):
    """
    Lightweight U-Net: encoder-decoder with skip connections.
    Input/Output: (B, 1, 256, 256) float32 in [0, 1].
    """
    def __init__(self, features=(32, 64, 128)):
        super().__init__()
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        in_ch = 1
        for f in features:
            self.encoders.append(ConvBlock(in_ch, f))
            self.pools.append(nn.MaxPool2d(2))
            in_ch = f

        self.bottleneck = ConvBlock(features[-1], features[-1] * 2)

        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        rev = list(reversed(features))
        in_ch = features[-1] * 2
        for f in rev:
            self.upconvs.append(nn.ConvTranspose2d(in_ch, f, 2, stride=2))
            self.decoders.append(ConvBlock(f * 2, f))
            in_ch = f

        self.final = nn.Conv2d(features[0], 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        skips = []
        for enc, pool in zip(self.encoders, self.pools):
            x = enc(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for up, dec, skip in zip(self.upconvs, self.decoders, reversed(skips)):
            x = up(x)
            x = torch.cat([x, skip], dim=1)
            x = dec(x)

        return self.sigmoid(self.final(x))


def get_model(device="cpu"):
    model = UNetRestorer()
    model.to(device)
    return model


def load_weights(model: UNetRestorer, path: str, device: str = "cpu") -> UNetRestorer:
    """
    Load weights into UNetRestorer, handling both:
    - Local key names  : encoders, bottleneck, decoders, upconvs, final
    - Colab key names  : encs, bot, decs, ups, out
    """
    state = torch.load(path, map_location=device)

    # Check if keys match directly
    model_keys = set(model.state_dict().keys())
    ckpt_keys  = set(state.keys())

    if model_keys == ckpt_keys:
        model.load_state_dict(state)
        return model

    # Remap Colab short names -> local full names
    remap = {
        "encs":  "encoders",
        "bot":   "bottleneck",
        "decs":  "decoders",
        "ups":   "upconvs",
        "out":   "final",
        ".b.":   ".block.",
    }
    new_state = {}
    for k, v in state.items():
        new_k = k
        for old, new in remap.items():
            new_k = new_k.replace(old, new)
        new_state[new_k] = v

    model.load_state_dict(new_state, strict=False)
    return model
