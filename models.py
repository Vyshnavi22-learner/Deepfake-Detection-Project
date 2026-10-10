import torch
import torch.nn as nn
from torchvision import models as tvm


class AudioCNN(nn.Module):
    def __init__(self, out_dim=64):
        super().__init__()
        def blk(i, o):
            return nn.Sequential(
                nn.Conv2d(i, o, 3, padding=1, bias=False),
                nn.BatchNorm2d(o),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )
        self.net = nn.Sequential(
            blk(1, 16),
            blk(16, 32),
            blk(32, out_dim),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )

    def forward(self, x):
        return self.net(x)


class ResNet18_LSTM_MM(nn.Module):
    """Baseline: frames -> ResNet18 -> LSTM, log-Mel -> AudioCNN, concat -> FC (2 classes: 0=Real, 1=Fake)."""

    def __init__(self, freeze_cnn=True, lstm_hidden=256, audio_dim=64):
        super().__init__()
        resnet = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])  # Feature extractor
        self.freeze_cnn = freeze_cnn
        if freeze_cnn:
            for q in self.cnn.parameters():
                q.requires_grad_(False)
        self.lstm = nn.LSTM(512, lstm_hidden, batch_first=True)
        self.audio_cnn = AudioCNN(audio_dim)
        self.feat_dim = lstm_hidden + audio_dim  # 256 + 64 = 320
        self.fc = nn.Linear(self.feat_dim, 2)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_cnn:
            self.cnn.eval()  # Keep BatchNorm fixed
        return self

    def features(self, v, a):
        B, T, C, H, W = v.shape  # v: float 0..1
        x = (v.reshape(B * T, C, H, W) - self.mean) / self.std
        if self.freeze_cnn and not v.requires_grad:
            with torch.no_grad():
                c = self.cnn(x)
        else:  # Grad-CAM++ explanation pass
            c = self.cnn(x)
        _, (h, _) = self.lstm(c.reshape(B, T, -1))
        return torch.cat([h[-1], self.audio_cnn(a)], dim=1)

    def forward(self, v, a):
        return self.fc(self.features(v, a))


class ProposedMultimodalFramework(nn.Module):
    """Baseline features + image/audio/text emotion embeddings (768-d each) -> feature-level fusion -> 2 classes."""

    def __init__(self, freeze_cnn=True, ie_dim=768, ae_dim=768, te_dim=768, proj=64, p_drop=0.15):
        super().__init__()
        self.video_net = ResNet18_LSTM_MM(freeze_cnn)
        self.img_emo_layer = nn.Linear(ie_dim, proj)  # 768 -> 64
        self.aud_emo_layer = nn.Linear(ae_dim, proj)  # 768 -> 64
        self.text_emo_layer = nn.Linear(te_dim, proj) # 768 -> 64
        self.p_drop = p_drop
        self.fusion_classifier = nn.Sequential(
            nn.Linear(self.video_net.feat_dim + 3 * proj, 128), # 320 + 192 = 512
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2)
        )

    def forward(self, v, a, ie, ae, te):
        f = self.video_net.features(v, a)
        if self.training and self.p_drop > 0:  # Modality dropout for robustness
            keep = lambda x: x * (torch.rand(x.shape[0], 1, device=x.device) > self.p_drop).float()
            ie, ae, te = keep(ie), keep(ae), keep(te)
        z = [f, torch.relu(self.img_emo_layer(ie)), torch.relu(self.aud_emo_layer(ae)), torch.relu(self.text_emo_layer(te))]
        return self.fusion_classifier(torch.cat(z, dim=1))
