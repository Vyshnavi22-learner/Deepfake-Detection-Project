import torch
import torch.nn as nn
from torchvision import models

class ResNet18_LSTM_MM(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = models.resnet18(pretrained=True)
        self.cnn = nn.Sequential(*list(resnet.children())[:-1])
        self.lstm = nn.LSTM(input_size=512, hidden_size=256, batch_first=True)
        self.audio_cnn = nn.Sequential(nn.Conv2d(1, 16, 3), nn.AdaptiveAvgPool2d((1,1)), nn.Flatten())
        self.fc = nn.Linear(256 + 16, 2) 

    def forward(self, v, a):
        batch, seq_len, c, h, w = v.size()
        v = v.view(batch * seq_len, c, h, w)
        c_out = self.cnn(v).view(batch, seq_len, -1)
        _, (h_n, _) = self.lstm(c_out)
        v_feat = h_n[-1]
        a_feat = self.audio_cnn(a)
        fused = torch.cat((v_feat, a_feat), dim=1)
        return self.fc(fused)

class ProposedMultimodalFramework(nn.Module):
    def __init__(self):
        super().__init__()
        self.video_net = ResNet18_LSTM_MM()
        self.img_emo_layer = nn.Linear(512, 64)
        self.aud_emo_layer = nn.Linear(128, 32)
        self.text_emo_layer = nn.Linear(768, 128)
        self.fusion_classifier = nn.Sequential(
            nn.Linear(2 + 64 + 32 + 128, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 2) 
        )

    def forward(self, v, a, ie, ae, te):
        va_feat = self.video_net(v, a)
        ie_feat = torch.relu(self.img_emo_layer(ie))
        ae_feat = torch.relu(self.aud_emo_layer(ae))
        te_feat = torch.relu(self.text_emo_layer(te))
        fused = torch.cat((va_feat, ie_feat, ae_feat, te_feat), dim=1)
        return self.fusion_classifier(fused)