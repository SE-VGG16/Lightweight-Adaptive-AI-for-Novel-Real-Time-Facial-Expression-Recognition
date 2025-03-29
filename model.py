# model.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
    
    def forward(self, x):
        attention = torch.sigmoid(self.conv(x))
        return x * attention

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
    
    def forward(self, x):
        b, c, _, _ = x.size()
        avg = self.avg_pool(x).view(b, c)
        attention = torch.sigmoid(self.fc(avg)).view(b, c, 1, 1)
        return x * attention

class AdaptiveFusion(nn.Module):
    def __init__(self, in_channels):
        super(AdaptiveFusion, self).__init__()
        self.spatial_att = SpatialAttention(in_channels)
        self.channel_att = ChannelAttention(in_channels)
    
    def forward(self, x):
        x = self.spatial_att(x)
        x = self.channel_att(x)
        return x

class AMFT(nn.Module):
    def __init__(self, num_classes=7, use_transformer=True, use_lstm=False, lstm_hidden=256):
        super(AMFT, self).__init__()
        # Backbone: ResNet-50 without the final classification layer
        resnet = models.resnet50(pretrained=True)
        self.backbone = nn.Sequential(*list(resnet.children())[:-2])  # (batch, 2048, H/32, W/32)
        
        self.adaptive_fusion = AdaptiveFusion(2048)
        
        # Optional Transformer block for feature enhancement
        self.use_transformer = use_transformer
        if use_transformer:
            encoder_layer = nn.TransformerEncoderLayer(d_model=2048, nhead=8)
            self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)
        
        # Global pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))
        self.use_lstm = use_lstm
        if use_lstm:
            self.lstm = nn.LSTM(2048, lstm_hidden, batch_first=True)
            self.classifier = nn.Linear(lstm_hidden, num_classes)
        else:
            self.classifier = nn.Linear(2048, num_classes)
    
    def forward(self, x):
        # x: (batch, C, H, W) or, if use_lstm, (batch, seq, C, H, W)
        if self.use_lstm:
            b, seq, C, H, W = x.size()
            features = []
            for i in range(seq):
                frame = x[:, i, :, :, :]
                feat = self.backbone(frame)
                feat = self.adaptive_fusion(feat)
                if self.use_transformer:
                    b, c, h, w = feat.size()
                    feat = feat.view(b, c, h*w).permute(2, 0, 1)  # shape: (S, B, C)
                    feat = self.transformer_encoder(feat)
                    feat = feat.permute(1, 2, 0).view(b, c, h, w)
                feat = self.avgpool(feat)
                feat = feat.view(b, -1)
                features.append(feat)
            features = torch.stack(features, dim=1)  # (b, seq, feature_dim)
            lstm_out, _ = self.lstm(features)
            out = self.classifier(lstm_out[:, -1, :])
        else:
            feat = self.backbone(x)
            feat = self.adaptive_fusion(feat)
            if self.use_transformer:
                b, c, h, w = feat.size()
                feat = feat.view(b, c, h*w).permute(2, 0, 1)  # (S, B, C)
                feat = self.transformer_encoder(feat)
                feat = feat.permute(1, 2, 0).view(b, c, h, w)
            feat = self.avgpool(feat)
            feat = feat.view(feat.size(0), -1)
            out = self.classifier(feat)
        return out

if __name__ == "__main__":
    # Quick test for image input
    model = AMFT(num_classes=7, use_transformer=True, use_lstm=False)
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print("Output shape:", output.shape)
