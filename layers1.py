import torch
import torch.nn as nn
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FlattenSmallWindow(nn.Module):
    def __init__(self, input_shape=(8, 8), output_dim=100):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),   # 输出: [B, 8, 8, 8]
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),  # 输出: [B, 16, 8, 8]
            nn.ReLU(),
            nn.Flatten(),                                # → [B, 1024]
            nn.Linear(16 * input_shape[0] * input_shape[1], output_dim),  # Linear(1024, 100)
            nn.ReLU(),
            nn.Dropout(0.3)
        )

    def forward(self, x):  # x: [B*31, 1, 8, 8]
        return self.cnn(x)  # [B*31, 100]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)  # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)   # 偶数位置
        pe[:, 1::2] = torch.cos(position * div_term)   # 奇数位置
        self.pe = pe.unsqueeze(0)  # [1, max_len, d_model]

    def forward(self, x):  # x: [B, T, D]
        x = x + self.pe[:, :x.size(1)].to(x.device)
        return x

class TemporalTransformer(nn.Module):
    def __init__(self, input_dim=100, d_model=128, nhead=4, num_layers=2, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model=d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=256,
            dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)

    def forward(self, x):  # x: [B, 31, 100]
        x = self.input_proj(x)               # → [B, 31, 128]
        x = self.pos_encoder(x)             # 加位置编码
        x = self.transformer(x)             # [B, 31, 128]
        x = x.permute(0, 2, 1)              # [B, 128, 31]
        x = self.pool(x).squeeze(-1)        # → [B, 128]
        return x


class E2EConv(nn.Module):
    def __init__(self, in_channels, num_output, kernel_h, kernel_w):
        super(E2EConv, self).__init__()
        self.conv_1xd_padding = nn.Conv2d(in_channels, num_output, kernel_size=(1, kernel_w), padding=(0, kernel_w // 2))
        self.conv_dx1 = nn.Conv2d(in_channels, num_output, kernel_size=(kernel_h, 1), padding=0)
        self.conv_1xd = nn.Conv2d(in_channels, num_output, kernel_size=(1, kernel_w), padding=0)

        nn.init.xavier_uniform_(self.conv_dx1.weight)
        nn.init.xavier_uniform_(self.conv_1xd.weight)
        nn.init.constant_(self.conv_dx1.bias, 0)
        nn.init.constant_(self.conv_1xd.bias, 0)

        self.dropout = nn.Dropout(p=0.5)
        self.relu = nn.ReLU()

        # 拼接后维度 [B, 11*8, 11, 11]， (1, 11) 卷积将每列特征变为 1
        self.e2n = nn.Conv2d(64, 64, kernel_size=(1, kernel_w))  # 输出 [B, 64, 11, 1]


    def forward(self, x):
        conv_1xd_padding = self.conv_1xd_padding(x)
        conv_dx1_out = self.conv_dx1(x)
        conv_dx1_out = conv_dx1_out.repeat(1, 1, x.shape[2], 1)
        conv_1xd_out = self.conv_1xd(x)
        conv_1xd_out = conv_1xd_out.repeat(1, 1, 1, x.shape[3])

        concat_dx1_dxd = torch.cat([conv_dx1_out] * conv_dx1_out.shape[2], dim=1)
        concat_1xd_dxd = torch.cat([conv_1xd_out] * conv_1xd_out.shape[2], dim=1)
        conv_1xd_padding = torch.cat([conv_1xd_padding] * conv_1xd_padding.shape[2], dim=1)

        sum_dxd = concat_dx1_dxd + concat_1xd_dxd

        # Dropout + ReLU
        conv_1xd_padding = self.dropout(self.relu(conv_1xd_padding))
        sum_dxd = self.dropout(self.relu(sum_dxd))

        # 添加 E2N 层进行压缩
        e2n_out = self.e2n(sum_dxd)  # 输出 [B, 64, 11, 1]

        # Flatten 后作为最终特征
        feature = e2n_out.view(e2n_out.size(0), -1)  # [B, 64*11] = [B, 704]

        return feature

class FusedEMGNet(nn.Module):
    def __init__(self, small_feat_dim=100, transformer_out_dim=128, cnn_feat_dim=512, num_classes=6):
        super().__init__()
        self.spatial_net = E2EConv(in_channels=1, num_output=8, kernel_h=8, kernel_w=8)
        self.small_feat_extractor = FlattenSmallWindow(output_dim=small_feat_dim)
        self.temporal_net = TemporalTransformer(input_dim=small_feat_dim, d_model=transformer_out_dim)

        # ===== 新增部分：投影 & 融合门控 =====
        self.big_proj = nn.Linear(cnn_feat_dim, 256)
        self.time_proj = nn.Linear(transformer_out_dim, 256)
        self.fuse_gate = nn.Sequential(
            nn.Linear(cnn_feat_dim + transformer_out_dim, 256),
            nn.Sigmoid()
        )

        # ===== 分类器（不变）=====
        self.classifier = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, big, small_seq):
        B = small_seq.shape[0]
        n_small = small_seq.shape[1]
        small_seq = small_seq.view(B * n_small, 1, 8, 8)
        small_feat = self.small_feat_extractor(small_seq)  # [B*31, 100]
        small_feat = small_feat.view(B, n_small, -1)        # [B, 31, 100]
        time_feat = self.temporal_net(small_feat)           # [B, 128]
        big_feat = self.spatial_net(big)                    # [B, 704]

        # ===== 门控融合 =====
        gate_input = torch.cat([big_feat, time_feat], dim=1)    # [B, 704+128]
        gate = self.fuse_gate(gate_input)                       # [B, 256]
        big_proj = self.big_proj(big_feat)                      # [B, 256]
        time_proj = self.time_proj(time_feat)                   # [B, 256]
        fused = gate * big_proj + (1 - gate) * time_proj        # [B, 256]

        return self.classifier(fused)                           # [B, num_classes]
