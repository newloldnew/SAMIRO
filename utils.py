import torch
import torch.nn as nn
import torch.nn.functional as F

class MeanEncoder(nn.Module):
    def __init__(self, shape):
        super().__init__()
        self.shape = shape

    def forward(self, x):
        return x

class VarianceEncoder(nn.Module):
    def __init__(self, shape, init=0.1, eps=1e-5):
        super().__init__()
        self.eps = eps
        init = (torch.as_tensor(init - eps).exp() - 1.0).log()
        b_shape = (1, shape[1], 1, 1)
        self.b = nn.Parameter(torch.full(b_shape, init))

    def forward(self, x):
        return F.softplus(self.b) + self.eps


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attn = torch.cat([avg_out, max_out], dim=1)
        attn = self.sigmoid(self.conv(attn))
        return attn

class SAMIROLossDemo(nn.Module):
    def __init__(self, size_cfg, lamda=0.1):
        super().__init__()
        self.lamda = lamda
        
        B = size_cfg["batch_size"]
        teacher_feature_maps = size_cfg["teacher_feature_maps"]
        student_feature_maps = size_cfg["student_feature_maps"]

        self.proj_convs = nn.ModuleList([
            nn.Conv2d(
                in_channels=s["C"], 
                out_channels=t["C"], 
                kernel_size=s["K"], 
                stride=(s["H"] // t["H"])
            ) for t, s in zip(teacher_feature_maps, student_feature_maps)
        ])

        self.mean_encoders = nn.ModuleList([
            MeanEncoder([B, t["C"], t["H"], t["W"]]) for t in teacher_feature_maps
        ])

        self.var_encoders = nn.ModuleList([
            VarianceEncoder([B, t["C"], t["H"], t["W"]]) for t in teacher_feature_maps
        ])

        self.spatial_attentions = nn.ModuleList([
            SpatialAttention() for _ in teacher_feature_maps
        ])

        self.layer_norms = nn.ModuleList([
            nn.LayerNorm([t["C"], t["H"], t["W"]], elementwise_affine=False)
            for t in teacher_feature_maps
        ])

    def forward(self, student_feats, teacher_feats):
        reg_loss = 0.0

        for s_feat, t_feat, conv, mean_enc, var_enc, attn, ln in zip(
            student_feats,
            teacher_feats,
            self.proj_convs,
            self.mean_encoders,
            self.var_encoders,
            self.spatial_attentions,
            self.layer_norms,
        ):
            s_proj = conv(s_feat)

            mean = mean_enc(s_proj)
            var = var_enc(s_proj)

            t_attn = attn(t_feat) * t_feat
            t_norm = ln(t_attn)
            s_norm = ln(mean)

            vlb = (s_norm - t_norm).pow(2).div(var) + var.log()
            reg_loss += vlb.mean()

        reg_loss *= self.lamda
        return F.relu(reg_loss)