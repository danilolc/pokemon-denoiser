import torch
import torch.nn as nn
import torch.nn.functional as F

# https://github.com/tcapelle/Diffusion-Models-pytorch/blob/main/LICENSE

def pos_encoding(t, channels):
    log_inv_freq = (-9.21034 / channels) * torch.arange(0.0, channels, 2.0, device=t.device)
    pos_enc_a = torch.sin(t[..., None] * torch.exp(log_inv_freq[None, :]))
    pos_enc_b = torch.cos(t[..., None] * torch.exp(log_inv_freq[None, :]))
    pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
    return pos_enc

#https://www.mdpi.com/2072-4292/11/9/1015/pdf
#@torch.compile
def ssim_loss(pred, target):
    mean_pred = torch.mean(pred, dim=(1, 2, 3))
    mean_target = torch.mean(target, dim=(1, 2, 3))

    var_pred = torch.var(pred, dim=(1, 2, 3))
    var_target = torch.var(target, dim=(1, 2, 3))

    cov = torch.mean(pred * target, dim=(1, 2, 3)) - mean_pred * mean_target

    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    ssim = (2 * mean_pred * mean_target + c1) * (2 * cov + c2) 
    ssim /= (mean_pred ** 2 + mean_target ** 2 + c1) * (var_pred + var_target + c2)

    return (1 - ssim) / 2

class GroupNorm(nn.Module):
    """Group Normalization is a method which normalizes the activation of the layer for better results across any batch size.
    Note : Weight Standardization is also shown to given better results when added with group norm

    Args:
        in_channels (int): Number of channels in the input tensor.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()

        # num_groups is according to the official code provided by the authors,
        # eps is for numerical stability
        # i think affine here is enabling learnable param for affine trasnform on calculated mean & standard deviation
        self.group_norm = nn.GroupNorm(
            num_groups=32, num_channels=in_channels, eps=1e-06, affine=True
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.group_norm(x)

class NonLocalBlock(nn.Module):
    """Attention mechanism similar to transformers but for CNNs, paper https://arxiv.org/abs/1805.08318

    Args:
        in_channels (int): Number of channels in the input tensor.
    """

    def __init__(self, in_channels:int) -> None:
        super().__init__()

        self.in_channels = in_channels

        # normalization layer
        self.norm = GroupNorm(in_channels)

        # query, key and value layers
        self.q = nn.Conv2d(in_channels, in_channels, 1)
        self.k = nn.Conv2d(in_channels, in_channels, 1)
        self.v = nn.Conv2d(in_channels, in_channels, 1)

        self.project_out = nn.Sequential(
            nn.Conv2d(in_channels, 4*in_channels, 1),
            nn.GELU(),
            nn.Conv2d(4*in_channels, in_channels, 1),
        )

        self.softmax = nn.Softmax(dim=2)

    def forward(self, x):

        batch, channels, height, width = x.size()
        assert channels == self.in_channels

        x = self.norm(x)

        # query, key and value layers
        q = self.q(x)
        k = self.k(x)
        v = self.v(x)

        # resizing the output from 4D to 3D to generate attention map
        q = q.reshape(batch, self.in_channels, height * width)
        k = k.reshape(batch, self.in_channels, height * width)
        v = v.reshape(batch, self.in_channels, height * width)

        # transpose the query tensor for dot product
        q = q.permute(0, 2, 1)

        # main attention formula
        scores = torch.bmm(q, k) * (self.in_channels**-0.5)
        weights = self.softmax(scores)
        weights = weights.permute(0, 2, 1)

        attention = torch.bmm(v, weights)

        # resizing the output from 3D to 4D to match the input
        attention = attention.reshape(batch, self.in_channels, height, width)
        attention = self.project_out(attention)

        # adding the identity to the output
        return x + attention

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
        )

    def forward(self, x):
        if self.residual:
            return x + self.double_conv(x)
        else:
            return self.double_conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.downsample = nn.Sequential(
            #nn.MaxPool2d(2),
            nn.Conv2d(in_channels, out_channels, kernel_size=2, stride=2),
            nn.GroupNorm(1, out_channels),
            nn.GELU(),
        )
        self.conv = nn.Sequential(
            DoubleConv(out_channels, out_channels),
            DoubleConv(out_channels, out_channels, residual=True),
        )

    def forward(self, x):
        x = self.downsample(x)
        x = self.conv(x)

        return x


class Up(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            DoubleConv(in_channels + skip_channels, out_channels),
            DoubleConv(out_channels, out_channels, residual=True),
        )

    def forward(self, x, skip_x):
        x = torch.nn.functional.interpolate(x, scale_factor=2.0)
        x = torch.cat([x, skip_x], dim=1)
        x = self.conv(x)

        return x

class ContourEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.step1 = nn.Sequential(
            DoubleConv(1, 64),
            nn.MaxPool2d(2),
        )
        self.step2 = nn.Sequential(
            DoubleConv(64, 128),
            nn.MaxPool2d(2),
        )
        self.step3 = nn.Sequential(
            DoubleConv(128, 256),
            nn.MaxPool2d(2),
        )

    def forward(self, c):
        c1 = self.step1(c)
        c2 = self.step2(c1)
        c3 = self.step3(c2)

        return c1, c2, c3

class TimeEncoder(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.emb_dim = emb_dim
        self.encoders = nn.ModuleList()
        for i in [64, 128, 256, 256, 128, 64]:
            self.encoders.append(nn.Linear(emb_dim, i))

    def forward(self, t):
        t = pos_encoding(t, self.emb_dim)
        return [enc(t)[:, :, None, None] for enc in self.encoders]


class UNet(nn.Module):
    def __init__(self, c_in=3, c_out=3, num_classes=10, time_dim=256):
        super().__init__()
        #self.label_emb = nn.Embedding(18, num_classes)

        self.enc = nn.Sequential(
            nn.Conv2d(c_in, 32, kernel_size=1),
            nn.GroupNorm(1, 32),
            nn.GELU(),
            
            DoubleConv(32, 32, residual=True),
            DoubleConv(32, 32, residual=True),
        )
        
        self.down1 = Down(32, 64)
        self.down2 = Down(64, 128)
        self.down3 = Down(128, 256)

        self.bot = nn.Sequential(
            DoubleConv(256, 256),
            DoubleConv(256, 256),
        )

        self.up1 = Up(256, 128, 128)
        self.up2 = Up(128, 64, 64)
        self.up3 = Up(64, 32, 32)
        
        self.dec = nn.Sequential(
            DoubleConv(32, 32, residual=True),
            DoubleConv(32, 32, residual=True),
            nn.Conv2d(32, c_out, kernel_size=1),
        )

        #self.c_encoder = ContourEncoder()
        self.t_encoder = TimeEncoder(time_dim)

        self.att1 = NonLocalBlock(64)
        self.att2 = NonLocalBlock(128)
        self.att3 = NonLocalBlock(256)
        self.att4 = NonLocalBlock(256)
        self.att5 = NonLocalBlock(128)
        self.att6 = NonLocalBlock(64)

    def forward(self, x, t, ty, c):
        #c1, c2, c3 = self.c_encoder(c.flatten(1,3))
        t1, t2, t3, t4, t5, t6 = self.t_encoder(t)
        
        x1 = self.enc(x)
        
        x2 = self.down1(x1)
        x2 = self.att1(x2 + t1)

        x3 = self.down2(x2)
        x3 = self.att2(x3 + t2)

        x4 = self.down3(x3)
        x4 = self.att3(x4 + t3)

        x4 = self.bot(x4)
        x4 = self.att4(x4 + t4)

        x4 = self.up1(x4, x3)
        x4 = self.att5(x4 + t5)

        x4 = self.up2(x4, x2)
        x4 = self.att6(x4 + t6)

        x4 = self.up3(x4, x1)

        return self.dec(x4)