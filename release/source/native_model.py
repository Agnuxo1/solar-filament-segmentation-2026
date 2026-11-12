"""Unchanged owned native H3 definitions. Recipe attribution: Dariush Afshar public write-up; own weights only. Source SHA 1d135d3d59594f963f740ab4a4fb4d386a7d9acd018c18c898cd6014d08cc3bc"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import models
NATIVE=2048
CROP=512
CENTRED=.7
DISK_RADIUS=950
VIEWS=8
D4=[(0,False),(0,True),(2,False),(2,True),(1,False),(1,True),(3,False),(3,True)]
_yy,_xx=np.ogrid[:NATIVE,:NATIVE]
DISK=((_yy-(NATIVE-1)/2)**2+(_xx-(NATIVE-1)/2)**2)<=DISK_RADIUS**2
ENCODERS = {  # stem taps at strides 2, 4, 8, 16, 32
    "resnet34": (64, 64, 128, 256, 512),
    "efficientnet_v2_s": (24, 48, 64, 160, 256),
}

class Encoder(nn.Module):
    def __init__(self, name: str, pretrained: bool):
        super().__init__()
        self.name = name
        if name == "resnet34":
            net = models.resnet34(weights="IMAGENET1K_V1" if pretrained else None)
            net.conv1 = self._one_channel(net.conv1)
            self.stages = nn.ModuleList([
                nn.Sequential(net.conv1, net.bn1, net.relu),
                nn.Sequential(net.maxpool, net.layer1), net.layer2, net.layer3, net.layer4])
        else:
            net = models.efficientnet_v2_s(weights="IMAGENET1K_V1" if pretrained else None).features
            net[0][0] = self._one_channel(net[0][0])
            self.stages = nn.ModuleList([
                nn.Sequential(net[0], net[1]), net[2], net[3], nn.Sequential(net[4], net[5]), net[6]])

    @staticmethod
    def _one_channel(conv: nn.Conv2d) -> nn.Conv2d:
        new = nn.Conv2d(1, conv.out_channels, conv.kernel_size, conv.stride, conv.padding, bias=False)
        with torch.no_grad():
            new.weight.copy_(conv.weight.sum(1, keepdim=True))
        return new

    def forward(self, x):
        taps = []
        for stage in self.stages:
            x = stage(x)
            taps.append(x)
        return taps

class Up(nn.Module):
    def __init__(self, cin: int, cskip: int, cout: int):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(cin + cskip, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True),
            nn.Conv2d(cout, cout, 3, padding=1, bias=False), nn.BatchNorm2d(cout), nn.ReLU(inplace=True))

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.body(x if skip is None else torch.cat([x, skip], 1))

class NativeUNet(nn.Module):
    def __init__(self, encoder: str, pretrained: bool = True):
        super().__init__()
        c = ENCODERS[encoder]
        self.encoder = Encoder(encoder, pretrained)
        self.u4, self.u3 = Up(c[4], c[3], 256), Up(256, c[2], 128)
        self.u2, self.u1 = Up(128, c[1], 64), Up(64, c[0], 32)
        self.u0 = Up(32, 0, 16)
        self.head = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        t = self.encoder(x)
        y = self.u1(self.u2(self.u3(self.u4(t[4], t[3]), t[2]), t[1]), t[0])
        return self.head(self.u0(y))

def loss_fn(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    p = torch.sigmoid(logits)
    inter = (p * target).sum((1, 2, 3))
    dice = 1 - (2 * inter + 1) / (p.sum((1, 2, 3)) + target.sum((1, 2, 3)) + 1)
    return bce + dice.mean()

def disk_stats(img: np.ndarray) -> tuple[float, float]:
    v = img[DISK].astype(np.float32) / 255
    return float(v.mean()), max(float(v.std()), 1e-3)

class Crops(Dataset):
    def __init__(self, imgs, targets, stats, boxes, length, seed):
        self.imgs, self.targets, self.stats, self.boxes = imgs, targets, stats, boxes
        self.length, self.seed = length, seed

    def __len__(self):
        return self.length

    def __getitem__(self, n):
        rng = np.random.default_rng((self.seed, n))
        i = int(rng.integers(len(self.imgs)))
        if self.boxes[i] and rng.random() < CENTRED:
            x, y, w, h = self.boxes[i][int(rng.integers(len(self.boxes[i])))]
            cx, cy = x + w / 2 + rng.uniform(-.25, .25) * CROP, y + h / 2 + rng.uniform(-.25, .25) * CROP
        else:
            cx, cy = rng.uniform(0, NATIVE, 2)
        x0 = int(np.clip(cx - CROP / 2, 0, NATIVE - CROP))
        y0 = int(np.clip(cy - CROP / 2, 0, NATIVE - CROP))
        a = self.imgs[i][y0:y0 + CROP, x0:x0 + CROP].astype(np.float32) / 255
        t = self.targets[i][y0:y0 + CROP, x0:x0 + CROP].astype(np.float32) / 255
        k = int(rng.integers(4))
        a, t = np.rot90(a, k), np.rot90(t, k)
        if rng.random() < 0.5:
            a, t = a[:, ::-1], t[:, ::-1]
        mean, std = self.stats[i]
        a = np.clip(a, 1e-4, 1) ** rng.uniform(0.85, 1.15)
        a = (a - mean) * rng.uniform(0.9, 1.1) + mean
        a = (a - mean) / std
        return (torch.from_numpy(np.ascontiguousarray(a))[None].float(),
                torch.from_numpy(np.ascontiguousarray(t))[None].float())

@torch.no_grad()
def predict(model, img: np.ndarray, device) -> np.ndarray:
    mean, std = disk_stats(img)
    x = torch.from_numpy((img.astype(np.float32) / 255 - mean) / std)[None, None].to(device)
    acc = torch.zeros_like(x)
    for k, flip in D4[:VIEWS]:
        v = torch.rot90(x, k, (2, 3))
        v = v.flip(3) if flip else v
        with torch.amp.autocast(device.type, enabled=device.type == "cuda"):
            p = torch.sigmoid(model(v).float())
        p = p.flip(3) if flip else p
        acc += torch.rot90(p, -k, (2, 3))
    return (acc[0, 0] / VIEWS * 255).round().to(torch.uint8).cpu().numpy()
