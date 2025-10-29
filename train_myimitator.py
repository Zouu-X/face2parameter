from __future__ import print_function
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.nn.functional as F
from torch.optim import lr_scheduler
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset
import json
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time
from torch.cuda.amp import autocast, GradScaler
import copy
import math

'''
    在服务器上训练只需上传这一个文件即可
'''

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

# Batch size during training
batch_size = 16
image_size = 512
num_epochs = 500
lr = 0.01
ngpu = 2

dataset_root = "/db-mnt/mnt/efs-mount/home/xiangxzou/"
params_path = os.path.join(dataset_root, "frontal_labels.json")
images_root = os.path.join(dataset_root, "images")
splits_root = os.path.join(dataset_root, "splits")
train_index_file = os.path.join(splits_root, "train.json")
val_index_file = os.path.join(splits_root, "val.json")


def _load_split(index_file):
    with open(index_file, encoding="utf-8") as f:
        records = json.load(f)
    samples = []
    for item in records:
        if isinstance(item, dict):
            key = item.get("key")
            rel_path = item.get("path") or item.get("image") or ""
        else:
            rel_path = str(item)
            key = os.path.splitext(os.path.basename(rel_path))[0]
        if not key or not rel_path:
            continue
        samples.append((key, rel_path))
    if not samples:
        raise ValueError(f"No valid entries found in split file: {index_file}")
    return samples

DEFAULT_IMG_TRANSFORM = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 再映射到 [-1,1]
])

class Imitator_Dataset(Dataset):
    def __init__(self, params_root, image_root, index_file, transform=None):
        self.image_root = image_root
        self.transform = transform or DEFAULT_IMG_TRANSFORM
        # self.transform = transform or T.ToTensor()
        with open(params_root, encoding='utf-8') as f:
            self.params = json.load(f)
        self.samples = _load_split(index_file)
        self.missing_keys = [key for key, _ in self.samples if key not in self.params]
        if self.missing_keys:
            print(f"WARNING: {len(self.missing_keys)} split entries missing params. They will be skipped.")
            self.samples = [(key, path) for key, path in self.samples if key in self.params]
        if not self.samples:
            raise ValueError("No samples available after filtering missing parameter entries.")

    def __getitem__(self, index):
        key, rel_path = self.samples[index]
        img_path = os.path.join(self.image_root, rel_path)
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)
        param = torch.tensor(self.params[key], dtype=torch.float32)
        return param, img

    def __len__(self):
        return len(self.samples)

train_dataset = Imitator_Dataset(params_path, images_root, train_index_file)
if os.path.exists(val_index_file):
    val_dataset = Imitator_Dataset(params_path, images_root, val_index_file)
else:
    val_dataset = None

train_dataloader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)
val_dataloader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    persistent_workers=True,
) if val_dataset is not None else None

preview_dir = os.path.join(dataset_root, "gen_image")
model_dir = os.path.join(dataset_root, "model")
metrics_path = os.path.join(dataset_root, "metrics.jpg")
os.makedirs(preview_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



# real_batch = next(iter(val_dataloader))
# plt.figure(figsize=(4, 4))
# plt.axis("off")
# plt.title("Training Images")
# plt.imshow(np.transpose(vutils.make_grid(real_batch[1].to(device)[:16], nrow=4, padding=2, normalize=True).cpu(), (1, 2, 0)))
# plt.show()
# vutils.save_image(vutils.make_grid(real_batch[1].to(device)[:16], nrow=4, padding=2, normalize=True).cpu(), "./a.jpg")



'''
    自定义Imitator
    1.conv，linear，embedding后加上sn
    2.指定层加上self-attention
    3.自定义bn
'''

# 采用sn做 normalization
def snconv2d(eps=1e-12, **kwargs):
    return nn.utils.spectral_norm(nn.Conv2d(**kwargs), eps=eps)

def snlinear(eps=1e-12, **kwargs):
    return nn.utils.spectral_norm(nn.Linear(**kwargs), eps=eps)

def sn_embedding(eps=1e-12, **kwargs):
    return nn.utils.spectral_norm(nn.Embedding(**kwargs), eps=eps)

# self-attention层
class SelfAttn(nn.Module):
    def __init__(self, in_channels, eps=1e-12):
        super(SelfAttn, self).__init__()
        self.in_channels = in_channels
        self.snconv1x1_theta = snconv2d(in_channels=in_channels, out_channels=in_channels//8,
                                        kernel_size=1, bias=False, eps=eps)
        self.snconv1x1_phi = snconv2d(in_channels=in_channels, out_channels=in_channels//8,
                                      kernel_size=1, bias=False, eps=eps)
        self.snconv1x1_g = snconv2d(in_channels=in_channels, out_channels=in_channels//2,
                                    kernel_size=1, bias=False, eps=eps)
        self.snconv1x1_o_conv = snconv2d(in_channels=in_channels//2, out_channels=in_channels,
                                         kernel_size=1, bias=False, eps=eps)
        self.maxpool = nn.MaxPool2d(2, stride=2, padding=0)
        self.softmax  = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        _, ch, h, w = x.size()
        # Theta path
        theta = self.snconv1x1_theta(x)
        theta = theta.view(-1, ch//8, h*w)
        # Phi path
        phi = self.snconv1x1_phi(x)
        phi = self.maxpool(phi)
        phi = phi.view(-1, ch//8, h*w//4)
        # Attn map
        attn = torch.bmm(theta.permute(0, 2, 1), phi)
        attn = self.softmax(attn)
        # g path
        g = self.snconv1x1_g(x)
        g = self.maxpool(g)
        g = g.view(-1, ch//2, h*w//4)
        # Attn_g - o_conv
        attn_g = torch.bmm(g, attn.permute(0, 2, 1))
        attn_g = attn_g.view(-1, ch//2, h, w)
        attn_g = self.snconv1x1_o_conv(attn_g)
        # Out
        out = x + self.gamma*attn_g
        return out

# 自定义bn（基于 ConditionalBatchNorm2d 结构）
class BigGANBatchNorm(nn.Module):
    """Conditional BatchNorm for generator.

    简化为：
    - 条件分支：`BN(affine=False)` + 线性层生成逐样本 `gamma/beta`，输出为 `BN(x) * (1 + gamma) + beta`。
    - 非条件分支：普通 `BN(affine=True)`。

    注：保持与现有调用一致，`truncation` 参数被忽略（固定 0.4）。
    """
    def __init__(self, num_features, condition_vector_dim=None, n_stats=51, eps=1e-4, conditional=True):
        super(BigGANBatchNorm, self).__init__()
        self.conditional = conditional

        if conditional:
            assert condition_vector_dim is not None
            self.bn = nn.BatchNorm2d(num_features, affine=False, eps=eps)
            # 使用光谱归一化的线性层以稳定训练
            self.gamma = snlinear(in_features=condition_vector_dim, out_features=num_features, bias=True, eps=eps)
            self.beta = snlinear(in_features=condition_vector_dim, out_features=num_features, bias=True, eps=eps)
        else:
            self.bn = nn.BatchNorm2d(num_features, affine=True, eps=eps)

    def forward(self, x, truncation, condition_vector=None):
        out = self.bn(x)
        if self.conditional:
            # 逐样本仿射：gamma 中心化为 1
            gamma = self.gamma(condition_vector).unsqueeze(-1).unsqueeze(-1)
            beta = self.beta(condition_vector).unsqueeze(-1).unsqueeze(-1)
            out = out * (1 + gamma) + beta
        return out

class GenBlock(nn.Module):
    def __init__(self, in_size, out_size, condition_vector_dim, reduction_factor=4, up_sample=False,
                 n_stats=51, eps=1e-12):
        super(GenBlock, self).__init__()
        self.up_sample = up_sample
        self.drop_channels = (in_size != out_size)
        middle_size = in_size // reduction_factor

        self.bn_0 = BigGANBatchNorm(in_size, condition_vector_dim, n_stats=n_stats, eps=eps, conditional=True)
        self.conv_0 = snconv2d(in_channels=in_size, out_channels=middle_size, kernel_size=1, eps=eps)

        self.bn_1 = BigGANBatchNorm(middle_size, condition_vector_dim, n_stats=n_stats, eps=eps, conditional=True)
        self.conv_1 = snconv2d(in_channels=middle_size, out_channels=middle_size, kernel_size=3, padding=1, eps=eps)

        self.bn_2 = BigGANBatchNorm(middle_size, condition_vector_dim, n_stats=n_stats, eps=eps, conditional=True)
        self.conv_2 = snconv2d(in_channels=middle_size, out_channels=middle_size, kernel_size=3, padding=1, eps=eps)

        self.bn_3 = BigGANBatchNorm(middle_size, condition_vector_dim, n_stats=n_stats, eps=eps, conditional=True)
        self.conv_3 = snconv2d(in_channels=middle_size, out_channels=out_size, kernel_size=1, eps=eps)

        self.relu = nn.ReLU()

    def forward(self, x, cond_vector, truncation):
        x0 = x

        x = self.bn_0(x, truncation, cond_vector)
        x = self.relu(x)
        x = self.conv_0(x)

        x = self.bn_1(x, truncation, cond_vector)
        x = self.relu(x)
        if self.up_sample:
            x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = self.conv_1(x)

        x = self.bn_2(x, truncation, cond_vector)
        x = self.relu(x)
        x = self.conv_2(x)

        x = self.bn_3(x, truncation, cond_vector)
        x = self.relu(x)
        x = self.conv_3(x)

        if self.drop_channels:
            new_channels = x0.shape[1] // 2
            x0 = x0[:, :new_channels, ...]
        if self.up_sample:
            x0 = F.interpolate(x0, scale_factor=2, mode='nearest')

        out = x + x0
        return out

class MyImitator(nn.Module):
    def __init__(self):
        super(MyImitator, self).__init__()

        # 1.加载配置文件
        with open("/Workspace/Users/xiangxzou@global.tencent.com/face2parameter/checkpoint/myimitator-256.json", "r", encoding='utf-8') as reader:
            text = reader.read()
        self.conf = BigGANConfig()
        for key, value in json.loads(text).items():
            self.conf.__dict__[key] = value

        # 定义网络结构
        # self.embeddings = nn.Linear(config.num_classes, config.continuous_params_size, bias=False)

        ch = self.conf.channel_width
        condition_vector_dim = 205

        self.gen_z = snlinear(in_features=condition_vector_dim, out_features=4*4*16*ch, eps=self.conf.eps)
        layers = []
        for i, layer in enumerate(self.conf.layers):
            if i == self.conf.attention_layer_position:    # 在指定层加上self-attention
                layers.append(SelfAttn(ch * layer[1], eps=self.conf.eps))
            layers.append(GenBlock(ch * layer[1],
                                   ch * layer[2],
                                   condition_vector_dim,
                                   up_sample=layer[0],
                                   n_stats=self.conf.n_stats,
                                   eps=self.conf.eps))
        self.layers = nn.ModuleList(layers)

        self.bn = BigGANBatchNorm(ch, n_stats=self.conf.n_stats, eps=self.conf.eps, conditional=False)
        self.relu = nn.ReLU()
        self.conv_to_rgb = snconv2d(in_channels=ch, out_channels=ch, kernel_size=3, padding=1, eps=self.conf.eps)
        self.tanh = nn.Tanh()

    def forward(self, cond_vector, truncation=0.4):
        # cond_vector = cond_vector.unsqueeze(2).unsqueeze(3)
        z = self.gen_z(cond_vector)    # cond_cector [batch_size, config.continuous_params_size], z [1, 4*4*16*self.conf.channel_width]

        # We use this conversion step to be able to use TF weights:
        # TF convention on shape is [batch, height, width, channels]
        # PT convention on shape is [batch, channels, height, width]
        z = z.view(-1, 4, 4, 16 * self.conf.channel_width)    # [batch_size, 4, 4, 2048]
        z = z.permute(0, 3, 1, 2).contiguous()    # [batch_size, 2048, 4, 4]

        for i, layer in enumerate(self.layers):
            if isinstance(layer, GenBlock):
                z = layer(z, cond_vector, truncation)
            else:
                z = layer(z)

        z = self.bn(z, truncation)    # [1, 128, 512, 512]
        z = self.relu(z)    # [1, 128, 512, 512]
        z = self.conv_to_rgb(z)    # [1, 128, 512, 512]
        z = z[:, :3, ...]    # [1, 3, 512, 512]
        z = self.tanh(z)    # [1, 3, 512, 512]
        return z

'''
    自定义Imitator的config
'''
class BigGANConfig(object):
    """ Configuration class to store the configuration of a `BigGAN`.
        Defaults are for the 128x128 model.
        layers tuple are (up-sample in the layer ?, input channels, output channels)
    """
    def __init__(self,
                 output_dim=512,
                 z_dim=512,
                 class_embed_dim=512,
                 channel_width=512,
                 num_classes=1000,
                 # (是否上采样，input_channels，output_channels)
                 layers=[(False, 16, 16),
                         (True, 16, 16),
                         (False, 16, 16),
                         (True, 16, 8),
                         (False, 8, 8),
                         (True, 8, 4),
                         (False, 4, 4),
                         (True, 4, 2),
                         (False, 2, 2),
                         (True, 2, 1)],
                 attention_layer_position=8,
                 eps=1e-4,
                 n_stats=51):
        """Constructs BigGANConfig. """
        self.output_dim = output_dim
        self.z_dim = z_dim
        self.class_embed_dim = class_embed_dim
        self.channel_width = channel_width
        self.num_classes = num_classes
        self.layers = layers
        self.attention_layer_position = attention_layer_position
        self.eps = eps
        self.n_stats = n_stats

    @classmethod
    def from_dict(cls, json_object):
        """Constructs a `BigGANConfig` from a Python dictionary of parameters."""
        config = BigGANConfig()
        for key, value in json_object.items():
            config.__dict__[key] = value
        return config

    @classmethod
    def from_json_file(cls, json_file):
        """Constructs a `BigGANConfig` from a json file of parameters."""
        with open(json_file, "r", encoding='utf-8') as reader:
            text = reader.read()
        return cls.from_dict(json.loads(text))

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


imitator = MyImitator()
if device.type == 'cuda':
    imitator = nn.DataParallel(imitator)
imitator.to(device)

# Initialize BCELoss function
criterion = nn.L1Loss()

# optimizer = optim.SGD(imitator.parameters(), lr=lr, momentum=0.9)
optimizer = optim.Adam(params=imitator.parameters(), lr=5e-5,
                           betas=(0.0, 0.999), weight_decay=0,
                           eps=1e-8)

use_amp = (device.type == 'cuda')
scaler = GradScaler(enabled=use_amp)

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, num_cycles=0.5):
    """Create a schedule with a learning rate that decreases following the
    values of the cosine function between the initial lr set in the optimizer
    to 0, after a warmup period during which it increases linearly from 0 to the
    initial lr.

    Args:
        optimizer: Wrapped optimizer.
        num_warmup_steps: Warmup steps where lr increases linearly 0 -> base_lr.
        num_training_steps: Total number of training steps.
        num_cycles: Cosine cycles (0.5 = single decay to zero).
    """
    def lr_lambda(current_step: int):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        return max(0.0, 0.5 * (1.0 + math.cos(math.pi * 2.0 * num_cycles * progress)))

    return lr_scheduler.LambdaLR(optimizer, lr_lambda)

total_step = len(train_dataloader)

# Warmup + cosine decay scheduler (per-step)
# Use 5 warmup epochs as a reasonable default for stability
warmup_epochs = 5
num_training_steps = num_epochs * total_step
num_warmup_steps = warmup_epochs * total_step
scheduler = get_cosine_schedule_with_warmup(
    optimizer=optimizer,
    num_warmup_steps=num_warmup_steps,
    num_training_steps=num_training_steps,
    num_cycles=0.5,
)
imitator.train()
train_loss_list = []
val_loss_list = []

# Early stopping state
early_stop_patience = 10
best_val_loss = float('inf')
epochs_no_improve = 0
for epoch in range(num_epochs):
    start = time.time()
    for i, (params, img) in enumerate(train_dataloader):
        optimizer.zero_grad()
        params = params.to(device)
        img = img.to(device)
        with autocast(enabled=use_amp):
            outputs = imitator(params)
            loss = criterion(outputs, img)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # Step LR scheduler per update
        scheduler.step()

        if (i % 500) == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, LR: {:.6e}, spend time: {:.4f}'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(), current_lr, time.time() - start))
            start = time.time()

    train_loss_list.append(loss.item())
    imitator.eval()
    with torch.no_grad():
        val_loss = 0
        for i, (params, img) in enumerate(val_dataloader):
            params = params.to(device)
            img = img.to(device)
            with autocast(enabled=use_amp):
                outputs = imitator(params)
                loss = criterion(outputs, img)
            val_loss += loss.item()

            if i == 1:
                vutils.save_image(
                    vutils.make_grid(outputs.to(device)[:16], nrow=4, padding=2, normalize=True).cpu(),
                    os.path.join(preview_dir, f"{epoch}.jpg"))
        val_loss = val_loss / len(val_dataloader)
        val_loss_list.append(val_loss)

        print('Epoch [{}/{}], val_loss: {:.6f}'
              .format(epoch + 1, num_epochs, val_loss))

        # Early stopping tracking and best model saving
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(imitator.state_dict(), os.path.join(model_dir, 'best_epoch_{}_val_loss_{:.6f}.pt'.format(epoch, val_loss)))
        else:
            epochs_no_improve += 1
        if (epoch % 10) == 0 or (epoch+1) == num_epochs:
            torch.save(imitator.state_dict(),
                       os.path.join(model_dir, 'epoch_{}_val_loss_{:.6f}_file.pt'.format(
                           epoch, val_loss)))
        if epoch >= 1:
            plt.figure()
            plt.subplot(121)
            plt.plot(np.arange(0, len(train_loss_list)), train_loss_list)
            plt.subplot(122)
            plt.plot(np.arange(0, len(val_loss_list)), val_loss_list)
            plt.savefig(metrics_path)
            plt.close("all")

    imitator.train()

    # Trigger early stopping if no improvement for `early_stop_patience` epochs
    if epochs_no_improve >= early_stop_patience:
        print(f"Early stopping triggered after {early_stop_patience} epochs without improvement. Best val_loss: {best_val_loss:.6f}")
        break
