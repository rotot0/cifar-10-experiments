import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, in_dim, hid_dim):
        super(MLP, self).__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            nn.GELU(),
            nn.Linear(hid_dim, in_dim)
        )
    
    def forward(self, x):
        return self.mlp(x)


class MixerBlock(nn.Module):
    def __init__(self, patches_dim, channels_dim):
        super(MixerBlock, self).__init__()

        self.norm1 = nn.LayerNorm(patches_dim)
        self.norm2 = nn.LayerNorm(channels_dim)
        self.mlp1 = MLP(patches_dim, patches_dim*4)
        self.mlp2 = MLP(channels_dim, channels_dim*4)
    
    def forward(self, x):
        out = self.norm1(x)
        out = self.mlp1(out)
        out += x
        x = out
        out = out.permute(0, 2, 1)
        out = self.mlp2(self.norm2(out))
        out = out.permute(0, 2, 1)
        out += x
        return out
    
class Mixer(nn.Module):
    def __init__(self, img_size, n_channels, p, emb_dim, num_classes=10, num_blocks=4):
        super(Mixer, self).__init__()

        self.p = p
        self.img_dim = img_size
        self.channels_dim = n_channels
        self.patches_dim = self.channels_dim * p * p
        self.n_pathces = (self.img_dim // p) ** 2
        self.emb_dim = emb_dim
        self.num_classes = num_classes

        self.proj = nn.Linear(self.patches_dim, self.emb_dim)
        self.blocks = nn.Sequential(*[MixerBlock(self.emb_dim, self.n_pathces) for _ in range(num_blocks)])
        self.last_norm = nn.LayerNorm(self.emb_dim)
        self.classifier = nn.Linear(self.emb_dim, self.num_classes)

    def forward(self, x):
        x_temp = x.unfold(2, self.p, self.p).unfold(3, self.p, self.p)
        x_temp = x_temp.reshape(-1 , self.channels_dim, self.n_pathces, self.p, self.p)
        x_patches = x_temp.permute(0, 2, 1, 3, 4).reshape(x_temp.size(0), self.n_pathces, -1)

        out = self.proj(x_patches)
        out = self.blocks(out)
        out = self.last_norm(out)
        out = out.mean(dim=1)
        return self.classifier(out)
