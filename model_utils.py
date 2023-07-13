import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch

# used in original DeepCubeA paper
class Residual(nn.Module):
    def __init__(self, nc):
        super(Residual, self).__init__()
        self.linear1 = nn.Linear(nc, nc)
        self.linear2 = nn.Linear(nc, nc)
        self.bn1 = nn.BatchNorm1d(nc)
        self.bn2 = nn.BatchNorm1d(nc)

    def forward(self, x):
        return F.relu(x + self.bn2(self.linear2(F.relu(self.bn1(self.linear1(x))))))

def get_model(nc=1024, nb=4):
    return nn.Sequential(
        nn.Linear(256, nc),
        *[Residual(nc) for _ in range(nb)],
        nn.Linear(nc, 1),
    )

# custom model based on ViT architecture
class TransformerBlock(nn.Module):
    def __init__(self, nc):
        super().__init__()
        self.att = nn.MultiheadAttention(nc, 8, batch_first=True)
        self.linear1 = nn.Linear(nc, 4*nc)
        self.linear2 = nn.Linear(4*nc, nc)
        self.norm1 = nn.LayerNorm(nc)
        self.norm2 = nn.LayerNorm(nc)

    def forward(self, x):
        x_norm = self.norm1(x)
        z_p = self.att(x_norm, x_norm, x_norm)[0] + x
        z = self.linear2(F.gelu(self.linear1(self.norm2(z_p)))) + z_p
        return z

class Transformer(nn.Module):
    def __init__(self, nc=256, nb=4):
        super().__init__()
        self.nc = nc
        self.nb = nb
        self.emb = nn.Embedding(16, nc)
        self.pos_emb = nn.Embedding(16, nc)
        self.blocks = nn.Sequential(*[TransformerBlock(nc) for _ in range(nb)])
        self.norm = nn.LayerNorm(16*nc)
        self.linear = nn.Linear(16*nc, 1)

    def forward(self, x):
        x = self.emb(x) + self.pos_emb.weight
        x = self.blocks(x)
        x = self.linear(self.norm(x.flatten(1)))
        return x

def get_saved_model(saved=True, save_file='slider_model_h.pth'):
    #model = get_model().eval()
    model = Transformer().eval()
    if saved:
        try:
            model.load_state_dict(torch.load(save_file))
            #model = torch.load(save_file).eval()
        except:
            print('Model not loaded from pretrained state.')
    return model
