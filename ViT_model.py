import torch
import torch.nn as nn
from einops import repeat


def pair(t):
    return t if isinstance(t, tuple) else (t, t)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        x = x.transpose(1, -1)
        # print(x.shape)
        x = self.norm(x)
        x = x.transpose(-1, 1)
        return self.fn(x, **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(hidden_dim, dim, 1),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 4, dim_head = 32, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.head_channels = dim // self.heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -2)
        self.to_keys = nn.Conv2d(dim, inner_dim, 1)
        self.to_queries = nn.Conv2d(dim, inner_dim, 1)
        self.to_values = nn.Conv2d(dim, inner_dim, 1)
        self.unifyheads = nn.Sequential(
            nn.Conv2d(inner_dim, dim, 1),
            nn.Dropout(dropout)) if project_out else nn.Identity()
        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        # self.to_out = nn.Sequential(
        #     nn.Linear(inner_dim, dim),
        #     nn.Dropout(dropout)
        # ) if project_out else nn.Identity()

    def forward(self, x):
        # qkv = self.to_qkv(x).chunk(3, dim = -1)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        b, _, h, w = x.shape
        k = self.to_keys(x).view(b, self.heads, self.head_channels, -1)
        q = self.to_queries(x).view(b, self.heads, self.head_channels, -1)
        v = self.to_values(x).view(b, self.heads, self.head_channels, -1)

        dots = k.transpose(-2, -1) @ q

        attn = self.attend(dots)

        out = torch.matmul(v, attn)
        out = out.view(b, -1, h, w)
        return self.unifyheads(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 32, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        
        # self.to_patch_embedding = nn.Sequential(
        #     Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_height, p2 = patch_width),
        #     nn.Linear(patch_dim, dim),
        # )
        self.to_patch_embedding = nn.Sequential(
            nn.Conv2d(channels, dim, 3, padding=1),
            nn.GELU(),
            nn.Conv2d(dim, dim, kernel_size=patch_size, stride=patch_size)
        )

        
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        reduced_size = image_size // patch_size
        shape = (reduced_size, reduced_size)
        dim += 1
        self.pos_embedding = nn.Parameter(torch.randn(dim, *shape))

        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, *shape))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(nn.Flatten(),
                                      nn.Linear(reduced_size**2, num_classes))
        self.reset_parameters()

    def forward(self, img):
        x = self.to_patch_embedding(img)

        b, n, h, w = x.shape

        cls_tokens = repeat(self.cls_token, '() n h w -> b n h w', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        
        x += self.pos_embedding[:, :(n + 1)]
        # x += self.pos_embedding
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]

        x = self.to_latent(x)

        
        return self.mlp_head(x)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.weight, 1.)
                nn.init.zeros_(m.bias)
    
    def separate_parameters(self):
        parameters_decay = set()
        parameters_no_decay = set()
        modules_weight_decay = (nn.Linear, nn.Conv2d)
        modules_no_weight_decay = (nn.LayerNorm,)

        for m_name, m in self.named_modules():
            for param_name, param in m.named_parameters():
                full_param_name = f"{m_name}.{param_name}" if m_name else param_name

                if isinstance(m, modules_no_weight_decay):
                    parameters_no_decay.add(full_param_name)
                elif param_name.endswith("bias"):
                    parameters_no_decay.add(full_param_name)
                elif isinstance(m, modules_weight_decay):
                    parameters_decay.add(full_param_name)
