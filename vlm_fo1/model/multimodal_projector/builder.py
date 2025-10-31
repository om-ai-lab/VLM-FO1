import torch
import torch.nn as nn
import re
from .honeybee import CAbstractor
from functools import partial
import numpy as np
from torch.nn.init import trunc_normal_
from torch.nn import functional as F
import math


class IdentityMap(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, *args, **kwargs):
        return x

    @property
    def config(self):
        return {"mm_projector_type": 'identity'}


class SimpleResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.pre_norm = nn.LayerNorm(channels)

        self.proj = nn.Sequential(
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels)
        )
    def forward(self, x):
        x = self.pre_norm(x)
        return x + self.proj(x)


def build_vision_projector(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_hidden_size, config.hidden_size)
    if projector_type == "cabstract":
        n_query = getattr(config, 'mm_projector_n_query', None)
        image_size = getattr(config, 'image_size', None)
        if not n_query:
            n_query = kwargs.get("mm_projector_n_query",144)
        if not image_size: 
            image_size = kwargs.get("image_size",336)
        vokens = int(image_size/14*image_size/14)
        print ("n_query",n_query)
        print ("image_size",image_size)
        print ("vokens",vokens)

        return CAbstractor(vokens, config.mm_hidden_size, config.hidden_size, num_queries=n_query)

    if projector_type == "tokenpacker":
        #TokenPacker(hidden_size=config.hidden_size, scale_factor=config.scale_factor)
        image_size = kwargs.get("image_size",448)
        return TokenPacker(hidden_size=config.hidden_size, mm_hidden_size=config.mm_hidden_size, raw_grid=int(image_size/14))


    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')

def build_vision_projector_aux(config, delay_load=False, **kwargs):
    projector_type = getattr(config, 'mm_projector_aux_type', 'linear')

    if projector_type == 'linear':
        return nn.Linear(config.mm_region_hidden_size, config.hidden_size)
    if projector_type == "cabstract":
        n_query = getattr(config, 'mm_projector_n_query', None)
        image_size = getattr(config, 'image_size', None)
        if not n_query:
            n_query = kwargs.get("mm_projector_n_query",144)
        if not image_size: 
            image_size = kwargs.get("image_size",336)
        vokens = int(image_size/14*image_size/14)
        print ("n_query",n_query)
        print ("image_size",image_size)
        print ("vokens",vokens)

        return CAbstractor(vokens, config.mm_region_hidden_size, config.hidden_size, num_queries=n_query)

    if projector_type == "tokenpacker":
        #TokenPacker(hidden_size=config.hidden_size, scale_factor=config.scale_factor)
        image_size = kwargs.get("image_size",448)
        return TokenPacker(hidden_size=config.hidden_size, mm_hidden_size=config.mm_region_hidden_size, raw_grid=int(image_size/14))


    mlp_gelu_match = re.match(r'^mlp(\d+)x_gelu$', projector_type)
    if mlp_gelu_match:
        mlp_depth = int(mlp_gelu_match.group(1))
        modules = [nn.Linear(config.mm_region_hidden_size, config.hidden_size)]
        for _ in range(1, mlp_depth):
            modules.append(nn.GELU())
            modules.append(nn.Linear(config.hidden_size, config.hidden_size))
        return nn.Sequential(*modules)

    if projector_type == 'identity':
        return IdentityMap()

    raise ValueError(f'Unknown projector type: {projector_type}')

class TokenPacker(nn.Module):
    def __init__(
            self,
            raw_grid=32,
            embed_dim=1024,
            num_heads=1024//128,
            hidden_size=4096,
            mm_hidden_size=3200,
            scale_factor=2,
            norm_layer=partial(nn.LayerNorm, eps=1e-6)
    ):
        super().__init__()
        if raw_grid%scale_factor!=0:
            raise ValueError("scale_factor must be divisible by grid size")
        self.raw_grid = raw_grid
        self.grid_size = raw_grid//scale_factor
        self.num_queries = self.grid_size ** 2
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.scale_factor = scale_factor
        kv_dim = mm_hidden_size
        self.q_proj_1 = nn.Linear(kv_dim, embed_dim, bias=False)

        k_modules = [nn.Linear(mm_hidden_size*4, 1024)]
        for _ in range(1,2):
            k_modules.append(nn.GELU())
            k_modules.append(nn.Linear(1024, 1024))
        self.k_proj_1 = nn.Sequential(*k_modules)

        v_modules = [nn.Linear(mm_hidden_size*4, 1024)]
        for _ in range(1,2):
            v_modules.append(nn.GELU())
            v_modules.append(nn.Linear(1024, 1024))
        self.v_proj_1 = nn.Sequential(*v_modules)

        self.ln_q_1 = norm_layer(embed_dim)
        self.ln_k_1 = norm_layer(embed_dim)
        self.ln_v_1 = norm_layer(embed_dim)

        self.clip_attn = nn.MultiheadAttention(embed_dim, num_heads)

        modules = [nn.Linear(1024, hidden_size)]
        for _ in range(1, 2):
            modules.append(nn.GELU())
            modules.append(nn.Linear(hidden_size, hidden_size))
        self.mlp = nn.Sequential(*modules)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def divide_feature(self, x, kernel_size, token_num, N, c):
        h = w = int(token_num**0.5)

        #print (x.shape)
        reshape_x = x.reshape(h, w, N, c).reshape(h//kernel_size, kernel_size, w, N, c)
        reshape_x = reshape_x.permute(0,2,1,3,4)
        reshape_x = reshape_x.reshape(h//kernel_size, w//kernel_size, kernel_size, kernel_size, N, c)
        reshape_x = reshape_x.permute(0,1,3,2,4,5).reshape(h//kernel_size, w//kernel_size, kernel_size*kernel_size, N, c)
        reshape_x = reshape_x.permute(2,0,1,3,4).reshape(kernel_size*kernel_size, -1, c)

        return reshape_x

    def forward(self, x, attn_mask=None):

        x_multi = x[1] # mulit-level
        x = x[0] # original single-level

        key = self.ln_k_1(self.k_proj_1(x_multi)).permute(1, 0, 2)
        value = self.ln_v_1(self.v_proj_1(x_multi)).permute(1, 0, 2)

        token_num, N, c = key.shape

        q = F.interpolate(x.reshape(x.shape[0],self.raw_grid,self.raw_grid,-1).float().permute(0,3,1,2), size=(self.grid_size, self.grid_size), mode='bilinear').permute(0,2,3,1) ## fix
        q = q.reshape(q.shape[0], -1, q.shape[-1]).to(x.dtype)

        query = self.ln_q_1(self.q_proj_1(q)).permute(1, 0, 2)

        reshape_query = self.divide_feature(query, 1, self.num_queries, N, c)
        reshape_key = self.divide_feature(key, self.scale_factor, token_num, N, c)
        reshape_value = self.divide_feature(value, self.scale_factor, token_num, N, value.shape[-1])

        out = self.clip_attn(
            reshape_query,
            reshape_key,
            reshape_value,
            attn_mask=attn_mask)[0]

        x = out
        x = x.reshape(self.num_queries, N, -1)
        x = x.permute(1, 0, 2)

        x = self.mlp(x)
        return x

    def _repeat(self, query, N: int):
        return query.unsqueeze(1).repeat(1, N, 1)


