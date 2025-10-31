from functools import partial
import torch
import torch.nn as nn
from einops import rearrange
from timm.layers import LayerNorm, LayerNorm2d
from timm.models.regnet import RegStage


def build_pos_embeds(
    pos_emb: bool, num_input_tokens: int, vision_hidden_size: int
):
    # pos emb
    if pos_emb:
        pos_emb = torch.nn.Parameter(torch.zeros(1, num_input_tokens, vision_hidden_size))
        nn.init.trunc_normal_(pos_emb, mean=0.0, std=0.02)
    else:
        pos_emb = None

    return pos_emb

def build_prenorm(prenorm, encoder_hidden_size):
    if prenorm:
        prenorm = LayerNorm(encoder_hidden_size)
    else:
        prenorm = None
    return prenorm


def build_mlp(depth, hidden_size, output_hidden_size):
    layers = [nn.Linear(hidden_size, output_hidden_size)]
    for _ in range(1, depth):
        layers.append(nn.SiLU())
        layers.append(nn.Linear(output_hidden_size, output_hidden_size))
    return nn.Sequential(*layers)


class CAbstractor(nn.Module):
    """Base projector class"""

    def __init__(
        self,
        num_input_tokens: int,
        encoder_hidden_size: int,
        output_hidden_size: int,
        hidden_size: int = 1024,
        depth: int = 3,
        mlp_depth: int = 2,
        num_queries: int = 144,
        pos_emb: bool = True,
        prenorm: bool = False
    ):
        super().__init__()
        self.num_input_tokens = num_input_tokens
        self.encoder_hidden_size = encoder_hidden_size
        self.output_hidden_size = output_hidden_size
        self.mlp_depth = mlp_depth
        self.depth = depth
        self.num_queries = num_queries
        self.hidden_size = hidden_size

        # pos emb
        self.pos_emb = build_pos_embeds(pos_emb, num_input_tokens, encoder_hidden_size)

        self.prenorm = build_prenorm(prenorm, encoder_hidden_size)

        self.build_net()

    def build_net(self):
        encoder_hidden_size = self.encoder_hidden_size
        hidden_size = self.hidden_size
        output_hidden_size = self.output_hidden_size
        depth = self.depth
        mlp_depth = self.mlp_depth
        n_queries = self.num_queries

        assert (n_queries ** 0.5).is_integer(), "n_queries must be square number"
        hw = int(n_queries ** 0.5)

        # RegBlock = ResBlock + SE
        RegBlock = partial(
            RegStage,
            stride=1,
            dilation=1,
            act_layer=nn.SiLU,
            norm_layer=LayerNorm2d,
        )

        s1 = RegBlock(
            depth,
            encoder_hidden_size,
            hidden_size,
        )
        sampler = nn.AdaptiveAvgPool2d((hw, hw))
        s2 = RegBlock(
            depth,
            hidden_size,
            hidden_size,
        )

        self.net = nn.Sequential(s1, sampler, s2)
        self.readout = build_mlp(mlp_depth, hidden_size, output_hidden_size)

    def _forward(self, x):
        # x: [B, L, dim]
        # x = x[:, 1:]  # drop cls token and 2d forward  @Kyusong, If we output CLS token from vision tower, u can use this
        hw = int(x.size(1) ** 0.5)
        x = rearrange(x, "b (h w) d -> b d h w", h=hw, w=hw)
        x = self.net(x)
        x = rearrange(x, "b d h w -> b (h w) d")
        x = self.readout(x)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, encoder_hidden_size) tensor from the visual backbone (CLIP visual encoder), including cls token.
        """
        if self.prenorm is not None:
            x = self.prenorm(x)

        if self.pos_emb is not None:
            x += self.pos_emb

        x = self._forward(x)  # (B, L, output_hidden_size)

        return x



if __name__ == "__main__":
    B = 2 # batch size
    L = 576 # number of input token
    H = 1024 # hidden size

    n_query = 256
    output_h = 4096

    x = torch.FloatTensor(B, L, H)
    m = CAbstractor(L, H, output_h, num_queries=n_query)
    y = m(x)
    print(y.shape) # B, N_Query, output_H
