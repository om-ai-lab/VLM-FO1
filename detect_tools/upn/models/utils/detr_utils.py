import copy
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import nn

from detect_tools.upn import POS_EMBEDDINGS
from detect_tools.upn.models.module import NestedTensor


@POS_EMBEDDINGS.register_module()
class PositionEmbeddingSine(nn.Module):
    """This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.

    Args:
        num_pos_feats (int): The channel of positional embeddings.
        temperature (float): The temperature used in positional embeddings.
        normalize (bool): Whether to normalize the positional embeddings.
        scale (float): The scale factor of positional embeddings.
    """

    def __init__(
        self,
        num_pos_feats: int = 64,
        temperature: int = 10000,
        normalize: bool = False,
        scale: float = None,
    ) -> None:
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor) -> torch.Tensor:
        """Forward function.

        Args:
            tensor_list (NestedTensor): NestedTensor wrapping the input tensor.

        Returns:
            torch.Tensor: Positional encoding in shape (B, num_pos_feats*2, H, W)
        """
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)
        return pos


@POS_EMBEDDINGS.register_module()
class PositionEmbeddingSineHW(nn.Module):
    """This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.

    Args:
        num_pos_feats (int): The channel of positional embeddings.
        temperatureH (float): The temperature used in positional embeddings.
        temperatureW (float): The temperature used in positional embeddings.
        normalize (bool): Whether to normalize the positional embeddings.
        scale (float): The scale factor of positional embeddings.
    """

    def __init__(
        self,
        num_pos_feats: int = 64,
        temperatureH: int = 10000,
        temperatureW: int = 10000,
        normalize: bool = False,
        scale: float = None,
    ) -> None:
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperatureH = temperatureH
        self.temperatureW = temperatureW
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    def forward(self, tensor_list: NestedTensor) -> torch.Tensor:
        """Forward function.

        Args:
            tensor_list (NestedTensor): NestedTensor wrapping the input tensor.

        Returns:
            torch.Tensor: Positional encoding in shape (B, num_pos_feats*2, H, W)
        """
        x = tensor_list.tensors
        mask = tensor_list.mask
        assert mask is not None
        not_mask = ~mask
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_tx = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_tx = self.temperatureW ** (2 * (dim_tx // 2) / self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_tx

        dim_ty = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_ty = self.temperatureH ** (2 * (dim_ty // 2) / self.num_pos_feats)
        pos_y = y_embed[:, :, :, None] / dim_ty

        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4
        ).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)

        return pos


@POS_EMBEDDINGS.register_module()
class PositionEmbeddingLearned(nn.Module):
    """Absolute pos embedding, learned.

    Args:
        num_pos_feats (int): The channel dimension of positional embeddings.
        num_row (int): The number of rows of the input feature map.
        num_col (int): The number of columns of the input feature map.
    """

    def __init__(
        self, num_row: int = 50, num_col: int = 50, num_pos_feats: int = 256
    ) -> None:
        super().__init__()
        self.row_embed = nn.Embedding(num_row, num_pos_feats)
        self.col_embed = nn.Embedding(num_col, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, tensor_list: NestedTensor) -> torch.Tensor:
        """Forward function.

        Args:
            tensor_list (NestedTensor): NestedTensor wrapping the input tensor.

        Returns:
            torch.Tensor: Positional encoding in shape (B, num_pos_feats*2, H, W)
        """
        x = tensor_list.tensors
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = (
            torch.cat(
                [
                    x_emb.unsqueeze(0).repeat(h, 1, 1),
                    y_emb.unsqueeze(1).repeat(1, w, 1),
                ],
                dim=-1,
            )
            .permute(2, 0, 1)
            .unsqueeze(0)
            .repeat(x.shape[0], 1, 1, 1)
        )
        return pos


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ("v2", "sine"):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSineHW(
            N_steps,
            temperatureH=args.pe_temperatureH,
            temperatureW=args.pe_temperatureW,
            normalize=True,
        )
    elif args.position_embedding in ("v3", "learned"):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding


def clean_state_dict(state_dict):
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if k[:7] == "module.":
            k = k[7:]  # remove `module.`
        new_state_dict[k] = v
    return new_state_dict


def get_activation_fn(activation: str, d_model: int = 256, batch_dim: int = 0):
    """Return an activation function given a string

    Args:
        activation (str): activation function name
        d_model (int, optional): d_model. Defaults to 256.
        batch_dim (int, optional): batch dimension. Defaults to 0.

    Returns:
        F: activation function
    """
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    if activation == "prelu":
        return nn.PReLU()
    if activation == "selu":
        return F.selu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}.")


def get_clones(module: nn.Module, N: int, layer_share: bool = False):
    """Copy module N times

    Args:
        module (nn.Module): module to copy
        N (int): number of copies
        layer_share (bool, optional): share the same layer. If true, the modules will
            share the same memory. Defaults to False.
    """
    if layer_share:
        return nn.ModuleList([module for _ in range(N)])
    else:
        return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def gen_sineembed_for_position(pos_tensor):
    # n_query, bs, _ = pos_tensor.size()
    # sineembed_tensor = torch.zeros(n_query, bs, 256)
    scale = 2 * math.pi
    dim_t = torch.arange(128, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (dim_t // 2) / 128)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack(
        (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
    ).flatten(2)
    pos_y = torch.stack(
        (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
    ).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack(
            (pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3
        ).flatten(2)

        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack(
            (pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3
        ).flatten(2)

        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


def get_sine_pos_embed(
    pos_tensor: torch.Tensor,
    num_pos_feats: int = 128,
    temperature: int = 10000,
    exchange_xy: bool = True,
):
    """generate sine position embedding from a position tensor
    Args:
        pos_tensor (torch.Tensor): shape: [..., n].
        num_pos_feats (int): projected shape for each float in the tensor.
        temperature (int): temperature in the sine/cosine function.
        exchange_xy (bool, optional): exchange pos x and pos y. \
            For example, input tensor is [x,y], the results will be [pos(y), pos(x)]. Defaults to True.
    Returns:
        pos_embed (torch.Tensor): shape: [..., n*num_pos_feats].
    """
    scale = 2 * math.pi
    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = temperature ** (
        2 * torch.div(dim_t, 2, rounding_mode="floor") / num_pos_feats
    )

    def sine_func(x: torch.Tensor):
        sin_x = x * scale / dim_t
        sin_x = torch.stack(
            (sin_x[..., 0::2].sin(), sin_x[..., 1::2].cos()), dim=3
        ).flatten(2)
        return sin_x

    pos_res = [
        sine_func(x) for x in pos_tensor.split([1] * pos_tensor.shape[-1], dim=-1)
    ]
    if exchange_xy:
        pos_res[0], pos_res[1] = pos_res[1], pos_res[0]
    pos_res = torch.cat(pos_res, dim=-1)
    return pos_res


def gen_encoder_output_proposals(
    memory: torch.Tensor,
    memory_padding_mask: torch.Tensor,
    spatial_shapes: torch.Tensor,
    learnedwh=None,
):
    """
    Input:
        - memory: bs, \sum{hw}, d_model
        - memory_padding_mask: bs, \sum{hw}
        - spatial_shapes: nlevel, 2
        - learnedwh: 2
    Output:
        - output_memory: bs, \sum{hw}, d_model
        - output_proposals: bs, \sum{hw}, 4
    """
    N_, S_, C_ = memory.shape
    base_scale = 4.0
    proposals = []
    _cur = 0
    for lvl, (H_, W_) in enumerate(spatial_shapes):
        mask_flatten_ = memory_padding_mask[:, _cur : (_cur + H_ * W_)].view(
            N_, H_, W_, 1
        )
        valid_H = torch.sum(~mask_flatten_[:, :, 0, 0], 1)
        valid_W = torch.sum(~mask_flatten_[:, 0, :, 0], 1)
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(0, H_ - 1, H_, dtype=torch.float32, device=memory.device),
            torch.linspace(0, W_ - 1, W_, dtype=torch.float32, device=memory.device),
        )
        grid = torch.cat([grid_x.unsqueeze(-1), grid_y.unsqueeze(-1)], -1)  # H_, W_, 2

        scale = torch.cat([valid_W.unsqueeze(-1), valid_H.unsqueeze(-1)], 1).view(
            N_, 1, 1, 2
        )
        grid = (grid.unsqueeze(0).expand(N_, -1, -1, -1) + 0.5) / scale

        if learnedwh is not None:
            wh = torch.ones_like(grid) * learnedwh.sigmoid() * (2.0**lvl)
        else:
            wh = torch.ones_like(grid) * 0.05 * (2.0**lvl)

        proposal = torch.cat((grid, wh), -1).view(N_, -1, 4)
        proposals.append(proposal)
        _cur += H_ * W_

    output_proposals = torch.cat(proposals, 1)
    output_proposals_valid = (
        (output_proposals > 0.01) & (output_proposals < 0.99)
    ).all(-1, keepdim=True)
    output_proposals = torch.log(output_proposals / (1 - output_proposals))  # unsigmoid
    output_proposals = output_proposals.masked_fill(
        memory_padding_mask.unsqueeze(-1), float("inf")
    )
    output_proposals = output_proposals.masked_fill(
        ~output_proposals_valid, float("inf")
    )

    output_memory = memory
    output_memory = output_memory.masked_fill(
        memory_padding_mask.unsqueeze(-1), float(0)
    )
    output_memory = output_memory.masked_fill(~output_proposals_valid, float(0))

    return output_memory, output_proposals
