from typing import Dict

import torch
import torch.nn as nn
import torch.utils.checkpoint as checkpoint

from detect_tools.upn import ENCODERS, build_encoder
from detect_tools.upn.models.utils import get_activation_fn, get_clones
from detect_tools.upn.ops.modules import MSDeformAttn


@ENCODERS.register_module()
class DeformableTransformerEncoderLayer(nn.Module):
    """Deformable Transformer Encoder Layer.

    Args:
        d_model (int): The dimension of keys/values/queries in
            :class:`MultiheadAttention`.
        d_ffn (int): The dimension of the feedforward network model.
        dropout (float): Probability of an element to be zeroed.
        activation (str): Activation function in the feedforward network.
            'relu' and 'gelu' are supported.
        n_levels (int): The number of levels in Multi-Scale Deformable Attention.
        n_heads (int): Parallel attention heads.
        n_points (int): Number of sampling points in Multi-Scale Deformable Attention.
        add_channel_attention (bool): If True, add channel attention.
    """

    def __init__(
        self,
        d_model: int = 256,
        d_ffn: int = 1024,
        dropout: float = 0.1,
        activation: str = "relu",
        n_levels: int = 4,
        n_heads: int = 8,
        n_points: int = 4,
        add_channel_attention: bool = False,
    ) -> None:
        super().__init__()

        # self attention
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = get_activation_fn(activation, d_model=d_ffn)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

        # channel attention
        self.add_channel_attention = add_channel_attention
        if add_channel_attention:
            self.activ_channel = get_activation_fn("dyrelu", d_model=d_model)
            self.norm_channel = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src: torch.Tensor) -> torch.Tensor:
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    def forward(
        self,
        src: torch.Tensor,
        pos: torch.Tensor,
        reference_points: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward function for `DeformableTransformerEncoderLayer`.

        Args:
            src (torch.Tensor): The input sequence of shape (S, N, E).
            pos (torch.Tensor): The position embedding of shape (S, N, E).
            reference_points (torch.Tensor): The reference points of shape (N, L, 2).
            spatial_shapes (torch.Tensor): The spatial shapes of feature levels.
            level_start_index (torch.Tensor): The start index of each level.
            key_padding_mask (torch.Tensor): The mask for keys with shape (N, S).
        """
        # self attention
        # import ipdb; ipdb.set_trace()
        src2 = self.self_attn(
            self.with_pos_embed(src, pos),
            reference_points,
            src,
            spatial_shapes,
            level_start_index,
            key_padding_mask,
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)

        # channel attn
        if self.add_channel_attention:
            src = self.norm_channel(src + self.activ_channel(src))

        return src


@ENCODERS.register_module()
class UPNEncoder(nn.Module):
    """Implementation of UPN Encoder.

    Args:
        num_layers (int): The number of layers in the TransformerEncoder.
        d_model (int, optional): The dimension of the input feature. Defaults to 256.
        encoder_layer_cfg (Dict): Config for the DeformableEncoderLayer.
        use_checkpoint (bool, optional): Whether to use checkpoint in the fusion layer for
            memory saving. Defaults to False.
        use_transformer_ckpt (bool, optional): Whether to use checkpoint for the deformableencoder.
        enc_layer_share (bool, optional): Whether to share the same memory for the encoder_layer.
            Defaults to False. This is used for all the sub-layers in the basic block.
    """

    def __init__(
        self,
        num_layers: int,
        d_model: int = 256,
        encoder_layer_cfg: Dict = None,
        use_checkpoint: bool = True,
        use_transformer_ckpt: bool = True,
        enc_layer_share: bool = False,
        multi_level_encoder_fusion: str = None,
    ):
        super().__init__()
        # prepare layers
        self.layers = []
        self.refImg_layers = []
        self.fusion_layers = []
        encoder_layer = build_encoder(encoder_layer_cfg)

        self.multi_level_encoder_fusion = multi_level_encoder_fusion
        self._initilize_memory_fusion_layers(
            multi_level_encoder_fusion, num_layers, d_model
        )

        if num_layers > 0:
            self.layers = get_clones(
                encoder_layer, num_layers, layer_share=enc_layer_share
            )
        else:
            self.layers = []
            del encoder_layer

        self.query_scale = None
        self.num_layers = num_layers
        self.d_model = d_model

        self.use_checkpoint = use_checkpoint
        self.use_transformer_ckpt = use_transformer_ckpt

    def _initilize_memory_fusion_layers(self, fusion_type, num_layers, d_model):
        if fusion_type is None:
            self.memory_fusion_layer = None
            return

        assert fusion_type in ["dense_net_fusion", "stable_dense_fusion"]
        if fusion_type == "stable_dense_fusion":
            self.memory_fusion_layer = nn.Sequential(
                nn.Linear(d_model * (num_layers + 1), d_model),
                nn.LayerNorm(d_model),
            )
            nn.init.constant_(self.memory_fusion_layer[0].bias, 0)
        elif fusion_type == "dense_net_fusion":
            self.memory_fusion_layer = nn.ModuleList()
            for i in range(num_layers):
                self.memory_fusion_layer.append(
                    nn.Sequential(
                        nn.Linear(
                            d_model * (i + 2), d_model
                        ),  # from second encoder layer, 512 -> 256 / 3rd: 768 -> 256
                        nn.LayerNorm(d_model),
                    )
                )
            for layer in self.memory_fusion_layer:
                nn.init.constant_(layer[0].bias, 0)
        else:
            raise NotImplementedError

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(
                torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device),
            )
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points

    def forward(
        self,
        src: torch.Tensor,
        pos: torch.Tensor,
        spatial_shapes: torch.Tensor,
        level_start_index: torch.Tensor,
        valid_ratios: torch.Tensor,
        key_padding_mask: torch.Tensor = None,
    ):
        """Forward function

        Args:
            src (torch.Tensor): Flattened Image features in shape [bs, sum(hi*wi), 256]
            pos (torch.Tensor): Position embedding for image feature in shape [bs, sum(hi*wi), 256]
            spatial_shapes (torch.Tensor): Spatial shape of each level in shape [num_level, 2]
            level_start_index (torch.Tensor): Start index of each level in shape [num_level]
            valid_ratios (torch.Tensor): Valid ratio of each level in shape [bs, num_level, 2]
            key_padding_mask (torch.Tensor): Padding mask for image feature in shape [bs, sum(hi*wi)]
            memory_refImg (torch.Tensor, optional): Text feature in shape [bs, n_ref, 256]. Defaults
                to None.
            refImg_padding_mask (torch.Tensor, optional): Padding mask for reference image feature
                in shape [bs, n_text]. Defaults to None.
            pos_refImg (torch.Tensor, optional): Position embedding for reference image in shape
                [bs, n_ref, 256]. Defaults to None.
            refImg_self_attention_masks (torch.Tensor, optional): Self attention mask for reference
                image feature in shape [bs, n_ref, n_ref]. Defaults to None.
        Outpus:
            torch.Tensor: Encoded image feature in shape [bs, sum(hi*wi), 256]
            torch.Tensor: Encoded reference image feature in shape [bs, n_ref, 256]
        """

        output = src
        # preparation and reshape
        if self.num_layers > 0:
            reference_points = self.get_reference_points(
                spatial_shapes, valid_ratios, device=src.device
            )

        # multi-level dense fusion
        output_list = [output]
        # main process
        for layer_id, layer in enumerate(self.layers):
            # main process
            if self.use_transformer_ckpt:
                output = checkpoint.checkpoint(
                    layer,
                    output,
                    pos,
                    reference_points,
                    spatial_shapes,
                    level_start_index,
                    key_padding_mask,
                )
            else:
                output = layer(
                    src=output,
                    pos=pos,
                    reference_points=reference_points,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index,
                    key_padding_mask=key_padding_mask,
                )

                output_list.append(output)
                if (
                    self.multi_level_encoder_fusion is not None
                    and self.multi_level_encoder_fusion == "dense_net_fusion"
                ):
                    output = self.memory_fusion_layer[layer_id](
                        torch.cat(output_list, dim=-1)
                    )

        if (
            self.multi_level_encoder_fusion is not None
            and self.multi_level_encoder_fusion == "stable_dense_fusion"
        ):
            output = self.memory_fusion_layer(torch.cat(output_list, dim=-1))

        return output
