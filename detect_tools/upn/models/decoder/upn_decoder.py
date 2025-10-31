from typing import Dict

import torch
import torch.nn as nn

from detect_tools.upn import DECODERS, build_decoder
from detect_tools.upn.models.module import MLP
from detect_tools.upn.models.utils import (gen_sineembed_for_position,
                                      get_activation_fn, get_clones,
                                      inverse_sigmoid)
from detect_tools.upn.ops.modules import MSDeformAttn


@DECODERS.register_module()
class DeformableTransformerDecoderLayer(nn.Module):
    """Deformable Transformer Decoder Layer. This is a modified version in Grounding DINO.
    After the query is attented to the image feature, it is further attented to the text feature.
    The execute order is: self_attn -> cross_attn to text -> cross_attn to image -> ffn
    Args:
        d_model (int): The dimension of keys/values/queries in :class:`MultiheadAttention`.
        d_ffn (int): The dimension of the feedforward network model.
        dropout (float): Probability of an element to be zeroed.
        activation (str): Activation function in the feedforward network.
            'relu' and 'gelu' are supported.
        n_levels (int): The number of levels in Multi-Scale Deformable Attention.
        n_heads (int): Parallel attention heads.
        n_points (int): Number of sampling points in Multi-Scale Deformable Attention.
        ffn_extra_layernorm (bool): If True, add an extra layernorm after ffn.
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
        ffn_extra_layernorm: bool = False,
    ) -> None:
        super().__init__()

        # cross attention for visual features
        self.cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm1 = nn.LayerNorm(d_model)

        # self attention for query
        self.self_attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout)
        self.dropout2 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model)

        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = get_activation_fn(activation, d_model=d_ffn, batch_dim=1)
        self.dropout3 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout4 = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model)
        if ffn_extra_layernorm:
            raise NotImplementedError("ffn_extra_layernorm not implemented")
            self.norm_ext = nn.LayerNorm(d_ffn)
        else:
            self.norm_ext = None

        self.key_aware_proj = None

    def rm_self_attn_modules(self):
        self.self_attn = None
        self.dropout2 = None
        self.norm2 = None

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, tgt):
        tgt2 = self.linear2(self.dropout3(self.activation(self.linear1(tgt))))

        tgt = tgt + self.dropout4(tgt2)
        tgt = self.norm3(tgt)
        return tgt

    def forward(
        self,
        tgt: torch.Tensor,
        tgt_query_pos: torch.Tensor = None,
        tgt_reference_points: torch.Tensor = None,
        memory: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None,
        memory_level_start_index: torch.Tensor = None,
        memory_spatial_shapes: torch.Tensor = None,
        self_attn_mask: torch.Tensor = None,
        cross_attn_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Forward function

        Args:
            tgt (torch.Tensor): Input target in shape (B, T, C)
            tgt_query_pos (torch.Tensor): Positional encoding of the query.
            tgt_query_sine_embed (torch.Tensor): Sine positional encoding of the query. Unused.
            tgt_key_padding_mask (torch.Tensor): Mask for target feature in shape (B, T).
            tgt_reference_points (torch.Tensor): Reference points for the query in shape (B, T, 4).
            memory_text (torch.Tensor): Input text embeddings in shape (B, num_token, C).
            text_attention_mask (torch.Tensor): Attention mask for text embeddings in shape
                (B, num_token).
            memory (torch.Tensor): Input image feature in shape (B, HW, C)
            memory_key_padding_mask (torch.Tensor): Mask for image feature in shape (B, HW)
            memory_level_start_index (torch.Tensor): Starting index of each level in memory.
            memory_spatial_shapes (torch.Tensor): Spatial shape of each level in memory.
            memory_pos (torch.Tensor): Positional encoding of memory. Unused.
            self_attn_mask (torch.Tensor): Mask used for self-attention.
            cross_attn_mask (torch.Tensor): Mask used for cross-attention.

        Returns:
            torch.Tensor: Output tensor in shape (B, T, C)
        """
        assert cross_attn_mask is None

        # self attention
        if self.self_attn is not None:
            q = k = self.with_pos_embed(tgt, tgt_query_pos)
            tgt2 = self.self_attn(q, k, tgt, attn_mask=self_attn_mask)[0]
            tgt = tgt + self.dropout2(tgt2)
            tgt = self.norm2(tgt)

        # attend to image features
        tgt2 = self.cross_attn(
            self.with_pos_embed(tgt, tgt_query_pos).transpose(0, 1),
            tgt_reference_points.transpose(0, 1).contiguous(),
            memory.transpose(0, 1),
            memory_spatial_shapes,
            memory_level_start_index,
            memory_key_padding_mask,
        ).transpose(0, 1)
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        # ffn
        tgt = self.forward_ffn(tgt)

        return tgt


@DECODERS.register_module()
class UPNDecoder(nn.Module):
    """Decoder used in UPN. Each layer is a DeformableTransformerDecoderLayer. The query
    will be abled to attend the image feature and text feature. The execute order is:
    self_attn -> cross_attn to image -> ffn

    Args:
        decoder_layer_cfg (Dict): Config for the DeformableTransformerDecoderLayer.
        num_layers (int): number of layers
        norm (nn.Module, optional): normalization layer. Defaults to None.
        return_intermediate (bool, optional): whether return intermediate results.
            Defaults to False.
        d_model (int, optional): dimension of the model. Defaults to 256.
        query_dim (int, optional): dimension of the query. Defaults to 4.
        modulate_hw_attn (bool, optional): whether modulate the attention weights
            by the height and width of the image feature. Defaults to False.
        num_feature_levels (int, optional): number of feature levels. Defaults to 1.
        deformable_decoder (bool, optional): whether use deformable decoder. Defaults to False.
        decoder_query_perturber ([type], optional): [description]. Defaults to None.
        dec_layer_number ([type], optional): [description]. Defaults to None.
        rm_dec_query_scale (bool, optional): [description]. Defaults to False.
        dec_layer_share (bool, optional): [description]. Defaults to False.
        dec_layer_dropout_prob ([type], optional): [description]. Defaults to None.
    """

    def __init__(
        self,
        decoder_layer_cfg: Dict,
        num_layers: int,
        norm: str = "layernorm",
        return_intermediate: bool = True,
        d_model: int = 256,
        query_dim: int = 4,
        modulate_hw_attn: bool = False,
        num_feature_levels: int = 1,
        deformable_decoder: bool = True,
        decoder_query_perturber=None,
        dec_layer_number=None,
        rm_dec_query_scale: bool = True,
        dec_layer_share: bool = False,
        dec_layer_dropout_prob=None,
        use_detached_boxes_dec_out: bool = False,
    ):
        super().__init__()

        decoder_layer = build_decoder(decoder_layer_cfg)
        if num_layers > 0:
            self.layers = get_clones(
                decoder_layer, num_layers, layer_share=dec_layer_share
            )
        else:
            self.layers = []
        self.num_layers = num_layers
        if norm == "layernorm":
            self.norm = nn.LayerNorm(d_model)
        self.return_intermediate = return_intermediate
        self.query_dim = query_dim
        assert query_dim in [2, 4], "query_dim should be 2/4 but {}".format(query_dim)
        self.num_feature_levels = num_feature_levels
        self.use_detached_boxes_dec_out = use_detached_boxes_dec_out

        self.ref_point_head = MLP(query_dim // 2 * d_model, d_model, d_model, 2)
        self.ref_point_head_point = MLP(
            d_model, d_model, d_model, 2
        )  # for point reference only
        if not deformable_decoder:
            self.query_pos_sine_scale = MLP(d_model, d_model, d_model, 2)
        else:
            self.query_pos_sine_scale = None

        if rm_dec_query_scale:
            self.query_scale = None
        else:
            raise NotImplementedError
            self.query_scale = MLP(d_model, d_model, d_model, 2)
        self.bbox_embed = None
        self.class_embed = None

        self.d_model = d_model
        self.modulate_hw_attn = modulate_hw_attn
        self.deformable_decoder = deformable_decoder

        if not deformable_decoder and modulate_hw_attn:
            self.ref_anchor_head = MLP(d_model, d_model, 2, 2)
        else:
            self.ref_anchor_head = None

        self.decoder_query_perturber = decoder_query_perturber
        self.box_pred_damping = None

        self.dec_layer_number = dec_layer_number
        if dec_layer_number is not None:
            assert isinstance(dec_layer_number, list)
            assert len(dec_layer_number) == num_layers

        self.dec_layer_dropout_prob = dec_layer_dropout_prob
        if dec_layer_dropout_prob is not None:
            assert isinstance(dec_layer_dropout_prob, list)
            assert len(dec_layer_dropout_prob) == num_layers
            for i in dec_layer_dropout_prob:
                assert 0.0 <= i <= 1.0

        self.rm_detach = None

    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: torch.Tensor = None,
        memory_mask: torch.Tensor = None,
        tgt_key_padding_mask: torch.Tensor = None,
        memory_key_padding_mask: torch.Tensor = None,
        pos: torch.Tensor = None,
        refpoints_unsigmoid: torch.Tensor = None,
        level_start_index: torch.Tensor = None,
        spatial_shapes: torch.Tensor = None,
        valid_ratios: torch.Tensor = None,
        memory_ref_image: torch.Tensor = None,
        refImg_padding_mask: torch.Tensor = None,
        memory_visual_prompt: torch.Tensor = None,
    ):
        """Forward function.

        Args:
            tgt (torch.Tensor): target feature, [bs, num_queries, d_model]
            memory (torch.Tensor): Image feature, [bs, hw, d_model]
            tgt_mask (torch.Tensor, optional): target mask for attention. Defaults to None.
            memory_mask (torch.Tensor, optional): image mask for attention. Defaults to None.
            tgt_key_padding_mask (torch.Tensor, optional): target mask for padding. Defaults to None.
            memory_key_padding_mask (torch.Tensor, optional): image mask for padding. Defaults to None.
            pos (torch.Tensor, optional): query position embedding
            refpoints_unsigmoid (torch.Tensor, optional): reference points. Defaults to None.
            level_start_index (torch.Tensor, optional): start index of each level. Defaults to None.
            spatial_shapes (torch.Tensor, optional): spatial shape of each level. Defaults to None.
            valid_ratios (torch.Tensor, optional): valid ratio of each level. Defaults to None.
            memory_ref_image (torch.Tensor, optional): reference image feature, [bs, num_ref, d_model]. Defaults to None.
            refImg_padding_mask (torch.Tensor, optional): padding mask for attention. Defaults to None.
        """
        output = tgt

        intermediate = []
        reference_points = refpoints_unsigmoid.sigmoid()
        ref_points = [reference_points]

        for layer_id, layer in enumerate(self.layers):

            if reference_points.shape[-1] == 4:
                reference_points_input = (
                    reference_points[:, :, None]
                    * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
                )  # nq, bs, nlevel, 4
            else:
                assert reference_points.shape[-1] == 2
                reference_points_input = (
                    reference_points[:, :, None] * valid_ratios[None, :]
                )
            query_sine_embed = gen_sineembed_for_position(
                reference_points_input[:, :, 0, :]
            )  # nq, bs, 256*2

            # conditional query
            if query_sine_embed.shape[-1] == 512:
                raw_query_pos = (
                    self.ref_point_head(query_sine_embed)
                    + self.ref_point_head_point(
                        torch.zeros_like(query_sine_embed)[:, :, :256]
                    )
                    * 0.0
                )
            else:
                raw_query_pos = (
                    self.ref_point_head_point(query_sine_embed)
                    + self.ref_point_head(
                        torch.zeros(
                            query_sine_embed.shape[0],
                            query_sine_embed.shape[1],
                            512,
                            device=query_sine_embed.device,
                        )
                    )
                    * 0.0
                )
            pos_scale = self.query_scale(output) if self.query_scale is not None else 1
            query_pos = pos_scale * raw_query_pos

            # main process
            output = layer(
                tgt=output,
                tgt_query_pos=query_pos,
                tgt_reference_points=reference_points_input,
                memory=memory,
                memory_key_padding_mask=memory_key_padding_mask,
                memory_level_start_index=level_start_index,
                memory_spatial_shapes=spatial_shapes,
                self_attn_mask=tgt_mask,
                cross_attn_mask=memory_mask,
            )
            if output.isnan().any() | output.isinf().any():
                print(f"output layer_id {layer_id} is nan")
                try:
                    num_nan = output.isnan().sum().item()
                    num_inf = output.isinf().sum().item()
                    print(f"num_nan {num_nan}, num_inf {num_inf}")
                except Exception as e:
                    print(e)

            # iter update
            if self.bbox_embed is not None:

                reference_before_sigmoid = inverse_sigmoid(reference_points)
                delta_unsig = self.bbox_embed[layer_id](output)
                outputs_unsig = delta_unsig + reference_before_sigmoid
                new_reference_points = outputs_unsig.sigmoid()

                if self.rm_detach and "dec" in self.rm_detach:
                    reference_points = new_reference_points
                else:
                    reference_points = new_reference_points.detach()

                if self.use_detached_boxes_dec_out:
                    ref_points.append(reference_points)
                else:
                    ref_points.append(new_reference_points)

            if self.return_intermediate:
                intermediate.append(self.norm(output))

        if self.return_intermediate:
            return [
                [itm_out.transpose(0, 1) for itm_out in intermediate],
                [itm_refpoint.transpose(0, 1) for itm_refpoint in ref_points],
            ]
        else:
            return self.norm(output).transpose(0, 1)
