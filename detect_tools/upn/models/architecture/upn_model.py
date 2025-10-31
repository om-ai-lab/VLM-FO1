import copy
from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from detect_tools.upn import ARCHITECTURES, build_architecture, build_backbone
from detect_tools.upn.models.module import (MLP, ContrastiveAssign, NestedTensor,
                                       nested_tensor_from_tensor_list)
from detect_tools.upn.models.utils import inverse_sigmoid


class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


@ARCHITECTURES.register_module()
class UPN(nn.Module):
    """Implementation of UPN"""

    def __init__(
        self,
        vision_backbone_cfg: Dict,
        transformer_cfg: Dict,
        num_queries: int,
        dec_pred_class_embed_share=True,
        dec_pred_bbox_embed_share=True,
        decoder_sa_type="sa",
    ):
        super().__init__()
        # build vision backbone
        self.backbone = build_backbone(vision_backbone_cfg)
        # build transformer
        self.transformer = build_architecture(transformer_cfg)

        self.hidden_dim = self.transformer.d_model

        # for dn training
        self.num_queries = num_queries
        self.num_feature_levels = self.transformer.num_feature_levels

        # prepare projection layer for vision feature
        self.input_proj = self.prepare_vision_feature_projection_layer(
            self.backbone,
            self.transformer.num_feature_levels,
            self.hidden_dim,
            self.transformer.two_stage_type,
        )
        # prepare prediction head
        self.prepare_prediction_head(
            dec_pred_class_embed_share,
            dec_pred_bbox_embed_share,
            self.hidden_dim,
            self.transformer.num_decoder_layers,
        )

        self.decoder_sa_type = decoder_sa_type
        assert decoder_sa_type in ["sa", "ca_label", "ca_content"]
        # self.replace_sa_with_double_ca = replace_sa_with_double_ca

        for layer in self.transformer.decoder.layers:
            layer.label_embedding = None
        self.label_embedding = None

        # build a unversal token
        self.transformer.fine_grained_prompt = nn.Embedding(1, self.hidden_dim)
        self.transformer.coarse_grained_prompt = nn.Embedding(1, self.hidden_dim)

        self._reset_parameters()

    def forward(self, samples: NestedTensor, prompt_type: str = None) -> Dict:
        """Foward function"""
        self.device = samples.device

        (
            src_flatten,
            lvl_pos_embed_flatten,
            level_start_index,
            spatial_shapes,
            valid_ratios,
            mask_flatten,
        ) = self.forward_backbone_encoder(samples)

        (
            hs,
            reference,
            ref_dict,
        ) = self.transformer(
            src_flatten,
            lvl_pos_embed_flatten,
            level_start_index,
            spatial_shapes,
            valid_ratios,
            mask_flatten,
            prompt_type,
        )

        # deformable-detr-line anchor update
        outputs_coord_list = []
        outputs_class = []

        for layer_idx, (layer_ref_sig, layer_bbox_embed, layer_hs) in enumerate(
            zip(reference[:-1], self.bbox_embed, hs)
        ):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            layer_outputs_unsig = layer_delta_unsig + inverse_sigmoid(layer_ref_sig)
            layer_outputs_unsig = layer_outputs_unsig.sigmoid()
            outputs_coord_list.append(layer_outputs_unsig)

        outputs_coord_list = torch.stack(outputs_coord_list)

        if ref_dict is None:
            # build a mock outputs_class for mask_dn training
            outputs_class = torch.zeros(
                outputs_coord_list.shape[0],
                outputs_coord_list.shape[1],
                outputs_coord_list.shape[2],
                self.hidden_dim,
            )
        else:
            outputs_class = torch.stack(
                [
                    layer_cls_embed(layer_hs, ref_dict)
                    for layer_cls_embed, layer_hs in zip(self.class_embed, hs)
                ]
            )

        out = {
            "pred_logits": outputs_class[-1],
            "pred_boxes": outputs_coord_list[-1],
        }
        out["ref_dict"] = ref_dict
        return out

    def forward_backbone_encoder(self, samples: NestedTensor) -> Tuple:
        # pass through backbone
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        features, poss = self.backbone(samples)
        # project features
        srcs = []
        masks = []
        for l, feat in enumerate(features):
            src, mask = feat.decompose()
            srcs.append(self.input_proj[l](src))  # downsample the feature map to 256
            masks.append(mask)
            assert mask is not None

        if self.num_feature_levels > len(
            srcs
        ):  # add more feature levels by downsampling the last feature map
            _len_srcs = len(srcs)
            for l in range(_len_srcs, self.num_feature_levels):
                if l == _len_srcs:
                    src = self.input_proj[l](features[-1].tensors)
                else:
                    src = self.input_proj[l](srcs[-1])
                m = samples.mask
                mask = F.interpolate(m[None].float(), size=src.shape[-2:]).to(
                    torch.bool
                )[0]
                pos_l = self.backbone.forward_pos_embed_only(
                    NestedTensor(src, mask)
                ).to(src.dtype)
                srcs.append(src)
                masks.append(mask)
                poss.append(pos_l)

        # prepare input for encoder with the following steps:
        # 1. flatten the feature maps and masks
        # 2. Add positional embedding and level embedding
        # 3. Calculate the valid ratio of each feature map based on the mask
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, poss)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)

            src = src.flatten(2).transpose(1, 2)  # bs, hw, c
            mask = mask.flatten(1)  # bs, hw
            pos_embed = pos_embed.flatten(2).transpose(1, 2)  # bs, hw, c
            if self.num_feature_levels > 1 and self.transformer.level_embed is not None:
                lvl_pos_embed = pos_embed + self.transformer.level_embed[lvl].view(
                    1, 1, -1
                )
            else:
                lvl_pos_embed = pos_embed
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(
            spatial_shapes, dtype=torch.long, device=src_flatten.device
        )
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1])
        )
        valid_ratios = torch.stack(
            [self.transformer.get_valid_ratio(m) for m in masks], 1
        )

        return (
            src_flatten,
            lvl_pos_embed_flatten,
            level_start_index,
            spatial_shapes,
            valid_ratios,
            mask_flatten,
        )

    def prepare_vision_feature_projection_layer(
        self,
        backbone: nn.Module,
        num_feature_levels: int,
        hidden_dim: int,
        two_stage_type: str,
    ) -> nn.ModuleList:
        """Prepare projection layer to map backbone feature to hidden dim.

        Args:
            backbone (nn.Module): Backbone.
            num_feature_levels (int): Number of feature levels.
            hidden_dim (int): Hidden dim.
            two_stage_type (str): Type of two stage.

        Returns:
            nn.ModuleList: Projection layer.
        """
        if num_feature_levels > 1:
            num_backbone_outs = len(backbone.num_channels)
            input_proj_list = []
            for _ in range(num_backbone_outs):
                in_channels = backbone.num_channels[_]
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(in_channels, hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
            for _ in range(num_feature_levels - num_backbone_outs):
                input_proj_list.append(
                    nn.Sequential(
                        nn.Conv2d(
                            in_channels, hidden_dim, kernel_size=3, stride=2, padding=1
                        ),
                        nn.GroupNorm(32, hidden_dim),
                    )
                )
                in_channels = hidden_dim
            input_proj = nn.ModuleList(input_proj_list)
        else:
            assert (
                two_stage_type == "no"
            ), "two_stage_type should be no if num_feature_levels=1 !!!"
            input_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Conv2d(backbone.num_channels[-1], hidden_dim, kernel_size=1),
                        nn.GroupNorm(32, hidden_dim),
                    )
                ]
            )
        return input_proj

    def prepare_prediction_head(
        self,
        dec_pred_class_embed_share: bool,
        dec_pred_bbox_embed_share: bool,
        hidden_dim: int,
        num_decoder_layers: int,
    ) -> Union[nn.ModuleList, nn.ModuleList]:
        """Prepare prediction head. Including class embed and bbox embed.

        Args:
            dec_pred_class_embed_share (bool): Whether to share class embed for all decoder layers.
            dec_pred_bbox_embed_share (bool): Whether to share bbox embed for all decoder layers.
            im (int): Hidden dim.
            num_decoder_layers (int): Number of decoder layers.

        """
        _class_embed = ContrastiveAssign()
        _bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(_bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(_bbox_embed.layers[-1].bias.data, 0)
        if dec_pred_bbox_embed_share:
            box_embed_layerlist = [_bbox_embed for _ in range(num_decoder_layers)]
        else:
            box_embed_layerlist = [
                copy.deepcopy(_bbox_embed) for i in range(num_decoder_layers)
            ]
        if dec_pred_class_embed_share:
            class_embed_layerlist = [_class_embed for i in range(num_decoder_layers)]
        else:
            class_embed_layerlist = [
                copy.deepcopy(_class_embed) for i in range(num_decoder_layers)
            ]
        bbox_embed = nn.ModuleList(box_embed_layerlist)
        class_embed = nn.ModuleList(class_embed_layerlist)
        self.bbox_embed = bbox_embed
        self.class_embed = class_embed

        # iniitalize bbox embed and class embed in transformer
        self.transformer.decoder.bbox_embed = bbox_embed
        self.transformer.decoder.class_embed = class_embed

        if self.transformer.two_stage_type != "no":
            if self.transformer.two_stage_bbox_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_bbox_embed = _bbox_embed
            else:
                self.transformer.enc_out_bbox_embed = copy.deepcopy(_bbox_embed)

            if self.transformer.two_stage_class_embed_share:
                assert dec_pred_class_embed_share and dec_pred_bbox_embed_share
                self.transformer.enc_out_class_embed = _class_embed
            else:
                self.transformer.enc_out_class_embed = copy.deepcopy(_class_embed)

            self.refpoint_embed = None

    def _reset_parameters(self):
        # init input_proj
        for proj in self.input_proj:

            nn.init.xavier_uniform_(proj[0].weight, gain=1)
            nn.init.constant_(proj[0].bias, 0)
