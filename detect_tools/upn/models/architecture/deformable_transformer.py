import math
from typing import Dict, List

import torch
import torch.nn as nn

from detect_tools.upn import ARCHITECTURES, build_decoder, build_encoder
from detect_tools.upn.models.utils import (gen_encoder_output_proposals,
                                      inverse_sigmoid)
from detect_tools.upn.ops.modules import MSDeformAttn


@ARCHITECTURES.register_module()
class DeformableTransformer(nn.Module):
    """Implementation of Deformable DETR.

    Args:
        encoder_cfg (Dict): Config for the TransformerEncoder.
        decoder_cfg (Dict): Config for the TransformerDecoder.
        num_queries (int): Number of queries. This is for matching part. Default: 900.
        d_model (int): Dimension of the model. Default: 256.
        num_feature_levels (int): Number of feature levels. Default: 1.
        binary_query_selection (bool): Whether to use binary query selection. Default: False.
            When using binary query selection, a linear with out channe =1 will be used to select
            topk proposals. Otherwise, we will use ContrastiveAssign to select topk proposals.
        learnable_tgt_init (bool): Whether to use learnable target init. Default: True. If False,
            we will use the topk encoder features as the target init.
        random_refpoints_xy (bool): Whether to use random refpoints xy. This is only used when
            two_stage_type is not 'no'. Default: False. If True, we will use random refpoints xy.
        two_stage_type (str): Type of two stage. Default: 'standard'. Options: 'no', 'standard'
        two_stage_learn_wh (bool): Whether to learn the width and height of anchor boxes. Default: False.
        two_stage_keep_all_tokens (bool): If False, the returned hs_enc, ref_enc, init_box_proposal
            will only be the topk proposals. Otherwise, we will return all the proposals from the
            encoder. Default: False.
        two_stage_bbox_embed_share (bool): Whether to share the bbox embedding between the two stages.
            Default: False.
        two_stage_class_embed_share (bool): Whether to share the class embedding between the two stages.
        rm_self_attn_layers (List[int]): The indices of the decoder layers to remove self-attention.
            Default: None.
        rm_detach (bool): Whether to detach the decoder output. Default: None.
        embed_init_tgt (bool): If true, the target embedding is learnable. Otherwise, we will use
            the topk encoder features as the target init. Default: True.
    """

    def __init__(
        self,
        encoder_cfg: Dict,
        decoder_cfg: Dict,
        mask_decoder_cfg: Dict = None,
        num_queries: int = 900,
        d_model: int = 256,
        num_feature_levels: int = 4,
        binary_query_selection: bool = False,
        # init query (target)
        learnable_tgt_init=True,
        random_refpoints_xy=False,
        # for two stage
        two_stage_type: str = "standard",
        two_stage_learn_wh: bool = False,
        two_stage_keep_all_tokens: bool = False,
        two_stage_bbox_embed_share: bool = False,
        two_stage_class_embed_share: bool = False,
        # evo of #anchors
        rm_self_attn_layers: List[int] = None,
        # for detach
        rm_detach: bool = None,
        with_encoder_out: bool = True,
    ) -> None:
        super().__init__()
        self.binary_query_selection = binary_query_selection
        self.num_queries = num_queries
        self.num_feature_levels = num_feature_levels
        self.rm_self_attn_layers = rm_self_attn_layers
        self.d_model = d_model
        self.two_stage_bbox_embed_share = two_stage_bbox_embed_share
        self.two_stage_class_embed_share = two_stage_class_embed_share

        if self.binary_query_selection:
            self.binary_query_selection_layer = nn.Linear(d_model, 1)

        # build encoder
        self.encoder = build_encoder(encoder_cfg)

        # build decoder
        self.decoder = build_decoder(decoder_cfg)
        self.num_decoder_layers = self.decoder.num_layers

        # build sole mask decoder
        if mask_decoder_cfg is not None:
            self.mask_decoder = build_decoder(mask_decoder_cfg)
        else:
            self.mask_decoder = None
        # level embedding
        if num_feature_levels > 1:
            self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        # learnable target embedding
        self.learnable_tgt_init = learnable_tgt_init
        assert learnable_tgt_init, "why not learnable_tgt_init"

        self.tgt_embed = nn.Embedding(num_queries, d_model)
        nn.init.normal_(self.tgt_embed.weight.data)

        # for two stage
        # TODO: this part is really confusing
        self.two_stage_type = two_stage_type
        self.two_stage_learn_wh = two_stage_learn_wh
        self.two_stage_keep_all_tokens = two_stage_keep_all_tokens
        assert two_stage_type in [
            "no",
            "standard",
        ], "unknown param {} of two_stage_type".format(two_stage_type)
        self.with_encoder_out = with_encoder_out
        if two_stage_type == "standard":
            # anchor selection at the output of encoder
            if with_encoder_out:
                self.enc_output = nn.Linear(d_model, d_model)
                self.enc_output_norm = nn.LayerNorm(d_model)

            if two_stage_learn_wh:
                # import ipdb; ipdb.set_trace()
                self.two_stage_wh_embedding = nn.Embedding(1, 2)
            else:
                self.two_stage_wh_embedding = None

        elif two_stage_type == "no":
            self.init_ref_points(
                num_queries, random_refpoints_xy
            )  # init self.refpoint_embed

        self.enc_out_class_embed = None  # this will be initialized outside of the model
        self.enc_out_bbox_embed = None  # this will be initialized outside of the model

        # remove some self_attn_layers or rm_detach
        self._reset_parameters()

        self.rm_self_attn_layers = rm_self_attn_layers
        if rm_self_attn_layers is not None:
            # assert len(rm_self_attn_layers) == num_decoder_layers
            print(
                "Removing the self-attn in {} decoder layers".format(
                    rm_self_attn_layers
                )
            )
            for lid, dec_layer in enumerate(self.decoder.layers):
                if lid in rm_self_attn_layers:
                    dec_layer.rm_self_attn_modules()

        self.rm_detach = rm_detach
        if self.rm_detach:
            assert isinstance(rm_detach, list)
            assert any([i in ["enc_ref", "enc_tgt", "dec"] for i in rm_detach])
        self.decoder.rm_detach = rm_detach

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        if self.num_feature_levels > 1 and self.level_embed is not None:
            nn.init.normal_(self.level_embed)

        if self.two_stage_learn_wh:
            nn.init.constant_(
                self.two_stage_wh_embedding.weight, math.log(0.05 / (1 - 0.05))
            )

    def init_ref_points(self, num_queries: int, random_refpoints_xy: bool = False):
        """Initialize learnable reference points for each query.

        Args:
            num_queries (int): number of queries
            random_refpoints_xy (bool, optional): whether to init the refpoints randomly.
                Defaults to False.
        """
        self.refpoint_embed = nn.Embedding(num_queries, 4)
        if random_refpoints_xy:
            self.refpoint_embed.weight.data[:, :2].uniform_(0, 1)
            self.refpoint_embed.weight.data[:, :2] = inverse_sigmoid(
                self.refpoint_embed.weight.data[:, :2]
            )
            self.refpoint_embed.weight.data[:, :2].requires_grad = False

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(~mask[:, :, 0], 1)
        valid_W = torch.sum(~mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio

    def forward(
        self,
        src_flatten: torch.Tensor,
        lvl_pos_embed_flatten: torch.Tensor,
        level_start_index: List[int],
        spatial_shapes: List[torch.Tensor],
        valid_ratios: List[torch.Tensor],
        mask_flatten: torch.Tensor,
        prompt_type: str,
    ) -> List[torch.Tensor]:
        """Forward function."""
        memory = self.encoder(
            src_flatten,
            pos=lvl_pos_embed_flatten,
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            key_padding_mask=mask_flatten,
        )
        batch_size = src_flatten.shape[0]
        crop_region_features = torch.zeros(batch_size, 1, self.d_model).to(
            memory.device
        )
        if prompt_type == "fine_grained_prompt":
            crop_region_features = (
                self.fine_grained_prompt.weight[0]
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(batch_size, 1, 1)
            )
        elif prompt_type == "coarse_grained_prompt":
            crop_region_features = (
                self.coarse_grained_prompt.weight[0]
                .unsqueeze(0)
                .unsqueeze(0)
                .repeat(batch_size, 1, 1)
            )
        pad_mask = torch.ones(batch_size, 1).to(crop_region_features.device).bool()
        self_attn_mask = torch.ones(batch_size, 1, 1).to(crop_region_features.device)
        ref_dict = dict(
            encoded_ref_feature=crop_region_features,
            pad_mask=pad_mask,
            self_attn_mask=self_attn_mask,
            prompt_type="universal_prompt",
        )

        (
            refpoint_embed,
            tgt,
            init_box_proposal,
        ) = self.get_two_stage_proposal(memory, mask_flatten, spatial_shapes, ref_dict)

        hs, references = self.decoder(
            tgt=tgt.transpose(0, 1),
            tgt_key_padding_mask=None,
            memory=memory.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            pos=lvl_pos_embed_flatten.transpose(0, 1),
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=None,
            # we ~ the mask . False means use the token; True means pad the token
        )
        hs_enc = ref_enc = None
        return (
            hs,
            references,
            ref_dict,
        )

    def get_two_stage_proposal(
        self,
        memory: torch.Tensor,
        mask_flatten: torch.Tensor,
        spatial_shapes: List[torch.Tensor],
        ref_dict: Dict,
    ) -> List[torch.Tensor]:
        """Two stage proposal generation for decoder

        Args:
            memory (torch.Tensor): Image encoded feature. [bs, n, 256]
            mask_flatten (torch.Tensor): Flattened mask. [bs, n]
            spatial_shapes (List[torch.Tensor]): Spatial shapes of each feature map. [bs, num_levels, 2]
            refpoint_embed_dn (torch.Tensor): Denosing refpoint embedding. [bs, num_dn_queries, 256]
            tgt_dn (torch.Tensor): Denosing target embedding. [bs, num_dn_queries, 256]
            ref_dict (Dict): A dict containing all kinds of reference image related features.
        """
        bs = memory.shape[0]
        input_hw = None
        output_memory, output_proposals = gen_encoder_output_proposals(
            memory, mask_flatten, spatial_shapes, input_hw
        )
        output_memory = self.enc_output_norm(self.enc_output(output_memory))

        if self.binary_query_selection:  # Unused
            topk_logits = self.binary_query_selection_layer(output_memory).squeeze(-1)
        else:
            if ref_dict is not None:
                enc_outputs_class_unselected = self.enc_out_class_embed(
                    output_memory, ref_dict
                )  # this is not a linear layer for prediction. But contrastive similaryity, shape [B, len_image, len_text]
            else:
                enc_outputs_class_unselected = self.enc_out_class_embed(output_memory)
            topk_logits = enc_outputs_class_unselected.max(-1)[
                0
            ]  # shape [B, len_image]
        enc_outputs_coord_unselected = (
            self.enc_out_bbox_embed(output_memory) + output_proposals
        )  # (bs, \sum{hw}, 4) unsigmoid
        topk = self.num_queries

        try:
            topk_proposals = torch.topk(topk_logits, topk, dim=1)[1]  # bs, nq
        except:
            raise ValueError(f"dadad {topk_logits.shape}")

        # gather boxes
        refpoint_embed_undetach = torch.gather(
            enc_outputs_coord_unselected,
            1,
            topk_proposals.unsqueeze(-1).repeat(1, 1, 4),
        )  # unsigmoid
        refpoint_embed_ = refpoint_embed_undetach.detach()
        init_box_proposal = torch.gather(
            output_proposals, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
        ).sigmoid()  # sigmoid
        # gather tgt
        tgt_undetach = torch.gather(
            output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.d_model)
        )
        tgt_ = (
            self.tgt_embed.weight[:, None, :].repeat(1, bs, 1).transpose(0, 1)
        )  # nq, bs, d_model
        refpoint_embed, tgt = refpoint_embed_, tgt_

        return (
            refpoint_embed,
            tgt,
            init_box_proposal,
        )
