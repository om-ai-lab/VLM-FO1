import torch
import torch.nn as nn
from typing import List, Union
import torch.nn.functional as F
from torchvision.ops import roi_align
import math

from vlm_fo1.model.multimodal_visual_prompt_encoder.simple_fpn import SimpleFP


def generate_2d_position_embedding(height, width, dim, device):
    """Generate a 2D positional encoding for a feature map.

    Args:
        height (int): Height of the feature map.
        width (int): Width of the feature map.
        dim (int): Dimensionality of the positional embedding (should match channel count).
        device: Torch device on which to allocate tensors.

    Returns:
        pos_embed (Tensor): Positional encoding of shape [H, W, dim].
    """
    # Generate grid coordinate vectors of length H and W
    y_pos = torch.arange(height, dtype=torch.float32, device=device)
    x_pos = torch.arange(width, dtype=torch.float32, device=device)
    
    # Normalize grid values to [0, 1]
    y_pos = y_pos / height
    x_pos = x_pos / width
    
    # Create mesh grid (Y: rows, X: cols)
    y_grid, x_grid = torch.meshgrid(y_pos, x_pos, indexing='ij')
    
    scale = 2 * math.pi
    # Calculate positions for sine/cosine encoding
    quarter_dim = dim // 4
    dim_t = torch.arange(quarter_dim, dtype=torch.float32, device=device)
    dim_t = 10000 ** (2 * (dim_t // 2) / quarter_dim) if quarter_dim > 0 else torch.tensor([1.0], device=device)
    
    # X direction encoding
    x_embed = x_grid.unsqueeze(-1) * scale  # [H, W, 1]
    pos_x = x_embed / dim_t  # [H, W, quarter_dim]
    pos_x = torch.stack((pos_x.sin(), pos_x.cos()), dim=-1).flatten(-2)  # Alternating sin/cos
    
    # Y direction encoding
    y_embed = y_grid.unsqueeze(-1) * scale  # [H, W, 1]
    pos_y = y_embed / dim_t  # [H, W, quarter_dim]
    pos_y = torch.stack((pos_y.sin(), pos_y.cos()), dim=-1).flatten(-2)  # Alternating sin/cos
    
    # Concatenate along the last dimension to make [H, W, dim]
    pos_embed = torch.cat([pos_y, pos_x], dim=-1)
    
    return pos_embed

def gen_sineembed_for_position(pos_tensor, dim_of_pos_feats):
    """Generate sine/cosine positional embedding for ROI position(s).

    Args:
        pos_tensor (Tensor): Shape [batch_size, N, 4] (format: [cx, cy, w, h] in normalized [0, 1])
        dim_of_pos_feats (int): Output embedding dimensionality (#positional channels).

    Returns:
        pos (Tensor): [batch_size, N, dim_of_pos_feats * (2, 4, ...)]
    """
    scale = 2 * math.pi
    dim_t = torch.arange(
        dim_of_pos_feats, dtype=torch.float32, device=pos_tensor.device
    )
    dim_t = 10000 ** (2 * (dim_t // 2) / dim_of_pos_feats)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale

    # Generate encodings for cx, cy
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack(
        (pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3
    ).flatten(2)
    pos_y = torch.stack(
        (pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3
    ).flatten(2)
    if pos_tensor.size(-1) == 2:
        # [cx, cy] input
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        # [cx, cy, w, h] input
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

        # Concatenate encodings for [cy, cx, w, h]
        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError("Unknown pos_tensor shape(-1):{}".format(pos_tensor.size(-1)))
    return pos


class HFREModule(nn.Module):
    """Hybrid Finegrained Region Encoder (HFREModule).

    Handles multi-level ROI region features, optional position embedding, and feature combination for hybrid visual prompt encoding.

    Args:
        roi_output_size (Optional[int]): Output spatial size for ROIAlign.
        region_feature_dim (int): The output dimension for region features.
        apply_position_embedding (bool): Whether positional embedding is used in region features.
        pos_embedding_strategy (str): 'bbox_based', 'feature_map_based', or 'hybrid'.
        use_vt_region_feature_only (bool): Only use vision tower features, skip auxiliary.
        use_vision_tower_region_feature (bool): Whether to include vision tower region features.
        region_feature_combination (str): Combination method: 'concat', 'mean', etc.
        use_separate_mlp_for_regions (bool): Whether to MLP project each region type separately.
        apply_region_layer_norm (bool): Whether to apply layernorm to region features.
        vision_tower_region_feature_dim (int): #channels for vision-tower region feature.
        vision_tower_spatial_scale (float): Spatial scale for vision-tower (for roi_align).
        use_simpleFPN_for_vt (bool): Whether to use FPN on the vision-tower output.
        aux_vision_tower_region_feature_dims (List[int]): Channel dimensions of auxiliary features list.
        aux_vision_tower_spatial_scale (float): Spatial scale for auxiliary vision-tower features.
    """

    def __init__(
        self,
        roi_output_size: int = None,                  # Output spatial size for ROI region features
        region_feature_dim: int = 1024,               # Output dimension for final region feature
        apply_position_embedding: bool = False,       # Whether to apply position embedding
        pos_embedding_strategy: str = 'bbox_based',   # Strategy: 'bbox_based', 'feature_map_based', 'hybrid'
        use_vt_region_feature_only: bool = False,     # Use vision tower (VT) region features only
        use_vision_tower_region_feature: bool = False,# Use vision tower region features (with others)
        region_feature_combination: str = 'concat',   # How to combine aux and vt region features
        use_separate_mlp_for_regions: bool = False,   # MLP-per-region
        apply_region_layer_norm: bool = False,        # Apply LayerNorm

        # Primary vision tower related
        vision_tower_region_feature_dim: int = 5120,    # Dim of the VT region feature
        vision_tower_spatial_scale: float = 1/14,       # Spatial scale of the VT for roi_align
        use_simpleFPN_for_vt: bool = False,             # Use simpleFPN for vision tower

        # Auxiliary vision tower related
        aux_vision_tower_region_feature_dims: List[int] = [256, 512, 1024, 2048],
        aux_vision_tower_spatial_scale: float = None,    # Scale for aux VT
    ):
        super(HFREModule, self).__init__()
        self.roi_output_size = roi_output_size
        self.region_feature_dim = region_feature_dim
        self.apply_position_embedding = apply_position_embedding
        self.pos_embedding_strategy = pos_embedding_strategy
        self.use_vt_region_feature_only = use_vt_region_feature_only
        self.use_vision_tower_region_feature = use_vision_tower_region_feature
        self.region_feature_combination = region_feature_combination
        self.use_separate_mlp_for_regions = use_separate_mlp_for_regions
        self.apply_region_layer_norm = apply_region_layer_norm

        self.vision_tower_region_feature_dim = vision_tower_region_feature_dim
        self.vision_tower_spatial_scale = vision_tower_spatial_scale
        self.use_simpleFPN_for_vt = use_simpleFPN_for_vt

        self.aux_vision_tower_region_feature_dims = aux_vision_tower_region_feature_dims
        self.aux_vision_tower_spatial_scale = aux_vision_tower_spatial_scale

        # Print configuration for debugging
        # print(f"output_size: {self.roi_output_size} use_vision_tower_region_feature: {self.use_vision_tower_region_feature} vision_tower_region_feature_dim: {self.vision_tower_region_feature_dim} "
        #       f"apply_position_embedding: {self.apply_position_embedding} region_feature_combination: {self.region_feature_combination} region_feature_dim: {self.region_feature_dim} use_vt_region_feature_only: {self.use_vt_region_feature_only} "
        #       f"use_simpleFPN_for_vt: {self.use_simpleFPN_for_vt} pos_embedding_strategy: {self.pos_embedding_strategy} "
        #       f"apply_region_layer_norm: {self.apply_region_layer_norm}")
        
        # Optional: FPN for the vision tower input if enabled
        if self.use_simpleFPN_for_vt:
            self.simple_fpn = SimpleFP(out_channels=512, norm="LN", square_pad=0, dim=1280, stride=14)

        # LayerNorm for auxiliary and VT region features, if enabled
        if self.apply_region_layer_norm:
            if self.use_vision_tower_region_feature:
                self.vt_region_norm = nn.LayerNorm(self.vision_tower_region_feature_dim)
            if not self.use_vt_region_feature_only:
                self.aux_region_norm = nn.LayerNorm(sum(self.aux_vision_tower_region_feature_dims))
            
        # Optionally, a projection MLP if using certain combination strategies
        if self.use_vision_tower_region_feature and self.region_feature_combination in ['mean', 'mean_sep_pos', 'mean_aux_pos', 'mean_sep_no_vt_pos']:
            self.vision_tower_region_feature_projector = nn.Sequential(
                nn.Linear(vision_tower_region_feature_dim, region_feature_dim),
                nn.GELU(),
                nn.Linear(region_feature_dim, region_feature_dim)
            )    
        
        # Two MLP heads if regions are projected separately (for concat mode)
        if self.use_vision_tower_region_feature and self.use_separate_mlp_for_regions:
            self.vt_region_mlp = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.GELU(),
                nn.Linear(1024, 1024)
            )
            self.aux_region_mlp = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.GELU(),
                nn.Linear(1024, 1024)
            )

    def _apply_feature_map_position_embedding(self, features):
        """Apply 2D position embedding to each feature map in a feature pyramid, if enabled.

        Args:
            features (List[Tensor]): Each is [B, C, H, W] per FPN level.

        Returns:
            List[Tensor]: Feature maps with position embedding applied, shape unchanged.
        """
        enhanced_features = []
        for level_idx, feature in enumerate(features):
            if self.apply_position_embedding and self.pos_embedding_strategy in ['feature_map_based', 'hybrid']:
                B, C, H, W = feature.shape

                # Generate position embedding matching channel dimension
                pos_embed = generate_2d_position_embedding(
                    H, W, C, feature.device
                )  # [H, W, C]
                
                # Reshape to [1, C, H, W] and add
                pos_embed = pos_embed.permute(2, 0, 1).unsqueeze(0)
                feature = feature + pos_embed.to(feature.dtype)
            enhanced_features.append(feature)
        return enhanced_features

    def extract_vt_region_feature(self, multi_level_features, boxes: Union[torch.Tensor, List[torch.Tensor]]) -> torch.Tensor:
        """Extract vision-tower region features via ROIAlign over FPN features, with spatial scaling.

        Args:
            multi_level_features (List[Tensor]): Per-FPN level features, [B, C, H, W].
            boxes (Union[Tensor, List[Tensor]]): ROI bounding boxes for roi_align.

        Returns:
            Tensor: [1, N, C=tower_channels]
        """
        if self.use_simpleFPN_for_vt:
            # If using FPN for vision tower: apply FPN and select fixed spatial scales (hardcoded stride)
            multi_level_features = self.simple_fpn(multi_level_features)
            roi_features_per_level = []
            # Hardcoded feature strides for each FPN stage; tweak if arch changes
            feature_strides = [3.5, 7, 14, 28]  
            for level_idx, level_feature in enumerate(multi_level_features):
                current_spatial_scale = 1.0 / feature_strides[level_idx]
                level_roi_feat = roi_align(
                    level_feature.float(),
                    boxes,
                    output_size=self.roi_output_size,
                    spatial_scale=current_spatial_scale
                )
                # Pool across H,W to get region feature per ROI
                level_roi_feat = level_roi_feat.mean(dim=(2, 3))
                roi_features_per_level.append(level_roi_feat)
            out_box_feat = torch.cat(roi_features_per_level, dim=1).unsqueeze(0)
        else:
            # If not using FPN: concatenate all feature levels on channel axis and ROI-align once
            concat_multi_level_feature = []
            concat_multi_level_feature = torch.cat(multi_level_features, dim=1)

            out_box_feat = roi_align(
                concat_multi_level_feature.float(),
                boxes,
                output_size=self.roi_output_size,
                spatial_scale=self.vision_tower_spatial_scale,
            )
            # Pool per ROI for (1, N, C_total)
            out_box_feat = out_box_feat.mean(dim=(2, 3)).reshape(
                1, out_box_feat.shape[0], out_box_feat.shape[1]
            )
        return out_box_feat
    
    def __call__(
        self,
        aux_multi_level_features: List[torch.Tensor],
        aux_boxes: Union[torch.Tensor, List[torch.Tensor]],
        vt_multi_level_features = None,
        vt_boxes: Union[torch.Tensor, List[torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Main forward. Extracts ROI region features with possible hybrid VT/aux, applies position embedding and combines as configured.

        Args:
            aux_multi_level_features (List[Tensor]): Auxiliary vision features (e.g., from FPN, [B, C, H, W]).
            aux_boxes (Union[Tensor, List[Tensor]]): ROIs in [N, 4] xyxy.
            vt_multi_level_features (optional): Vision tower features.
            vt_boxes (optional): Vision tower's box coordinates ([N, 4]).

        Returns:
            Tensor: Region features of shape [1, N, C], N=#ROIs.
        """
        if self.use_vt_region_feature_only:
            # Only use VT region features (skip aux completely)
            out_box_feat = self.extract_vt_region_feature(vt_multi_level_features, vt_boxes)

            if self.apply_position_embedding:
                # Add position embedding to VT region feature
                pos_boxes = vt_boxes[0]  # (N, 4)
                pos_boxes = pos_boxes.to(out_box_feat.dtype)
                vt_max_height = max([feature.shape[-2] for feature in vt_multi_level_features])
                vt_max_width = max([feature.shape[-1] for feature in vt_multi_level_features])
                original_img_width = vt_max_width / self.vision_tower_spatial_scale
                original_img_height = vt_max_height / self.vision_tower_spatial_scale
                # Normalize box coordinates by image size
                pos_boxes[:, [0, 2]] = pos_boxes[:, [0, 2]] / original_img_width
                pos_boxes[:, [1, 3]] = pos_boxes[:, [1, 3]] / original_img_height
                # Convert from (x1, y1, x2, y2) to (cx, cy, w, h)
                pos_boxes[:, 2] = pos_boxes[:, 2] - pos_boxes[:, 0]
                pos_boxes[:, 3] = pos_boxes[:, 3] - pos_boxes[:, 1]
                pos_boxes[:, 0] = pos_boxes[:, 0] + pos_boxes[:, 2] / 2
                pos_boxes[:, 1] = pos_boxes[:, 1] + pos_boxes[:, 3] / 2
                # Add sine/cos position embedding
                pos_embed = gen_sineembed_for_position(pos_boxes.unsqueeze(0), self.region_feature_dim // 4)
                out_box_feat = out_box_feat + pos_embed
            return out_box_feat
        
        # Otherwise: hybrid mode (aux + possibly VT region features)
        aux_boxes[0] = aux_boxes[0].float()
        
        # Collect all auxiliary features at the same (max) spatial size for channel concat
        concat_multi_level_feature = []
        max_height = max([feature.shape[2] for feature in aux_multi_level_features])
        max_width = max([feature.shape[3] for feature in aux_multi_level_features])
        
        # Optionally apply 2D position encoding at the feature map level (before concat/roi_align)
        if self.pos_embedding_strategy in ['feature_map_based', 'hybrid']:
            # Option: compute stride info for each level for debugging/extension
            feature_strides = []
            for feature in aux_multi_level_features:
                stride = max_height / feature.shape[2]
                feature_strides.append(stride)
            aux_multi_level_features = self._apply_feature_map_position_embedding(
                aux_multi_level_features
            )
        
        # Interpolate all features to (max_height,max_width), then concat along channel
        for level, feature in enumerate(aux_multi_level_features):
            if level != 0:
                concat_multi_level_feature.append(
                    F.interpolate(
                        feature.float(),
                        size=(max_height, max_width),
                        mode="bilinear",
                        align_corners=False,
                    )
                )
            else:
                concat_multi_level_feature.append(feature.float())
        concat_multi_level_feature = torch.cat(concat_multi_level_feature, dim=1)

        # Extract region feature for all boxes using roi_align
        out_box_aux_feat = roi_align(
            concat_multi_level_feature,
            aux_boxes,
            output_size=self.roi_output_size,
            spatial_scale=self.aux_vision_tower_spatial_scale
        )
        
        # Pool H,W to get final shape (1, Nbox, C)
        out_box_aux_feat = out_box_aux_feat.mean(dim=(2, 3)).reshape(
            1, out_box_aux_feat.shape[0], out_box_aux_feat.shape[1]
        )

        if self.apply_region_layer_norm:
            out_box_aux_feat = self.aux_region_norm.float()(out_box_aux_feat)
        
        if self.use_vision_tower_region_feature:
            # If also using vision-tower features
            out_box_vt_feat = self.extract_vt_region_feature(vt_multi_level_features, vt_boxes)
            if self.apply_region_layer_norm:
                out_box_vt_feat = self.vt_region_norm.float()(out_box_vt_feat)
            if self.region_feature_combination in ['mean', 'mean_aux_pos']:
                # Combine by mean
                out_box_feat = (out_box_aux_feat + out_box_vt_feat) / 2
            elif self.region_feature_combination in ['concat', 'concat_aux_pos']:
                # Optionally MLP each before concat
                if self.use_separate_mlp_for_regions:
                    original_vt_dtype = out_box_vt_feat.dtype
                    original_aux_dtype = out_box_aux_feat.dtype
                    out_box_vt_feat = self.vt_region_mlp(out_box_vt_feat.to(self.vt_region_mlp[0].weight.dtype)).to(original_vt_dtype)
                    out_box_aux_feat = self.aux_region_mlp(out_box_aux_feat.to(self.aux_region_mlp[0].weight.dtype)).to(original_aux_dtype)
                out_box_feat = torch.cat([out_box_aux_feat, out_box_vt_feat], dim=-1)
            elif self.region_feature_combination in ['concat_sep_pos', 'mean_sep_pos', 'concat_sep_no_vt_pos', 'mean_sep_no_vt_pos']:
                # Compute position embedding separately for aux and vt features
                # Use `aux_boxes` for aux and `vt_boxes` for vt
                vt_dim = 5120 if self.region_feature_combination == 'concat_sep_pos' else 2880

                # Aux region: positional embedding using aux_boxes
                aux_pos_boxes = aux_boxes[0].to(out_box_aux_feat.dtype)  # (N, 4)
                aux_original_img_width = max_width / self.aux_vision_tower_spatial_scale
                aux_original_img_height = max_height / self.aux_vision_tower_spatial_scale
                
                aux_pos_boxes[:, [0, 2]] = aux_pos_boxes[:, [0, 2]] / aux_original_img_width
                aux_pos_boxes[:, [1, 3]] = aux_pos_boxes[:, [1, 3]] / aux_original_img_height
                aux_pos_boxes[:, 2] = aux_pos_boxes[:, 2] - aux_pos_boxes[:, 0]
                aux_pos_boxes[:, 3] = aux_pos_boxes[:, 3] - aux_pos_boxes[:, 1]
                aux_pos_boxes[:, 0] = aux_pos_boxes[:, 0] + aux_pos_boxes[:, 2] / 2
                aux_pos_boxes[:, 1] = aux_pos_boxes[:, 1] + aux_pos_boxes[:, 3] / 2
                aux_pos_embed = gen_sineembed_for_position(
                    aux_pos_boxes.unsqueeze(0), 2880 // 4
                )
                out_box_aux_feat = out_box_aux_feat + aux_pos_embed
                
                # Only apply VT position embedding in these combos:
                # For *_no_vt_pos: skip vt feature position embedding
                if self.region_feature_combination in ['concat_sep_no_vt_pos', 'mean_sep_no_vt_pos']:
                    pass
                else:
                    # VT region: positional embedding using vt_boxes
                    vt_pos_boxes = vt_boxes[0].to(out_box_vt_feat.dtype)  # (N, 4)
                    vt_max_height = max([feature.shape[2] for feature in vt_multi_level_features])
                    vt_max_width = max([feature.shape[3] for feature in vt_multi_level_features])
                    vt_original_img_width = vt_max_width / self.vision_tower_spatial_scale
                    vt_original_img_height = vt_max_height / self.vision_tower_spatial_scale

                    vt_pos_boxes[:, [0, 2]] = vt_pos_boxes[:, [0, 2]] / vt_original_img_width
                    vt_pos_boxes[:, [1, 3]] = vt_pos_boxes[:, [1, 3]] / vt_original_img_height
                    vt_pos_boxes[:, 2] = vt_pos_boxes[:, 2] - vt_pos_boxes[:, 0]
                    vt_pos_boxes[:, 3] = vt_pos_boxes[:, 3] - vt_pos_boxes[:, 1]
                    vt_pos_boxes[:, 0] = vt_pos_boxes[:, 0] + vt_pos_boxes[:, 2] / 2
                    vt_pos_boxes[:, 1] = vt_pos_boxes[:, 1] + vt_pos_boxes[:, 3] / 2
                    vt_pos_embed = gen_sineembed_for_position(
                        vt_pos_boxes.unsqueeze(0), vt_dim // 4
                    )
                    out_box_vt_feat = out_box_vt_feat + vt_pos_embed

                # Merge aux and vt region features (by cat or mean)
                if self.region_feature_combination in ['concat_sep_pos', 'concat_sep_no_vt_pos']:
                    out_box_feat = torch.cat([out_box_aux_feat, out_box_vt_feat], dim=-1)
                elif self.region_feature_combination in ['mean_sep_pos', 'mean_sep_no_vt_pos']:
                    out_box_feat = (out_box_aux_feat + out_box_vt_feat) / 2
                

        # If enabled: add single positional embedding (bbox-based, not separate for each region type)
        if self.apply_position_embedding and self.region_feature_combination not in ['concat_sep_pos', 'mean_sep_pos', 'concat_sep_no_vt_pos', 'mean_sep_no_vt_pos']:
            # Only apply if position embedding strategy matches
            apply_bbox_pos_embed = (self.pos_embedding_strategy == 'bbox_based' or self.pos_embedding_strategy == 'hybrid')
            
            if apply_bbox_pos_embed:
                # Use vt_boxes unless not enabled or configured otherwise
                if self.use_vision_tower_region_feature and vt_boxes is not None and self.region_feature_combination not in ['concat_aux_pos', 'mean_aux_pos']:
                    pos_boxes = vt_boxes[0]  # (N, 4)
                    vt_max_height = max([feature.shape[-2] for feature in vt_multi_level_features])
                    vt_max_width = max([feature.shape[-1] for feature in vt_multi_level_features])
                    vt_spatial_scale = self.vision_tower_spatial_scale
                    original_img_width = vt_max_width / vt_spatial_scale
                    original_img_height = vt_max_height / vt_spatial_scale
                else:
                    max_width = max([feature.shape[3] for feature in aux_multi_level_features])
                    max_height = max([feature.shape[2] for feature in aux_multi_level_features])
                    pos_boxes = aux_boxes[0]  # (N, 4)
                    original_img_width = max_width / self.aux_vision_tower_spatial_scale
                    original_img_height = max_height / self.aux_vision_tower_spatial_scale
                
                pos_boxes = pos_boxes.to(out_box_feat.dtype)
                pos_boxes[:, [0, 2]] = pos_boxes[:, [0, 2]] / original_img_width
                pos_boxes[:, [1, 3]] = pos_boxes[:, [1, 3]] / original_img_height
                # Convert box to center format
                pos_boxes[:, 2] = pos_boxes[:, 2] - pos_boxes[:, 0]
                pos_boxes[:, 3] = pos_boxes[:, 3] - pos_boxes[:, 1]
                pos_boxes[:, 0] = pos_boxes[:, 0] + pos_boxes[:, 2] / 2
                pos_boxes[:, 1] = pos_boxes[:, 1] + pos_boxes[:, 3] / 2
                pos_embed = gen_sineembed_for_position(
                    pos_boxes.unsqueeze(0), self.region_feature_dim // 4
                )
                out_box_feat = out_box_feat + pos_embed

        return out_box_feat