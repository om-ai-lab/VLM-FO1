import torch
import torch.nn as nn
import torch.nn.functional as F

from vlm_fo1.model.multimodal_encoder.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
from torchvision.transforms import ToPILImage

class VisionFeaturesGather:
    """
    Collects and manages intermediate features for multi-level visual representation extraction
    (used for region feature/ROIAlign task). Each forward pass (per image) builds up a list of features.
    """
    def __init__(self) -> None:
        self.features_list = []
        self.grid_thw = None
        self.window_index = None
        self.merge_size = None
    
    def reset(self):
        """Clear all states before starting a new feature-gathering process."""
        self.features_list.clear()
        self.grid_thw = None
        self.window_index = None
        self.merge_size = None
    
    def set_params(self, grid_thw, window_index, merge_size):
        """Store spatial and merge information for the current image or batch."""
        self.grid_thw = grid_thw
        self.window_index = window_index
        self.merge_size = merge_size

    def append(self, element):
        """Append a set of features (typically per layer in encoder)."""
        self.features_list.append(element)
    
    def extract_multi_level_features(self):
        """
        Assemble all gathered multi-level features into canonical tensor forms.

        The goal: for each visual sample, produce a list of region-aligned feature maps
        (e.g., multiple stage outputs for downstream region patching/ROIAlign).

        Returns:
            List of features, where each element is a list [stage1, stage2, ...] for one image.
        """
        # Concatenate all feature tensors along hidden dimension: [seq_len, hidden_size * k]
        concat_features = torch.cat(self.features_list, dim=1)
        merge_unit = self.merge_size * self.merge_size
        seq_len = concat_features.shape[0]

        # Rearrange into [windows, merge_unit, hidden_dim*layers]
        concat_features = concat_features.reshape(seq_len // merge_unit, merge_unit, -1)
        reverse_indices = torch.argsort(self.window_index)
        concat_features = concat_features[reverse_indices, :, :]
        concat_features = concat_features.reshape(seq_len, -1)
        
        # Split features for each image/video by product of grid h and w (per sample)
        split_size = (self.grid_thw[:, 1] * self.grid_thw[:, 2]).tolist()
        split_features = list(torch.split(concat_features, split_size, dim=0))
        assert len(split_features) == self.grid_thw.shape[0]
        for i in range(len(split_features)):
            # Recover original grid shape and merge windowing into stages, then split
            _, grid_h, grid_w = self.grid_thw[i]
            merge_h = grid_h // self.merge_size
            merge_w = grid_w // self.merge_size
            split_features[i] = split_features[i].reshape(merge_h, merge_w, merge_unit, -1)
            split_features[i] = split_features[i].reshape(merge_h, merge_w, self.merge_size, self.merge_size, -1)
            split_features[i] = split_features[i].permute(0, 2, 1, 3, 4)
            split_features[i] = split_features[i].flatten(start_dim=0, end_dim=-2)
            # Split [h, w, dim] into k tensors [1, dim/k, h, w] (for compatibility with multi-stage vision encoding)
            hidden_dim = split_features[i].shape[-1]
            split_dim = hidden_dim // len(self.features_list)
            split_features[i] = split_features[i].reshape(grid_h, grid_w, -1)
            split_features[i] = [
                split_features[i][..., j*split_dim:(j+1)*split_dim].permute(2, 0, 1).unsqueeze(0)
                for j in range(len(self.features_list))
            ]

        return split_features

# Global gather object to pass into Qwen2_5_VisionTransformer for monkey-patched feature gathering
GATHER = VisionFeaturesGather()

# --------------------------------- Monkey Patch ---------------------------------------
def custom_forward(self, hidden_states: torch.Tensor, grid_thw: torch.Tensor) -> torch.Tensor:
    """
    Custom forward used with monkey patch to support multi-level feature extraction.
    Applies patch embedding, window partition, position embedding, and passes through all blocks.
    Optionally collects features at each 'fullatt' block for multi-region support.

    Args:
        hidden_states (`torch.Tensor` of shape `(seq_len, hidden_size)`):
            The final hidden states of the model.
        grid_thw (`torch.Tensor` of shape `(num_images_or_videos, 3)`):
            Temporal, height, width of each feature sequence.

    Returns:
        `torch.Tensor`: Final hidden states after MLP head (merger).
    """
    hidden_states = self.patch_embed(hidden_states)
    rotary_pos_emb = self.rot_pos_emb(grid_thw)
    window_index, cu_window_seqlens = self.get_window_index(grid_thw)
    cu_window_seqlens = torch.tensor(
        cu_window_seqlens,
        device=hidden_states.device,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_window_seqlens = torch.unique_consecutive(cu_window_seqlens)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    hidden_states = hidden_states[window_index, :, :]
    hidden_states = hidden_states.reshape(seq_len, -1)
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len // self.spatial_merge_unit, self.spatial_merge_unit, -1)
    rotary_pos_emb = rotary_pos_emb[window_index, :, :]
    rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
    emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
    position_embeddings = (emb.cos(), emb.sin())

    cu_seqlens = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0]).cumsum(
        dim=0,
        # FA2 requires that cu_seqlens_q must have dtype int32
        # torch.onnx.export requires that cu_seqlens_q must match grid_thw dtype
        # See https://github.com/huggingface/transformers/pull/34852 for more info
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    # If monkey-patched feature gather enabled, prepare to collect intermediate features
    if hasattr(self, 'vision_features_gather'):
        self.vision_features_gather.reset()
        self.vision_features_gather.set_params(grid_thw, window_index, self.spatial_merge_size)

    # Forward pass through all transformer blocks; collect intermediate features if needed
    for layer_num, blk in enumerate(self.blocks):
        if layer_num in self.fullatt_block_indexes:
            cu_seqlens_now = cu_seqlens
        else:
            cu_seqlens_now = cu_window_seqlens
        if self.gradient_checkpointing and self.training:
            hidden_states = self._gradient_checkpointing_func(
                blk.__call__, hidden_states, cu_seqlens_now, None, position_embeddings, use_reentrant=False
            )
        else:
            hidden_states = blk(hidden_states, cu_seqlens=cu_seqlens_now, position_embeddings=position_embeddings)
        
        if hasattr(self, 'vision_features_gather'):
            # Capture hidden states at all 'full attention' blocks as multi-level features
            if layer_num in self.fullatt_block_indexes:
                # This property is set by monkey patching
                self.vision_features_gather.append(hidden_states.clone())
    
    hidden_states = self.merger(hidden_states)
    reverse_indices = torch.argsort(window_index)
    hidden_states = hidden_states[reverse_indices, :]

    return hidden_states

def init_vision_features_gather(self, vision_features_gather):
    """
    Helper method for monkey patch to inject a VisionFeaturesGather instance into model.
    """
    self.vision_features_gather = vision_features_gather

def replace_qwen_vit_forward():
    """
    Monkey-patch Qwen2_5_VisionTransformer to use custom forward with multi-level feature support.
    """
    Qwen2_5_VisionTransformerPretrainedModel.forward = custom_forward
    Qwen2_5_VisionTransformerPretrainedModel.init_vision_features_gather = init_vision_features_gather


class Qwen2_5_VlVisionTower(nn.Module):
    """
    Vision backbone wrapper for Qwen2.5-VL (Vision Transformer).
    Handles both standard and region-level (multi-level) encoding with optional monkey patch logic.
    """
    def __init__(self, image_tower, args, delay_load=False, min_pixels=56*56, max_pixels=2048*2048):
        super().__init__()

        self.is_loaded = False

        self.image_tower_name = image_tower
        
        # Determine if multi-level region feature is to be enabled (monkey patch required)
        self.use_vision_tower_region_feature = getattr(args, 'mm_use_vision_tower_region_feature', False)
        if self.use_vision_tower_region_feature:
            replace_qwen_vit_forward()    # Monkey patch: add multi-level feature extraction logic
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.delay_load = delay_load
        print (f"Qwen2_5_VlVisionTower loading_info: delay_load: {delay_load} min_pixels: {min_pixels} max_pixels: {max_pixels}")

        if not delay_load:
            self.load_model()
        else:
            # Defer actual model loading to support (e.g.) model parallel or delayed download scenarios
            self.cfg_only = args.vision_config

    def load_model(self, model_path=None, image_size=336, is_train=True):
        """
        Actually load Qwen2.5 Vision Tower backbone and processor.
        Sets up the image tower and patch feed pipeline.
        """
        self.image_tower = Qwen2_5_VisionTransformerPretrainedModel._from_config(self.cfg_only, attn_implementation="flash_attention_2", torch_dtype=torch.bfloat16)
        # print(f'Qwen2_5_VlVisionTower loading_info: {loading_info}')

        if model_path is not None:
            self.image_processor = Qwen2VLImageProcessor.from_pretrained(model_path, min_pixels=self.min_pixels, max_pixels=self.max_pixels)
        else:
            self.image_processor = Qwen2VLImageProcessor.from_pretrained(self.image_tower_name, min_pixels=self.min_pixels, max_pixels=self.max_pixels)

        if self.use_vision_tower_region_feature:
            # Setup gather instance for monkey-patched feature extraction
            self.image_tower.init_vision_features_gather(GATHER)
        self.is_loaded = True
    
    def convert_image_format(self, image):
        """
        Convert raw image tensor to pre-processed model input tensor and grid shape, using appropriate processor.
        Handles PIL conversion and applies preprocessor for Qwen2.5-VL.
        """
        pil_image = ToPILImage()(image)
        inputs = self.image_processor(images=pil_image, videos=None, return_tensors="pt")
        return inputs['pixel_values'], inputs['image_grid_thw']

    def forward(self, images, image_grid_thws=[]):
        """
        Forward pass for a batch (list) of images.
        Returns image features, gridTHWs, and optional multi-level features for each input image.
        """
        if type(images) is list:
            image_features = []
            multi_level_features_list = []
            output_image_grid_thws = []

            for i, image in enumerate(images):
                # If no grid provided, convert and infer via processor
                if image_grid_thws is None or len(image_grid_thws) == 0:
                    image, image_grid_thw = self.convert_image_format(image=image)
                else:
                    image_grid_thw = image_grid_thws[i]
                image_forward_out = self.image_tower(image.to(device=self.device, dtype=self.dtype), grid_thw=image_grid_thw.to(device=self.device))
                image_feature = image_forward_out.unsqueeze(0).to(self.dtype)

                image_features.append(image_feature)
                output_image_grid_thws.append(image_grid_thw)

                # If region feature mode enabled, collect multi-level features for this image
                if self.use_vision_tower_region_feature:
                    multi_level_features_list.append(self.get_multi_level_features()[0])
                
        else:
            raise NotImplementedError("Qwen2_5_VlVisionTower only supports list-of-image input")

        return image_features, output_image_grid_thws, multi_level_features_list
    
    def get_multi_level_features(self):
        """
        Get the current (last-processed) multi-level region features from the VisionFeaturesGather helper.
        Used in region-feature/ROIAlign branches.
        """
        multi_level_features = self.image_tower.vision_features_gather.extract_multi_level_features()
        return multi_level_features

    @property
    def dummy_feature(self):
        """Returns a zero-vector feature, for use as fallback/null visual token."""
        return torch.zeros(1, self.hidden_size, device=self.device, dtype=self.dtype)

    @property
    def dtype(self):
        """Report vision tower's expected/active tensor dtype (inferred from real weights)."""
        return self.image_tower.dtype

    @property
    def device(self):
        """Report vision tower's tensor device (cuda/cpu) for autoflow/compatibility."""
        return self.image_tower.device

    @property
    def config(self):
        """Yield config, for both loaded-and-ready and 'config only' modes (delay load etc)."""
        if self.is_loaded:
            return self.image_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        """Return backbone output hidden size (for proj or post-processing modules)."""
        return self.config.out_hidden_size

    @property
    def num_patches(self):
        """Return number of vision tokens (patches) in processed image."""
        return (self.config.image_size // self.config.patch_size) ** 2

