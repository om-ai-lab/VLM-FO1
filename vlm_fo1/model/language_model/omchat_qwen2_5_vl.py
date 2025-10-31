from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn

from transformers import Qwen2_5_VLConfig, AutoConfig, AutoModelForCausalLM
from vlm_fo1.model.multimodal_encoder.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLModel, Qwen2_5_VLForConditionalGeneration, Qwen2_5_VLCausalLMOutputWithPast
from vlm_fo1.model.multimodal_encoder.qwen2_5_vl_encoder import Qwen2_5_VlVisionTower
from vlm_fo1.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_REGION_INDEX, QWEN2_5_VL_IMAGE_TOKEN, QWEN2_5_VL_IMAGE_TOKEN_INDEX

from ..omchat_arch import OmChatMetaModel, OmChatMetaForCausalLM

# Custom config which extends Qwen2_5_VLConfig for OmChat multimodal model
class OmChatQwen25VLConfig(Qwen2_5_VLConfig):
    model_type = "omchat_qwen2_5_vl"
    rotary_type = "normal_rotary"
    multi_scale_im = None
    vision_tower_aux = None

# Core model definition: inherits from OmChat and Qwen multimodal base
class OmChatQwen25VLModel(OmChatMetaModel, Qwen2_5_VLModel):
    config_class = OmChatQwen25VLConfig

    def __init__(self, config: Qwen2_5_VLConfig):
        super(OmChatQwen25VLModel, self).__init__(config)

# Main class for multimodal CausalLM
class OmChatQwen25VLForCausalLM(Qwen2_5_VLForConditionalGeneration, OmChatMetaForCausalLM):
    config_class = OmChatQwen25VLConfig

    def __init__(self, config, delay_load=True):
        # Ensure config has delay_load property
        if not hasattr(config, 'delay_load'):
            config.delay_load = delay_load
        super(Qwen2_5_VLForConditionalGeneration, self).__init__(config)
        self.model = OmChatQwen25VLModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.rope_deltas = None  # cache rope_deltas here

        self.post_init()

    # Encode input images into feature representations
    def encode_images(self, images, images_grid_thw=None):
        # If vision_tower is Qwen2.5-specific, use its custom forward signature
        if isinstance(self.get_model().get_vision_tower(), Qwen2_5_VlVisionTower):
            image_features = self.get_model().get_vision_tower()(images, images_grid_thw)
            image_features, image_grid_thws, multi_level_features = image_features
            # If multiple images, handle concatenation
            if type(image_features) is list:
                # List has items of shape (1, seq_len, dim)
                token_length_list = [i.shape[1] for i in image_features]
                image_features = torch.cat(image_features, dim=1)      # Concatenate to (1, total_seq_len, dim)
        else:
            image_features = self.get_model().get_vision_tower()(images)
            image_grid_thws = None
            multi_level_features = None

        image_features = self.get_model().mm_projector(image_features)

        # Split concatenated image features back by original lengths (for multi-image case)
        if isinstance(self.get_model().get_vision_tower(), Qwen2_5_VlVisionTower):
            start = 0
            new_image_features = []
            # Split according to token_length_list
            for length in token_length_list:
                end = start + length
                new_image_features.append(image_features[:, start:end, :].squeeze(0))
                start = end
            image_features = new_image_features
        
        return image_features, image_grid_thws, multi_level_features
    
    # Encode region regions (bounding boxes) into features, optionally using auxiliary vision tower
    def encode_regions(self, images, bbox_list, vt_multi_level_features=None, vt_images_size=None):
        aux_image_features_list = self.get_model().get_vision_tower_aux()(images)
        region_features = []
        if getattr(self.config, "mm_use_vision_tower_region_feature", False):
            image_features_list = vt_multi_level_features
            for batch_idx, (image_features, aux_image_features) in enumerate(zip(image_features_list, aux_image_features_list)):

                if getattr(self.config, "mm_use_simpleFPN_for_vt", False):
                    multilevel_visual_feats = image_features[-1]
                else:
                    multilevel_visual_feats = image_features
                multilevel_aux_visual_feats = aux_image_features["image_features"]
                boxes = bbox_list[batch_idx]

                # If no boxes provided, use dummy box (covers tiny region)
                if boxes is None or len(boxes) == 0:
                    boxes = torch.tensor([[0, 10, 0, 10]], device=multilevel_aux_visual_feats[0].device, dtype=torch.float32)

                boxes = boxes.to(torch.float32).to(multilevel_aux_visual_feats[0].device)
                current_image_height, current_image_width = images[batch_idx].shape[-2:]
                original_height, original_width = vt_images_size[batch_idx]
                # Scale bounding boxes from original image size to processed size
                scale_height = original_height / current_image_height
                scale_width = original_width / current_image_width
                vt_boxes = boxes * torch.tensor([scale_width, scale_height, scale_width, scale_height], device=boxes.device)

                extracted_region_feat = self.get_model().object_vp_extractor(
                    aux_multi_level_features=multilevel_aux_visual_feats,
                    vt_multi_level_features=multilevel_visual_feats,
                    aux_boxes=[boxes],
                    vt_boxes=[vt_boxes]
                ).squeeze(0).to(multilevel_aux_visual_feats[0].dtype)
                region_feat = self.get_model().mm_projector_aux(extracted_region_feat) # [num_bbox, 2048]
                region_features.append(region_feat)
        else:
            # Extract region features only from auxiliary vision tower
            for batch_idx, image_features in enumerate(aux_image_features_list):
                multilevel_visual_feats = image_features["image_features"]
                last_feat = image_features["last_feat"]
                boxes = bbox_list[batch_idx]

                if boxes is None or len(boxes) == 0:
                    boxes = torch.tensor([[0, 10, 0, 10]], device=multilevel_visual_feats[0].device, dtype=torch.float32)
               
                multi_level_aux_features = multilevel_visual_feats
                boxes = boxes.to(torch.float32).to(multi_level_aux_features[0].device)
                extracted_region_feat = self.get_model().object_vp_extractor(
                    multi_level_aux_features,
                    [boxes],
                ).squeeze(0).to(multi_level_aux_features[0].dtype)
                region_feat = self.get_model().mm_projector_aux(extracted_region_feat) # [num_bbox, 2880]
                region_features.append(region_feat)

        return region_features
    
    def get_model(self):
        # Getter for model. Used to access backbone/model internals.
        return self.model

    # Convert sequence of input_ids/labels/images/boxes to multimodal embedding and associated masks/ids for transformer input.
    def prepare_inputs_labels_for_qwen2_5_vl_multimodal(
        self, input_ids, position_ids, attention_mask, past_key_values, labels, images, images_aux=None, bbox_list=None, image_grid_thws=None
    ):
        # ========================== Above this line, input parsing and batching =============================
        vision_tower = self.get_vision_tower()
        video_tower = self.get_video_tower()
        vision_tower_aux = self.get_vision_tower_aux()
        # Fast-path for non-multimodal case or first step in generation (i.e. only one token in input)
        if (vision_tower is None and video_tower is None) or images is None or input_ids.shape[1] == 1:
            if past_key_values is not None and (vision_tower is not None or video_tower is not None) and images is not None and input_ids.shape[1] == 1:

                target_shape = past_key_values[-1][-1].shape[-2] + 1
                attention_mask = torch.cat((attention_mask, torch.ones(
                    (attention_mask.shape[0], target_shape - attention_mask.shape[1]),
                    dtype=attention_mask.dtype,
                    device=attention_mask.device
                )), dim=1)

                position_ids=None
                cache_position = torch.tensor([target_shape - 1],device=attention_mask.device)
            return input_ids, position_ids, attention_mask, past_key_values, None, labels, None, cache_position

        # Indices for images (3D or 2D tensors) and videos (4D tensors)
        image_idx = [idx for idx, img in enumerate(images) if img.ndim == 3 or img.ndim == 2]
        is_all_image = len(image_idx) == len(images)
        video_idx = [idx for idx, vid in enumerate(images) if vid.ndim == 4]
        
        # Stack image and video tensors accordingly for mini-batch processing
        if isinstance(vision_tower, Qwen2_5_VlVisionTower):
            images_minibatch = [images[idx] for idx in image_idx] if len(image_idx) > 0 else []  # list of [c,h,w], can have variable shapes
        else:
            images_minibatch = torch.stack([images[idx] for idx in image_idx]) if len(image_idx) > 0 else []  # tensor [mini_b, c, h, w]
        videos_minibatch = torch.stack([images[idx] for idx in video_idx]) if len(video_idx) > 0 else []  # tensor [mini_b, c, t, h, w]

        # Auxiliary batch for region encoding, if relevant
        if vision_tower_aux is not None and images_aux is not None:
            images_minibatch_aux = [images_aux[idx].unsqueeze(0) for idx in image_idx] if len(image_idx) > 0 else []  # list of [1, c, h, w]

        # tmp_image_features will be indexed to scatter extracted image/video features into original batch positions
        tmp_image_features = [None] * (len(image_idx) + len(video_idx))
        if getattr(images_minibatch, 'ndim', 0) == 4 or (type(images_minibatch) is list and len(images_minibatch) > 0):  # batch consists of images, [mini_b, c, h, w]
            if vision_tower is not None:
                image_features_minibatch, image_grid_thws_minibatch, vt_multi_level_features_minibatch = self.encode_images(images_minibatch, image_grid_thws)  # [mini_b, l, c]
            else:
                image_features_minibatch = torch.randn(1).to(self.device)  # dummy feature for video-only training under tuning

            # Map extracted image features back to their places in the original batch
            for i, pos in enumerate(image_idx):
                tmp_image_features[pos] = image_features_minibatch[i]
            
            # Handle auxiliary region features if enabled and boxes provided
            if vision_tower_aux is not None and bbox_list is not None and len(bbox_list) > 0:
                if isinstance(self.get_model().get_vision_tower(), Qwen2_5_VlVisionTower):
                    patch_size = self.get_model().get_vision_tower().config.patch_size
                    vt_images_size_minibatch = [im_grid_thw[0][-2:]*patch_size for im_grid_thw in image_grid_thws]
                    region_features = self.encode_regions(images_minibatch_aux, bbox_list, vt_multi_level_features_minibatch, vt_images_size_minibatch)  # [mini_b, l, c]
            else:
                region_features = None

        # Same as above, but for video features if any
        if getattr(videos_minibatch, 'ndim', 0) == 5:  # batch consists of videos, [mini_b, c, t, h, w]
            video_features_minibatch = self.encode_videos(videos_minibatch)  # fake list [mini_b, t, l, c]
            for i, pos in enumerate(video_idx):
                tmp_image_features[pos] = video_features_minibatch[i]

        # Flatten image feature slot list to proper order for current batch
        new_tmp = []
        for image in tmp_image_features:
            # If multi-image per item, flatten out
            if isinstance(image, list):
                t = len(image)
                for i in range(t):
                    new_tmp.append(image[i])
            else:
                new_tmp.append(image)
        image_features = new_tmp

        # =========================== Now, build multimodal input & target sequences =========================

        if getattr(self.config, 'tune_mm_mlp_adapter', False) and getattr(self.config, 'mm_use_im_start_end', False):
            raise NotImplementedError

        _labels = labels
        _position_ids = position_ids
        _attention_mask = attention_mask

        # Default construction of masks etc.
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.bool)
        else:
            attention_mask = attention_mask.bool()
        if position_ids is None:
            position_ids = torch.arange(0, input_ids.shape[1], dtype=torch.long, device=input_ids.device)
        if labels is None:
           labels = torch.full_like(input_ids, IGNORE_INDEX)

        # For each batch item, strip padded tokens based on attention_mask
        input_ids = [cur_input_ids[cur_attention_mask] for cur_input_ids, cur_attention_mask in zip(input_ids, attention_mask)]
        labels = [cur_labels[cur_attention_mask] for cur_labels, cur_attention_mask in zip(labels, attention_mask)]

        # If neither region auxiliary nor bboxes present: process classic image-text input
        if vision_tower_aux is None and (bbox_list is None or all(x is None for x in bbox_list)):
            new_input_embeds = []
            new_labels = []
            new_input_ids = []
            cur_image_idx = 0
            image_nums_in_batch = []
            
            for batch_idx, cur_input_ids in enumerate(input_ids):
                num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
                image_nums_in_batch.append(num_images)
                # If there are no image markers, just get text features
                if num_images == 0:
                    cur_image_features = image_features[cur_image_idx]
                    cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                    cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0]], dim=0)
                    new_input_embeds.append(cur_input_embeds)
                    new_labels.append(labels[batch_idx])
                    new_input_ids.append(cur_input_ids)
                    cur_image_idx += 1
                    continue

                # Split on image token indices: replace them with image features after conversion
                image_token_indices = [-1] + torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist() + [cur_input_ids.shape[0]]
                cur_input_ids_noim = []
                cur_labels = labels[batch_idx]
                cur_labels_noim = []
                for i in range(len(image_token_indices) - 1):
                    cur_input_ids_noim.append(cur_input_ids[image_token_indices[i]+1:image_token_indices[i+1]])
                    cur_labels_noim.append(cur_labels[image_token_indices[i]+1:image_token_indices[i+1]])
                split_sizes = [x.shape[0] for x in cur_labels_noim]
                cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_noim))
                cur_input_embeds_no_im = torch.split(cur_input_embeds, split_sizes, dim=0)

                cur_new_input_embeds = []
                cur_new_labels = []
                cur_new_input_ids = []
                for i in range(num_images + 1):
                    # Interleave text and image features
                    cur_new_input_embeds.append(cur_input_embeds_no_im[i])
                    cur_new_labels.append(cur_labels_noim[i])
                    cur_new_input_ids.append(cur_input_ids_noim[i])
                    if i < num_images:
                        cur_image_features = image_features[cur_image_idx].to(self.device)
                        cur_image_idx += 1
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                        cur_new_input_ids.append(torch.full((cur_image_features.shape[0],), self.config.image_token_id, device=cur_labels.device, dtype=cur_labels.dtype))
                cur_new_input_embeds = torch.cat(cur_new_input_embeds)
                cur_new_labels = torch.cat(cur_new_labels)
                cur_new_input_ids = torch.cat(cur_new_input_ids)

                new_input_embeds.append(cur_new_input_embeds)
                new_labels.append(cur_new_labels)
                new_input_ids.append(cur_new_input_ids)
        # If region markers or region features enabled in config
        else:
            new_input_embeds = []
            new_labels = []
            new_input_ids = []
            cur_image_idx = 0
            image_nums_in_batch = []
            
            for batch_idx, cur_input_ids in enumerate(input_ids):
                cur_region_idx = 0 
                # Detect image and region special token counts
                num_images = (cur_input_ids == IMAGE_TOKEN_INDEX).sum()
                num_regions = (cur_input_ids == DEFAULT_REGION_INDEX).sum() if DEFAULT_REGION_INDEX in cur_input_ids else 0
                image_nums_in_batch.append(num_images)

                # If no markers, just do text embedding for this item
                if num_images == 0 and num_regions == 0:
                    cur_image_features = image_features[cur_image_idx]
                    cur_region_features = region_features[cur_region_idx]
                    cur_input_embeds_1 = self.get_model().embed_tokens(cur_input_ids)
                    cur_input_embeds = torch.cat([cur_input_embeds_1, cur_image_features[0:0], cur_region_features[0:0]], dim=0)
                    new_input_embeds.append(cur_input_embeds)
                    new_labels.append(labels[batch_idx])
                    new_input_ids.append(cur_input_ids)
                    cur_image_idx += 1
                    continue

                # Get all special marker indices (image/region)
                image_indices = torch.where(cur_input_ids == IMAGE_TOKEN_INDEX)[0].tolist()
                region_indices = torch.where(cur_input_ids == DEFAULT_REGION_INDEX)[0].tolist() if num_regions > 0 else []
                all_special_indices = sorted([-1] + image_indices + region_indices + [cur_input_ids.shape[0]])

                # Split out plain text chunks between special markers
                cur_input_ids_segments = []
                cur_labels = labels[batch_idx]
                cur_labels_segments = []
                
                for i in range(len(all_special_indices) - 1):
                    cur_input_ids_segments.append(cur_input_ids[all_special_indices[i]+1:all_special_indices[i+1]])
                    cur_labels_segments.append(cur_labels[all_special_indices[i]+1:all_special_indices[i+1]])

                # Project text ids to word embeddings
                split_sizes = [x.shape[0] for x in cur_labels_segments]
                cur_input_embeds = self.get_model().embed_tokens(torch.cat(cur_input_ids_segments))
                if num_regions == 0 and vision_tower_aux is not None and region_features is not None:
                    cur_region_features = region_features[cur_region_idx]
                    temp_input_embeds = torch.cat([cur_input_embeds, cur_region_features[0:0]], dim=0)
                    cur_input_embeds = temp_input_embeds
                
                cur_input_embeds_segments = torch.split(cur_input_embeds, split_sizes, dim=0)

                # Reassemble text and image/region segments in order
                cur_new_input_embeds = []
                cur_new_labels = []
                cur_new_input_ids = []
                
                for i in range(len(all_special_indices) - 1):
                    # Insert current text segment
                    cur_new_input_embeds.append(cur_input_embeds_segments[i])
                    cur_new_labels.append(cur_labels_segments[i])
                    cur_new_input_ids.append(cur_input_ids_segments[i])
                    # If next is image, insert feature representation
                    if all_special_indices[i+1] in image_indices:
                        cur_image_features = image_features[cur_image_idx].to(self.device)
                        cur_image_idx += 1
                        cur_new_input_embeds.append(cur_image_features)
                        cur_new_labels.append(torch.full((cur_image_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                        cur_new_input_ids.append(torch.full((cur_image_features.shape[0],), self.config.image_token_id, device=cur_labels.device, dtype=cur_labels.dtype))
                        
                    # If next is region token, insert extracted region features
                    elif all_special_indices[i+1] in region_indices:
                        cur_region_features = region_features[batch_idx][cur_region_idx].to(self.device).unsqueeze(0)
                        cur_region_idx += 1
                        cur_new_input_embeds.append(cur_region_features)
                        
                        cur_new_labels.append(torch.full((cur_region_features.shape[0],), IGNORE_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                        cur_new_input_ids.append(torch.full((cur_region_features.shape[0],), DEFAULT_REGION_INDEX, device=cur_labels.device, dtype=cur_labels.dtype))
                # Combine for this batch item
                cur_new_input_embeds = torch.cat(cur_new_input_embeds)
                cur_new_labels = torch.cat(cur_new_labels)
                cur_new_input_ids = torch.cat(cur_new_input_ids)
                new_input_embeds.append(cur_new_input_embeds)
                new_labels.append(cur_new_labels)
                new_input_ids.append(cur_new_input_ids)
        # Truncate sequences to maximum model length, if image+region tokens caused overflow
        tokenizer_model_max_length = getattr(self.config, 'tokenizer_model_max_length', None)
        if tokenizer_model_max_length is not None:
            new_input_embeds = [x[:tokenizer_model_max_length] for x in new_input_embeds]
            new_labels = [x[:tokenizer_model_max_length] for x in new_labels]

        # Pad sequences in the batch to same length; compute batch masks
        max_len = max(x.shape[0] for x in new_input_embeds)
        batch_size = len(new_input_embeds)

        new_input_embeds_padded = []
        new_labels_padded = torch.full((batch_size, max_len), IGNORE_INDEX, dtype=new_labels[0].dtype, device=new_labels[0].device)
        new_input_ids_padded = torch.full((batch_size, max_len), self.config.bos_token_id, dtype=new_input_ids[0].dtype, device=new_input_ids[0].device)
        attention_mask = torch.zeros((batch_size, max_len), dtype=attention_mask.dtype, device=attention_mask.device)
        position_ids = torch.zeros((batch_size, max_len), dtype=position_ids.dtype, device=position_ids.device)

        # Left or right padding as per config; fill padded tensors
        for i, (cur_new_embed, cur_new_labels, cur_new_input_ids) in enumerate(zip(new_input_embeds, new_labels, new_input_ids)):
            cur_len = cur_new_embed.shape[0]
            if getattr(self.config, 'tokenizer_padding_side', 'right') == "left":
                # Left pad: add zeros before text tokens/features
                new_input_embeds_padded.append(torch.cat((
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device),
                    cur_new_embed
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, -cur_len:] = cur_new_labels
                    attention_mask[i, -cur_len:] = True
                    position_ids[i, -cur_len:] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)
            else:
                # Right pad: add zeros after text tokens/features
                new_input_embeds_padded.append(torch.cat((
                    cur_new_embed,
                    torch.zeros((max_len - cur_len, cur_new_embed.shape[1]), dtype=cur_new_embed.dtype, device=cur_new_embed.device)
                ), dim=0))
                if cur_len > 0:
                    new_labels_padded[i, :cur_len] = cur_new_labels
                    new_input_ids_padded[i, :cur_len] = cur_new_input_ids
                    attention_mask[i, :cur_len] = True
                    position_ids[i, :cur_len] = torch.arange(0, cur_len, dtype=position_ids.dtype, device=position_ids.device)

        new_input_embeds = torch.stack(new_input_embeds_padded, dim=0)
        new_input_ids = new_input_ids_padded

        # Only set new_labels if original labels were not None
        if _labels is None:
            new_labels = None
        else:
            new_labels = new_labels_padded

        # Similarly handle provided attention_mask/position_ids overrides
        if _attention_mask is None:
            attention_mask = None
        else:
            attention_mask = attention_mask.to(dtype=_attention_mask.dtype)

        if _position_ids is None:
            position_ids = None

        # For Qwen2.5 vision towers, use and concatenate image_grid_thws for positional computations
        if isinstance(self.get_model().get_vision_tower(), Qwen2_5_VlVisionTower):
            image_grid_thws = []
            cur_image_idx = 0
            for num_images in image_nums_in_batch:
                if num_images == 0:
                    cur_image_idx += 1
                    continue
                image_grid_thws += image_grid_thws_minibatch[cur_image_idx:cur_image_idx+num_images]
                cur_image_idx += num_images
            
            if len(image_grid_thws) > 0:
                image_grid_thws = torch.cat(image_grid_thws, dim=0)
            else:
                image_grid_thws = None
            
            rope_index_kwargs = {
                "input_ids": new_input_ids,
                "image_grid_thw": image_grid_thws,
                "video_grid_thw": None,
                "attention_mask": attention_mask,
            }

            # Compute new position_ids and rope_deltas for transformer (for rotary embeddings)
            position_ids, rope_deltas = self.get_rope_index(**rope_index_kwargs)
            cache_position = torch.arange(new_input_embeds.shape[1], device=new_input_embeds.device)
        else:
            rope_deltas = None
            cache_position = None
        # Final output is a tuple mimicking HuggingFace prepare_inputs_for_generation return
        return None, position_ids, attention_mask, past_key_values, new_input_embeds, new_labels, rope_deltas, cache_position
    
    # Patch forward() of HF CausalLM to allow multimodal embedding with images/regions
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        images: Optional[torch.FloatTensor] = None,
        images_aux: Optional[torch.FloatTensor] = None,
        bbox_list: Optional[torch.FloatTensor] = None,
        image_grid_thws: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, Qwen2_5_VLCausalLMOutputWithPast]:
        
        if inputs_embeds is None:
            (
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                inputs_embeds,
                labels,
                rope_deltas,
                cache_position
            ) = self.prepare_inputs_labels_for_qwen2_5_vl_multimodal(
                input_ids,
                position_ids,
                attention_mask,
                past_key_values,
                labels,
                images,
                images_aux,
                bbox_list,
                image_grid_thws
            )
        
        if rope_deltas is not None:
            self.rope_deltas = rope_deltas

        # Call base CausalLM forward, with possibly replaced multimodal embeddings
        out = super().forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            rope_deltas=rope_deltas,
            cache_position=cache_position,
            second_per_grid_ts=second_per_grid_ts,
            return_dict=return_dict
        )
        return out

    # Prepare model input dict for autoregressive generation (for use with generation methods like generate())
    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        images: Optional[torch.FloatTensor] = None,
        images_aux: Optional[torch.FloatTensor] = None,
        bbox_list: Optional[torch.FloatTensor] = None,
        image_grid_thws: Optional[torch.FloatTensor] = None,
        **kwargs,
    ):
        # Wrap parent logic so extra multimodal kwargs are preserved
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            images=images,
            images_aux=images_aux,
            bbox_list=bbox_list,
            image_grid_thws=image_grid_thws,
        )   
        return model_inputs
    
# Register our config and model with HuggingFace transformers registry
AutoConfig.register("omchat_qwen2_5_vl", OmChatQwen25VLConfig)
AutoModelForCausalLM.register(OmChatQwen25VLConfig, OmChatQwen25VLForCausalLM)
