# coding=utf-8
# Copyright 2024 Microsoft and the HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional

from transformers import AutoConfig
from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)

class DavitConfig(PretrainedConfig):
    r"""
    This is the configuration class to store the configuration of a [`Florence2VisionModel`]. It is used to instantiate a Florence2VisionModel
    according to the specified arguments, defining the model architecture. Instantiating a configuration with the 
    defaults will yield a similar configuration to that of the Florence2VisionModel architecture.

    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        drop_path_rate (`float`, *optional*, defaults to 0.1):
            The dropout rate of the drop path layer.
        patch_size (`List[int]`, *optional*, defaults to [7, 3, 3, 3]):
            The patch size of the image.
        patch_stride (`List[int]`, *optional*, defaults to [4, 2, 2, 2]):
            The patch stride of the image.
        patch_padding (`List[int]`, *optional*, defaults to [3, 1, 1, 1]):
            The patch padding of the image.
        patch_prenorm (`List[bool]`, *optional*, defaults to [false, true, true, true]):
            Whether to apply layer normalization before the patch embedding layer.
        enable_checkpoint (`bool`, *optional*, defaults to False):
            Whether to enable checkpointing.
        dim_embed (`List[int]`, *optional*, defaults to [256, 512, 1024, 2048]):
            The dimension of the embedding layer.
        num_heads (`List[int]`, *optional*, defaults to [8, 16, 32, 64]):
            The number of attention heads.
        num_groups (`List[int]`, *optional*, defaults to [8, 16, 32, 64]):
            The number of groups.
        depths (`List[int]`, *optional*, defaults to [1, 1, 9, 1]):
            The depth of the model.
        window_size (`int`, *optional*, defaults to 12):
            The window size of the model.
        projection_dim (`int`, *optional*, defaults to 1024):
            The dimension of the projection layer.
        visual_temporal_embedding (`dict`, *optional*):
            The configuration of the visual temporal embedding.
        image_pos_embed (`dict`, *optional*):
            The configuration of the image position embedding.
        image_feature_source (`List[str]`, *optional*, defaults to ["spatial_avg_pool", "temporal_avg_pool"]):
            The source of the image feature.
    Example:

    ```python
    >>> from transformers import Florence2VisionConfig, Florence2VisionModel

    >>> # Initializing a Florence2 Vision style configuration
    >>> configuration = Florence2VisionConfig()

    >>> # Initializing a model (with random weights)
    >>> model = Florence2VisionModel(configuration)

    >>> # Accessing the model configuration
    >>> configuration = model.config
    ```"""

    model_type = "florence2_vision"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        drop_path_rate=0.1,
        patch_size=[7, 3, 3, 3],
        patch_stride=[4, 2, 2, 2],
        patch_padding=[3, 1, 1, 1],
        patch_prenorm=[False, True, True, True],
        enable_checkpoint=False,
        dim_embed=[256, 512, 1024, 2048],
        num_heads=[8, 16, 32, 64],
        num_groups=[8, 16, 32, 64],
        depths=[1, 1, 9, 1],
        window_size=12,
        projection_dim=1024,
        visual_temporal_embedding=None,
        image_pos_embed=None,
        image_feature_source=["spatial_avg_pool", "temporal_avg_pool"],
        **kwargs,
    ):
        self.drop_path_rate = drop_path_rate
        self.patch_size = patch_size
        self.patch_stride = patch_stride
        self.patch_padding = patch_padding
        self.patch_prenorm = patch_prenorm
        self.enable_checkpoint = enable_checkpoint
        self.dim_embed = dim_embed
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.depths = depths
        self.window_size = window_size
        self.projection_dim = projection_dim
        self.visual_temporal_embedding = visual_temporal_embedding
        self.image_pos_embed = image_pos_embed
        self.image_feature_source = image_feature_source

        super().__init__(**kwargs)



