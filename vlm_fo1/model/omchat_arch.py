from abc import ABC, abstractmethod

from vlm_fo1.model.multimodal_encoder.builder import build_vision_tower, build_vision_tower_aux
from vlm_fo1.model.multimodal_projector.builder import build_vision_projector, build_vision_projector_aux
from vlm_fo1.model.multimodal_visual_prompt_encoder.hybrid_finegrained_region_encoder import HFREModule

class OmChatMetaModel:
    def __init__(self, config):
        super(OmChatMetaModel, self).__init__(config)
        # print('----------------------delay_load:', config.delay_load)
        if getattr(config, "mm_vision_tower", None) is not None:
            self.vision_tower = build_vision_tower(config, delay_load=getattr(config, 'delay_load', True))
        if getattr(config, "mm_vision_tower", None) is not None:
            self.mm_projector = build_vision_projector(config)
        if getattr(config, "mm_vision_tower_aux", None) is not None:
            self.vision_tower_aux = build_vision_tower_aux(config, delay_load=getattr(config, 'delay_load', True))
            self.object_vp_extractor = HFREModule(
                roi_output_size=getattr(config, "mm_roi_output_size", 7),
                region_feature_dim=config.mm_region_hidden_size,
                apply_position_embedding=getattr(config, "mm_apply_position_embedding", True),
                pos_embedding_strategy=getattr(config, "mm_pos_embedding_strategy", "bbox_based"),
                use_vt_region_feature_only=getattr(config, "mm_use_vt_region_feature_only", False),
                use_vision_tower_region_feature=getattr(config, "mm_use_vision_tower_region_feature", False),
                region_feature_combination=getattr(config, "mm_region_feature_combination", "concat"),
                apply_region_layer_norm=getattr(config, "mm_apply_region_layer_norm", False),                
                vision_tower_region_feature_dim=self.get_vision_tower().config.hidden_size * 4 if not getattr(config, "mm_use_simpleFPN_for_vt", False) else 2048,
                vision_tower_spatial_scale=1/self.get_vision_tower().config.patch_size,
                use_simpleFPN_for_vt=getattr(config, "mm_use_simpleFPN_for_vt", False),
                aux_vision_tower_spatial_scale=0.25,
                aux_vision_tower_region_feature_dims=[256, 512, 1024, 2048],
            )
        if getattr(config, "mm_vision_tower_aux", None) is not None:
            self.mm_projector_aux = build_vision_projector_aux(config)

    def get_vision_tower(self):
        vision_tower = getattr(self, 'vision_tower', None)
        if type(vision_tower) is list:
            vision_tower = vision_tower[0]
        return vision_tower

    def get_vision_tower_aux(self):
        vision_tower_aux = getattr(self, 'vision_tower_aux', None)
        if type(vision_tower_aux) is list:
            vision_tower_aux = vision_tower_aux[0]
        return vision_tower_aux

    def get_video_tower(self):
        video_tower = getattr(self, 'video_tower', None)
        if type(video_tower) is list:
            video_tower = video_tower[0]
        return video_tower


class OmChatMetaForCausalLM(ABC):

    @abstractmethod
    def get_model(self):
        pass

    def get_vision_tower(self):
        return self.get_model().get_vision_tower()
    
    def get_vision_tower_aux(self):
        return self.get_model().get_vision_tower_aux()

    def get_video_tower(self):
        return self.get_model().get_vision_tower()

    def encode_videos(self, videos):  # [mini_b, c, t, h, w]
        video_features = self.get_model().get_video_tower()(videos)  # [mini_b, t, n, c]
        video_features = self.get_model().mm_projector.forward_video(video_features)
        return video_features