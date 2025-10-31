# Builders for different vision tower backbones (MM encoder visual modules)
from .qwen2_5_vl_encoder import Qwen2_5_VlVisionTower    # Main Qwen2.5 vision tower
from .davit_aux_encoder import DavitVisionTower as DavitVisionTowerAux  # Auxiliary DaViT vision tower

def build_vision_tower(vision_tower_cfg, **kwargs):
    """
    Use model config to construct the main vision tower.

    vision_tower_cfg: should have attribute mm_vision_tower
    Returns: instance of configured vision backbone
    """
    vision_tower_name = getattr(vision_tower_cfg, 'mm_vision_tower', None)
    print(vision_tower_cfg)  # Debug print of the config being used
    
    # Check for the Qwen2.5-VL vision model in tower name
    if "qwen2.5-vl" in vision_tower_name.lower():
        return Qwen2_5_VlVisionTower(vision_tower_name, args=vision_tower_cfg, **kwargs) 

    # Raise a clear error for unknown towers
    raise ValueError(f'Unknown vision tower: {vision_tower_name}')

def build_vision_tower_aux(vision_tower_cfg, **kwargs):
    """
    Use model config to construct the auxiliary (helper) vision tower.

    vision_tower_cfg: should have attribute mm_vision_tower_aux
    Returns: instance of configured auxiliary vision backbone
    """
    vision_tower_aux = getattr(vision_tower_cfg, 'mm_vision_tower_aux', None)
    # Optionally print config for debugging
    # print(vision_tower_cfg)

    # Check for the DaViT auxiliary vision model in tower name
    if 'davit' in vision_tower_aux.lower():
        return DavitVisionTowerAux(vision_tower_aux, args=vision_tower_cfg, **kwargs)

    # Raise a clear error if tower type is unknown
    raise ValueError(f'Unknown aux vision tower: {vision_tower_aux}')
