from vlm_fo1.model.multimodal_encoder.base_encoder import AbsVisionTower
from vlm_fo1.model.multimodal_encoder.davit.configuration_davit import DavitConfig
from vlm_fo1.model.multimodal_encoder.davit.configs import model_configs, img_cfg
from vlm_fo1.model.multimodal_encoder.davit.modeling_davit import DaViT
from vlm_fo1.model.multimodal_encoder.davit.image_processing_clip import CLIPImageProcessor

# Auxiliary DaViT-based vision tower for multi-modal encoder framework.
# This class manages configuration, processing, and dynamic instantiation of DaViT models.
class DavitVisionTower(AbsVisionTower):
    def __init__(self, vision_tower_name, args, delay_load=False, image_size=768, aspect_ratio='squash'):
        """
        Args:
            vision_tower_name: Identifier string for model variant (usually a file name or config section).
            args: Parent MM model/global config (currently ignored).
            delay_load: If True, only config is loaded, not the weights/model (for e.g., lazy instantiation).
            image_size: Target size to which images are resized (unless aspect_ratio=='dynamic').
            aspect_ratio: Controls how input aspect ratio is handled ('squash', 'dynamic', etc.).
        """
        super().__init__()
        self.is_loaded = False
        self.vision_tower_name = vision_tower_name
        self.aspect_ratio = aspect_ratio
        self.image_size = image_size

        # In this implementation, training flag is ignored (always uses pretrained weights).
        is_train = False
        if not delay_load:
            self.load_model(is_train, self.image_size, self.aspect_ratio)
        else:
            # Only load/prepare configuration (not model weights or modules)
            cfg_dict = model_configs[self.vision_tower_name.split('/')[-1].replace('.pth', '')]
            vision_cfg = DavitConfig.from_dict(cfg_dict)
            vision_cfg.image_size = image_size
            self.cfg_only = vision_cfg
        
    def load_model(self, is_train=False, image_size=768, aspect_ratio='squash'):
        """
        Actually loads the DaViT model (with weights) and its image processor.
        Sets up resizing/aspect handling as needed.
        """
        cfg_dict = model_configs[self.vision_tower_name.split('/')[-1].replace('.pth', '')]
        vision_cfg = DavitConfig.from_dict(cfg_dict)
        vision_cfg.image_size = image_size
        self.image_tower = DaViT.from_config(config=vision_cfg, enable_checkpoint=True)
        self.image_tower.config = vision_cfg
        img_cfg['resize_mode'] = aspect_ratio
        # If using 'dynamic' aspect ratio, disable resizing for the processor
        if aspect_ratio == 'dynamic':    # dynamic aspect ratio means no resizing, use the original image size, and the image_size parameter is not used
            img_cfg['do_resize'] = False
        self.image_processor = CLIPImageProcessor(**img_cfg)

        self.is_loaded = True

    def forward(self, images):
        """
        Runs the auxiliary DaViT encoder.
        Args:
            images: Torch tensor, or list of tensors, of images to encode.
        Returns:
            List of image feature outputs (typically 4-stage outputs per image).
        """
        # If input is a list of images, encode each separately.
        if type(images) is list:
            image_features = []
            for image in images:
                # Forward pass: returns 4-stage outputs; caller must handle downstream selection/merging.
                image_features.append(self.image_tower.forward(image.to(device=self.device, dtype=self.dtype))) # this returns 4 stage output
            return image_features
        else:
            # Single image: compute features, return as a length-1 list for consistency.
            # image_features = self.image_tower.forward(images.to(device=self.device, dtype=self.dtype)) # this returns 4 stage output
            # return [image_features]  # return the last layer for now
            raise NotImplementedError

    @property
    def dtype(self):
        # Expose main tensor dtype to external utilities (e.g., for caller to move data to right dtype).
        return self.image_tower.convs[0].proj.weight.dtype

    @property
    def device(self):
        # Expose main parameter device so inputs and other dependent modules use matching device.
        return self.image_tower.convs[0].proj.weight.device

    @property
    def config(self):
        # Get configuration in loaded or 'config only' state
        if self.is_loaded:
            return self.image_tower.config
        else:
            return self.cfg_only

    @property
    def hidden_size(self):
        # Hidden size: sum of embedding dims (all multi-stage outputs).
        return sum(self.image_tower.embed_dims)

