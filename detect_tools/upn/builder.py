from mmengine import Registry, build_from_cfg

BACKBONES = Registry("backbone")
POS_EMBEDDINGS = Registry("position_embedding")
FUSERS = Registry("fuser")
ENCODERS = Registry("encoder")
DECODERS = Registry("decoder")
ARCHITECTURES = Registry("architecture")


def build_backbone(cfg):
    """Build encoder."""
    return build_from_cfg(cfg, BACKBONES)


def build_position_embedding(cfg):
    """Build position embedding."""
    return build_from_cfg(cfg, POS_EMBEDDINGS)


def build_fuser(cfg):
    """Build fuser."""
    return build_from_cfg(cfg, FUSERS)


def build_encoder(cfg):
    """Build encoder."""
    return build_from_cfg(cfg, ENCODERS)


def build_decoder(cfg):
    """Build decoder."""
    return build_from_cfg(cfg, DECODERS)


def build_architecture(cfg):
    """Build architecture."""

    return build_from_cfg(cfg, ARCHITECTURES)
