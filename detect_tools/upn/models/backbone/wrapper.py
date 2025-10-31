from typing import Dict, List, Tuple, Union

import torch
import torch.nn as nn

from detect_tools.upn import BACKBONES, build_backbone, build_position_embedding
from detect_tools.upn.models.module import NestedTensor
from detect_tools.upn.models.utils import clean_state_dict


class FrozenBatchNorm2d(torch.nn.Module):
    """
    BatchNorm2d where the batch statistics and the affine parameters are fixed.

    Copy-paste from torchvision.misc.ops with added eps before rqsrt,
    without which any other models than torchvision.models.resnet[18,34,50,101]
    produce nans.
    """

    def __init__(self, n):
        super(FrozenBatchNorm2d, self).__init__()
        self.register_buffer("weight", torch.ones(n))
        self.register_buffer("bias", torch.zeros(n))
        self.register_buffer("running_mean", torch.zeros(n))
        self.register_buffer("running_var", torch.ones(n))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        num_batches_tracked_key = prefix + "num_batches_tracked"
        if num_batches_tracked_key in state_dict:
            del state_dict[num_batches_tracked_key]

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, x):
        # move reshapes to the beginning
        # to make it fuser-friendly
        w = self.weight.reshape(1, -1, 1, 1)
        b = self.bias.reshape(1, -1, 1, 1)
        rv = self.running_var.reshape(1, -1, 1, 1)
        rm = self.running_mean.reshape(1, -1, 1, 1)
        eps = 1e-5
        scale = w * (rv + eps).rsqrt()
        bias = b - rm * scale
        return x * scale + bias


class Joiner(nn.Module):
    """A wrapper for the backbone and the position embedding.

    Args:
        backbone_cfg (Dict): Config dict to build backbone.
        position_embedding_cfg (Dict): Config dict to build position embedding.
    """

    def __init__(self, backbone: nn.Module, position_embedding: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone
        self.pos_embed = position_embedding

    def forward(
        self, tensor_list: NestedTensor
    ) -> Union[List[NestedTensor], List[torch.Tensor]]:
        """Forward function.

        Args:
            tensor_list (NestedTensor): NestedTensor wrapping the input tensor.

        Returns:
            [List[NestedTensor]: A list of feature map in NestedTensor format.
            List[torch.Tensor]: A list of position encoding.
        """

        xs = self.backbone(tensor_list)
        out: List[NestedTensor] = []
        pos = []
        for layer_idx, x in xs.items():
            out.append(x)
            # position encoding
            pos.append(self.pos_embed(x).to(x.tensors.dtype))

        return out, pos

    def forward_pos_embed_only(self, x: NestedTensor) -> torch.Tensor:
        """Forward function for position embedding only. This is used to generate additional layer

        Args:
            x (NestedTensor): NestedTensor wrapping the input tensor.

        Returns:
            [List[torch.Tensor]: A list of position encoding.
        """
        return self.pos_embed(x)


@BACKBONES.register_module()
class SwinWrapper(nn.Module):
    """A wrapper for swin transformer.

    Args:
        backbone_cfg Union[Dict, str]: Config dict to build backbone. If given a str name, we
            will call `get_swin_config` to get the config dict.
        dilation (bool): Whether to use dilation in stage 4.
        position_embedding_cfg (Dict): Config dict to build position embedding.
        lr_backbone (float): Learning rate of the backbone.
        return_interm_layers (List[int]): Which layers to return.
        backbone_freeze_keywords (List[str]): List of keywords to freeze the backbone.
        use_checkpoint (bool): Whether to use checkpoint. Default: False.
        ckpt_path (str): Checkpoint path. Default: None.
        use_pretrained_ckpt (bool): Whether to use pretrained checkpoint. Default: True.
    """

    def __init__(
        self,
        backbone_cfg: Union[Dict, str],
        dilation: bool,
        position_embedding_cfg: Dict,
        lr_backbone: float,
        return_interm_indices: List[int],
        backbone_freeze_keywords: List[str],
        use_checkpoint: bool = False,
        backbone_ckpt_path: str = None,
    ) -> None:
        super(SwinWrapper, self).__init__()
        pos_embedding = build_position_embedding(position_embedding_cfg)
        train_backbone = lr_backbone > 0
        if not train_backbone:
            raise ValueError("Please set lr_backbone > 0")
        assert return_interm_indices in [[0, 1, 2, 3], [1, 2, 3], [3]]

        # build backbone
        if isinstance(backbone_cfg, str):
            assert (
                backbone_cfg
                in backbone_cfg
                in [
                    "swin_T_224_1k",
                    "swin_B_224_22k",
                    "swin_B_384_22k",
                    "swin_L_224_22k",
                    "swin_L_384_22k",
                ]
            )
            pretrain_img_size = int(backbone_cfg.split("_")[-2])
            backbone_cfg = get_swin_config(
                backbone_cfg,
                pretrain_img_size,
                out_indices=tuple(return_interm_indices),
                dilation=dilation,
                use_checkpoint=use_checkpoint,
            )
        backbone = build_backbone(backbone_cfg)

        # freeze some layers
        if backbone_freeze_keywords is not None:
            for name, parameter in backbone.named_parameters():
                for keyword in backbone_freeze_keywords:
                    if keyword in name:
                        parameter.requires_grad_(False)
                        break

        # load checkpoint
        if backbone_ckpt_path is not None:
            print("Loading backbone checkpoint from {}".format(backbone_ckpt_path))
            checkpoint = torch.load(backbone_ckpt_path, map_location="cpu")["model"]
            from collections import OrderedDict

            def key_select_function(keyname):
                if "head" in keyname:
                    return False
                if dilation and "layers.3" in keyname:
                    return False
                return True

            _tmp_st = OrderedDict(
                {
                    k: v
                    for k, v in clean_state_dict(checkpoint).items()
                    if key_select_function(k)
                }
            )
            _tmp_st_output = backbone.load_state_dict(_tmp_st, strict=False)
            print(str(_tmp_st_output))

        bb_num_channels = backbone.num_features[4 - len(return_interm_indices) :]
        assert len(bb_num_channels) == len(
            return_interm_indices
        ), f"len(bb_num_channels) {len(bb_num_channels)} != len(return_interm_indices) {len(return_interm_indices)}"

        model = Joiner(backbone, pos_embedding)
        model.num_channels = bb_num_channels
        self.num_channels = bb_num_channels
        self.model = model

    def forward(
        self, tensor_list: NestedTensor
    ) -> Union[List[NestedTensor], List[torch.Tensor]]:
        """Forward function.

        Args:
            tensor_list (NestedTensor): NestedTensor wrapping the input tensor.

        Returns:
            [List[NestedTensor]: A list of feature map in NestedTensor format.
            List[torch.Tensor]: A list of position encoding.
        """

        return self.model(tensor_list)

    def forward_pos_embed_only(self, tensor_list: NestedTensor) -> torch.Tensor:
        """Forward function to get position embedding only.

        Args:
            tensor_list (NestedTensor): NestedTensor wrapping the input tensor.

        Returns:
            torch.Tensor: Position embedding.
        """
        return self.model.forward_pos_embed_only(tensor_list)


def get_swin_config(modelname: str, pretrain_img_size: Tuple[int, int], **kw):
    """Get swin config dict.

    Args:
        modelname (str): Name of the model.
        pretrain_img_size (Tuple[int, int]): Image size of the pretrain model.
        kw (Dict): Other key word arguments.

    Returns:
        Dict: Config dict.
        str: Path to the pretrained checkpoint.
    """
    assert modelname in [
        "swin_T_224_1k",
        "swin_B_224_22k",
        "swin_B_384_22k",
        "swin_L_224_22k",
        "swin_L_384_22k",
    ]
    model_para_dict = {
        "swin_T_224_1k": dict(
            type="SwinTransformer",
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=7,
        ),
        "swin_B_224_22k": dict(
            type="SwinTransformer",
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=7,
        ),
        "swin_B_384_22k": dict(
            type="SwinTransformer",
            embed_dim=128,
            depths=[2, 2, 18, 2],
            num_heads=[4, 8, 16, 32],
            window_size=12,
        ),
        "swin_L_224_22k": dict(
            type="SwinTransformer",
            embed_dim=192,
            depths=[2, 2, 18, 2],
            num_heads=[6, 12, 24, 48],
            window_size=7,
        ),
        "swin_L_384_22k": dict(
            type="SwinTransformer",
            embed_dim=192,
            depths=[2, 2, 18, 2],
            num_heads=[6, 12, 24, 48],
            window_size=12,
        ),
    }
    kw_cgf = model_para_dict[modelname]
    kw_cgf.update(kw)
    kw_cgf.update(dict(pretrain_img_size=pretrain_img_size))
    return kw_cgf
