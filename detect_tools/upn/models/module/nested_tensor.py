from typing import List, Union

import torch
import torchvision


class NestedTensor(object):
    """Define a NestedTensor class

    Args:
        tensors (torch.Tensor): Tensor with shape [batch, C, H, W] or [C, H, W]
        mask (Union[torch.Tensor, str]): mask with shape [batch, H, W] or [H, W]. If mask
            is 'auto', it will be generated automatically by summing the tensor along
            the channel dimension. Mask is used to indicate the padding area.
    """

    def __init__(
        self, tensors: torch.Tensor, mask: Union[torch.Tensor, str] = "auto"
    ) -> None:
        self.tensors = tensors
        self.mask = mask
        if mask == "auto":
            self.mask = torch.zeros_like(tensors).to(tensors.device)
            if self.mask.dim() == 3:
                self.mask = self.mask.sum(0).to(bool)
            elif self.mask.dim() == 4:
                self.mask = self.mask.sum(1).to(bool)
            else:
                raise ValueError(
                    "tensors dim must be 3 or 4 but {}({})".format(
                        self.tensors.dim(), self.tensors.shape
                    )
                )

    def imgsize(self) -> List[torch.Tensor]:
        """get the img size of the tensor

        Returns:
            list[torch.Tensor]: list of tensor with shape [2] which is [H, W]
        """
        res = []
        for i in range(self.tensors.shape[0]):
            mask = self.mask[i]
            maxH = (~mask).sum(0).max()
            maxW = (~mask).sum(1).max()
            res.append(torch.Tensor([maxH, maxW]))
        return res

    def to(self, device: torch.device):
        """Move tensors and mask to the given device

        Args:
            device (torch.device): device to move

        Returns:
            NestedTensor: moved NestedTensor
        """
        cast_tensor = self.tensors.to(device)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def to_img_list_single(
        self, tensor: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        """remove the padding for one image

        Args:
            tensor (torch.Tensor): tensor with shape [C, H, W]
            mask (torch.Tensor): mask with shape [H, W]

        Returns:
            torch.Tensor: tensor with shape [C, maxH, maxW]
        """
        assert tensor.dim() == 3, "dim of tensor should be 3 but {}".format(
            tensor.dim()
        )
        maxH = (~mask).sum(0).max()
        maxW = (~mask).sum(1).max()
        img = tensor[:, :maxH, :maxW]
        return img

    def to_img_list(self) -> List[torch.Tensor]:
        """remove the padding and convert to img list

        Returns:
            list[torch.Tensor]: list of tensor with shape [C, maxH, maxW]
        """
        if self.tensors.dim() == 3:
            return self.to_img_list_single(self.tensors, self.mask)
        else:
            res = []
            for i in range(self.tensors.shape[0]):
                tensor_i = self.tensors[i]
                mask_i = self.mask[i]
                res.append(self.to_img_list_single(tensor_i, mask_i))
            return res

    @property
    def device(self):
        return self.tensors.device

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

    @property
    def shape(self):
        return {"tensors.shape": self.tensors.shape, "mask.shape": self.mask.shape}


def _max_by_axis(the_list):
    # type: (List[List[int]]) -> List[int]
    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes


def nested_tensor_from_tensor_list(
    tensor_list: List[torch.Tensor], fixed_img_size=None
):
    if fixed_img_size is not None:
        if isinstance(fixed_img_size, (list, tuple)):
            assert (
                len(fixed_img_size) == 2
            ), "image size should be a tuple or list with two elements"
        elif isinstance(fixed_img_size, int):
            fixed_img_size = [fixed_img_size, fixed_img_size]

    if tensor_list[0].ndim == 3:
        if torchvision._is_tracing():
            # nested_tensor_from_tensor_list() does not export well to ONNX
            # call _onnx_nested_tensor_from_tensor_list() instead
            return _onnx_nested_tensor_from_tensor_list(tensor_list)

        # TODO make it support different-sized images
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])

        if fixed_img_size is not None:
            c, orig_h, orig_w = max_size
            assert (
                orig_h <= fixed_img_size[0] and orig_w <= fixed_img_size[1]
            ), f"{orig_h} {orig_w} the fixed output image size should be larger than original image"
            max_size = [c, fixed_img_size[0], fixed_img_size[1]]

        # min_size = tuple(min(s) for s in zip(*[img.shape for img in tensor_list]))
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], : img.shape[2]] = False
    else:
        raise ValueError("not supported")
    return NestedTensor(tensor, mask)


@torch.jit.unused
def _onnx_nested_tensor_from_tensor_list(
    tensor_list: List[torch.Tensor],
) -> NestedTensor:
    max_size = []
    for i in range(tensor_list[0].dim()):
        max_size_i = torch.max(
            torch.stack([img.shape[i] for img in tensor_list]).to(torch.float32)
        ).to(torch.int64)
        max_size.append(max_size_i)
    max_size = tuple(max_size)

    padded_imgs = []
    padded_masks = []
    for img in tensor_list:
        padding = [(s1 - s2) for s1, s2 in zip(max_size, tuple(img.shape))]
        padded_img = torch.nn.functional.pad(
            img, (0, padding[2], 0, padding[1], 0, padding[0])
        )
        padded_imgs.append(padded_img)

        m = torch.zeros_like(img[0], dtype=torch.int, device=img.device)
        padded_mask = torch.nn.functional.pad(
            m, (0, padding[2], 0, padding[1]), "constant", 1
        )
        padded_masks.append(padded_mask.to(torch.bool))

    tensor = torch.stack(padded_imgs)
    mask = torch.stack(padded_masks)

    return NestedTensor(tensor, mask=mask)
