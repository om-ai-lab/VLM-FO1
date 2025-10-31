from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveAssign(nn.Module):

    def __init__(
        self,
        cal_bias: nn.Module = None,
    ) -> None:
        """Lanuage-Image Contrastive Assignment used to calculate the similarity between
        the text and the image.

        Args:
            cal_bias (nn.Module, optional): The bias used to calculate the similarity.
                Defaults to None.
            max_text_len (int, optional): The max length of the text. Defaults to 256.
        """
        super().__init__()
        self.cal_bias = cal_bias

    def forward(self, x: torch.Tensor, ref_dict: Dict):

        y = ref_dict["encoded_ref_feature"]
        res = x @ y.transpose(-1, -2)
        return res
