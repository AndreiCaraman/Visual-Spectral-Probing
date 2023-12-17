from typing import Optional, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from transformers import ViTPreTrainedModel, ViTModel, ViTMAEConfig
from transformers.modeling_outputs import ImageClassifierOutput


class PrismViTConfig(ViTMAEConfig):
    model_type = "PrismViT"

    def __init__(self, filter_type: dict = {}, **kwargs):
        r"""
        Arg:
            filter_mode (`Dict[str, List[int]]`, *optional*, defaults to `{}`):
                Filter mode of the model. The key is the name of the mode, the args are the associated values.

                Availible modes are:

                bypass: filter bypassed when no filter_mode argument is provided

                band: select the frequency band indicated in the args; no filter weights training is performed
                band mode requires three arguments: 'max_freqs', 'start_idx', 'end_idx'

                auto: filter learning; only this mode enables filter weight training
                auto mode requires the argument 'max_freqs'
        """

        super().__init__(**kwargs)

        self.filter_type = filter_type


class PrismViT(ViTPreTrainedModel):
    config_class = PrismViTConfig

    def __init__(self, config: PrismViTConfig):
        super().__init__(config)

        # load transformer
        self.vit = ViTModel(config, add_pooling_layer=False).from_pretrained(
            "facebook/vit-mae-base"
        )

        # Filter variables
        self.frq_filter = self.parse_filter_type(config.filter_type)

        # Classifier variables
        self.num_labels = config.num_labels

        # Classifier head
        self.classifier = (
            nn.Linear(self.vit.config.hidden_size, config.num_labels)
            if config.num_labels > 0
            else nn.Identity()
        )

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        pixel_values: Optional[
            torch.Tensor
        ] = None,  # torch.FloatTensor of shape (batch_size, num_channels, height, width)
        labels: Optional[torch.Tensor] = None,
    ) -> Union[tuple, ImageClassifierOutput]:
        outputs = self.vit(pixel_values)
        sequence_output = outputs[0]

        # apply filter if set
        if self.frq_filter is not None:
            sequence_output = self.filter(sequence_output)

        # classification
        logits = self.classifier(sequence_output[:, 0, :])

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        return ImageClassifierOutput(loss=loss, logits=logits)

    def parse_filter_type(self, filter_type):
        if not filter_type:
            frq_filter = None
            return frq_filter

        elif filter_type["filter_name"] == "band":
            if len(filter_type["filter_args"]) != 3:
                raise ValueError(
                    "[Error] band filter requires three arguments 'max_freqs', 'start_idx', 'end_idx'."
                )
            frq_filter = PrismViT.gen_band_filter(
                filter_type["filter_args"][0],
                filter_type["filter_args"][1],
                filter_type["filter_args"][2],
            )
            return nn.Parameter(frq_filter, requires_grad=False)

        elif filter_type["filter_name"] == "eqalloc":
            if len(filter_type["filter_args"]) != 3:
                raise ValueError(
                    "[Error] eqalloc filter requires three arguments 'max_freqs', 'num_bands', 'band_idx'. Exiting."
                )
            frq_filter = PrismViT.gen_equal_allocation_filters(
                filter_type["filter_args"][0], filter_type["filter_args"][1]
            )[filter_type["filter_args"][2]]

            return nn.Parameter(frq_filter, requires_grad=False)

        elif filter_type["filter_name"] == "auto":
            if len(filter_type["filter_args"]) != 1:
                raise ValueError("[Error] auto requires the argument 'max_freqs'.")
            frq_filter = PrismViT.gen_auto_filter_initialization(
                filter_type["filter_args"][0]
            )
            return nn.Parameter(frq_filter, requires_grad=True)

        else:
            raise ValueError(
                f"[Error] Unknown filter type '{filter_type['filter_name']}'."
            )

    def filter(self, embeddings):
        """
        Decomposes embeddings into the specified frequency bands.

        :param embeddings: tensor of embeddings with shape (batch_size, sequence_length, hidden_dim)
        :return: filtered embeddings with the same shape as the input
        """

        emb_filtered = torch.zeros_like(embeddings)

        sequence_length = embeddings.shape[1]

        # apply discrete cosine transform
        for img_idx in range(embeddings.shape[0]):
            # run DCT over length of sequence
            seq_frequencies = self.dct(
                embeddings[img_idx].T
            )  # (hidden_dim, sequence_length)

            # pool filter to sequence length (filter_size, ) -> (1, sequence_length)
            frq_filter = F.adaptive_avg_pool1d(
                self.frq_filter[None, :], sequence_length
            )

            # normalize filter values to [0, 1] for learnable filter
            frq_filter = torch.sigmoid(frq_filter)

            # element-wise multiplication of filter with each row (no expansion needed due to prior pooling)
            seq_filtered = seq_frequencies * frq_filter
            # perform inverse DCT
            emb_recomposed = self.idct(seq_filtered).T  # (sequence_length, hidden_dim)
            emb_filtered[img_idx, :, :] = emb_recomposed

        return emb_filtered

    @staticmethod
    def gen_band_filter(filter_size, start_idx, end_idx):
        assert (
            start_idx < end_idx < filter_size
        ), f"[Error] Range {start_idx}-{end_idx} out of range for filter with {filter_size} frequencies."

        filter = torch.zeros(filter_size)
        filter[start_idx : end_idx + 1] = 1

        return filter

    @staticmethod
    def gen_equal_allocation_filters(filter_size, num_bands):
        filters = []
        assert (
            num_bands <= filter_size
        ), f"[Error] Cannot equally allocate more bands than there are frequencies ({num_bands} < {filter_size})."

        # get number of frequencies to equally allocate
        num_equal = filter_size // num_bands
        # get remainder of bands to unequally distribute (max. num_bands - 1)
        num_unequal = filter_size % num_bands
        # allocate remainder bands from the outside in (starting at band 0)
        extra_bands = set()
        cursor = 0
        for eidx in range(num_unequal):
            # if even iteration, add to left
            if eidx % 2 == 0:
                extra_bands.add(eidx + cursor)
            # if uneven, add to right
            else:
                extra_bands.add(num_bands - cursor - 1)
                cursor += 1

        # equally allocate cleanly divisible bands
        for bidx in range(num_bands):
            cur_filter = torch.zeros(filter_size)
            start_idx = bidx * num_equal
            end_idx = start_idx + num_equal + int(bidx in extra_bands)
            cur_filter[start_idx:end_idx] = 1
            filters.append(cur_filter)

        return filters

    @staticmethod
    def gen_auto_filter_initialization(filter_size):
        return torch.ones(filter_size)

    @staticmethod
    def dct(embeddings, norm=None):
        """
        (Implemented as in https://github.com/zh217/torch-dct)
        Discrete Cosine Transform, Type II (a.k.a. the DCT)

        For the meaning of the parameter `norm`, see:
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html

        :param x: the input signal
        :param norm: the normalization, None or 'ortho'
        :return: the DCT-II of the signal over the last dimension
        """
        orig_shape = embeddings.shape
        N = embeddings.shape[-1]
        embeddings = embeddings.contiguous().view(-1, N)

        v = torch.cat([embeddings[:, ::2], embeddings[:, 1::2].flip([1])], dim=1)

        Vc = torch.view_as_real(torch.fft.fft(v, dim=1))

        k = (
            -torch.arange(N, dtype=embeddings.dtype, device=embeddings.device)[None, :]
            * np.pi
            / (2 * N)
        )
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V = Vc[:, :, 0] * W_r - Vc[:, :, 1] * W_i

        if norm == "ortho":
            V[:, 0] /= np.sqrt(N) * 2
            V[:, 1:] /= np.sqrt(N / 2) * 2

        V = 2 * V.view(*orig_shape)

        return V

    @staticmethod
    def idct(frequencies, norm=None):
        """
        (Implemented as in https://github.com/zh217/torch-dct)
        The inverse to DCT-II, which is a scaled Discrete Cosine Transform, Type III
        Our definition of idct is that idct(dct(x)) == x
        For the meaning of the parameter `norm`, see:
        https://docs.scipy.org/doc/scipy-0.14.0/reference/generated/scipy.fftpack.dct.html
        :param X: the input signal
        :param norm: the normalization, None or 'ortho'
        :return: the inverse DCT-II of the signal over the last dimension
        """

        orig_shape = frequencies.shape
        N = orig_shape[-1]

        X_v = frequencies.contiguous().view(-1, orig_shape[-1]) / 2

        if norm == "ortho":
            X_v[:, 0] *= np.sqrt(N) * 2
            X_v[:, 1:] *= np.sqrt(N / 2) * 2

        k = (
            torch.arange(
                orig_shape[-1], dtype=frequencies.dtype, device=frequencies.device
            )[None, :]
            * np.pi
            / (2 * N)
        )
        W_r = torch.cos(k)
        W_i = torch.sin(k)

        V_t_r = X_v
        V_t_i = torch.cat([X_v[:, :1] * 0, -X_v.flip([1])[:, :-1]], dim=1)

        V_r = V_t_r * W_r - V_t_i * W_i
        V_i = V_t_r * W_i + V_t_i * W_r

        V = torch.cat([V_r.unsqueeze(2), V_i.unsqueeze(2)], dim=2)

        v = torch.fft.irfft(torch.view_as_complex(V), n=V.shape[1], dim=1)
        frequencies = v.new_zeros(v.shape)
        frequencies[:, ::2] += v[:, : N - (N // 2)]
        frequencies[:, 1::2] += v.flip([1])[:, : N // 2]

        return frequencies.view(*orig_shape)
