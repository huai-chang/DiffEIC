import time
import torch
import torch as th
import torch.nn as nn

from typing import Dict, Mapping, Any

import os
import math
import pyiqa
import einops
import numpy as np
from pytorch_lightning.utilities.types import EPOCH_OUTPUT
from utils.utils import *

from ldm.modules.diffusionmodules.util import (
    conv_nd,
    linear,
    zero_module,
    timestep_embedding,
    checkpoint
)

from .spaced_sampler import SpacedSampler
from einops import rearrange
from ldm.modules.attention import BasicTransformerBlock, SpatialTransformer
from ldm.models.diffusion.ddpm import LatentDiffusion
from ldm.modules.diffusionmodules.openaimodel import (
    UNetModel,
    TimestepEmbedSequential,
    ResBlock as ResBlock_orig,
    Downsample,
    Upsample,
    AttentionBlock,
    TimestepBlock
)
from ldm.util import log_txt_as_img, exists, instantiate_from_config, default

class CDDM(nn.Module):
    def __init__(
            self,
            image_size,
            in_channels,
            model_channels,
            out_channels,
            hint_channels,
            num_res_blocks,
            attention_resolutions,
            dropout=0,
            channel_mult=(1, 2, 4, 8),
            conv_resample=True,
            dims=2,
            use_checkpoint=False,
            use_fp16=False,
            num_heads=-1,
            num_head_channels=-1,
            num_heads_upsample=-1,
            use_scale_shift_norm=False,
            resblock_updown=False,
            use_new_attention_order=False,
            use_spatial_transformer=False,  # custom transformer support
            transformer_depth=1,  # custom transformer support
            context_dim=None,  # custom transformer support
            n_embed=None,  # custom support for prediction of discrete ids into codebook of first stage vq model
            legacy=False,
            use_linear_in_transformer=False,
            control_model_ratio=1.0,        # ratio of the control model size compared to the base model. [0, 1]
            learn_embedding=True,
            control_scale=1.0
    ):
        super().__init__()

        self.learn_embedding = learn_embedding
        self.control_model_ratio = control_model_ratio
        self.out_channels = out_channels
        self.dims = 2
        self.model_channels = model_channels
        self.control_scale = control_scale

        ################# start control model variations #################
        base_model = UNetModel(
            image_size=image_size, in_channels=in_channels, model_channels=model_channels,
            out_channels=out_channels, num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions, dropout=dropout, channel_mult=channel_mult,
            conv_resample=conv_resample, dims=dims, use_checkpoint=use_checkpoint,
            use_fp16=use_fp16, num_heads=num_heads, num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample, use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown, use_new_attention_order=use_new_attention_order,
            use_spatial_transformer=use_spatial_transformer, transformer_depth=transformer_depth,
            context_dim=context_dim, n_embed=n_embed, legacy=legacy,
            use_linear_in_transformer=use_linear_in_transformer,
        )  # initialise control model from base model
        self.control_model = ControlModule(
            image_size=image_size, in_channels=in_channels, model_channels=model_channels, hint_channels=hint_channels,
            out_channels=out_channels, num_res_blocks=num_res_blocks,
            attention_resolutions=attention_resolutions, dropout=dropout, channel_mult=channel_mult,
            conv_resample=conv_resample, dims=dims, use_checkpoint=use_checkpoint,
            use_fp16=use_fp16, num_heads=num_heads, num_head_channels=num_head_channels,
            num_heads_upsample=num_heads_upsample, use_scale_shift_norm=use_scale_shift_norm,
            resblock_updown=resblock_updown, use_new_attention_order=use_new_attention_order,
            use_spatial_transformer=use_spatial_transformer, transformer_depth=transformer_depth,
            context_dim=context_dim, n_embed=n_embed, legacy=legacy,
            use_linear_in_transformer=use_linear_in_transformer,
            control_model_ratio=control_model_ratio,
        )  # initialise pretrained model

        ################# end control model variations #################

        self.enc_zero_convs_out = nn.ModuleList([])

        self.middle_block_out = None
        self.middle_block_in = None

        self.dec_zero_convs_out = nn.ModuleList([])

        ch_inout_ctr = {'enc': [], 'mid': [], 'dec': []}
        ch_inout_base = {'enc': [], 'mid': [], 'dec': []}

        ################# Gather Channel Sizes #################
        for module in self.control_model.input_blocks:
            if isinstance(module[0], nn.Conv2d):
                ch_inout_ctr['enc'].append((module[0].in_channels, module[0].out_channels))
            elif isinstance(module[0], (ResBlock, ResBlock_orig)):
                ch_inout_ctr['enc'].append((module[0].channels, module[0].out_channels))
            elif isinstance(module[0], Downsample):
                ch_inout_ctr['enc'].append((module[0].channels, module[-1].out_channels))

        for module in base_model.input_blocks:
            if isinstance(module[0], nn.Conv2d):
                ch_inout_base['enc'].append((module[0].in_channels, module[0].out_channels))
            elif isinstance(module[0], (ResBlock, ResBlock_orig)):
                ch_inout_base['enc'].append((module[0].channels, module[0].out_channels))
            elif isinstance(module[0], Downsample):
                ch_inout_base['enc'].append((module[0].channels, module[-1].out_channels))

        ch_inout_ctr['mid'].append((self.control_model.middle_block[0].channels, self.control_model.middle_block[-1].out_channels))
        ch_inout_base['mid'].append((base_model.middle_block[0].channels, base_model.middle_block[-1].out_channels))

        for module in self.control_model.output_blocks:
            if isinstance(module[0], nn.Conv2d):
                ch_inout_ctr['dec'].append((module[0].in_channels, module[0].out_channels))
            elif isinstance(module[0], (ResBlock, ResBlock_orig)):
                ch_inout_ctr['dec'].append((module[0].channels, module[0].out_channels))
            elif isinstance(module[-1], Upsample):
                ch_inout_ctr['dec'].append((module[0].channels, module[-1].out_channels))

        for module in base_model.output_blocks:
            if isinstance(module[0], nn.Conv2d):
                ch_inout_base['dec'].append((module[0].in_channels, module[0].out_channels))
            elif isinstance(module[0], (ResBlock, ResBlock_orig)):
                ch_inout_base['dec'].append((module[0].channels, module[0].out_channels))
            elif isinstance(module[-1], Upsample):
                ch_inout_base['dec'].append((module[0].channels, module[-1].out_channels))

        self.ch_inout_ctr = ch_inout_ctr
        self.ch_inout_base = ch_inout_base

        ################# Build zero convolutions #################
        self.middle_block_out = self.make_zero_conv(ch_inout_ctr['mid'][-1][1], ch_inout_base['mid'][-1][1])

        self.dec_zero_convs_out.append(
            self.make_zero_conv(ch_inout_ctr['enc'][-1][1], ch_inout_base['mid'][-1][1])
        )
        for i in range(1, len(ch_inout_ctr['enc'])):
            self.dec_zero_convs_out.append(
                self.make_zero_conv(ch_inout_ctr['enc'][-(i + 1)][1], ch_inout_base['dec'][i - 1][1])
            )
        for i in range(len(ch_inout_ctr['enc'])):
            self.enc_zero_convs_out.append(self.make_zero_conv(
                in_channels=ch_inout_ctr['enc'][i][1], out_channels=ch_inout_base['enc'][i][1])
                    )

        scale_list = [1.] * len(self.enc_zero_convs_out) + [1.] + [1.] * len(self.dec_zero_convs_out)
        self.register_buffer('scale_list', torch.tensor(scale_list) * self.control_scale)

    def make_zero_conv(self, in_channels, out_channels=None):
        self.in_channels = in_channels
        self.out_channels = out_channels or in_channels
        return TimestepEmbedSequential(
            zero_module(conv_nd(self.dims, in_channels, out_channels, 1, padding=0))
        )

    def forward(self, x, hint, timesteps, context, base_model, **kwargs):
        t_emb = timestep_embedding(timesteps, self.model_channels, repeat_only=False)
        emb = self.control_model.time_embed(t_emb)
        emb_base = base_model.time_embed(t_emb)

        h_base = x.type(base_model.dtype)
        h_ctr = torch.cat((h_base, hint), dim=1)
        hs_base = []
        hs_ctr = []
        it_enc_convs_out = iter(self.enc_zero_convs_out)
        it_dec_convs_out = iter(self.dec_zero_convs_out)
        scales = iter(self.scale_list)

        ###################### Cross Control  ######################

        # input blocks (encoder)
        for module_base, module_ctr in zip(base_model.input_blocks, self.control_model.input_blocks):
            h_base = module_base(h_base, emb_base, context)
            h_ctr = module_ctr(h_ctr, emb, context)
            
            h_base = h_base + next(it_enc_convs_out)(h_ctr, emb) * next(scales)

            hs_base.append(h_base)
            hs_ctr.append(h_ctr)

        # mid blocks (bottleneck)
        h_base = base_model.middle_block(h_base, emb_base, context)
        h_ctr = self.control_model.middle_block(h_ctr, emb, context)

        h_base = h_base + self.middle_block_out(h_ctr, emb) * next(scales)

        # output blocks (decoder)
        for module_base in base_model.output_blocks:
            h_base = h_base + next(it_dec_convs_out)(hs_ctr.pop(), emb) * next(scales)

            h_base = th.cat([h_base, hs_base.pop()], dim=1)
            h_base = module_base(h_base, emb_base, context)

        return base_model.out(h_base)
    
class ControlModule(nn.Module):
    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        hint_channels,
        out_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=None,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=-1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        use_spatial_transformer=False,    # custom transformer support
        transformer_depth=1,              # custom transformer support
        context_dim=None,                 # custom transformer support
        n_embed=None,                     # custom support for prediction of discrete ids into codebook of first stage vq model
        legacy=True,
        disable_self_attentions=None,
        num_attention_blocks=None,
        disable_middle_self_attn=False,
        use_linear_in_transformer=False,
        control_model_ratio=1.0,
    ):
        super().__init__()
        if use_spatial_transformer:
            assert context_dim is not None, 'Fool!! You forgot to include the dimension of your cross-attention conditioning...'

        if context_dim is not None:
            assert use_spatial_transformer, 'Fool!! You forgot to use the spatial transformer for your cross-attention conditioning...'
            from omegaconf.listconfig import ListConfig
            if type(context_dim) == ListConfig:
                context_dim = list(context_dim)


        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        if num_heads == -1:
            assert num_head_channels != -1, 'Either num_heads or num_head_channels has to be set'

        if num_head_channels == -1:
            assert num_heads != -1, 'Either num_heads or num_head_channels has to be set'

        self.image_size = image_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        if isinstance(num_res_blocks, int):
            self.num_res_blocks = len(channel_mult) * [num_res_blocks]
        else:
            if len(num_res_blocks) != len(channel_mult):
                raise ValueError("provide num_res_blocks either as an int (globally constant) or "
                                 "as a list/tuple (per-level) with the same length as channel_mult")
            self.num_res_blocks = num_res_blocks
        if disable_self_attentions is not None:
            # should be a list of booleans, indicating whether to disable self-attention in TransformerBlocks or not
            assert len(disable_self_attentions) == len(channel_mult)
        if num_attention_blocks is not None:
            assert len(num_attention_blocks) == len(self.num_res_blocks)
            assert all(map(lambda i: self.num_res_blocks[i] >= num_attention_blocks[i], range(len(num_attention_blocks))))
            print(f"Constructor of UNetModel received num_attention_blocks={num_attention_blocks}. "
                  f"This option has LESS priority than attention_resolutions {attention_resolutions}, "
                  f"i.e., in cases where num_attention_blocks[i] > 0 but 2**i not in attention_resolutions, "
                  f"attention will still not be set.")

        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = th.float16 if use_fp16 else th.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample
        self.predict_codebook_ids = n_embed is not None

        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),
            linear(time_embed_dim, time_embed_dim),
        )

        model_channels = int(model_channels * control_model_ratio)
        self.model_channels = model_channels
        self.control_model_ratio = control_model_ratio

        if self.num_classes is not None:
            if isinstance(self.num_classes, int):
                self.label_emb = nn.Embedding(num_classes, time_embed_dim)
            elif self.num_classes == "continuous":
                print("setting up linear c_adm embedding layer")
                self.label_emb = nn.Linear(1, time_embed_dim)
            else:
                raise ValueError()

        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels+hint_channels, model_channels, 3, padding=1)
                )
            ]
        )
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                if ds in attention_resolutions:
                    if num_head_channels == -1:
                        dim_head = ch // num_heads
                    else:
                        num_head_channels = find_denominator(ch, self.num_head_channels)
                        num_heads = ch // num_head_channels
                        dim_head = num_head_channels
                    if legacy:
                        dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
                    if exists(disable_self_attentions):
                        disabled_sa = disable_self_attentions[level]
                    else:
                        disabled_sa = False

                    if not exists(num_attention_blocks) or nr < num_attention_blocks[level]:
                        layers.append(
                            AttentionBlock(
                                ch,
                                use_checkpoint=use_checkpoint,
                                num_heads=num_heads,
                                num_head_channels=dim_head,
                                use_new_attention_order=use_new_attention_order,
                            ) if not use_spatial_transformer else SpatialTransformer(
                                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                                disable_self_attn=disabled_sa, use_linear=use_linear_in_transformer,
                                use_checkpoint=use_checkpoint
                            )
                        )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch,
                            conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        if num_head_channels == -1:
            dim_head = ch // num_heads
        else:
            num_heads = ch // num_head_channels
            dim_head = num_head_channels
        if legacy:
            dim_head = ch // num_heads if use_spatial_transformer else num_head_channels
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                out_channels=ch,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=dim_head,
                use_new_attention_order=use_new_attention_order,
            ) if not use_spatial_transformer else SpatialTransformer(  # always uses a self-attn
                ch, num_heads, dim_head, depth=transformer_depth, context_dim=context_dim,
                disable_self_attn=disable_middle_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

def find_denominator(number, start):
    if start >= number:
        return number
    while (start != 0):
        residual = number % start
        if residual == 0:
            return start
        start -= 1

def normalization(channels):
    """
    Make a standard normalization layer.
    :param channels: number of input channels.
    :return: an nn.Module for normalization.
    """
    # if find_denominator(channels, 32) < 32:
    #     print(f'[USING GROUPNORM OVER LESS CHANNELS ({find_denominator(channels, 32)}) FOR {channels} CHANNELS]')
    return GroupNorm_leq32(find_denominator(channels, 32), channels)

class GroupNorm_leq32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)
    
class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.
    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm

        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        self.updown = up or down

        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.
        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = th.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h
    
class DiffEIC(LatentDiffusion):

    def __init__(
        self, 
        control_stage_config: Mapping[str, Any], 
        control_key: str,
        sd_locked: bool,
        # synch_control: bool,
        learning_rate: float,
        aux_learning_rate: float,
        l_bpp_weight: float,
        l_guide_weight: float,
        sync_path: str, 
        synch_control: bool,
        ckpt_path_pre: str,
        preprocess_config: Mapping[str, Any],
        calculate_metrics: Mapping[str, Any],
        *args, 
        **kwargs
    ) -> "DiffEIC":
        super().__init__(*args, **kwargs)
        self.control_model = instantiate_from_config(control_stage_config)
        self.preprocess_model = instantiate_from_config(preprocess_config)
        if sync_path is not None:
            self.sync_control_weights_from_base_checkpoint(sync_path, synch_control=synch_control)
        if ckpt_path_pre is not None:
            self.load_preprocess_ckpt(ckpt_path_pre=ckpt_path_pre)

        self.control_key = control_key
        self.sd_locked = sd_locked

        self.learning_rate = learning_rate
        self.aux_learning_rate = aux_learning_rate
        self.l_bpp_weight = l_bpp_weight
        self.l_guide_weight = l_guide_weight

        self.calculate_metrics = calculate_metrics
        self.metric_funcs = {}
        for _, opt in calculate_metrics.items(): 
            mopt = opt.copy()
            name = mopt.pop('type', None)
            mopt.pop('better', None)
            self.metric_funcs[name] = pyiqa.create_metric(name, device=self.device, **mopt)

    def apply_condition_encoder(self, control, x):
        c_latent, likelihoods, q_likelihoods = self.preprocess_model(control, x)
        return c_latent, likelihoods, q_likelihoods
    
    @torch.no_grad()
    def apply_condition_compress(self, control, stream_path, H, W):
        ref = self.encode_first_stage(control * 2 - 1).mode() * self.scale_factor
        out = self.preprocess_model.compress(control, ref)
        shape = out["shape"]
        with Path(stream_path).open("wb") as f:
            write_body(f, shape, out["strings"])
        size = filesize(stream_path)
        bpp = float(size) * 8 / (H * W)
        return bpp

    @torch.no_grad()
    def apply_condition_decompress(self, stream_path):
        with Path(stream_path).open("rb") as f:
            strings, shape = read_body(f)
        c_latent = self.preprocess_model.decompress(strings, shape)
        return c_latent
    
    def get_input(self, batch, k, bs=None, *args, **kwargs):
        x, c = super().get_input(batch, self.first_stage_key, bs=bs, *args, **kwargs) 
        control = batch[self.control_key]
        if bs is not None:
            control = control[:bs]
        control = control.to(self.device)
        control = einops.rearrange(control, 'b h w c -> b c h w')
        control = control.to(memory_format=torch.contiguous_format).float()

        c_latent, likelihoods, q_likelihoods = self.apply_condition_encoder(control, x)
        N , _, H, W = control.shape
        num_pixels = N * H * W
        bpp = sum((torch.log(likelihood).sum() / (-math.log(2) * num_pixels)) for likelihood in likelihoods)
        q_bpp = sum((torch.log(likelihood).sum() / (-math.log(2) * num_pixels)) for likelihood in q_likelihoods)
        return x, dict(c_crossattn=[c], c_latent=[c_latent], bpp=bpp, q_bpp=q_bpp, control=[control])
    
    def apply_model(self, x_noisy, t, cond, *args, **kwargs):
        assert isinstance(cond, dict)
        diffusion_model = self.model.diffusion_model

        cond_txt = torch.cat(cond['c_crossattn'], 1)
        cond_hint = torch.cat(cond['c_latent'], 1)

        eps = self.control_model(
            x=x_noisy, timesteps=t, context=cond_txt, hint=cond_hint, base_model=diffusion_model)
        
        return eps
    
    @torch.no_grad()
    def get_unconditional_conditioning(self, N):
        return self.get_learned_conditioning([""] * N)
    
    @torch.no_grad()
    def log_images(self, batch, sample_steps=50, bs=2):
        log = dict()
        z, c = self.get_input(batch, self.first_stage_key, bs=bs)
        bpp = c["q_bpp"]
        bpp_img = [f'{bpp:2f}']*4
        c_latent = c["c_latent"][0]
        control = c["control"][0]
        c = c["c_crossattn"][0]

        log["hq"] = (self.decode_first_stage(z) + 1) / 2
        log["control"] = control
        log["text"] = (log_txt_as_img((512, 512), bpp_img, size=16) + 1) / 2
        
        samples = self.sample_log(
            cond={"c_crossattn": [c], "c_latent": [c_latent]},
            steps=sample_steps
        )
        x_samples = self.decode_first_stage(samples)
        log["samples"] = (x_samples + 1) / 2

        return log, bpp
    
    @torch.no_grad()
    def sample_log(self, cond, steps):
        sampler = SpacedSampler(self)
        b, c, h, w = cond["c_latent"][0].shape
        shape = (b, self.channels, h, w)

        samples = sampler.sample(
            steps, shape, cond, unconditional_guidance_scale=1.0,
            unconditional_conditioning=None
        )
        return samples
    
    def configure_optimizers(self):
        lr = self.learning_rate
        params = list(self.control_model.parameters())
        params += list(param for name, param in self.preprocess_model.named_parameters() 
                       if not name.endswith('.quantiles'))
        if not self.sd_locked:
            params += list(self.model.diffusion_model.output_blocks.parameters())
            params += list(self.model.diffusion_model.out.parameters())
        opt = torch.optim.AdamW(params, lr=lr)

        aux_lr = self.aux_learning_rate
        aux_params = list(param for name, param in self.preprocess_model.named_parameters() 
                       if name.endswith('.quantiles'))
        aux_opt =  torch.optim.AdamW(aux_params, lr=aux_lr)

        return opt, aux_opt
    
    def p_losses(self, x_start, cond, t, noise=None):
        loss_dict = {}
        prefix = 'T' if self.training else 'V'

        noise = default(noise, lambda: torch.randn_like(x_start))
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        model_output = self.apply_model(x_noisy, t, cond)

        if self.parameterization == "x0":
            target = x_start
        elif self.parameterization == "eps":
            target = noise
        elif self.parameterization == "v":
            target = self.get_v(x_start, noise, t)
        else:
            raise NotImplementedError()

        loss_simple = self.get_loss(model_output, target, mean=False).mean([1, 2, 3])
        loss_dict.update({f'{prefix}/l_simple': loss_simple.mean()})

        logvar_t = self.logvar[t].to(self.device)
        loss = loss_simple / torch.exp(logvar_t) + logvar_t
        if self.learn_logvar:
            loss_dict.update({f'{prefix}/l_gamma': loss.mean()})
            loss_dict.update({'logvar': self.logvar.data.mean()})

        loss = self.l_simple_weight * loss.mean()

        loss_bpp = cond['bpp']
        guide_bpp = cond['q_bpp']
        loss_dict.update({f'{prefix}/l_bpp': loss_bpp.mean()})
        loss_dict.update({f'{prefix}/q_bpp': guide_bpp.mean()})
        loss += self.l_bpp_weight * loss_bpp

        c_latent = cond['c_latent'][0][:,:4,:,:]
        loss_guide = self.get_loss(c_latent, x_start)
        loss_dict.update({f'{prefix}/l_guide': loss_guide.mean()})
        loss += self.l_guide_weight * loss_guide
        loss_dict.update({f'{prefix}/loss': loss})

        return loss, loss_dict
    
    def training_step(self, batch, batch_idx, optimizer_idx):
        if optimizer_idx == 0:
            for k in self.ucg_training:
                p = self.ucg_training[k]["p"]
                val = self.ucg_training[k]["val"]
                if val is None:
                    val = ""
                for i in range(len(batch[k])):
                    if self.ucg_prng.choice(2, p=[1 - p, p]):
                        batch[k][i] = val

            loss, loss_dict = self.shared_step(batch)

            self.log_dict(loss_dict, prog_bar=True,
                        logger=True, on_step=True, on_epoch=True)

            self.log("global_step", self.global_step,
                    prog_bar=True, logger=True, on_step=True, on_epoch=False)

            if self.use_scheduler:
                lr = self.optimizers().param_groups[0]['lr']
                self.log('lr_abs', lr, prog_bar=True, logger=True, on_step=True, on_epoch=False)

            return loss
        
        if optimizer_idx == 1:
            aux_loss = self.preprocess_model.aux_loss()
            self.log("aux_loss", aux_loss,
                    prog_bar=True, logger=True, on_step=True, on_epoch=False)
            return aux_loss
        
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        out = []
        log, bpp = self.log_images(batch, bs=None)
        out.append(bpp.cpu())
        # save images
        save_dir = os.path.join(self.logger.save_dir, "validation", f'{self.global_step}')
        os.makedirs(save_dir, exist_ok=True)
        image = log["samples"].detach().cpu()
        image = image.numpy().squeeze().transpose(1,2,0)
        image = (image * 255).clip(0, 255).astype(np.uint8)
        path = os.path.join(save_dir, f'{batch_idx}.png')
        Image.fromarray(image).save(path)

        control = log["control"].detach().cpu()
        control = control.numpy().squeeze().transpose(1,2,0)
        control = (control * 255).clip(0, 255).astype(np.uint8)

        metric_data = [img2tensor(image).unsqueeze(0) / 255.0, img2tensor(control).unsqueeze(0) / 255.0]

        for name, _ in self.calculate_metrics.items():
            out.append(self.metric_funcs[name](*metric_data))

        
        return out
    
    def validation_epoch_end(self, outputs: EPOCH_OUTPUT):
        outputs = np.array(outputs)
        avg_out = sum(outputs)/len(outputs)
        self.log("avg_bpp", avg_out[0],
                    prog_bar=True, logger=True, on_step=False, on_epoch=True)
        
        for i, (name, _) in enumerate(self.calculate_metrics.items()):
            self.log(f"avg_{name}", avg_out[i+1],
                    prog_bar=True, logger=True, on_step=False, on_epoch=True)
        
    def load_preprocess_ckpt(self, ckpt_path_pre):
        ckpt = torch.load(ckpt_path_pre)
        self.preprocess_model.load_state_dict(ckpt)
        print(['CONTROL WEIGHTS LOADED'])
        
    def sync_control_weights_from_base_checkpoint(self, path, synch_control=True):
        ckpt_base = torch.load(path)  # load the base model checkpoints

        if synch_control:
            # add copy for control_module weights from the base model
            for key in list(ckpt_base['state_dict'].keys()):
                if "diffusion_model." in key:
                    if 'control_model.control' + key[15:] in self.state_dict().keys():
                        if ckpt_base['state_dict'][key].shape != self.state_dict()['control_model.control' + key[15:]].shape:
                            if len(ckpt_base['state_dict'][key].shape) == 1:
                                dim = 0
                                control_dim = self.state_dict()['control_model.control' + key[15:]].size(dim)
                                ckpt_base['state_dict']['control_model.control' + key[15:]] = torch.cat([
                                    ckpt_base['state_dict'][key],
                                    ckpt_base['state_dict'][key]
                                ], dim=dim)[:control_dim]
                            else:
                                dim = 0
                                control_dim_0 = self.state_dict()['control_model.control' + key[15:]].size(dim)
                                dim = 1
                                control_dim_1 = self.state_dict()['control_model.control' + key[15:]].size(dim)
                                ckpt_base['state_dict']['control_model.control' + key[15:]] = torch.cat([
                                    ckpt_base['state_dict'][key],
                                    ckpt_base['state_dict'][key]
                                ], dim=dim)[:control_dim_0, :control_dim_1, ...]
                        else:
                            ckpt_base['state_dict']['control_model.control' + key[15:]] = ckpt_base['state_dict'][key]
            
        res_sync = self.load_state_dict(ckpt_base['state_dict'], strict=False)
        print(f'[{len(res_sync.missing_keys)} keys are missing from the model (hint processing and cross connections included)]')