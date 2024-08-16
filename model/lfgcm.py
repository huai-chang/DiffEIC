import torch
import torch.nn as nn
from compressai.models import CompressionModel
from compressai.entropy_models import GaussianConditional, EntropyBottleneck
from compressai.ops import quantize_ste
from compressai.ans import BufferedRansEncoder, RansDecoder
from utils.func import update_registered_buffers, get_scale_table
from utils.ckbd import *
from model.layers import *
from thop import profile
    
class Encoder(nn.Module):
    def __init__(self, in_nc, mid_nc, out_nc, prior_nc, sft_ks):
        super().__init__()

        self.sft_1_8 = nn.Sequential(
            conv3x3(4, prior_nc),
            nn.GELU(),
            conv1x1(prior_nc, prior_nc)
        )

        self.sft_1_16 = nn.Sequential(
            conv3x3(prior_nc, prior_nc, stride=2),
            nn.GELU(),
            conv1x1(prior_nc, prior_nc)
        )

        self.sft_1_16_2 = nn.Sequential(
            conv3x3(prior_nc, prior_nc),
            nn.GELU(),
            conv1x1(prior_nc, prior_nc)
        )

        self.g_a1 = nn.Sequential(
            ResidualBlockWithStride(in_nc, mid_nc[0]),
            ResidualBottleneck(mid_nc[0]),
            ResidualBottleneck(mid_nc[0]),
            ResidualBottleneck(mid_nc[0]),
            ResidualBlockWithStride(mid_nc[0], mid_nc[1]),
            ResidualBottleneck(mid_nc[1]),
            ResidualBottleneck(mid_nc[1]),
            ResidualBottleneck(mid_nc[1]),
            ResidualBlockWithStride(mid_nc[1], mid_nc[2]),
            ResidualBottleneck(mid_nc[2]),
            ResidualBottleneck(mid_nc[2]),
            ResidualBottleneck(mid_nc[2]),
        )

        self.g_a1_ref = SFT(mid_nc[2], prior_nc, ks=sft_ks)

        self.g_a2  = nn.Sequential(
            ResidualBlockWithStride(mid_nc[2], mid_nc[3]),
            ResidualBottleneck(mid_nc[3]),
            ResidualBottleneck(mid_nc[3]),
            ResidualBottleneck(mid_nc[3]),
        )
        self.g_a2_ref = SFT(mid_nc[3], prior_nc, ks=sft_ks)


        self.g_a3 = conv3x3(mid_nc[3],mid_nc[3])
        self.g_a4 = SFTResblk(mid_nc[3], prior_nc, ks=sft_ks)
        self.g_a5 = SFTResblk(mid_nc[3], prior_nc, ks=sft_ks)

        self.g_a6 = conv3x3(mid_nc[3], out_nc)

    def forward(self, x, feature):
        sft_feature = self.sft_1_8(feature)
        x = self.g_a1(x)
        x = self.g_a1_ref(x, sft_feature)

        sft_feature = self.sft_1_16(sft_feature)
        x = self.g_a2(x)
        x = self.g_a2_ref(x, sft_feature)

        sft_feature = self.sft_1_16_2(sft_feature)
        x = self.g_a3(x)
        x = self.g_a4(x, sft_feature)
        x = self.g_a5(x, sft_feature)
        x = self.g_a6(x)

        return x
    
class Decoder(nn.Module):
    def __init__(self, N, M, out_nc, prior_nc, sft_ks):
        super().__init__()

        self.sft_feature_gs1 = nn.Sequential(
            conv3x3(M + N//4, prior_nc * 4),
            nn.GELU(),
            conv3x3(prior_nc * 4, prior_nc * 2),
            nn.GELU(),
            conv3x3(prior_nc * 2, prior_nc)
        )
        self.sft_feature_gs2 = nn.Sequential(
            conv3x3(prior_nc, prior_nc),
            nn.GELU(),
            conv1x1(prior_nc, prior_nc)
        )
        self.sft_feature_gs3 = nn.Sequential(
            deconv(prior_nc, prior_nc, 3),
            nn.GELU(),
            conv1x1(prior_nc, prior_nc)
        )

        self.g_s0 = SFTResblk(M, prior_nc, ks=sft_ks)
        self.g_s1 = SFTResblk(M, prior_nc, ks=sft_ks)

        self.g_s2 = nn.Sequential(
            conv3x3(M,N),
            ResidualBottleneck(N),
            ResidualBottleneck(N),
            ResidualBottleneck(N))
        self.g_s2_ref = SFT(N, prior_nc, ks=sft_ks)

        self.g_s3 = nn.Sequential(
            ResidualBlockUpsample(N, N),
            ResidualBottleneck(N),
            ResidualBottleneck(N),
            ResidualBottleneck(N))
        self.g_s3_ref = SFT(N, prior_nc, ks=sft_ks)

        self.g_s4 = conv3x3(N, out_nc)

    def forward(self, x, ref):
        sft_feature = self.sft_feature_gs1(torch.cat([ref,x], dim=1))
        x = self.g_s0(x, sft_feature)
        x = self.g_s1(x, sft_feature)

        sft_feature = self.sft_feature_gs2(sft_feature)
        x = self.g_s2(x)
        x = self.g_s2_ref(x, sft_feature)

        sft_feature = self.sft_feature_gs3(sft_feature)
        x = self.g_s3(x)
        x = self.g_s3_ref(x, sft_feature)

        x = self.g_s4(x)

        return x
    
class HyperEncoder(nn.Module):
    def __init__(self, N, M, prior_nc, sft_ks):
        super().__init__()
        self.sft_feature = nn.Sequential(
            conv3x3(4, prior_nc, stride=2),
            nn.GELU(),
            conv1x1(prior_nc, prior_nc)
        )

        self.sft_feature_h1 = nn.Sequential(
            conv3x3(M+prior_nc, prior_nc*4),
            nn.GELU(),
            conv3x3(prior_nc*4, prior_nc*2),
            nn.GELU(),
            conv3x3(prior_nc*2, prior_nc)
        )

        self.sft_feature_h2 = nn.Sequential(
            conv(prior_nc, prior_nc, 3),
            nn.GELU(),
            conv(prior_nc, prior_nc, 1, 1)
        )

        self.sft_feature_h3 = nn.Sequential(
            conv(prior_nc, prior_nc, 3),
            nn.GELU(),
            conv(prior_nc, prior_nc, 1, 1)
        )

        self.h_a0 = conv3x3(M, N)
        self.h_a1 = SFT(N, prior_nc, ks=sft_ks)
        self.h_a2 = nn.GELU()

        self.h_a3 = conv(N, N)
        self.h_a4 = SFT(N, prior_nc, ks=sft_ks)
        self.h_a5 = nn.GELU()

        self.h_a6 = conv(N, N)
        self.h_a7 = SFTResblk(N, prior_nc, ks=sft_ks)
        self.h_a8 = SFTResblk(N, prior_nc, ks=sft_ks)
        self.h_a9 = conv1x1(N,N)

    def forward(self, x, feature):
        sft_feature = self.sft_feature(feature)
        sft_feature = self.sft_feature_h1(torch.cat([sft_feature, x], dim=1))
        x = self.h_a0(x)
        x = self.h_a1(x, sft_feature)
        x = self.h_a2(x)

        sft_feature = self.sft_feature_h2(sft_feature)
        x = self.h_a3(x)
        x = self.h_a4(x, sft_feature)
        x = self.h_a5(x)

        sft_feature = self.sft_feature_h3(sft_feature)
        x = self.h_a6(x)
        x = self.h_a7(x, sft_feature)
        x = self.h_a8(x, sft_feature)
        x = self.h_a9(x)

        return x
    
class HyperDecoder(nn.Module):
    def __init__(self, N, M):
        super().__init__()
        self.hyper_dec = nn.Sequential(
            deconv(N, M),
            nn.GELU(),
            deconv(M, M * 3 // 2),
            nn.GELU(),
            deconv(M * 3 // 2, M * 2, kernel_size=3, stride=1),
        )

    def forward(self, x):
        return self.hyper_dec(x)
    
class ChannelContextEX(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fushion = nn.Sequential(
            nn.Conv2d(in_dim, 224, kernel_size=5, stride=1, padding=2),
            nn.GELU(),
            nn.Conv2d(224, 128, kernel_size=5, stride=1, padding=2),
            nn.GELU(),
            nn.Conv2d(128, out_dim, kernel_size=5, stride=1, padding=2)
        )

    def forward(self, channel_params):
        channel_params = self.fushion(channel_params)
        return channel_params

class EntropyParametersEX(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.fusion = nn.Sequential(
            nn.Conv2d(in_dim, out_dim * 5 // 3, 1),
            nn.GELU(),
            nn.Conv2d(out_dim * 5 // 3, out_dim * 4 // 3, 1),
            nn.GELU(),
            nn.Conv2d(out_dim * 4 // 3, out_dim, 1),
        )

    def forward(self, params):
        gaussian_params = self.fusion(params)
        return gaussian_params

class LFGCM(CompressionModel):
    def __init__(self, in_nc, out_nc, enc_mid, N, M, prior_nc, sft_ks, slice_num, slice_ch):
        super().__init__()

        self.slice_num = slice_num
        self.slice_ch = slice_ch

        self.encoder = Encoder(in_nc, enc_mid, M, prior_nc, sft_ks)
        self.hyper_enc = HyperEncoder(N, M, prior_nc, sft_ks)
        self.hyper_dec = HyperDecoder(N, M)
        self.decoder = Decoder(N, M, out_nc, prior_nc, sft_ks)

        self.f_c = nn.Sequential(
            deconv(N, N//2),
            nn.GELU(),
            deconv(N//2, N//4),
            nn.GELU(),
            deconv(N//4, N//4, kernel_size=3, stride=1),
        )

        self.local_context = nn.ModuleList(
            nn.Conv2d(in_channels=slice_ch[i], out_channels=slice_ch[i] * 2, kernel_size=5, stride=1, padding=2)
            for i in range(len(slice_ch))
        )

        self.channel_context = nn.ModuleList(
            ChannelContextEX(in_dim=sum(slice_ch[:i]), out_dim=slice_ch[i] * 2) if i else None
            for i in range(slice_num)
        )

        # Use channel_ctx and hyper_params
        self.entropy_parameters_anchor = nn.ModuleList(
            EntropyParametersEX(in_dim=M * 2 + slice_ch[i] * 2, out_dim=slice_ch[i] * 2)
            if i else EntropyParametersEX(in_dim=M * 2, out_dim=slice_ch[i] * 2)
            for i in range(slice_num)
        )

        # Entropy parameters for non-anchors
        # Use spatial_params, channel_ctx and hyper_params
        self.entropy_parameters_nonanchor = nn.ModuleList(
            EntropyParametersEX(in_dim=M * 2 + slice_ch[i] * 4, out_dim=slice_ch[i] * 2)
            if i else  EntropyParametersEX(in_dim=M * 2 + slice_ch[i] * 2, out_dim=slice_ch[i] * 2)
            for i in range(slice_num)
        )

        self.entropy_bottleneck = EntropyBottleneck(N)
        self.gaussian_conditional = GaussianConditional(None)

    def forward(self, x, ref):
        y = self.encoder(x, ref)
        z = self.hyper_enc(y, ref)
        _, z_likelihoods = self.entropy_bottleneck(z)
        _, q_z_likelihoods = self.entropy_bottleneck(z, False)
        z_offset = self.entropy_bottleneck._get_medians()
        z_hat = quantize_ste(z - z_offset) + z_offset

        # Hyper-parameters
        hyper_params = self.hyper_dec(z_hat)

        y_slices = [y[:, sum(self.slice_ch[:i]):sum(self.slice_ch[:(i + 1)]), ...] for i in range(len(self.slice_ch))]
        y_hat_slices = []
        y_likelihoods = []
        q_likelihoods = []
        for idx, y_slice in enumerate(y_slices):
            """
            Split y to anchor and non-anchor
            anchor :
                0 1 0 1 0
                1 0 1 0 1
                0 1 0 1 0
                1 0 1 0 1
                0 1 0 1 0
            non-anchor:
                1 0 1 0 1
                0 1 0 1 0
                1 0 1 0 1
                0 1 0 1 0
                1 0 1 0 1
            """
            slice_anchor, slice_nonanchor = ckbd_split(y_slice)
            if idx == 0:
                # Anchor
                params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # round anchor
                slice_anchor = quantize_ste(slice_anchor - means_anchor) + means_anchor
                
                # Non-anchor
                # local_ctx: [B, H, W, 2 * C]
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # merge means and scales of anchor and nonanchor
                scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                means_slice = ckbd_merge(means_anchor, means_nonanchor)
                _, y_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice)
                _, q_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice, False)
                # round slice_nonanchor
                slice_nonanchor = quantize_ste(slice_nonanchor - means_nonanchor) + means_nonanchor
                y_hat_slice = slice_anchor + slice_nonanchor
                y_hat_slices.append(y_hat_slice)
                y_likelihoods.append(y_slice_likelihoods)
                q_likelihoods.append(q_slice_likelihoods)
            else:
                channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
                # Anchor(Use channel context and hyper params)
                params_anchor = self.entropy_parameters_anchor[idx](torch.cat([channel_ctx, hyper_params], dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # split means and scales of anchor
                scales_anchor = ckbd_anchor(scales_anchor)
                means_anchor = ckbd_anchor(means_anchor)
                # round anchor
                slice_anchor = quantize_ste(slice_anchor - means_anchor) + means_anchor
                
                # Non-anchor
                # ctx_params: [B, H, W, 2 * C]
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, channel_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # split means and scales of nonanchor
                scales_nonanchor = ckbd_nonanchor(scales_nonanchor)
                means_nonanchor = ckbd_nonanchor(means_nonanchor)
                # merge means and scales of anchor and nonanchor
                scales_slice = ckbd_merge(scales_anchor, scales_nonanchor)
                means_slice = ckbd_merge(means_anchor, means_nonanchor)
                _, y_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice)
                _, q_slice_likelihoods = self.gaussian_conditional(y_slice, scales_slice, means_slice, False)
                # round slice_nonanchor
                slice_nonanchor = quantize_ste(slice_nonanchor - means_nonanchor) + means_nonanchor
                y_hat_slice = slice_anchor + slice_nonanchor
                y_hat_slices.append(y_hat_slice)
                y_likelihoods.append(y_slice_likelihoods)
                q_likelihoods.append(q_slice_likelihoods)

        y_hat = torch.cat(y_hat_slices, dim=1)
        y_likelihoods = torch.cat(y_likelihoods, dim=1)
        q_likelihoods = torch.cat(q_likelihoods, dim=1)

        ref = self.f_c(z_hat)
        output = self.decoder(y_hat, ref)

        return output, [y_likelihoods, z_likelihoods], [q_likelihoods, q_z_likelihoods]
    
    def compress(self, x, ref):
        y = self.encoder(x, ref)
        z = self.hyper_enc(y, ref)

        torch.backends.cudnn.deterministic = True
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])
        hyper_params = self.hyper_dec(z_hat)

        y_slices = [y[:, sum(self.slice_ch[:i]):sum(self.slice_ch[:(i + 1)]), ...] for i in range(len(self.slice_ch))]
        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        encoder = BufferedRansEncoder()
        symbols_list = []
        indexes_list = []
        y_strings = []

        for idx, y_slice in enumerate(y_slices):
            slice_anchor, slice_nonanchor = ckbd_split(y_slice)
            if idx == 0:
                # Anchor
                params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # round and compress anchor
                slice_anchor = compress_anchor(self.gaussian_conditional, slice_anchor, scales_anchor, means_anchor, symbols_list, indexes_list)
                # Non-anchor
                # local_ctx: [B,2 * C, H, W]
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # round and compress nonanchor
                slice_nonanchor = compress_nonanchor(self.gaussian_conditional, slice_nonanchor, scales_nonanchor, means_nonanchor, symbols_list, indexes_list)
                y_slice_hat = slice_anchor + slice_nonanchor
                y_hat_slices.append(y_slice_hat)

            else:
                # Anchor
                channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
                params_anchor = self.entropy_parameters_anchor[idx](torch.cat([channel_ctx, hyper_params], dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # round and compress anchor
                slice_anchor = compress_anchor(self.gaussian_conditional, slice_anchor, scales_anchor, means_anchor, symbols_list, indexes_list)
                # Non-anchor
                # local_ctx: [B,2 * C, H, W]
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, channel_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # round and compress nonanchor
                slice_nonanchor = compress_nonanchor(self.gaussian_conditional, slice_nonanchor, scales_nonanchor, means_nonanchor, symbols_list, indexes_list)
                y_hat_slices.append(slice_nonanchor + slice_anchor)

        encoder.encode_with_indexes(symbols_list, indexes_list, cdf, cdf_lengths, offsets)
        y_string = encoder.flush()
        y_strings.append(y_string)

        torch.backends.cudnn.deterministic = False
        return {
            "strings": [y_strings, z_strings],
            "shape": z.size()[-2:]
        }
    
    def decompress(self, strings, shape):
        torch.backends.cudnn.deterministic = True

        y_strings = strings[0][0]
        z_strings = strings[1]
        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        hyper_params = self.hyper_dec(z_hat)

        y_hat_slices = []

        cdf = self.gaussian_conditional.quantized_cdf.tolist()
        cdf_lengths = self.gaussian_conditional.cdf_length.reshape(-1).int().tolist()
        offsets = self.gaussian_conditional.offset.reshape(-1).int().tolist()
        decoder = RansDecoder()
        decoder.set_stream(y_strings)

        for idx in range(self.slice_num):
            if idx == 0:
                # Anchor
                params_anchor = self.entropy_parameters_anchor[idx](hyper_params)
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # decompress anchor
                slice_anchor = decompress_anchor(self.gaussian_conditional, scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets)
                # Non-anchor
                # local_ctx: [B,2 * C, H, W]
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # decompress non-anchor
                slice_nonanchor = decompress_nonanchor(self.gaussian_conditional, scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets)
                y_hat_slice = slice_nonanchor + slice_anchor
                y_hat_slices.append(y_hat_slice)
            else:
                # Anchor
                channel_ctx = self.channel_context[idx](torch.cat(y_hat_slices, dim=1))
                params_anchor = self.entropy_parameters_anchor[idx](torch.cat([channel_ctx, hyper_params], dim=1))
                scales_anchor, means_anchor = params_anchor.chunk(2, 1)
                # decompress anchor
                slice_anchor = decompress_anchor(self.gaussian_conditional, scales_anchor, means_anchor, decoder, cdf, cdf_lengths, offsets)
                # Non-anchor
                # local_ctx: [B,2 * C, H, W]
                local_ctx = self.local_context[idx](slice_anchor)
                params_nonanchor = self.entropy_parameters_nonanchor[idx](torch.cat([local_ctx, channel_ctx, hyper_params], dim=1))
                scales_nonanchor, means_nonanchor = params_nonanchor.chunk(2, 1)
                # decompress non-anchor
                slice_nonanchor = decompress_nonanchor(self.gaussian_conditional, scales_nonanchor, means_nonanchor, decoder, cdf, cdf_lengths, offsets)
                y_hat_slice = slice_nonanchor + slice_anchor
                y_hat_slices.append(y_hat_slice)

        y_hat = torch.cat(y_hat_slices, dim=1)
        torch.backends.cudnn.deterministic = False
        ref = self.f_c(z_hat)
        output = self.decoder(y_hat, ref)

        return output
    
    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(scale_table, force=force)
        updated |= super().update(force=force)
        return updated


if __name__ == "__main__":
    x = torch.randn(1,3,512,512)
    y = torch.randn(1,4,64,64)
    
    model = LFGCM(3,4,[192,192,192,192],128,192,64,3,5,[16,16,32,64,64])
    z = model(x,y)
