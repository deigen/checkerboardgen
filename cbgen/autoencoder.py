import os
import importlib
import json
import copy
import random
import torch
from torch import nn
import numpy as np
import torch.nn.functional as F
from torch import einsum
from einops import rearrange
from contextlib import contextmanager
from omegaconf import OmegaConf

try:
    import pytorch_lightning as pl
    have_pl = True
except ImportError:
    have_pl = False

from util.misc import instantiate_from_config

from .ae_encoder import Encoder, Decoder


def load_model(config_path, checkpoint_path, cull_threshold=None, set_initialized=True, params=None):
    config = OmegaConf.load(config_path)
    if params is not None:
        for k, v in params.items():
            config.model.params[k] = v
    model = instantiate_from_config(config.model)
    if checkpoint_path is not None:
        model.init_from_ckpt(checkpoint_path)
        if set_initialized:
            model.quantize.initialized = torch.tensor(True, dtype=torch.bool)
    if hasattr(model, 'loss'):
        model.loss = None
    if cull_threshold is not None:
        norms = model.quantize.embedding.weight.norm(dim=1)
        keep = norms > cull_threshold
        print(f"VQ Autoencoder: Culling {len(keep) - keep.sum()} / {len(keep)} embeddings")
        model.quantize.embedding.weight.data = model.quantize.embedding.weight.data[keep]
        model.quantize.n_e = keep.sum().item()
        model.n_embed = keep.sum().item()
    return model


def rescale_with_interpolate(x, size=None, scale=None):
    assert x.ndim == 4 and x.shape[2] == x.shape[3], 'x must be a 4D square tensor'
    assert (size is not None) ^ (scale is not None), "one of size or scale should be given"
    w = x.shape[-1]
    if scale is not None:
        size = int(torch.round(torch.as_tensor(w * scale)).item())
    if isinstance(size, (tuple, list)) or (isinstance(size, torch.Tensor) and size.ndim > 0):
        assert len(size) == 2 and size[0] == size[1], "only square sizes are supported"
        size = size[0]
    if size == w:
        return x
    return F.interpolate(x, size=size, mode='bilinear', align_corners=False)



class VQModel(pl.LightningModule if have_pl else nn.Module):
    def __init__(self,
                 ddconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 p_train_random_scale=0.9,
                 quantize_prob=1.0,
                 quantize_iter_start=1000,
                 requant_pyramid_levels=-1,
                 scale_ratio=2,
                 ignore_keys=[],
                 image_key="image",
                 resolution=None,
                 colorize_nlabels=None,
                 normalize=False,
                 monitor=None,
                 batch_resize_range=None,
                 scheduler_config=None,
                 lr_g_factor=1.0,
                 ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.image_key = image_key
        self.requant_pyramid_levels = requant_pyramid_levels
        self.scale_ratio = scale_ratio
        if self.requant_pyramid_levels == -1:
            self.requant_pyramid_levels = 1000  # all levels
        if not resolution:
            resolution = ddconfig['resolution']


        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)

        self.resolution = resolution
        self.stride = 2**(self.encoder.num_resolutions-1)

        # feature sizes: from smallest size (largest stride) first
        final_size = self.resolution // self.stride  # input feature map size
        scale_sizes = [0, 1]  # include zero size and 1x1 size
        while scale_sizes[-1] < final_size:
            scale_sizes.append(scale_sizes[-1] * scale_ratio)
        scale_sizes = sorted(set([int(np.round(s)) for s in scale_sizes]))
        assert scale_sizes[-1] == final_size
        self.scales_feat_sizes = scale_sizes

        self.p_train_random_scale = p_train_random_scale
        self.quantize_prob=quantize_prob
        self.apply_normalization = normalize
        self.quantize_iter_start = quantize_iter_start
        self.quantize = VectorQuantizer(n_embed, embed_dim,
                                        normalize=normalize,
                                        beta=0.001,
                                        )
        self.quant_conv = torch.nn.Conv2d(self.encoder.z_channels, embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, self.encoder.z_channels, 1)
        if colorize_nlabels is not None:
            assert type(colorize_nlabels)==int
            self.register_buffer("colorize", torch.randn(3, colorize_nlabels, 1, 1))
        self.register_buffer("cluster_usage_train", torch.zeros(n_embed))
        self.cluster_usage_alpha = 0.01
        if monitor is not None:
            self.monitor = monitor
        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}.")

        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)
        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor

    def init_from_ckpt(self, path, ignore_keys=list()):
        sd = torch.load(path, map_location="cpu", weights_only=False)
        if 'state_dict' in sd:
            sd = sd['state_dict']
        if 'model' in sd and not any(k.startswith('encoder.') for k in sd.keys()):
            sd = sd['model']
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startswith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]
        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")
        if len(missing) > 0:
            print(f"Missing Keys: {missing}")
            print(f"Unexpected Keys: {unexpected}")

    def normalize(self, h, dim=1):
        if self.apply_normalization:
            h = F.normalize(h, p=2, dim=dim)
        return h

    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return self.normalize(h)

    def decode(self, h):
        h = self.normalize(h)
        h = self.post_quant_conv(h)
        dec = self.decoder(h)
        return dec

    def rescale(self, x, size=None, scale=None):
        return rescale_with_interpolate(x, size=size, scale=scale)

    def set_scale_sizes(self, feat_sizes):
        self.scales_feat_sizes = feat_sizes

    def rescale_latents(self, z, size=None, scale=None, fullres_size=None):
        return self.normalize(self.rescale(z, size=size, scale=scale))

    def apply_requantization(self, x, update_cluster_usage=False, initializing_quantizer=False):
        # reconstructs x1 from z0 + quantize(z1 - z0)
        # quantization pyramid
        pyr = self.quantization_pyramid(x, initializing_quantizer=initializing_quantizer)
        zq_pyr = pyr['zq_pyr']
        q_losses = pyr['q_losses']
        #dq_inds = pyr['dq_inds']
        q_inds = pyr['q_inds']

        recon = self.decode(zq_pyr[-1])
        qloss = sum(q_losses)

        log_dict = {'train/quantization_loss': qloss,
                    }

        if update_cluster_usage and not initializing_quantizer:
            with torch.no_grad():
                reinit_unused = self.global_step > self.quantize_iter_start + 1000 and self.global_step % 1000 == 0
                ind = torch.cat([xi.flatten() for xi in q_inds])
                perplexity, usage = self.update_cluster_usage(ind, reinit_unused=reinit_unused)
                log_dict[f"train/cluster_perplexity"] = perplexity
                log_dict[f"train/cluster_usage"] = usage

        return recon, qloss, log_dict

    def quantization_pyramid(self, x, initializing_quantizer=False):
        # create pyramid of z by recoding x at each scale
        z = self.encode(x)
        z_pyr = [z]
        curr_size_i = self.scales_feat_sizes.index(z.shape[2])
        assert curr_size_i != -1
        while curr_size_i > 0 and len(z_pyr) < self.requant_pyramid_levels:
            curr_size_i -= 1
            size = self.scales_feat_sizes[curr_size_i] * self.stride
            if size == 0:
                break  # stop before scale 0
            imgs_s = self.rescale(x, size=size)
            z = self.encode(imgs_s)  # B, C, H, W  
            z_pyr.append(z)

        z_pyr = z_pyr[::-1]  # z_pyr now from smallest to largest

        # loss weights for each scale
        scale_npix = np.asarray([zi.shape[2]*zi.shape[3] for zi in z_pyr])
        scale_loss_weights = np.sqrt(scale_npix)
        scale_loss_weights /= scale_loss_weights.sum()

        # compute quantization
        zq_pyr = []
        q_losses = []
        q_inds = []

        for i in range(len(z_pyr)):
            z_target = z_pyr[i].detach()

            # direct quantization of current scale
            zq, qloss, ind = self.quantize_with_init(
                z_target,
                scale_index=i,
                initializing_quantizer=initializing_quantizer)

            # Store results
            zq_pyr.append(zq)
            q_inds.append(ind)
            q_losses.append(qloss * scale_loss_weights[i])

        return {'zq_pyr': zq_pyr,  # pyramid after quantization
                'q_losses': q_losses,  # quantization losses at each scale
                'q_inds': q_inds,  # quantized code indices at each scale
                }

    def quantize_with_init(self, z, scale_index, initializing_quantizer=False):
        if initializing_quantizer:
            self.quantize.init_update(z, scale_index)
            q, qloss, ind = z, torch.tensor(0.).to(z), None
        else:
            q, qloss, ind = self.quantize(z)
        return q, qloss, ind

    def forward(self, input, requantize=None):
        if requantize is None:
            requantize = self.quantize.initialized
        if requantize:
            recon, qloss, _ = self.apply_requantization(input)
        else:
            z = self.encode(input)
            recon = self.decode(z)
            qloss = torch.tensor(0.).to(z)

        return recon, qloss

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]
        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()
        if self.batch_resize_range is not None:
            lower_size = self.batch_resize_range[0]
            upper_size = self.batch_resize_range[1]
            if self.global_step <= 4:
                # do the first few batches with max size to avoid later oom
                new_resize = upper_size
            else:
                new_resize = np.random.choice(np.arange(lower_size, upper_size+16, 16))
            if new_resize != x.shape[2]:
                x = F.interpolate(x, size=new_resize, mode="bicubic")
            x = x.detach()
        return x

    def training_step(self, batch, batch_idx, optimizer_idx=0):
        inputs = self.get_input(batch, self.image_key)

        initializing_quantizer = (self.global_step < self.quantize_iter_start)

        if random.random() < self.p_train_random_scale:
            scale_i = random.choice(range(1, len(self.scales_feat_sizes)))  # exclude zero scale
        else:
            scale_i = len(self.scales_feat_sizes) - 1  # full size
        scale = self.scales_feat_sizes[scale_i] / self.scales_feat_sizes[-1]
        x = self.rescale(inputs, scale=scale)
        recon, qloss, log_dict_rq = self.apply_requantization(
            x, update_cluster_usage=True, initializing_quantizer=initializing_quantizer)

        self.log_dict(log_dict_rq, prog_bar=False, logger=True, on_step=True, on_epoch=False)
        if not qloss.requires_grad:
            qloss.requires_grad = True
        return qloss

    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        return log_dict

    def _validation_step(self, batch, batch_idx, suffix=""):
        x = self.get_input(batch, self.image_key)
        xrec, qloss = self(x)
        self.log(f"val{suffix}/qloss", qloss,
                   prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
        return self.log_dict

    def configure_optimizers(self):
        lr = self.lr_g_factor * self.learning_rate
        print("lr", lr)
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr, betas=(0.5, 0.9))
        return [opt_ae], []

    @torch.no_grad()
    def update_cluster_usage(self, predicted_indices, cluster_usage_buffer=None, reinit_unused=False):
        # usage tracking for embeddings
        # perplexity part originally based on https://github.com/karpathy/deep-vector-quantization/blob/main/model.py
        if cluster_usage_buffer is None:
            cluster_usage_buffer = self.cluster_usage_train
        distrib = cluster_usage_buffer
        x = F.one_hot(predicted_indices.reshape(-1), self.n_embed).float()
        avg_counts = x.mean(0)
        distrib *= (1 - self.cluster_usage_alpha)
        distrib += avg_counts * self.cluster_usage_alpha
        p = distrib / distrib.sum()
        perplexity = (-p * torch.log(p + 1e-10)).sum().exp()
        used = (p > 1e-4/self.n_embed)
        #print('perplexity: %.3f, used: %.4f' % (perplexity, used.float().mean()))
        if reinit_unused:
            # TODO rank 0 only
            # reinitialize unused embeddings
            unused = torch.nonzero(~used, as_tuple=False).squeeze(-1)
            if len(unused) > 0:
                print(f"Reinitializing {len(unused)} unused embeddings")
                m = self.quantize.embedding.weight.data[used].mean(0)
                s = self.quantize.embedding.weight.data[used].std(0)
                self.quantize.embedding.weight.data[unused] = (
                    m + s * torch.randn_like(self.quantize.embedding.weight.data[unused]))
        return perplexity, used.float().mean()

    def log_images(self, batch, only_inputs=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if only_inputs:
            log["inputs"] = x
            return log
        #xrec, _ = self(x)
        if random.random() < 0.5:
            xrec, _ = self(x)
        else:
            xrec, _ = self(self.rescale(x, scale=0.5))
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x


class UnitNorm(nn.Module):
    def forward(self, x):
        return F.normalize(x, p=2, dim=1)


class VectorQuantizer(nn.Module):
    """
    Based on LDM VQ code, https://github.com/CompVis/latent-diffusion
    """
    def __init__(self, n_e, e_dim, beta, dist_p=2, normalize=False):
        super().__init__()
        self.n_e = n_e
        self.e_dim = e_dim
        self.beta = beta
        self.dist_p = dist_p
        self.init_n = {}
        self.init_moment1 = {}
        self.init_moment2 = {}
        self.normalize = normalize
        self.register_buffer("initialized", torch.tensor(False, dtype=torch.bool))
        
        self.embedding = nn.Embedding(self.n_e, self.e_dim)
        if self.normalize:
            torch.nn.utils.parametrize.register_parametrization(
                self.embedding, "weight", UnitNorm()
            )
            self.embedding._is_hf_initialized = True  # flag as init done

    def forward(self, z, temp=None, rescale_logits=False, return_logits=False):
        assert temp is None or temp==1.0, "Only for interface compatible with Gumbel"
        assert rescale_logits==False, "Only for interface compatible with Gumbel"
        assert return_logits==False, "Only for interface compatible with Gumbel"
        if not self.initialized:
            self.init_embeddings()
        # reshape z -> (batch, height, width, channel) and flatten
        z = rearrange(z, 'b c h w -> b h w c').contiguous()
        if self.normalize:
            z = F.normalize(z, p=2, dim=-1)
        z_flattened = z.view(-1, self.e_dim)

        # distances from z to embeddings
        d = torch.cdist(z_flattened, self.embedding.weight, p=self.dist_p)

        # compute loss for embedding
        # do quantization
        min_encoding_indices = torch.argmin(d, dim=1)
        z_q = self.embedding(min_encoding_indices).view(z.shape)

        # compute loss for embedding and z
        if self.dist_p == 2:
            loss_d1 = torch.mean((z_q.detach() - z)**2)
            loss_d2 = torch.mean((z_q - z.detach()) ** 2)
        elif self.dist_p == 1:
            loss_d1 = torch.mean(torch.abs(z_q.detach() - z))
            loss_d2 = torch.mean(torch.abs(z_q - z.detach()))
        else:
            raise NotImplementedError()

        loss = self.beta * loss_d1 + loss_d2 if self.beta > 0 else loss_d2

        # preserve gradients
        z_q = z + (z_q - z).detach()

        # reshape back to match original input shape
        min_encoding_indices = min_encoding_indices.view(*z.shape[:-1])
        z_q = rearrange(z_q, 'b h w c -> b c h w').contiguous()

        return z_q, loss, min_encoding_indices

    def init_update(self, z, scale_ind):
        # gather init stats
        if self.initialized:
            return
        # z is (b,c,h,w)
        m1 = z.mean(dim=(0,2,3))
        m2 = (z**2).mean(dim=(0,2,3))
        n = z.shape[0]*z.shape[2]*z.shape[3]
        if scale_ind not in self.init_n:
            self.init_n[scale_ind] = 0
        if self.init_n[scale_ind] == 0:
            self.init_moment1[scale_ind] = m1
            self.init_moment2[scale_ind] = m2
        else:
            a = 1/(1+self.init_n[scale_ind])  # exponential moving average factor
            self.init_moment1[scale_ind] = self.init_moment1[scale_ind] * (1-a) + a * m1
            self.init_moment2[scale_ind] = self.init_moment2[scale_ind] * (1-a) + a * m2
        self.init_n[scale_ind] += n

    def init_embeddings(self):
        assert not self.initialized
        total_n = sum(self.init_n.values())
        assert total_n > 10, "not enough data to init"
        w_inits = []
        num_embed, embed_dim = self.embedding.weight.shape
        for scale_ind in sorted(self.init_n.keys()):
            print(f"Scale {scale_ind}: init n = {self.init_n[scale_ind]}")
            mean = self.init_moment1[scale_ind]
            std = torch.sqrt(self.init_moment2[scale_ind] - self.init_moment1[scale_ind] ** 2)
            n = max(1, int(self.n_e * self.init_n[scale_ind] / total_n))
            if scale_ind == max(self.init_n.keys()):
                n = self.n_e - sum(w.shape[0] for w in w_inits)
            print(f"Initializing {n} embeddings with mean and std: {mean[:5]}..., {std[:5]}...")
            w_inits.append(torch.randn(n, embed_dim).to(mean) * std[None, :] + mean[None, :])
        w_init = torch.cat(w_inits, dim=0)
        assert w_init.shape[0] == num_embed

        if self.normalize:
            # Set the original (unconstrained) parameter through the parametrization
            self.embedding.parametrizations.weight.original.data.copy_(
                F.normalize(w_init, p=2, dim=1).to(self.embedding.weight.device)
            )
        else:
            # For non-parametrized weights, set directly
            self.embedding.weight.data.copy_(w_init.to(self.embedding.weight.device))

        self.initialized = torch.tensor(True, dtype=torch.bool)

    def get_codebook_entry(self, indices, shape):
        # shape specifying (batch, height, width, channel)
        # get quantized latent vectors
        z_q = self.embedding(indices)

        if shape is not None:
            z_q = z_q.view(shape)
            # reshape back to match original input shape
            z_q = z_q.permute(0, 3, 1, 2).contiguous()

        return z_q

class Resid(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x
