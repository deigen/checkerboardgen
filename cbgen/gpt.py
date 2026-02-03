# Modified from:
#   VQGAN:    https://github.com/CompVis/taming-transformers/blob/master/taming/modules/transformer/mingpt.py
#   DiT:      https://github.com/facebookresearch/DiT/blob/main/models.py  
#   nanoGPT:  https://github.com/karpathy/nanoGPT/blob/master/model.py
#   llama:    https://github.com/facebookresearch/llama/blob/main/llama/model.py
#   gpt-fast: https://github.com/pytorch-labs/gpt-fast/blob/main/model.py
#   PixArt:   https://github.com/PixArt-alpha/PixArt-alpha/blob/master/diffusion/model/nets/PixArt_blocks.py
#   LlamaGen: https://github.com/FoundationVision/LlamaGen/blob/main/autoregressive/models/gpt.py
#   PAR:      https://github.com/YuqingWang1029/PAR/blob/main/autoregressive/models/gpt.py

from dataclasses import dataclass
from typing import Optional, List


import torch
import torch.nn as nn
from torch.nn import functional as F


def find_multiple(n: int, k: int):
    if n % k == 0:
        return n
    return n + k - (n % k)

@dataclass
class ModelArgs:
    dim: int = 4096
    n_layer: int = 32
    n_head: int = 32
    n_kv_head: Optional[int] = None
    multiple_of: int = 256  # make SwiGLU hidden layer size multiple of large power of 2
    ffn_dim_multiplier: Optional[float] = None
    rope_base: float = 10000
    norm_eps: float = 1e-5
    initializer_range: float = 0.02
    
    token_dropout_p: float = 0.1
    attn_dropout_p: float = 0.0
    resid_dropout_p: float = 0.1
    ffn_dropout_p: float = 0.1
    drop_path_rate: float = 0.0

    num_classes: int = 1000
    caption_dim: int = 2048
    class_dropout_prob: float = 0.1
    model_type: str = 'c2i'

    vocab_size: int = 16384
    cls_token_num: int = 1
    block_size: int = 256
    max_batch_size: int = 32
    max_seq_len: int = 2048
    spe_token_num: int = 3
    ar_token_num: int = 4

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)


#################################################################################
#                      Embedding Layers for Class Labels                        #
#################################################################################
class LabelEmbedder(nn.Module):
    """
    Embeds class labels into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, num_classes, hidden_size, dropout_prob):
        super().__init__()
        use_cfg_embedding = dropout_prob > 0
        self.embedding_table = nn.Embedding(num_classes + use_cfg_embedding, hidden_size)
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob

    def token_drop(self, labels, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(labels.shape[0], device=labels.device) < self.dropout_prob
        else:
            drop_ids = force_drop_ids == 1
        labels = torch.where(drop_ids, self.num_classes, labels)
        return labels

    def forward(self, labels, train, force_drop_ids=None):
        use_dropout = self.dropout_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            labels = self.token_drop(labels, force_drop_ids)
        embeddings = self.embedding_table(labels).unsqueeze(1)
        return embeddings

class SpecialTokenEmbedding(nn.Module):
    def __init__(self, num_special_tokens, hidden_size):
        super().__init__()
        self.num_special_tokens = num_special_tokens
        self.hidden_size = hidden_size
        self.special_embeddings = nn.Embedding(num_special_tokens, hidden_size)

    def forward(self):
        special_tokens = torch.arange(self.num_special_tokens, device=self.special_embeddings.weight.device)
        special_embeddings = self.special_embeddings(special_tokens)
        return special_embeddings


#################################################################################
#                      Embedding Layers for Text Feature                        #
#################################################################################
class CaptionEmbedder(nn.Module):
    """
    Embeds text caption into vector representations. Also handles label dropout for classifier-free guidance.
    """
    def __init__(self, in_channels, hidden_size, uncond_prob, token_num=120):
        super().__init__()
        self.cap_proj = MLP(in_features=in_channels, hidden_features=hidden_size, out_features=hidden_size)
        self.register_buffer("uncond_embedding", nn.Parameter(torch.randn(token_num, in_channels) / in_channels ** 0.5))
        self.uncond_prob = uncond_prob

    def token_drop(self, caption, force_drop_ids=None):
        """
        Drops labels to enable classifier-free guidance.
        """
        if force_drop_ids is None:
            drop_ids = torch.rand(caption.shape[0], device=caption.device) < self.uncond_prob
        else:
            drop_ids = force_drop_ids == 1
        caption = torch.where(drop_ids[:, None, None], self.uncond_embedding, caption)
        return caption

    def forward(self, caption, train, force_drop_ids=None):
        use_dropout = self.uncond_prob > 0
        if (train and use_dropout) or (force_drop_ids is not None):
            caption = self.token_drop(caption, force_drop_ids)
        embeddings = self.cap_proj(caption)
        return embeddings


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        return x


#################################################################################
#                                  GPT Model                                    #
#################################################################################
class RMSNorm(torch.nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        hidden_dim = 4 * config.dim
        hidden_dim = int(2 * hidden_dim / 3)
        # custom dim factor multiplier
        if config.ffn_dim_multiplier is not None:
            hidden_dim = int(config.ffn_dim_multiplier * hidden_dim)
        hidden_dim = find_multiple(hidden_dim, config.multiple_of)

        self.w1 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w3 = nn.Linear(config.dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, config.dim, bias=False)
        self.ffn_dropout = nn.Dropout(config.ffn_dropout_p)

    def forward(self, x):
        return self.ffn_dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_head, head_dim, dtype):
        super().__init__()
        cache_shape = (max_batch_size, n_head, max_seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]
        k_out = self.k_cache
        v_out = self.v_cache
        k_out[:, :, input_pos] = k_val
        v_out[:, :, input_pos] = v_val

        return k_out, v_out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        assert config.dim % config.n_head == 0
        self.dim = config.dim
        self.head_dim = config.dim // config.n_head
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head if config.n_kv_head is not None else config.n_head
        total_kv_dim = (self.n_head + 2 * self.n_kv_head) * self.head_dim

        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_kv_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.kv_cache = None

        # regularization
        self.attn_dropout_p = config.attn_dropout_p
        self.resid_dropout = nn.Dropout(config.resid_dropout_p)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor = None,
        input_pos: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        key_freqs_cis: Optional[torch.Tensor] = None,
        past_key_value: Optional['DynamicCache'] = None,
        cache_position: Optional[torch.LongTensor] = None,
        layer_idx: int = 0,
    ):
        from transformers.cache_utils import DynamicCache

        bsz, seqlen, _ = x.shape
        kv_size = self.n_kv_head * self.head_dim
        xq, xk, xv = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        xq = xq.view(bsz, seqlen, self.n_head, self.head_dim)
        xk = xk.view(bsz, seqlen, self.n_kv_head, self.head_dim)
        xv = xv.view(bsz, seqlen, self.n_kv_head, self.head_dim)

        # Detect if freqs_cis is complex (from ar.py) or real (traditional gpt.py)
        is_complex_rope = freqs_cis is not None and torch.is_complex(freqs_cis)

        if is_complex_rope:
            # Use complex RoPE (matches modeling_llama.py / ar.py)
            xq = apply_rotary_emb_complex(xq, freqs_cis)
            # Use separate freqs_cis for keys if provided (for dual RoPE mixing)
            xk = apply_rotary_emb_complex(xk, key_freqs_cis if key_freqs_cis is not None else freqs_cis)
        else:
            # Use traditional gpt.py RoPE
            xq = apply_rotary_emb(xq, freqs_cis)
            # Use separate freqs_cis for keys if provided (for dual RoPE mixing)
            xk = apply_rotary_emb(xk, key_freqs_cis if key_freqs_cis is not None else freqs_cis)

        xq, xk, xv = map(lambda x: x.transpose(1, 2), (xq, xk, xv))

        # Use DynamicCache if provided, otherwise fall back to custom KV cache
        if past_key_value is not None:
            # Using DynamicCache from transformers
            if cache_position is None:
                cache_position = torch.arange(seqlen, device=x.device)
            cache_kwargs = {"cache_position": cache_position}
            keys, values = past_key_value.update(xk, xv, layer_idx, cache_kwargs)
        elif self.kv_cache is not None:
            # Legacy KVCache support
            keys, values = self.kv_cache.update(input_pos, xk, xv)
        else:
            keys, values = xk, xv

        keys = keys.repeat_interleave(self.n_head // self.n_kv_head, dim=1)
        values = values.repeat_interleave(self.n_head // self.n_kv_head, dim=1)

        output = F.scaled_dot_product_attention(
            xq, keys, values,
            attn_mask=mask,
            is_causal=True if mask is None else False, # is_causal=False is for KV cache
            dropout_p=self.attn_dropout_p if self.training else 0)

        output = output.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        output = self.resid_dropout(self.wo(output))
        return output


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs, drop_path: float, class_conditional: bool = False):
        super().__init__()
        self.attention = Attention(config)
        self.feed_forward = FeedForward(config)
        self.class_conditional = class_conditional
        # Use affine=False for RMSNorm when using AdaLN
        self.attention_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.drop_path = nn.Identity()

        if self.class_conditional:
            from .adaln import AdaLNModulation
            class_embedding_dim = getattr(config, 'class_embedding_dim', config.dim)
            self.input_modulation = AdaLNModulation(
                config.dim, class_embedding_dim, with_gate=False)
            self.post_attention_modulation = AdaLNModulation(
                config.dim, class_embedding_dim, with_gate=False)

    def forward(
        self, x: torch.Tensor, freqs_cis: torch.Tensor, start_pos: int,
        mask: Optional[torch.Tensor] = None,
        key_freqs_cis: Optional[torch.Tensor] = None,
        label_embeddings: Optional[torch.Tensor] = None,
        past_key_value: Optional['DynamicCache'] = None,
        cache_position: Optional[torch.LongTensor] = None,
        layer_idx: int = 0,
    ):
        # Pre-attention norm + AdaLN
        h_norm = self.attention_norm(x)
        if self.class_conditional and label_embeddings is not None:
            h_norm = self.input_modulation(h_norm, label_embeddings)

        h = x + self.drop_path(self.attention(
            h_norm,
            freqs_cis,  # Can be complex (from ar.py) or real (traditional gpt.py)
            start_pos,
            mask,
            key_freqs_cis=key_freqs_cis,  # Can be complex (per-head mixed) or real
            past_key_value=past_key_value,
            cache_position=cache_position,
            layer_idx=layer_idx,
        ))

        # Pre-FFN norm + AdaLN
        out_norm = self.ffn_norm(h)
        if self.class_conditional and label_embeddings is not None:
            out_norm = self.post_attention_modulation(out_norm, label_embeddings)

        out = h + self.drop_path(self.feed_forward(out_norm))
        return out


class Transformer(nn.Module):
    def __init__(self, config: ModelArgs):
        super().__init__()
        self.config = config
        self.vocab_size = config.vocab_size
        self.n_layer = config.n_layer
        self.block_size = config.block_size
        self.num_classes = config.num_classes
        self.model_type = config.model_type
        self.cls_token_num = config.cls_token_num
        self.spe_token_num = config.spe_token_num
        self.ar_token_num = config.ar_token_num

        if self.model_type == 'c2i':
            self.cls_embedding = LabelEmbedder(config.num_classes, config.dim, config.class_dropout_prob)
        elif self.model_type == 't2i':
            self.cls_embedding = CaptionEmbedder(config.caption_dim, config.dim, config.class_dropout_prob)
        else:
            raise Exception("please check model type")
        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.tok_dropout = nn.Dropout(config.token_dropout_p)

        self.spe_tok_embeddings = SpecialTokenEmbedding(self.spe_token_num, config.dim)


        # transformer blocks
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.n_layer)]
        self.layers = torch.nn.ModuleList()
        for layer_id in range(config.n_layer):
            self.layers.append(TransformerBlock(config, dpr[layer_id]))

        # output layer
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        # 2d rotary pos embedding
        grid_size = int(self.block_size ** 0.5)
        assert grid_size * grid_size == self.block_size
        self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num, spe_token_num=self.spe_token_num, ar_token_num=self.ar_token_num)
        
        max_len = self.freqs_cis.shape[0]
        group_mask = torch.tril(torch.ones(max_len, max_len, dtype=torch.bool))
        group_mask[:, 0] = True
        group_size = self.spe_token_num + 1
        
        for i in range(0, max_len // group_size):
            start = self.ar_token_num + i * group_size
            end = start + group_size
            group_mask[start:end, :end] = True
        self.group_mask = group_mask

        # KVCache
        self.max_batch_size = -1
        self.max_seq_length = -1

        self.initialize_weights()

    def initialize_weights(self):        
        # Initialize nn.Linear and nn.Embedding
        self.apply(self._init_weights)

        # Zero-out output layers:
        nn.init.constant_(self.output.weight, 0)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def setup_caches(self, max_batch_size, max_seq_length, dtype):
        # if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
        #     return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_head, head_dim, dtype)

        group_mask = torch.tril(torch.ones(self.max_seq_length,  self.max_seq_length, dtype=torch.bool))
        group_mask[:, 0] = True
        group_size = self.spe_token_num + 1
        for i in range(0, self.max_seq_length // group_size):
            start = i * group_size + self.ar_token_num
            end = start + group_size
            group_mask[start:end, :end] = True
        self.causal_mask = group_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)
        
        # causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))
        # self.causal_mask = causal_mask.unsqueeze(0).repeat(self.max_batch_size, 1, 1)
        grid_size = int(self.config.block_size ** 0.5)
        assert grid_size * grid_size == self.block_size
        self.freqs_cis = precompute_freqs_cis_2d(grid_size, self.config.dim // self.config.n_head, self.config.rope_base, self.cls_token_num, spe_token_num = self.spe_token_num, ar_token_num=self.ar_token_num)

    def forward(
        self, 
        idx: torch.Tensor, 
        cond_idx: torch.Tensor,  # cond_idx_or_embed
        input_pos:  Optional[torch.Tensor] = None, 
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        valid: Optional[torch.Tensor] = None,
    ):
        if idx is not None and cond_idx is not None: # training or naive inference
            cond_embeddings = self.cls_embedding(cond_idx, train=self.training)[:,:self.cls_token_num]
            token_embeddings = self.tok_embeddings(idx)
            spe_embeddings = self.spe_tok_embeddings().unsqueeze(0).expand(cond_embeddings.shape[0], -1, -1)
            # token_embeddings = torch.cat((cond_embeddings, token_embeddings), dim=1)
            token_embeddings_first, token_embeddings_last = token_embeddings[:,:self.ar_token_num], token_embeddings[:,self.ar_token_num:]
            token_embeddings = torch.cat((cond_embeddings, token_embeddings_first, spe_embeddings, token_embeddings_last), dim=1)
            h = self.tok_dropout(token_embeddings)

            mask = self.group_mask[:token_embeddings.shape[1], :token_embeddings.shape[1]]
            batch_size = cond_embeddings.shape[0]
            mask = mask.unsqueeze(0).unsqueeze(0).repeat(batch_size, 1, 1, 1)
            mask = mask.to(h.device)

            self.freqs_cis = self.freqs_cis.to(h.device)
        else:
            if cond_idx is not None: # prefill in inference
                self.start = False
                token_embeddings = self.cls_embedding(cond_idx, train=self.training)[:,:self.cls_token_num]
                # spe_embeddings = self.spe_tok_embeddings().unsqueeze(0).expand(token_embeddings.shape[0], -1, -1)
                # token_embeddings = torch.cat((token_embeddings, spe_embeddings), dim=1)
            else: # decode_n_tokens(kv cache) in inferenceå
                if idx.shape[1]>1 and not self.start:
                    token_embeddings = self.tok_embeddings(idx)
                    spe_embeddings = self.spe_tok_embeddings().unsqueeze(0).expand(token_embeddings.shape[0], -1, -1)
                    token_embeddings = torch.cat((token_embeddings[:,-1:], spe_embeddings), dim=1)
                    # token_embeddings = spe_embeddings.to(token_embeddings.device)
                    self.start = True
                else:
                    token_embeddings = self.tok_embeddings(idx)
            
            bs = token_embeddings.shape[0]
            mask = self.causal_mask[:bs, None, input_pos]
            h = self.tok_dropout(token_embeddings)
            self.freqs_cis = self.freqs_cis
        
        if self.training:
            freqs_cis = self.freqs_cis[:token_embeddings.shape[1]]
        else:
            freqs_cis = self.freqs_cis[input_pos]
        # transformer blocks
        for layer in self.layers:
            h = layer(h, freqs_cis, input_pos, mask)
        
        # output layers
        h = self.norm(h)
        logits = self.output(h).float()
        
        if self.training:
            # logits = logits[:, self.cls_token_num - 1:].contiguous()
            logits = logits[:, self.cls_token_num -1 : self.cls_token_num -1 + self.block_size].contiguous()

        # if we are given some desired targets also calculate the loss
        loss = None
        if valid is not None:
            loss_all = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none')
            valid_all = valid[:,None].repeat(1, targets.shape[1]).view(-1)
            loss = (loss_all * valid_all).sum() / max(valid_all.sum(), 1)
        elif targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))

        return logits, loss


    def get_fsdp_wrap_module_list(self) -> List[nn.Module]:
        return list(self.layers)



#################################################################################
#                      Rotary Positional Embedding Functions                    #
#################################################################################
# https://github.com/pytorch-labs/gpt-fast/blob/main/model.py 
def precompute_freqs_cis(seq_len: int, n_elem: int, base: int = 10000, cls_token_num=120):
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs) # (seq_len, head_dim // 2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1) # (cls_token_num+seq_len, head_dim // 2, 2)
    cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache]) # (cls_token_num+seq_len, head_dim // 2, 2)
    return cond_cache 


def precompute_freqs_cis_2d(grid_size: int, n_elem: int, base: int = 10000, cls_token_num=120, spe_token_num=3, ar_token_num=4):
    # split the dimension into half, one for x and one for y
    half_dim = n_elem // 2
    freqs = 1.0 / (base ** (torch.arange(0, half_dim, 2)[: (half_dim // 2)].float() / half_dim))
    t = torch.arange(grid_size, device=freqs.device)
    freqs = torch.outer(t, freqs) # (grid_size, head_dim // 2)
    freqs_grid = torch.concat([
        freqs[:, None, :].expand(-1, grid_size, -1),
        freqs[None, :, :].expand(grid_size, -1, -1),
    ], dim=-1)  # (grid_size, grid_size, head_dim // 2)
    cache_grid = torch.stack([torch.cos(freqs_grid), torch.sin(freqs_grid)], dim=-1) # (grid_size, grid_size, head_dim // 2, 2)
    sub_num = int(ar_token_num**0.5)

    cache_grid = cache_grid.reshape(sub_num, grid_size//sub_num, sub_num, grid_size//sub_num, half_dim, 2)
    cache_grid = cache_grid.permute(1, 3, 0, 2, 4, 5)
    cache = cache_grid.flatten(0, 3)
    cache_one, cache_two = cache[:ar_token_num], cache[ar_token_num:]
    sep_cache = torch.zeros(spe_token_num, n_elem // 2, 2)
    cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache_one, sep_cache, cache_two])
    # cond_cache = torch.cat([torch.zeros(cls_token_num, n_elem // 2, 2), cache]) # (cls_token_num+grid_size**2, head_dim // 2, 2)
    return cond_cache 


def apply_rotary_emb(x: torch.Tensor, freqs_cis: torch.Tensor):
    # x: (bs, seq_len, n_head, head_dim)
    # freqs_cis (seq_len, head_dim // 2, 2) OR (bs, seq_len, n_head, head_dim // 2, 2) for per-head
    xshaped = x.float().reshape(*x.shape[:-1], -1, 2) # (bs, seq_len, n_head, head_dim//2, 2)

    # Check if per-head freqs_cis (5D) or shared freqs_cis (3D)
    if freqs_cis.dim() == 5:
        # Per-head RoPE: (bs, seq_len, n_head, head_dim//2, 2)
        assert freqs_cis.shape == xshaped.shape, f"Per-head freqs_cis shape {freqs_cis.shape} doesn't match x shape {xshaped.shape}"
        # No broadcasting needed
        pass
    elif freqs_cis.dim() == 3:
        # Shared RoPE across heads: (seq_len, head_dim // 2, 2)
        freqs_cis = freqs_cis.view(1, xshaped.size(1), 1, xshaped.size(3), 2) # (1, seq_len, 1, head_dim//2, 2)
    else:
        raise ValueError(f"freqs_cis must be 3D or 5D, got shape {freqs_cis.shape}")

    x_out2 = torch.stack([
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
    ], dim=-1)
    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


def rotate_half(x):
    """Rotates half the hidden dims of the input (for complex RoPE)."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_emb_complex(x: torch.Tensor, freqs_cis_complex: torch.Tensor):
    """
    Apply rotary position embedding using complex representation.
    This matches the approach used in modeling_llama.py and ar.py.

    Args:
        x: Input tensor of shape (bs, seq_len, n_head, head_dim)
        freqs_cis_complex: Complex RoPE tensor of shape:
            - (B, 1, S, head_dim) complex for shared across heads, OR
            - (B, num_heads, S, head_dim) complex for per-head

    Returns:
        Tensor of shape (bs, seq_len, n_head, head_dim) with RoPE applied

    Pairing scheme: Rotation θ_i applies to pair (x_i, x_{i+head_dim/2})
    This differs from the standard gpt.py pairing of (x_{2i}, x_{2i+1})
    """
    # freqs_cis_complex: (B, 1, S, head_dim) or (B, num_heads, S, head_dim) complex
    # x: (bs, seq_len, n_head, head_dim)

    # Handle broadcasting: if freqs_cis has num_heads=1, expand to match x
    if freqs_cis_complex.shape[1] == 1 and freqs_cis_complex.shape[1] != x.shape[2]:
        # (B, 1, S, head_dim) -> (B, num_heads, S, head_dim)
        freqs_cis_complex = freqs_cis_complex.expand(-1, x.shape[2], -1, -1)

    # Permute to match x's layout: (B, num_heads, S, head_dim) -> (B, S, num_heads, head_dim)
    cos = freqs_cis_complex.real.permute(0, 2, 1, 3)  # (B, S, num_heads, head_dim)
    sin = freqs_cis_complex.imag.permute(0, 2, 1, 3)  # (B, S, num_heads, head_dim)

    # Handle batch dimension broadcasting if needed
    if cos.shape[0] == 1 and x.shape[0] != 1:
        cos = cos.expand(x.shape[0], -1, -1, -1)
        sin = sin.expand(x.shape[0], -1, -1, -1)

    # Apply rotate_half style RoPE
    x_embed = (x * cos) + (rotate_half(x) * sin)
    return x_embed.type_as(x)



#################################################################################
#                                GPT Configs                                    #
#################################################################################
### text-conditional
def GPT_7B(**kwargs):
    return Transformer(ModelArgs(n_layer=32, n_head=32, dim=4096, **kwargs)) # 6.6B

def GPT_3B(**kwargs):
    return Transformer(ModelArgs(n_layer=24, n_head=32, dim=3200, **kwargs)) # 3.1B

def GPT_1B(**kwargs):
    return Transformer(ModelArgs(n_layer=22, n_head=32, dim=2048, **kwargs)) # 1.2B

### class-conditional
def GPT_XXXL(**kwargs):
    return Transformer(ModelArgs(n_layer=48, n_head=40, dim=2560, **kwargs)) # 3.9B

def GPT_XXL(**kwargs):
    return Transformer(ModelArgs(n_layer=48, n_head=24, dim=1536, **kwargs)) # 1.4B

def GPT_XL(**kwargs):
    return Transformer(ModelArgs(n_layer=36, n_head=20, dim=1280, **kwargs)) # 775M

def GPT_L(**kwargs):
    return Transformer(ModelArgs(n_layer=24, n_head=16, dim=1024, **kwargs)) # 343M

def GPT_B(**kwargs):
    return Transformer(ModelArgs(n_layer=12, n_head=12, dim=768, **kwargs)) # 111M
        

GPT_models = {
    'GPT-B': GPT_B, 'GPT-L': GPT_L, 'GPT-XL': GPT_XL, 'GPT-XXL': GPT_XXL, 'GPT-XXXL': GPT_XXXL,
    'GPT-1B': GPT_1B, 'GPT-3B': GPT_3B, 'GPT-7B': GPT_7B,
}


#################################################################################
#           Simplified GPT Transformer for AR Integration                      #
#################################################################################

class GPTTransformerAR(nn.Module):
    """
    Simplified GPT Transformer designed for integration with AR model.
    Key differences from standard Transformer:
    - No built-in embeddings (ar.py handles these)
    - Accepts inputs_embeds instead of token indices
    - Supports dual RoPE with per-head mixing
    - Uses DynamicCache from transformers
    - Supports AdaLN for class conditioning
    - Custom 4D attention mask support
    """
    def __init__(self, config: ModelArgs, position_embedding_module=None, class_conditional: bool = False):
        super().__init__()
        self.config = config
        self.n_layer = config.n_layer
        self.class_conditional = class_conditional

        # Transformer blocks
        dpr = [x.item() for x in torch.linspace(0, config.drop_path_rate, config.n_layer)]
        self.layers = nn.ModuleList()
        for layer_id in range(config.n_layer):
            self.layers.append(TransformerBlock(config, dpr[layer_id], class_conditional=class_conditional))

        # Output norm
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)

        # Position embeddings module (provided externally, similar to LlamaModel)
        if position_embedding_module is not None:
            self.rotary_emb = position_embedding_module
        else:
            # Fallback: create simple position embeddings (not used in ar.py integration)
            self.rotary_emb = None

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize nn.Linear and nn.Embedding
        self.apply(self._init_weights)

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)

    def forward(
        self,
        inputs_embeds: torch.FloatTensor,
        sequence_order: torch.LongTensor,
        past_sequence_order: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional['DynamicCache'] = None,
        label_embeds: Optional[torch.FloatTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
    ):
        """
        Forward pass matching the interface used by ar.py with LlamaModel.

        Args:
            inputs_embeds: Pre-computed token embeddings (B, S, C)
            sequence_order: Current position IDs for RoPE (B, S)
            past_sequence_order: Previous sampled position IDs for dual RoPE (B, S)
            attention_mask: Custom 4D attention mask (B, 1, S, S) or None
            past_key_values: KV cache (DynamicCache from transformers)
            label_embeds: Class embeddings for AdaLN modulation (B, 1, C)
            cache_position: Cache positions (S,) - used for compatibility
            use_cache: Whether to use/update KV cache

        Returns:
            Dictionary with 'last_hidden_state' and optionally 'past_key_values'
        """
        from transformers.cache_utils import DynamicCache
        from transformers.modeling_outputs import BaseModelOutputWithPast

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache()

        hidden_states = inputs_embeds
        B, S, C = hidden_states.shape

        # Generate RoPE embeddings via position embedding module
        if self.rotary_emb is not None:
            curr_rope, past_rope = self.rotary_emb(sequence_order, past_sequence_order)
        else:
            raise ValueError("position_embedding_module must be provided")

        # curr_rope, past_rope are (B, 1, S, head_dim) complex tensors
        # We can now pass these directly to the attention layers!
        # The apply_rotary_emb_complex function will handle them correctly.

        # Setup cache position if using cache
        if cache_position is None and use_cache:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + S, device=inputs_embeds.device
            )

        # Process through transformer blocks
        for layer_num, layer in enumerate(self.layers):
            # Apply per-head mixing for key position embeddings
            if self.rotary_emb is not None and hasattr(self.rotary_emb, 'apply_mixing'):
                # Get mixed RoPE for keys
                # This returns (B, num_heads, S, head_dim) complex tensor
                key_freqs_cis = self.rotary_emb.apply_mixing((curr_rope, past_rope), layer_num)
            else:
                # Use same RoPE for keys and queries
                key_freqs_cis = curr_rope

            hidden_states = layer(
                hidden_states,
                freqs_cis=curr_rope,  # Complex tensor (B, 1, S, head_dim)
                start_pos=None,  # Not used with DynamicCache
                mask=attention_mask,
                key_freqs_cis=key_freqs_cis,  # Complex tensor (B, num_heads, S, head_dim) or (B, 1, S, head_dim)
                label_embeddings=label_embeds,
                past_key_value=past_key_values,
                cache_position=cache_position,
                layer_idx=layer_num,
            )

        # Final norm
        hidden_states = self.norm(hidden_states)

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
        )
