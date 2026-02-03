import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from cbgen.gpt import GPTTransformerAR, ModelArgs as GPTModelArgs
from transformers.cache_utils import DynamicCache

from cbgen.autoencoder import load_model as load_vae
from cbgen.checkerboard import progressive_checkerboard

_default_llama_config = dict(
    hidden_size=256,
    intermediate_size=1024,
    num_attention_heads=8,
    num_hidden_layers=8,
    rms_norm_eps=1.0e-6,
)

class AR(nn.Module):
    def __init__(self, img_size, autoenc=None, llama_config=None,
                 class_cond=False, num_classes=0, uncond_training_prob=0.1,
                 scale_min_size=1, scale_ratio=2, num_blocks_per_scale=8,
                 random_replace=0.0, random_replace_sampled=False,
                 random_block_size=False,
                 head_weights_l2_loss=1e-3,
                 use_head_mlp=True,
                 num_mixing_layers=2,
                 single_scale=False,
    ):
        super().__init__()
        self.img_size = img_size
        self.class_cond = class_cond
        self.num_classes = num_classes
        self.uncond_training_prob = uncond_training_prob
        self.scale_ratio = scale_ratio
        self.num_blocks_per_scale = num_blocks_per_scale
        self.random_replace = random_replace
        self.random_replace_sampled = random_replace_sampled
        self.random_block_size = random_block_size
        self.head_weights_l2_loss = head_weights_l2_loss
        self.use_head_mlp = use_head_mlp
        self.num_mixing_layers = num_mixing_layers
        self.single_scale = single_scale
        assert num_classes > 0 if class_cond else num_classes == 0

        # load and freeze autoencoder
        assert isinstance(autoenc, dict)
        self.autoenc = load_vae(**autoenc)
        self.autoenc = self.freeze(self.autoenc)
        self.vae_stride = 2**(self.autoenc.encoder.num_resolutions-1)
        self.vq_num_tokens = self.autoenc.n_embed

        vae_size = self.img_size // self.vae_stride

        # feature sizes: from smallest size (largest stride) first
        scale_sizes = [vae_size]
        if not self.single_scale:
            while scale_sizes[-1] > scale_min_size:
                scale_sizes.append(scale_sizes[-1] / scale_ratio)
            if scale_sizes[-1] == 0:
                scale_sizes = scale_sizes[:-1]
            scale_sizes.append(scale_min_size)
            scale_sizes = sorted(set([int(np.round(s)) for s in scale_sizes]))
        self.scale_sizes = scale_sizes

        self.autoenc.set_scale_sizes(self.scale_sizes)

        self.num_scales = len(self.scale_sizes)

        self.init_model(llama_config)

    def init_model(self, llama_config):
        # Convert llama_config dict to GPT ModelArgs
        config_dict = llama_config or _default_llama_config

        # Map parameters to GPT ModelArgs
        gpt_config = GPTModelArgs(
            dim=config_dict.get('hidden_size', _default_llama_config['hidden_size']),
            n_layer=config_dict.get('num_hidden_layers', _default_llama_config['num_hidden_layers']),
            n_head=config_dict.get('num_attention_heads', _default_llama_config['num_attention_heads']),
            n_kv_head=config_dict.get('num_key_value_heads', None),
            multiple_of=256,
            ffn_dim_multiplier=None,
            norm_eps=config_dict.get('rms_norm_eps', _default_llama_config['rms_norm_eps']),
            initializer_range=0.02,
            # Dropout settings for llama/gpt
            token_dropout_p=0.0,
            attn_dropout_p=0.0,
            resid_dropout_p=0.0,
            ffn_dropout_p=0.0,
            drop_path_rate=0.0,
            # Class conditioning
            class_embedding_dim=config_dict.get('class_embedding_dim', None),
        )

        # Rephrase intermediate_size if provided
        if 'intermediate_size' in config_dict:
            # Calculate ffn_dim_multiplier to achieve desired intermediate_size
            default_hidden = int(2 * gpt_config.dim * 4 / 3)
            gpt_config.ffn_dim_multiplier = config_dict['intermediate_size'] / default_hidden

        # Set up attention position embeddings for multi-scale
        self.class_cond_pos_id = -1
        self.empty_token_pos_id = -2
        attn_position_embedding = AttnPositionEmbedding(
            self.scale_sizes, gpt_config, self.class_cond_pos_id, self.empty_token_pos_id,
            num_mixing_layers=self.num_mixing_layers,
        )

        # Initialize GPT Transformer AR model
        self.llama = GPTTransformerAR(
            gpt_config,
            position_embedding_module=attn_position_embedding,
            class_conditional=self.class_cond
        )
        self.input_embedding_dim = gpt_config.dim

        # Additional embeddings and layers for inputs
        if self.class_cond:
            self.class_embeddings = nn.Embedding(self.num_classes + 1, self.input_embedding_dim)
        num_position_embeddings = sum([x ** 2 for x in self.scale_sizes])
        self.embed_positions = nn.Embedding(num_position_embeddings, self.input_embedding_dim)
        q_dim = self.autoenc.embed_dim
        self.embed_codevecs_mlp = MLP(q_dim, self.input_embedding_dim, max(64, q_dim), 2, norm=False)
        self.src_latents_proj = nn.Linear(self.input_embedding_dim, self.input_embedding_dim)

        # Linear combination layer to merge pos, src, and gt embeddings
        self.combine_inputs = nn.Linear(self.input_embedding_dim * 4, self.input_embedding_dim)

        # Learned standin embedding for first block (when no previous gt available)
        self.empty_token_emb = nn.Parameter(torch.zeros(self.input_embedding_dim))
        self.empty_token_pos = nn.Parameter(torch.zeros(self.input_embedding_dim))

        # output head
        self.head_mlp = None
        if self.use_head_mlp:
            self.head_mlp = MLP(q_dim, gpt_config.dim, max(64, q_dim), 2, norm=False, bias=False)
        self.lm_head = nn.Linear(gpt_config.dim, self.vq_num_tokens, bias=True)
        nn.init.zeros_(self.lm_head.weight)
        nn.init.zeros_(self.lm_head.bias)

    def freeze(self, m):
        for param in m.parameters():
            param.requires_grad = False
        m.eval()
        m.train = lambda mode=True: m  # disable train()
        return m

    def preprocess(self, imgs, scale_sizes=None, cached_pyramid=None):
        """
        Preprocess images to multi-scale quantized codes.

        Args:
            imgs: Either images (B, C, H, W) or tuple (zq_pyr, q_inds) if cached_pyramid=True
            scale_sizes: List of scale sizes to process (default: self.scale_sizes)
            cached_pyramid: If True, imgs contains pre-computed (zq_pyr, q_inds) from cache

        Returns:
            upscaled_latents: Upscaled latents from previous scale (for conditioning)
            latents: Target quantized latents at current scale
            codes: Target quantized code indices at current scale
        """
        device = next(self.autoenc.parameters()).device

        if scale_sizes is None:
            scale_sizes = self.scale_sizes

        if cached_pyramid:
            # imgs is actually cached q_inds
            # the current cache format is a 1-tuple of (q_inds,) (previous versions also had zq_pyr)
            assert len(imgs) == 1, 'cached_pyramid imgs must be tuple of (q_inds,)'
            q_inds_cached = imgs[0]

            # Filter cached pyramid (maybe with more scales) to requested scale_sizes
            q_inds_cached = [q for q in q_inds_cached if q.shape[1] in scale_sizes]
            assert [q.shape[1] for q in q_inds_cached] == scale_sizes

            # Move to device and extract pyramids
            q_inds = [q.to(device) for q in q_inds_cached]
            zq_pyr = [self.autoenc.quantize.embedding(q) for q in q_inds]
            zq_pyr = [z.permute(0, 3, 1, 2) for z in zq_pyr]  # B, C, H, W

        else:
            # Standard path: run VQ encoding
            imgs = imgs.to(device)
            with torch.cuda.amp.autocast(enabled=False):
                pyramid = self.autoenc.quantization_pyramid(imgs)
            zq_pyr = pyramid['zq_pyr']
            q_inds = pyramid['q_inds']

        upscaled_latents = []
        latents = []
        codes = []

        with torch.cuda.amp.autocast(enabled=False):
            # Common processing for both cached and non-cached paths
            for i, size in enumerate(scale_sizes):
                # Find the index of this scale_size in the full scale_sizes list
                scale_idx = self.scale_sizes.index(size)
                qvecs = zq_pyr[scale_idx]
                qinds = q_inds[scale_idx]
                zprev = latents[-1] if len(latents) > 0 else torch.zeros_like(qvecs)

                if self.random_replace and self.training and i > 0:
                    # random replace qvecs (used for inputs as samples)
                    # not qinds (used for training target)
                    z_rep = qvecs if self.random_replace_sampled else zprev
                    z_rep = z_rep.clone().permute(0, 2, 3, 1)
                    replace_mask = torch.rand(z_rep.shape[:-1], device=z_rep.device) < self.random_replace
                    nreplace = replace_mask.sum().item()
                    replace_inds = torch.randint(0, self.autoenc.n_embed, (nreplace,), device=z_rep.device)
                    z_rep[replace_mask] = self.autoenc.quantize.embedding.weight[replace_inds]
                    z_rep = z_rep.permute(0, 3, 1, 2)
                    if self.random_replace_sampled:
                        qvecs = z_rep
                    else:
                        zprev = z_rep

                upscaled_latents.append(self.autoenc.rescale_latents(zprev, size=size))

                assert qvecs.shape[2:] == (size, size)  # B, C, H, W
                assert qinds.shape[1:] == (size, size)  # B, H, W
                latents.append(qvecs)
                codes.append(qinds)  # B, H, W

        return upscaled_latents, latents, codes

    def embed_inputs(self, vecs):
        shape = vecs.shape
        result = self.embed_codevecs_mlp(vecs.reshape(-1, shape[-1])).reshape(shape[:-1] + (-1,))
        return result

    def load_state_dict(self, state_dict, strict=True):
        # check if src_latents_proj exists in state_dict
        if 'src_latents_proj.weight' not in state_dict:
            # init/set to identity
            state_dict['src_latents_proj.weight'] = torch.eye(self.input_embedding_dim)
            state_dict['src_latents_proj.bias'] = torch.zeros(self.input_embedding_dim)
        # check size mismatches in the following keys, and use current init parmams if mismatched
        check_keys = [
            'llama.rotary_emb.position_embeddings.parametrizations.weight.original',
            'embed_positions.weight',
            'class_embeddings.weight',
        ]
        model_params = dict(self.named_parameters())
        for k in check_keys:
            if k in state_dict and k in model_params:
                if state_dict[k].shape != model_params[k].shape:
                    print(f'Warning: skipping loading of {k} due to size mismatch {state_dict[k].shape} vs {model_params[k].shape}')
                    state_dict[k] = model_params[k].data
        super().load_state_dict(state_dict, strict=strict)
        

    def head(self, hidden_states):
        if self.head_mlp is not None:
            vq_embeds = self.autoenc.quantize.embedding.weight.detach()  # vq_num_tokens, code_dim
            #w = w + self.head_mlp(vq_embeds)  # vq_num_tokens, hidden_size
            w = self.head_mlp(vq_embeds)  # vq_num_tokens, hidden_size
            # Apply L2 regularization via gradient modification instead of loss term
            w = L2RegGrad.apply(w, self.head_weights_l2_loss)
            w = w + self.lm_head.weight  # vq_num_tokens, hidden_size
            b = self.lm_head.bias  # vq_num_tokens
            logits = F.linear(hidden_states, w, b)  # B, L, vq_num_tokens
            return logits
        else:
            return self.lm_head(hidden_states)

    def create_4d_causal_mask_from_block_ids(self, block_ids):
        # Create blockwise causal mask from block IDs
        # block_ids: (B, L) tensor where each element is the block ID for that position
        # Returns: (B, 1, L, L) mask with 0.0 for allowed attention, -1e9 for masked
        B, L = block_ids.shape
        # Position i can attend to position j if i is in the same or an earlier block than j
        block_ids_i = block_ids.unsqueeze(2)  # B, L, 1
        block_ids_j = block_ids.unsqueeze(1)  # B, 1, L
        mask = (block_ids_j <= block_ids_i)  # B, L, L
        mask = torch.where(mask, 0.0, -1e9).unsqueeze(1)  # B, 1, L, L
        return mask

    def forward(self, imgs, labels, cached_load=False):
        # imgs: B, C, H, W in [-1, 1] OR tuple (zq_pyr, q_inds) if cached_load=True
        # labels: B,

        batch_size = labels.shape[0]
        scale_sizes = self.scale_sizes

        src_embeds = []
        pos_embeds = []
        gt_embeds = []
        gt_codes = []
        order = []
        num_tokens_per_scale = []
        scale_position_start = 0

        with torch.no_grad():
            upscaled_latents, latents, codes = self.preprocess(
                imgs, scale_sizes=scale_sizes, cached_pyramid=cached_load
            )

        # apply ordering within each scale using the progressive checkerboard scan order
        for scale_i, scale_size in enumerate(scale_sizes):
            z0 = upscaled_latents[scale_i]
            z = latents[scale_i]
            q = codes[scale_i]

            B, C, Hs, Ws = z.shape

            z0 = z0.reshape(B, C, Hs*Ws).permute(0, 2, 1).contiguous()  # B, Hs*Ws, C
            z = z.reshape(B, C, Hs*Ws).permute(0, 2, 1).contiguous()  # B, Hs*Ws, C
            q = q.reshape(B, Hs*Ws)

            # use the progressive checkerboard order
            order_s = torch.argsort(progressive_checkerboard(Hs).reshape(-1).expand(B, -1).to(z.device))

            z0 = torch.gather(z0, 1, order_s.unsqueeze(-1).expand(-1, -1, C))
            z = torch.gather(z, 1, order_s.unsqueeze(-1).expand(-1, -1, C))
            q = torch.gather(q, 1, order_s)

            pos_embeds.append(self.embed_positions(order_s + scale_position_start))
            src_embeds.append(self.src_latents_proj(self.embed_inputs(z0)))

            gt_embeds.append(self.embed_inputs(z))
            gt_codes.append(q)

            order.append(order_s + scale_position_start)
            num_tokens_per_scale.append(Hs*Ws)

            scale_position_start += Hs*Ws


        pos_embeds = torch.cat(pos_embeds, dim=1)  # B, L, C
        src_embeds = torch.cat(src_embeds, dim=1)  # B, L, C
        gt_embeds = torch.cat(gt_embeds, dim=1)  # B, L, C
        gt_codes = torch.cat(gt_codes, dim=1)  # B, L
        order = torch.cat(order, dim=1)  # B, L
        B, L, _ = pos_embeds.shape  # L = total num tokens over all scales
        assert L == scale_position_start
        assert L == pos_embeds.shape[1]
        assert L == src_embeds.shape[1]
        assert L == gt_embeds.shape[1]
        assert L == gt_codes.shape[1]
        assert L == order.shape[1]

        sequence = []
        sequence_order = []
        shifted_sequence_order = []
        block_ids = []  # track which block each position belongs to

        class_cond_seq_len = 0
        if self.class_cond:
            uncond = (torch.rand(B) < self.uncond_training_prob).to(z.device)
            labels = torch.where(uncond, self.num_classes, labels)
            class_embeds = self.class_embeddings(labels).unsqueeze(1)  # B, 1, C
            sequence.append(class_embeds)
            sequence_order.append(self.class_cond_pos_id + torch.zeros((B, 1), dtype=torch.long, device=z.device))
            shifted_sequence_order.append(self.class_cond_pos_id + torch.zeros((B, 1), dtype=torch.long, device=z.device))
            block_ids.append(torch.zeros((B, 1), dtype=torch.long, device=z.device))
            class_cond_seq_len = 1

        scale_position_start = 0
        current_block_id = 1  # start from 1 (0 is class conditioning)
        for scale_i, scale_size in enumerate(scale_sizes):
            B, _, Hs, Ws = latents[scale_i].shape
            C = self.input_embedding_dim
            n = Hs*Ws

            if self.training and self.random_block_size:
                block_size = random_block_size(n)
            else:
                block_size = max(1, n // self.num_blocks_per_scale)

            pos_embeds_s = pos_embeds[:, scale_position_start:scale_position_start+n, :]
            src_embeds_s = src_embeds[:, scale_position_start:scale_position_start+n, :]
            gt_embeds_s = gt_embeds[:, scale_position_start:scale_position_start+n, :]
            order_s = order[:, scale_position_start:scale_position_start+n]  # includes scale_position_start offset

            # Create gt embeddings from previous block 
            # At position i, use gt from position i - block_size
            shifted_pos_embeds = torch.empty_like(pos_embeds_s)
            shifted_pos_embeds[:, :block_size, :] = self.empty_token_pos.expand(B, block_size, -1)
            shifted_pos_embeds[:, block_size:, :] = pos_embeds_s[:, :-block_size, :]
            shifted_gt_embeds = torch.empty_like(gt_embeds_s)
            shifted_gt_embeds[:, :block_size, :] = self.empty_token_emb.expand(B, block_size, -1)
            shifted_gt_embeds[:, block_size:, :] = gt_embeds_s[:, :-block_size, :]
            shifted_order_s = torch.empty_like(order_s)
            shifted_order_s[:, :block_size] = self.empty_token_pos_id
            shifted_order_s[:, block_size:] = order_s[:, :-block_size]

            # Create block IDs based on position in sequence: position // block_size
            # This groups consecutive positions in the progressive scan order into blocks
            block_ids_s = torch.arange(n, device=z.device) // block_size  # n
            block_ids_s = block_ids_s.unsqueeze(0).expand(B, -1) + current_block_id  # B, n

            # Update current_block_id for next scale (ceiling division to account for partial last block)
            num_blocks_s = (n + block_size - 1) // block_size

            # sometimes fold remainder into same block at end
            if n % block_size != 0 and self.training and torch.rand(1).item() < 0.5:
                rem = n % block_size
                # remainder positions share block ID of previous block
                # and shifted "sampled" inputs are set to empty (no info from gt)
                _last_block_id = n // block_size - 1
                block_ids_s[:, -rem:] = _last_block_id + current_block_id
                shifted_pos_embeds[:, -rem:, :] = self.empty_token_pos.expand(B, rem, -1)
                shifted_gt_embeds[:, -rem:, :] = self.empty_token_emb.expand(B, rem, -1)
                shifted_order_s[:, -rem:] = self.empty_token_pos_id
                num_blocks_s -= 1  # last partial block merged into previous

            # Combine pos, src, and shifted gt targets using linear layer
            combined = torch.cat([pos_embeds_s, src_embeds_s, shifted_pos_embeds, shifted_gt_embeds], dim=-1)  # B, n, 4*C
            sequence_s = self.combine_inputs(combined)  # B, n, C

            # Append scale's sequence to overall sequence
            sequence.append(sequence_s)
            sequence_order.append(order_s)
            shifted_sequence_order.append(shifted_order_s)
            block_ids.append(block_ids_s)

            # Increments for next scale
            current_block_id += num_blocks_s
            scale_position_start += Hs*Ws

        sequence = torch.cat(sequence, dim=1)  # B, L, C
        sequence_order = torch.cat(sequence_order, dim=1)  # B, L
        shifted_sequence_order = torch.cat(shifted_sequence_order, dim=1)  # B, L
        block_ids = torch.cat(block_ids, dim=1)  # B, L


        # call the transformer model
        outputs = self.llama(
            inputs_embeds=sequence,
            label_embeds=class_embeds if self.class_cond else None,
            sequence_order=sequence_order,
            past_sequence_order=shifted_sequence_order,
            attention_mask=self.create_4d_causal_mask_from_block_ids(block_ids),
        )

        hidden_states = outputs.last_hidden_state  # B, L, C

        # remove class conditioning from sequence for loss calculation
        if class_cond_seq_len > 0:
            hidden_states = hidden_states[:, class_cond_seq_len:, :]
            sequence = sequence[:, class_cond_seq_len:, :]
            sequence_order = sequence_order[:, class_cond_seq_len:]

        # check sequence_order has no empty_token_pos_id 
        # (this id used to also be used for between-scale buffer embeds, but these are now removed)
        assert torch.all(sequence_order != self.empty_token_pos_id), 'sequence_order has unexpected empty_token_pos_id values'

        assert hidden_states.shape[1] == L
        assert sequence.shape[1] == L
        assert sequence_order.shape[1] == L

        token_pred_hiddens = hidden_states  # B, L, C
        logits = self.head(token_pred_hiddens)  # B, L, vocab_size

        # Compute loss
        V = logits.shape[-1]
        assert V == self.vq_num_tokens
        # flat loss
        loss = nn.functional.cross_entropy(
            logits.reshape(-1, V), gt_codes.reshape(-1), reduction='mean')

        # stats
        stats = {}
        stats['loss_total'] = loss.item()
        start = 0
        for scale_i, n in enumerate(num_tokens_per_scale):
            logits_s = logits[:, start:start+n, :]  # B, n, vocab_size
            gt_codes_s = gt_codes[:, start:start+n]  # B, n

            # Compute loss stat
            loss_s = nn.functional.cross_entropy(
                logits_s.reshape(B*n, -1), gt_codes_s.reshape(B*n), reduction='mean')

            # Compute top-1 accuracy for monitoring
            pred_codes = logits_s.argmax(dim=-1)  # B, n
            top1_correct = (pred_codes == gt_codes_s).float().mean()

            # Store stats with scale identifier
            scale_size = self.scale_sizes[scale_i]
            stats[f'loss_scale_size{scale_size}'] = loss_s.item()
            stats[f'top1_acc_scale_size{scale_size}'] = top1_correct.item()

            start += n

        return loss, stats

    @torch.no_grad()
    @torch.cuda.amp.autocast()
    def sample(self, batch_size=1, labels=None, temperature=1.0, cfg=0.0, cfg_start_step=0, last_scale_only=False):
        device = next(self.parameters()).device

        token_positions = []
        token_sample_ts = []
        token_values = []
        token_probs = []
        token_positions_entropy = []

        nsamples = batch_size

        cfg_enable = (cfg > 0)

        conditional_sampling = (labels is not None)
        if conditional_sampling:
            assert self.class_cond, 'must be class conditional if providing labels'
            assert labels.shape[0] == batch_size and labels.ndim == 1
            if cfg_enable:
                unc = torch.as_tensor([self.num_classes] * len(labels), device=labels.device, dtype=labels.dtype)
                labels = torch.cat((labels, unc), dim=0)
                batch_size *= 2
            
        assert batch_size in (nsamples, 2*nsamples)  # bsize for uncond or cond

        B = batch_size
        C = self.input_embedding_dim

        sequence = torch.zeros((B, 0, C), device=device)  # B, L, C (with L=0)
        sequence_order = torch.zeros((B, 0), dtype=torch.long, device=device)  # B, L
        sampled_sequence_order = torch.zeros((B, 0), dtype=torch.long, device=device)  # B, L

        past_key_values = DynamicCache()

        t = 0

        scale_position_start = 0

        latents_by_scale = []

        if conditional_sampling:
            class_embeds = self.class_embeddings(labels).unsqueeze(1)  # B, 1, C
            sequence = torch.cat([sequence, class_embeds], dim=1)
            sequence_order = torch.cat(
                [sequence_order, self.class_cond_pos_id + torch.zeros((B, 1), dtype=torch.long, device=device)],
                dim=1)
            sampled_sequence_order = torch.cat(
                [sampled_sequence_order, self.class_cond_pos_id + torch.zeros((B, 1), dtype=torch.long, device=device)],
                dim=1)

        # Track block IDs for blockwise causal masking
        block_ids = torch.zeros((B, 0), dtype=torch.long, device=device)
        if conditional_sampling:
            block_ids = torch.cat([block_ids, torch.zeros((B, 1), dtype=torch.long, device=device)], dim=1)
        current_block_id = 1

        for scale_i, scale_size in enumerate(self.scale_sizes):
            Hs = Ws = scale_size

            # get the positions in progressive checkerboard order
            positions = torch.argsort(progressive_checkerboard(Hs).reshape(-1).to(device)) + scale_position_start

            if len(latents_by_scale) == 0:
                z0 = torch.zeros((B, self.autoenc.embed_dim, Hs, Ws), device=device)
            else:
                z0 = latents_by_scale[-1]
                z0 = self.autoenc.rescale_latents(z0, size=(Hs, Ws))

            z0 = z0.reshape(B, self.autoenc.embed_dim, Hs*Ws).permute(0, 2, 1).contiguous()

            z_curr = torch.zeros_like(z0)

            block_size = max(1, len(positions) // self.num_blocks_per_scale)

            # First block: use learned standins for prev block samples
            prev_block_sampled_embeds = self.empty_token_emb.unsqueeze(0).unsqueeze(0).expand(B, block_size, -1)
            prev_block_pos_embeds = self.empty_token_pos.unsqueeze(0).unsqueeze(0).expand(B, block_size, -1)
            prev_block_pos_ids = self.empty_token_pos_id + torch.zeros((block_size,), dtype=torch.long, device=device)

            while len(positions) > 0:

                # update cache up to current sequence element
                cache_len = past_key_values.get_seq_length()
                if sequence.shape[1] > 0 and cache_len < sequence.shape[1]:
                    # Create mask for cache update
                    cache_mask = self.create_4d_causal_mask_from_block_ids(block_ids)[:, :, cache_len:, :]  # new q x all k
                    outputs = self.llama(
                        inputs_embeds=sequence[:, cache_len:, :],
                        sequence_order=sequence_order[:, cache_len:],
                        past_sequence_order=sampled_sequence_order[:, cache_len:],
                        label_embeds=class_embeds if conditional_sampling else None,
                        past_key_values=past_key_values,
                        attention_mask=cache_mask,
                        use_cache=True,
                    )
                    assert past_key_values is outputs.past_key_values
                    assert past_key_values.get_seq_length() == sequence.shape[1]

                pos = positions[:block_size]
                pos_s = pos - scale_position_start  # relative to current scale

                n_pos = len(pos)

                # Get embeddings for current positions
                pos_embeds = self.embed_positions(pos.unsqueeze(0)).expand(B, -1, -1)
                src_embeds = self.src_latents_proj(self.embed_inputs(z0[:, pos_s, :].reshape(B*n_pos, -1))).reshape(B, n_pos, -1)

                # Combine pos, src, and prev_block_gt using linear layer
                # Slice prev_block_sampled_embeds to match n_pos (handles remainder at end of scale)
                # TODO handle folding remainder at end into last block, not adding an extra one
                assert n_pos == block_size or (n_pos == len(positions) and n_pos < block_size)
                combined = torch.cat([pos_embeds, src_embeds, prev_block_pos_embeds[:, :n_pos, :], prev_block_sampled_embeds[:, :n_pos, :]], dim=-1)  # B, n_pos, 4*C
                combined_input = self.combine_inputs(combined)  # B, n_pos, C

                # Run transformer on selected positions
                outputs = self.llama(
                    inputs_embeds=combined_input,
                    sequence_order=pos.unsqueeze(0).expand(B, -1),
                    past_sequence_order=prev_block_pos_ids[:n_pos].unsqueeze(0).expand(B, -1),
                    label_embeds=class_embeds if conditional_sampling else None,
                    past_key_values=past_key_values,
                    attention_mask=torch.zeros((B, 1, n_pos, n_pos + past_key_values.get_seq_length()), device=device),  # new q x all k
                    use_cache=True,
                )
                hiddens = outputs.last_hidden_state  # B, n_pos, C

                assert outputs.past_key_values is past_key_values

                logits = self.head(hiddens)  # B, n_pos, vocab_size

                if conditional_sampling and cfg_enable:
                    cond_logits = logits[:nsamples, :, :]  # B, n_pos, vocab_size
                    uncond_logits = logits[nsamples:, :, :]  # B, n_pos, vocab_size
                    if t >= cfg_start_step:
                        logits = cond_logits + cfg * (cond_logits - uncond_logits)
                    else:
                        logits = cond_logits

                # Sample tokens from logits
                if temperature > 0:
                    probs = F.softmax(logits / temperature, dim=-1)  # B, n_pos, vocab_size
                else:
                    probs = torch.zeros_like(logits)
                    probs.scatter_(2, logits.argmax(dim=-1, keepdim=True), 1.0)

                # sample in joint space of positions and tokens
                ent = -(probs * torch.log(probs + 1e-8)).sum(dim=-1)  # B, n_pos
                p = probs.reshape(-1, probs.shape[-1])  # B*n_pos, vocab_size
                if temperature > 0:
                    tokens = torch.multinomial(p, num_samples=1)[:, 0]
                else:
                    tokens = torch.argmax(p, dim=-1)

                tokens = tokens.reshape(-1, n_pos)

                # store sampled tokens and stats
                token_values.append(tokens)
                token_positions.append(pos)
                token_sample_ts.append(t + torch.zeros_like(pos))
                token_probs.append(p[torch.arange(len(p)), tokens.reshape(-1)].reshape(-1, n_pos))
                token_positions_entropy.append(ent)

                if conditional_sampling and cfg_enable:
                    tokens = torch.cat((tokens, tokens), dim=0)
                    assert tokens.shape == (B, n_pos)

                # calculate input embeddings for selected pos/tokens
                q = self.autoenc.quantize.embedding(tokens).reshape(B*n_pos, -1)  # B*n_pos, code_dim
                z_i = q  # B*n_pos, code_dim

                # update current latents
                z_curr[:, pos_s, :] = z_i.reshape(B, n_pos, -1)

                # Save sampled embeddings for next block's prev_block_gt conditioning
                sampled_embeds = self.embed_inputs(z_i).reshape(B, n_pos, C)
                prev_block_sampled_embeds = sampled_embeds
                prev_block_pos_embeds = pos_embeds
                prev_block_pos_ids = pos

                # remove selected positions remaining
                positions = positions[n_pos:]

                # update sequence and sequence_order
                sequence = torch.cat((sequence, combined_input), dim=1)

                # Create sequence_order
                sequence_order = torch.cat([sequence_order, pos.expand(B, -1)], dim=1)
                sampled_sequence_order = torch.cat([sampled_sequence_order, prev_block_pos_ids.unsqueeze(0).expand(B, -1)], dim=1)

                # Assign block ID to this block of positions
                block_id_tensor = torch.full((B, n_pos), current_block_id, dtype=torch.long, device=device)
                block_ids = torch.cat([block_ids, block_id_tensor], dim=1)
                current_block_id += 1

                # advance in token selection loop
                t += 1

            # advance in scale loop

            # set latents for this scale
            latents_by_scale.append(z_curr.reshape(B, Hs, Ws, -1).permute(0, 3, 1, 2).contiguous())  # B, C, Hs, Ws

            scale_position_start += Hs*Ws

        if last_scale_only:
            # return only last scale decoded img without extra detailed stats
            z = latents_by_scale[-1]
            with torch.cuda.amp.autocast(enabled=False):
                img = self.autoenc.decode(z)
            return img

        token_values = torch.cat(token_values, dim=1)  # B, L  over all scales
        token_positions = torch.cat(token_positions, dim=0).unsqueeze(0).expand(B, -1)  # B, L
        token_sample_ts = torch.cat(token_sample_ts, dim=0).unsqueeze(0).expand(B, -1)
        token_probs = torch.cat(token_probs, dim=1)  # B, L
        token_positions_entropy = torch.cat(token_positions_entropy, dim=1)  # B, L

        # remove unconditional batch elements if present
        if conditional_sampling and cfg_enable:
            token_values = token_values[:nsamples, :]
            token_positions = token_positions[:nsamples, :]
            token_sample_ts = token_sample_ts[:nsamples, :]
            token_probs = token_probs[:nsamples, :]
            token_positions_entropy = token_positions_entropy[:nsamples, :]
            latents_by_scale = [z[:nsamples, :] for z in latents_by_scale]
            # set batch_size back
            B = batch_size = nsamples

        # reorder to orig raster order
        order = torch.argsort(token_positions, dim=1).to(device)  # B, L
        #assert torch.all(token_positions[order] == torch.arange(H*W, device=device))
        token_values = torch.gather(token_values, 1, order)  # B, L
        token_positions = torch.gather(token_positions, 1, order)  # B, L
        token_sample_ts = torch.gather(token_sample_ts, 1, order)
        token_probs = torch.gather(token_probs, 1, order)  # B, L
        token_positions_entropy = torch.gather(token_positions_entropy, 1, order)  # B, L

        results = []
        start = 0
        for scale_i, scale_size in enumerate(self.scale_sizes):
            H = W = scale_size
            end = start + H*W

            # decode image
            z = latents_by_scale[scale_i]
            with torch.cuda.amp.autocast(enabled=False):
                img = self.autoenc.decode(z)

            # map of selection order
            selection_order = order[:, start:end].view(B, H, W)
            selection_ts = token_sample_ts[:, start:end].view(B, H, W)

            # token probs map
            value_prob_map = token_probs[:, start:end].view(B, H, W)

            selection_entropy = token_positions_entropy[:, start:end].view(B, H, W)

            # TODO make this a dict return
            results.append((img, selection_ts, value_prob_map, selection_entropy))

            start = end

        return results


def random_block_size(n):
    mid = int(np.sqrt(n))
    while True:
        x = np.random.randint(1, mid + 1)
        if np.random.rand() < 0.5:
            return x
        elif x != mid or (n % mid != 0):
            return n // x
        else:
            pass # reject/retry not to double-count mid


class AttnPositionEmbedding(nn.Module):
    def __init__(self, scale_sizes, config, cls_cond_pos_id, empty_token_pos_id, num_mixing_layers=None):
        super().__init__()
        # Support both LlamaConfig and GPTModelArgs
        if hasattr(config, 'hidden_size'):
            # LlamaConfig
            hidden_size = config.hidden_size
            num_heads = config.num_attention_heads
            num_layers = config.num_hidden_layers
        else:
            # GPTModelArgs
            hidden_size = config.dim
            num_heads = config.n_head
            num_layers = config.n_layer

        # create inits
        inits = []
        scale_inds = []
        self.scale_sizes = scale_sizes
        # 2D spatial rope init
        rope_theta = self.scale_sizes[-1] ** 2
        total_freq_dim = hidden_size // num_heads
        freq_dim = total_freq_dim - total_freq_dim // 8  # leave some dims for scale embeddings
        freq_dim = freq_dim + (freq_dim % 2)  # make even
        for scale_i, scale in enumerate(self.scale_sizes):
            idx = scale
            img_idx = torch.arange(idx**2, dtype=torch.int32).reshape(idx**2, 1)
            frequencies_x = img_idx % idx  # get the coordinates of the 2d matrix along x
            frequencies_y = img_idx // idx  # get the coordinates of the 2d matrix along y
            rope_freq = 1.0 / (rope_theta ** (torch.arange(0, freq_dim, 2)[: (freq_dim // 2)].float() / freq_dim))
            stride = self.scale_sizes[-1] / scale
            freqs_x = ((stride * frequencies_x + 1)[..., None] * rope_freq[None, None, :]).repeat_interleave(2, dim=-1)
            freqs_y = ((stride * frequencies_y + 1)[..., None] * rope_freq[None, None, :]).repeat_interleave(2, dim=-1)
            freqs = torch.cat([freqs_x, freqs_y], dim=-1).float().contiguous()[..., ::2]
            freq_cis = torch.view_as_complex(torch.stack([torch.cos(freqs), torch.sin(freqs)], dim=-1))
            inits.append(freq_cis.squeeze(1))  # (seq_len, dim)
            scale_inds.append(scale_i + torch.zeros(idx**2, dtype=torch.int32))

        # inits for scale embeddings
        # append scale init into rope embeddings joint space
        num_scales = len(self.scale_sizes)
        code_len = total_freq_dim - freq_dim  # num dims left over for scale embeddings init
        freqs = math.pi/2 / (10.0 ** (torch.arange(code_len) / code_len))
        for s in range(num_scales):
            scale_rope = torch.exp(1.0j * freqs * (s + 1))
            inits[s] = torch.cat([inits[s], scale_rope.unsqueeze(0).expand(inits[s].shape[0], -1)], dim=1)
    
        # init embeddings for class cond and empty token
        self.class_cond_pos_id = cls_cond_pos_id
        self.empty_token_pos_id = empty_token_pos_id
        self.class_cond_embedding_index = sum(map(len, inits))
        self.empty_embedding_index = self.class_cond_embedding_index + 1
        inits.append(torch.ones((1, total_freq_dim), dtype=torch.cfloat))  # class cond
        inits.append(torch.ones((1, total_freq_dim), dtype=torch.cfloat))  # empty token

        inits = torch.cat(inits, dim=0)
        self.scale_inds = torch.cat(scale_inds, dim=0)

        # init embeddings for positions (space + scale)
        self.position_embeddings = nn.Embedding(inits.shape[0], total_freq_dim*2)
        self.position_embeddings.weight.data[:] = torch.view_as_real(inits).view(*self.position_embeddings.weight.data.shape)

        # param constraints for unit norm and init flags
        for emb in (self.position_embeddings,):
            torch.nn.utils.parametrize.register_parametrization(
                emb, "weight", UnitNormComplex()
            )
            emb._is_hf_initialized = True  # flag as init done
        
        # mixing coeffs for each transformer layer and attention head
        if num_mixing_layers in [None, 'all']:
            num_mixing_layers = num_layers
        self.num_mixing_layers = num_mixing_layers
        self.num_heads = num_heads
        self.mixing_coeffs = nn.Parameter(torch.zeros(num_mixing_layers, num_heads, dtype=torch.float32))

    def forward(self, current_position_ids, past_position_ids):
        '''
        current_position_ids: (B, S) long
            position ids for the locations currently being predicted/sampled at
            each sequence position
        past_position_ids: (B, L) long
            position ids for sampled locations from the last sampling block,
            included in this sequence position
        '''
        ropes = []
        for position_ids in (current_position_ids, past_position_ids):
            shp = position_ids.shape  # (B, S)
            ids = position_ids.clone().view(-1)
            ids[ids == self.class_cond_pos_id] = self.class_cond_embedding_index
            ids[ids == self.empty_token_pos_id] = self.empty_embedding_index
            f = torch.view_as_complex(self.position_embeddings(ids).view(*shp, -1, 2))  # (B, S, head_dim) complex
            ropes.append(f)

        # ropes[0], ropes[1] shape: (B, S, head_dim)
        # Expand to (B, num_heads, S, head_dim) for per-head mixing
        ropes = [r.unsqueeze(1) for r in ropes]  # (B, 1, S, head_dim)
        return ropes

    def apply_mixing(self, ropes, layer_idx):
        if layer_idx >= self.num_mixing_layers:
            # no mixing for higher layers
            return ropes[0]

        # Apply rope mixing for this layer
        alpha = torch.sigmoid(self.mixing_coeffs[layer_idx])  # (num_heads,)
        alpha = alpha.view(1, -1, 1, 1)
        f = alpha * ropes[0] + (1 - alpha) * ropes[1]

        return f


class UnitNormComplex(nn.Module):
    def forward(self, x):
        shp = x.shape
        x = x.reshape(-1, x.shape[-1]//2, 2)
        x = x / x.norm(dim=-1, p=2, keepdim=True)
        return x.reshape(*shp)


class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, n_layers, act=nn.SiLU, norm=False, bias=True):
        super().__init__()
        layers = []
        for i in range(n_layers):
            layers.append(nn.Linear(in_dim if i == 0 else hidden_dim, hidden_dim))
            layers.append(act())
            if norm:
                layers.append(nn.LayerNorm(hidden_dim))
        layers.append(nn.Linear(hidden_dim if n_layers > 0 else in_dim, out_dim, bias=bias))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)



class L2RegGrad(torch.autograd.Function):
    """
    Custom autograd function that applies L2 regularization via gradient modification.
    Instead of adding a loss term (lambda/2 * ||w||^2), this modifies the gradient
    directly: grad_out = grad_in + strength * activation
    """
    @staticmethod
    def forward(ctx, activation, strength):
        ctx.save_for_backward(activation)
        ctx.strength = strength
        return activation

    @staticmethod
    def backward(ctx, grad_output):
        activation, = ctx.saved_tensors
        strength = ctx.strength
        # Apply L2 regularization: grad = grad + strength * activation
        # Note: using + (not -) because the grad should correspond
        # to a loss term of (strength/2) * ||w||^2, more pos values increase loss
        grad_input = grad_output + strength * activation
        return grad_input, None  # None for strength gradient

