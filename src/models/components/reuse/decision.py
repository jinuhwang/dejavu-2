import torch
from torch import nn

class ReuseThreshold(nn.Module):
    def __init__(
            self, 
            threshold,
        ):
        super().__init__()
        self.threshold = threshold

    def forward(self, importance, similarity, compressed_map, **kwargs):
        # [B, current_token, cached_token] => [B, current_token]
        most_similar_score, most_similar_idx = similarity.max(dim=-1)

        reuse_decision = most_similar_score - self.threshold

        reuse_decision = reuse_decision.unsqueeze(-1)

        return reuse_decision, most_similar_idx, None

class ReuseMLP(nn.Module):
    def __init__(
            self,
            inner_dim=64,
            layer_pattern=None,
            out_dim=1,
            use_compressed_info=False,
            disable_bias=False,
            use_norm=False,
            initialize=None,
            dropout=0.25,
            add_residual=False,
            use_reference_type=False,
            use_prev_reuse_map=False,
            # Local context feature options (A/B)
            local_ctx_mode: str = 'none',   # 'none' | 'conv' | 'neighbors'
            local_ctx_rank: int = 32,
            local_ctx_kernel: int = 3,
            local_ctx_dropout: float = 0.0,
            verbose=False,
        ):
        super().__init__()

        self.use_compressed_info = use_compressed_info
        if use_compressed_info:
            input_dim = 3
        else:
            input_dim = 2

        self.use_reference_type = use_reference_type
        if use_reference_type:
            input_dim += 3

        self.use_norm = use_norm
        if self.use_norm:
            input_dim += 1

        self.use_prev_reuse_map = use_prev_reuse_map
        if self.use_prev_reuse_map:
            input_dim += 1

        # Local context settings
        self.local_ctx_mode = (local_ctx_mode or 'none').lower()
        self.local_ctx_rank = int(local_ctx_rank)
        self.local_ctx_kernel = int(local_ctx_kernel)
        self.local_ctx_dropout = float(local_ctx_dropout)
        if self.local_ctx_mode not in ('none', 'conv', 'neighbors'):
            raise ValueError(f"Invalid local_ctx_mode={self.local_ctx_mode}")
        if self.local_ctx_mode != 'none':
            # We will concatenate 3 scalars per token: [cos_local/mean, var, edge/max]
            input_dim += 3
            # Lazily initialized modules/buffers for building local context
            self.ctx_proj = None
            self.register_buffer('shape_hw', torch.zeros(2, dtype=torch.int32), persistent=False)
            self.register_buffer('nbr_idx', torch.empty(0, dtype=torch.long), persistent=False)

        self.blocks = nn.ModuleList()
        self.blocks_add_residual = []
        dims = []
        last_dim = input_dim
        linear_count = layer_pattern.count('l')
        for i in range(linear_count):
            if i == linear_count - 1:
                output_dim = out_dim
            else:
                output_dim = inner_dim
            dims.append((last_dim, output_dim))
            last_dim = output_dim
        if verbose:
            print(f"Layer dims: {dims}")

        last_dim = input_dim
        block = []
        add_residual = False
        for i in range(len(layer_pattern)):
            if layer_pattern[i] == 'l':
                if block is not None:
                    block = nn.Sequential(*block)
                    self.blocks.append(block)
                    self.blocks_add_residual.append(add_residual)

                input_dim, output_dim = dims.pop(0)

                # Create new block
                block = []
                add_residual = add_residual and input_dim == output_dim

                block.append(nn.Linear(input_dim, output_dim, bias=not disable_bias))
                last_dim = output_dim
            elif layer_pattern[i] == 'r':
                block.append(nn.ReLU())
            elif layer_pattern[i] == 'b':
                block.append(nn.BatchNorm1d(last_dim))
            elif layer_pattern[i] == 'd':
                block.append(nn.Dropout(dropout))
            elif layer_pattern[i] == 'L':
                block.append(nn.LayerNorm(last_dim))
            else:
                raise ValueError(f"Invalid layer pattern {layer_pattern[i]}")

        # Add last block
        block = nn.Sequential(*block)
        self.blocks.append(block)
        self.blocks_add_residual.append(add_residual)
        
        if initialize is not None:
            if initialize == "adafuse":
                def initialize_adafuse(submodule):
                    if isinstance(submodule, torch.nn.Linear):
                        torch.nn.init.normal_(submodule.weight, 0, 0.001)
                        torch.nn.init.constant_(submodule.bias, 0)
                    return
                self.blocks.apply(initialize_adafuse)
            else:
                raise ValueError(f"Invalid initialize method {initialize}")
        

    def forward(self, importance, similarity, compressed_map, ref_type=None, prev_reuse_map=None, **kwargs):
        if self.use_norm:
            similarity, norm = similarity

        B, N, N_ = similarity.shape
        most_similar_score, most_similar_idx = similarity.max(dim=-1)

        if self.use_compressed_info:
            mlp_input = torch.cat(
                (
                    importance.view(B, N, 1),
                    most_similar_score.view(B, N, 1),
                    compressed_map.view(B, N, 1)
                ),
                dim=-1
            )
        else:
            mlp_input = torch.cat(
                (
                    importance.view(B, N, 1),
                    most_similar_score.view(B, N, 1)
                ),
                dim=-1
            )

        if ref_type is not None:
            assert self.use_reference_type, "Reference type is not enabled"
            mlp_input = torch.cat(
                (
                    mlp_input,
                    ref_type.unsqueeze(1).expand(-1, N, -1)
                ),
                dim=-1
            )

        if self.use_prev_reuse_map:
            mlp_input = torch.cat(
                (
                    mlp_input,
                    prev_reuse_map.unsqueeze(-1) # [B, N + 1] -> [B, N, 1]
                ),
                dim=-1
            )

        if self.use_norm:
            mlp_input = torch.cat((mlp_input, norm), dim=-1)

        # Optional: append local context features computed upstream
        local_ctx = kwargs.get('local_ctx', None)
        if local_ctx is not None:
            # Expect shape (B, N, C_ctx)
            mlp_input = torch.cat((mlp_input, local_ctx), dim=-1)

        # [B, 2, N-1]
        return self.forward_inner(mlp_input), most_similar_idx, None

    def forward_inner(self, mlp_input):
        B, N, dim = mlp_input.shape
        x = mlp_input.view(B*N, dim)

        for block, add_residual in zip(self.blocks, self.blocks_add_residual):
            if add_residual:
                residual = x
            x = block(x)
            if add_residual:
                x = x + residual

        reuse_decision = x.view(B, N, -1)
        return reuse_decision 

    # --- Local context builders (used by ReuseModule) ---
    @torch.no_grad()
    def _infer_hw(self, P: int) -> tuple[int, int]:
        # Try square grid; fallback to 1xP
        h = int(round(P ** 0.5))
        if h * h == P:
            return h, h
        return 1, P

    def _ensure_proj(self, D: int, device):
        if self.ctx_proj is None:
            self.ctx_proj = nn.Linear(D, self.local_ctx_rank, bias=False).to(device)

    def _ensure_neighbors(self, H: int, W: int, device):
        if self.nbr_idx.numel() == 0 or int(self.shape_hw[0]) != H or int(self.shape_hw[1]) != W:
            k = self.local_ctx_kernel
            assert k % 2 == 1
            r = k // 2
            grid = torch.arange(H * W, device=device).view(H, W)
            idxs = []
            ys = torch.arange(H, device=device)
            xs = torch.arange(W, device=device)
            for dy in range(-r, r + 1):
                for dx in range(-r, r + 1):
                    y = (ys[:, None] + dy).clamp(0, H - 1)
                    x = (xs[None, :] + dx).clamp(0, W - 1)
                    idxs.append(grid[y, x].reshape(-1))
            self.nbr_idx = torch.stack(idxs, dim=1)  # (P, k*k)
            self.shape_hw[:] = torch.tensor([H, W], dtype=torch.int32, device=device)

    def build_local_ctx(self, tokens_no_cls: torch.Tensor) -> torch.Tensor:
        """Compute local context scalars per token.

        tokens_no_cls: (B, P, D)
        Returns: (B, P, 3)
        """
        if self.local_ctx_mode == 'none':
            return None
        B, P, D = tokens_no_cls.shape
        device = tokens_no_cls.device
        H, W = int(self.shape_hw[0].item()), int(self.shape_hw[1].item())
        if H * W != P:
            h, w = self._infer_hw(P)
            H, W = h, w
            self.shape_hw[:] = torch.tensor([H, W], dtype=torch.int32, device=device)

        self._ensure_proj(D, device)
        x = self.ctx_proj(tokens_no_cls)  # (B, P, r)
        r = x.shape[-1]
        xg = x.transpose(1, 2).reshape(B, r, H, W)

        if self.local_ctx_mode == 'conv':
            k = self.local_ctx_kernel
            mean = torch.nn.functional.avg_pool2d(xg, k, stride=1, padding=k // 2, count_include_pad=False)
            mean2 = torch.nn.functional.avg_pool2d(xg * xg, k, stride=1, padding=k // 2, count_include_pad=False)
            var = (mean2 - mean * mean).mean(dim=1, keepdim=False)  # (B, H, W)

            # Sobel edges
            sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], device=device, dtype=xg.dtype).view(1, 1, 3, 3)
            sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], device=device, dtype=xg.dtype).view(1, 1, 3, 3)
            gx = torch.nn.functional.conv2d(xg, sobel_x.expand(r, 1, 3, 3), padding=1, groups=r)
            gy = torch.nn.functional.conv2d(xg, sobel_y.expand(r, 1, 3, 3), padding=1, groups=r)
            edge = (gx * gx + gy * gy).mean(dim=1)  # (B, H, W)

            def l2n(z, eps=1e-6):
                return z / (z.norm(dim=1, keepdim=True) + eps)
            cos_local = (l2n(xg) * l2n(mean)).sum(dim=1)  # (B, H, W)

            var_f = var.view(B, P, 1)
            edge_f = edge.view(B, P, 1)
            cos_f = cos_local.view(B, P, 1)
            return torch.cat([cos_f, var_f, edge_f], dim=-1)

        elif self.local_ctx_mode == 'neighbors':
            self._ensure_neighbors(H, W, device)
            K2 = self.nbr_idx.shape[1]
            flat = self.nbr_idx.reshape(-1)
            nbrs = x[:, flat, :].view(B, P, K2, r)
            center = x.unsqueeze(2)  # (B, P, 1, r)
            sim = torch.nn.functional.cosine_similarity(center, nbrs, dim=-1)  # (B, P, K2)
            feat_mean = sim.mean(-1, keepdim=True)
            feat_max = sim.max(-1, keepdim=True).values
            feat_var = sim.var(-1, unbiased=False, keepdim=True)
            return torch.cat([feat_mean, feat_var, feat_max], dim=-1)
        else:
            return None

class TopKDecision(nn.Module):
    def __init__(self, k, use_norm=False):
        super().__init__()
        self.k = k
        self.use_norm = use_norm

    def forward(self, importance, similarity, compressed_map, **kwargs):
        if self.use_norm:
            similarity, norm = similarity

        most_similar_score, most_similar_idx = similarity.max(dim=-1)

        topk = torch.topk(most_similar_score, self.k, dim=1)

        B, N, _ = similarity.shape

        reuse_decision = torch.ones_like(most_similar_score)
        reuse_decision = torch.scatter(reuse_decision, 1, topk.indices, 0)
        reuse_decision = reuse_decision.unsqueeze(-1)

        return reuse_decision, most_similar_idx, None

class TldrDecision(nn.Module):
    def __init__(self, k, use_norm=False):
        super().__init__()
        self.k = k
        self.use_norm = use_norm

    def forward(self, importance, similarity, compressed_map, **kwargs):

        similarity = (similarity + 1) / 2 # Move to [0, 1]

        most_similar_score, most_similar_idx = similarity.max(dim=-1)
        edge_idx = most_similar_idx.argsort(dim=-1, descending=True)
        unm_idx = edge_idx[..., self.k:, :] # Recomputed Tokens
        src_idx = edge_idx[..., :self.k, :] # Reused Tokens

        src_so = importance.gather(dim=-1, index=src_idx)

        # Wait, the importance doesn't seem to participate in the decision making?

        return most_similar_score, most_similar_idx, None
