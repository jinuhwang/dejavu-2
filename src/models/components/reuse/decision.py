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