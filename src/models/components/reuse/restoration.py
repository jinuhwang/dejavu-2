import torch
from torch import nn
import warnings

class PassthroughRestoration(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        current_pre_proj,
        most_similar_pre_proj,
        most_similar_hidden_states,
        most_similar_query_states,
        most_similar_key_states,
        most_similar_value_states,
    ):
        return most_similar_hidden_states, most_similar_query_states, most_similar_key_states, most_similar_value_states

class DiffRestoration(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        current_pre_proj,
        most_similar_pre_proj,
        most_similar_hidden_states,
        most_similar_query_states,
        most_similar_key_states,
        most_similar_value_states,
    ):
        diff = current_pre_proj - most_similar_pre_proj

        hidden_states = most_similar_hidden_states + diff
        query_states = most_similar_query_states + diff
        key_states = most_similar_key_states + diff
        value_states = most_similar_value_states + diff

        return hidden_states, query_states, key_states, value_states


class MLPRestoration(nn.Module):
    def __init__(self, input_dim=768, inner_dim=64, disable_bias=False, dropout=0.0):
        super().__init__()
        self.use_bias = use_bias = not disable_bias
        dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.h_mlp = nn.Sequential(
            nn.Linear(input_dim, inner_dim, bias=use_bias),
            nn.ReLU(),
            dropout,
            nn.Linear(inner_dim, input_dim, bias=use_bias),
        )
        self.q_mlp = nn.Sequential(
            nn.Linear(input_dim, inner_dim, bias=use_bias),
            nn.ReLU(),
            dropout,
            nn.Linear(inner_dim, input_dim, bias=use_bias),
        )
        self.k_mlp = nn.Sequential(
            nn.Linear(input_dim, inner_dim, bias=use_bias),
            nn.ReLU(),
            dropout,
            nn.Linear(inner_dim, input_dim, bias=use_bias),
        )
        self.v_mlp = nn.Sequential(
            nn.Linear(input_dim, inner_dim, bias=use_bias),
            nn.ReLU(),
            dropout,
            nn.Linear(inner_dim, input_dim, bias=use_bias),
        )

    def forward_mlp(self, diff):
        hidden_states = self.h_mlp(diff)
        query_states = self.q_mlp(diff)
        key_states = self.k_mlp(diff)
        value_states = self.v_mlp(diff)

        return hidden_states, query_states, key_states, value_states

    def forward(
        self,
        current_pre_proj,
        most_similar_pre_proj,
        most_similar_hidden_states,
        most_similar_query_states,
        most_similar_key_states,
        most_similar_value_states,
    ):
        diff = current_pre_proj - most_similar_pre_proj

        h, q, k, v = self.forward_mlp(diff)

        hidden_states = most_similar_hidden_states + h
        query_states = most_similar_query_states + q
        key_states = most_similar_key_states + k
        value_states = most_similar_value_states + v

        return hidden_states, query_states, key_states, value_states


class MergedMLPRestoration(nn.Module):
    def __init__(self, restoration_module: MLPRestoration, input_dim=768, inner_dim=64):
        super().__init__()

        self.use_bias = restoration_module.use_bias

        self.w1 = nn.Parameter(torch.stack((restoration_module.h_mlp[0].weight, restoration_module.q_mlp[0].weight, restoration_module.k_mlp[0].weight, restoration_module.v_mlp[0].weight)).transpose(1, 2))
        self.w2 = nn.Parameter(torch.stack((restoration_module.h_mlp[2].weight, restoration_module.q_mlp[2].weight, restoration_module.k_mlp[2].weight, restoration_module.v_mlp[2].weight)).transpose(1, 2))
        if restoration_module.use_bias:
            self.b1 = nn.Parameter(torch.stack((restoration_module.h_mlp[0].bias, restoration_module.q_mlp[0].bias, restoration_module.k_mlp[0].bias, restoration_module.v_mlp[0].bias)).unsqueeze(1))
            self.b2 = nn.Parameter(torch.stack((restoration_module.h_mlp[2].bias, restoration_module.q_mlp[2].bias, restoration_module.k_mlp[2].bias, restoration_module.v_mlp[2].bias)).unsqueeze(1))
        else:
            self.b1 = self.b2 = None

        # self.hqkv_mlp = nn.Sequential(
        #     nn.Conv1d(4 * input_dim, 4 * inner_dim, 1, groups=4),
        #     nn.ReLU(),
        #     nn.Conv1d(4 * inner_dim, 4 * input_dim, 1, groups=4),
        # )
        warnings.warn("This model is not equivalent to the original model at the moment")

    def forward(
        self,
        current_pre_proj,
        most_similar_pre_proj,
        most_similar_hidden_states,
        most_similar_query_states,
        most_similar_key_states,
        most_similar_value_states,
    ):
        diff = current_pre_proj - most_similar_pre_proj

        h, q, k, v = self.forward_mlp(diff)

        hidden_states = most_similar_hidden_states + h
        query_states = most_similar_query_states + q
        key_states = most_similar_key_states + k
        value_states = most_similar_value_states + v

        return hidden_states, query_states, key_states, value_states
    
    def forward_mlp(self, diff):
        *dim_other, dim = diff.shape
        x = diff.view(-1, dim).unsqueeze(0).expand(4, -1, -1)
        x = torch.bmm(x, self.w1)
        if self.use_bias:
            x = x + self.b1
        x = torch.relu(x)
        x = torch.bmm(x, self.w2)
        if self.use_bias:
            x = x + self.b2
        x = x.view(4, *dim_other, dim)

        return x[0], x[1], x[2], x[3]
 


if __name__ == '__main__':
    # Check the computation result matches
    # Dummy data for testing
    input_dim = 768
    inner_dim = 64
    batch_size = 1
    seq_len = 197

    # Instantiate models
    original_model = MLPRestoration(input_dim, inner_dim)
    merged_model = MergedMLPRestoration(original_model)

    # Random tensors for input
    current_pre_proj = torch.randn(batch_size, seq_len, input_dim)
    most_similar_pre_proj = torch.randn(batch_size, seq_len, input_dim)
    most_similar_hidden_states = torch.randn(batch_size, seq_len, input_dim)
    most_similar_query_states = torch.randn(batch_size, seq_len, input_dim)
    most_similar_key_states = torch.randn(batch_size, seq_len, input_dim)
    most_similar_value_states = torch.randn(batch_size, seq_len, input_dim)

    # Get outputs
    original_output = original_model(current_pre_proj, most_similar_pre_proj, most_similar_hidden_states,
                                     most_similar_query_states, most_similar_key_states, most_similar_value_states)
    merged_output = merged_model(current_pre_proj, most_similar_pre_proj, most_similar_hidden_states,
                                 most_similar_query_states, most_similar_key_states, most_similar_value_states)

    # Compare outputs
    for original, merged in zip(original_output, merged_output):
        print(f"Original vs Merged close: {torch.allclose(original, merged, atol=1e-6)}")