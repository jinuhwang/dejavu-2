import math
from .naive import NaiveVitFlopsLogger
from .config import ViTConfig
from .ops import mm_flops, layer_norm_1d_flops, softmax_1d_flops

class EventfulTransformerFlopsLogger(NaiveVitFlopsLogger):
    def __init__(self, model_config: ViTConfig, batch_size, r):
        super().__init__(model_config, batch_size)

        # number of tokens to reuse
        self.r = r
        # print(f"tokens: {self.num_tokens}")

        assert self.r <= self.num_tokens

        self.flops_gating_module = 0
        self.flops_delta_gating_module = 0

    def reset_flops(self):
        super().reset_flops()
        self.flops_gating_module = 0
        self.flops_delta_gating_module = 0


    def sum_flops(self):
        self.flops_total = self.flops_qkv_projection + self.flops_attention + self.flops_ffn + \
            self.flops_gating_module + self.flops_delta_gating_module
        
    def set_flops_of_encoder(self, i=None):
        # 1st layer norm
        self.flops_qkv_projection += self.batch_size * self.num_tokens * layer_norm_1d_flops(dim=self.latent_vector_size)

        # MHA, output dimension: [Batch, N, D]
        self.set_flops_of_multi_head_attention()

        num_tokens = 1 if i == self.num_encoder_layers - 1 else self.r

        self.flops_ffn += self.batch_size * mm_flops(m=num_tokens,
                                                     k=self.latent_vector_size,
                                                     n=self.latent_vector_size)

        # 1st residual
        # self.flops_ffn += self.batch_size * num_tokens * self.latent_vector_size
        self.flops_ffn += self.batch_size * self.num_tokens * self.latent_vector_size
        # self.flops_ffn += self.batch_size * self.r * self.latent_vector_size

        # 2nd layer norm
        # self.flops_ffn += self.batch_size * num_tokens * layer_norm_1d_flops(dim=self.latent_vector_size)
        self.flops_ffn += self.batch_size * self.num_tokens * layer_norm_1d_flops(dim=self.latent_vector_size)
        # self.flops_ffn += self.batch_size * self.r * layer_norm_1d_flops(dim=self.latent_vector_size)
        
        # MLP
        self.set_flops_of_mlp(num_tokens)
        
        # 2nd residual
        # self.flops_ffn += self.batch_size * num_tokens * self.latent_vector_size
        self.flops_ffn += self.batch_size * self.num_tokens * self.latent_vector_size
        # self.flops_ffn += self.batch_size * self.r * self.latent_vector_size


    def set_flops_of_multi_head_attention(self):
        # gating module
        # q, k, v -> q', k', v' ([N, D] -> [M, D] / N: num_tokens, M: r)
        # self.flops_gating_module +=  self.batch_size * self.get_flops_of_gating_module(num_tokens=self.num_tokens,
        #                                                                                latent_vector_size=self.latent_vector_size)
        self.flops_gating_module +=  self.batch_size * self.get_flops_of_gating_module(num_tokens=self.r,
                                                                                       latent_vector_size=self.latent_vector_size)


        # qkv linear
        # batch_size x [N, D] x [D, D]
        self.flops_qkv_projection += 3 * self.batch_size * mm_flops(m=self.r,
                                                                    k=self.latent_vector_size,
                                                                    n=self.latent_vector_size)
        
        # MHA
        assert self.latent_vector_size % self.num_heads == 0
        latent_vector_size_per_head = int(self.latent_vector_size / self.num_heads)

        # attention scores
        # Aw = q x k^t, output dimentions: [batch_size, num_heads, num_tokens, num_tokens]
        self.flops_attention += self.batch_size * self.num_heads * mm_flops(m=self.num_tokens,
                                                                            k=latent_vector_size_per_head,
                                                                            n=self.num_tokens)
        
        # scaling
        # Aw = Aw / scale
        # we don't consider flops of calculating the scaling constant (it is negligible)
        # TODO: previous submission
        self.flops_attention += self.batch_size * self.num_heads * (self.num_tokens ** 2 + 1)

        # attention weights
        # Aw = softmax(Aw)
        self.flops_attention += self.batch_size * self.num_heads * self.num_tokens * softmax_1d_flops(dim=self.num_tokens)

        # delta gating module
        self.flops_delta_gating_module += self.batch_size * self.num_heads * self.get_flops_of_delta_gating_module(num_tokens=self.r,
                                                                                                                   latent_vector_size=latent_vector_size_per_head)
        self.flops_delta_gating_module += self.batch_size * self.num_heads * self.get_flops_of_delta_gating_module(num_tokens=self.r,
                                                                                                                   latent_vector_size=self.num_tokens)
        # attention outputs
        # A = Aw' x v'
        self.flops_attention += self.batch_size * self.num_heads * mm_flops(m=self.num_tokens,
                                                                            k=self.num_tokens,
                                                                            n=latent_vector_size_per_head)

        # # prev_Aw x delta_v
        # self.flops_attention += self.batch_size * self.num_heads * mm_flops(m=self.num_tokens,
        #                                                                     k=self.r,
        #                                                                     n=latent_vector_size_per_head)
        
        # # prev_delta_A x (prev_v - delta_v)
        # self.flops_attention += self.batch_size * self.num_heads * (self.r * latent_vector_size_per_head)
        # self.flops_attention += self.batch_size * self.num_heads * mm_flops(m=self.num_tokens,
        #                                                                     k=self.r,
        #                                                                     n=latent_vector_size_per_head)

        # # prev_attention_output + (prev_Aw x delta_v) + (prev_delta_A x (prev_v - delta_v))
        # self.flops_attention += 2 * (self.batch_size * self.num_heads * (self.num_tokens * latent_vector_size_per_head))

        # gating module
        self.flops_gating_module += self.get_flops_of_gating_module(num_tokens=self.r,
                                                                    latent_vector_size=self.latent_vector_size)

        # linear layer
        # self.flops_attention += self.batch_size * mm_flops(m=self.r,
        #                                                    k=self.latent_vector_size,
        #                                                    n=self.latent_vector_size)

    def set_flops_of_mlp(self, num_tokens):
        # gating module
        self.flops_gating_module += self.get_flops_of_gating_module(num_tokens=num_tokens,
                                                                    latent_vector_size=self.latent_vector_size)

        # 1st linear layer
        self.flops_ffn += self.batch_size * mm_flops(m=num_tokens,
                                                     k=self.latent_vector_size,
                                                     n=self.mlp_hidden_size)

        # gelu
        self.flops_ffn += self.batch_size * self.r * self.mlp_hidden_size

        # 2nd linear layer
        self.flops_ffn += self.batch_size * mm_flops(m=num_tokens,
                                                     k=self.mlp_hidden_size,
                                                     n=self.latent_vector_size)
    
    def get_flops_of_gating_module(self, num_tokens, latent_vector_size):
        # cacluate e = c - u
        flops = num_tokens * latent_vector_size

        # calculate
        for _ in range(num_tokens):
            # elementwise power(D, 2)
            flops += latent_vector_size

            # reduce
            flops += latent_vector_size - 1

            # sqrt
            flops += 1

        # select r tokens
        flops += int(math.ceil(num_tokens * math.log2(num_tokens) + self.r))

        return flops
    
    def get_flops_of_delta_gating_module(self, num_tokens, latent_vector_size):
        return self.get_flops_of_gating_module(num_tokens=num_tokens,
                                               latent_vector_size=latent_vector_size)


def get_eventful_flops(model_size, r, batch_size=256):
    assert model_size in ["base", "large"], "model size must be either base or large"
    model_config = ViTConfig(model_size)
    eventful_transformer_flops_logger = EventfulTransformerFlopsLogger(model_config=model_config,
                                                                   batch_size=batch_size,
                                                                   r=r)
    return eventful_transformer_flops_logger.get_flops()