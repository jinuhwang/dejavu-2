from .naive import NaiveVitFlopsLogger
from .config import ViTConfig, RunConfig
from .ops import mm_flops, layer_norm_1d_flops, softmax_1d_flops


class CmcFlopsLogger(NaiveVitFlopsLogger):
    def __init__(self, model_config: ViTConfig, batch_size, avg_reuse_ratio, reuse_start_before_mlp=False):
        super().__init__(model_config, batch_size)

        self.avg_reuse_ratio = avg_reuse_ratio
        self.reuse_start_before_mlp = reuse_start_before_mlp
        self.flops_overhead = 0

    def reset_flops(self):
        super().reset_flops()
        self.flops_overhead = 0

    def set_flops_of_encoder(self, i=None):
        if not self.reuse_start_before_mlp:
            super().set_flops_of_encoder(i)
        else:
            # 1st layer norm
            num_tokens = int(self.num_tokens * (1. - self.avg_reuse_ratio))
            self.flops_qkv_projection += self.batch_size * num_tokens * layer_norm_1d_flops(dim=self.latent_vector_size)

            # MHA, output dimension: [Batch, N, D]
            self.set_flops_of_multi_head_attention()

            num_tokens = 1 if i == self.num_encoder_layers - 1 else int(self.num_tokens * (1. - self.avg_reuse_ratio))

            self.flops_ffn += self.batch_size * mm_flops(m=num_tokens,
                                                     k=self.latent_vector_size,
                                                     n=self.latent_vector_size)

            # 1st residual
            # self.flops_ffn += self.batch_size * num_tokens * self.latent_vector_size
            self.flops_ffn += self.batch_size * num_tokens * self.latent_vector_size

            # 2nd layer norm
            # self.flops_ffn += self.batch_size * num_tokens * layer_norm_1d_flops(dim=self.latent_vector_size)
            self.flops_ffn += self.batch_size * num_tokens * layer_norm_1d_flops(dim=self.latent_vector_size)
            
            # MLP
            self.set_flops_of_mlp(num_tokens)
        
            # 2nd residual
            # self.flops_ffn += self.batch_size * num_tokens * self.latent_vector_size
            self.flops_ffn += self.batch_size * num_tokens * self.latent_vector_size

    def set_flops_of_multi_head_attention(self):
        # qkv linear
        # batch_size x [N, D] x [D, D]
        # CMC only apply the reuse computation on QKV projection
        self.flops_qkv_projection += 3 * self.batch_size * mm_flops(m=int(self.num_tokens * (1. - self.avg_reuse_ratio)),
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
        self.flops_attention += self.batch_size * self.num_heads * (self.num_tokens ** 2 + 1)

        # attention weights
        # Aw = softmax(Aw)
        self.flops_attention += self.batch_size * self.num_heads * self.num_tokens * softmax_1d_flops(dim=self.num_tokens)

        # attention outputs
        # A = Aw x v
        self.flops_attention += self.batch_size * self.num_heads * mm_flops(m=self.num_tokens,
                                                                            k=self.num_tokens,
                                                                            n=latent_vector_size_per_head)



def get_cmc_flops(model_size, avg_reuse_ratio, batch_size=256, reuse_start_before_mlp=False):
    assert model_size in ["base", "large"], "model size must be either base or large"
    model_config = ViTConfig(model_size)
    cmc_flops_logger = CmcFlopsLogger(model_config=model_config,
                                     batch_size=batch_size,
                                     avg_reuse_ratio=avg_reuse_ratio,
                                     reuse_start_before_mlp=reuse_start_before_mlp)
    return cmc_flops_logger.get_flops()
