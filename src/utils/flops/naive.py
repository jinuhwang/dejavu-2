from .config import ViTConfig
from .ops import mm_flops, layer_norm_1d_flops, softmax_1d_flops

class NaiveVitFlopsLogger:
    def __init__(self, model_config: ViTConfig, batch_size):
        self.model_config = model_config
        self.batch_size = batch_size # Batch
        self.latent_vector_size = self.model_config.hidden_size # D
        self.num_heads = self.model_config.nhead
        self.num_encoder_layers = self.model_config.nlayer
        self.num_tokens = self.model_config.npatch # N, this include class token
        self.mlp_hidden_size = self.model_config.mlp_size

        self.flops_qkv_projection = 0
        self.flops_attention = 0
        self.flops_ffn = 0
        self.flops_total = 0

    def reset_flops(self):
        self.flops_qkv_projection = 0
        self.flops_attention = 0
        self.flops_ffn = 0
        self.flops_total = 0

    def sum_flops(self):
        self.flops_total = self.flops_qkv_projection + self.flops_attention + self.flops_ffn

    def get_flops(self):
        self.reset_flops()

        # transformer encoder layers
        for i in range(self.num_encoder_layers):
            self.set_flops_of_encoder(i)

        # final output project (reference: CLIP)
        # single linear layer with a class token
        self.flops_ffn += self.batch_size * mm_flops(m=1,
                                                     k=self.latent_vector_size,
                                                     n=self.latent_vector_size)
        
        self.sum_flops()
        
        return self.flops_total

    def set_flops_of_encoder(self, i=None):
        # 1st layer norm
        self.flops_qkv_projection += self.batch_size * self.num_tokens * layer_norm_1d_flops(dim=self.latent_vector_size)

        # MHA, output dimension: [Batch, N, D]
        self.set_flops_of_multi_head_attention()

        num_tokens = 1 if i == self.num_encoder_layers - 1 else self.num_tokens

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

    def set_flops_of_multi_head_attention(self, num_tokens=None):
        # qkv linear
        # batch_size x [N, D] x [D, D]
        if num_tokens is None:
            num_tokens = self.num_tokens

        self.flops_qkv_projection += 3 * self.batch_size * mm_flops(m=num_tokens,
                                                                    k=self.latent_vector_size,
                                                                    n=self.latent_vector_size)
        
        # MHA
        assert self.latent_vector_size % self.num_heads == 0
        latent_vector_size_per_head = int(self.latent_vector_size / self.num_heads)

        # attention scores
        # Aw = q x k^t, output dimentions: [batch_size, num_heads, num_tokens, num_tokens]
        self.flops_attention += self.batch_size * self.num_heads * mm_flops(m=num_tokens,
                                                                            k=latent_vector_size_per_head,
                                                                            n=num_tokens)

        # scaling
        # Aw = Aw / scale
        # we don't consider flops of calculating the scaling constant (it is negligible)
        # TODO: previous submission
        # self.flops_attention += self.batch_size * self.num_heads * (self.num_tokens ** 2)
        self.flops_attention += self.batch_size * self.num_heads * (num_tokens ** 2 + 1)

        # attention weights
        # Aw = softmax(Aw)
        self.flops_attention += self.batch_size * self.num_heads * num_tokens * softmax_1d_flops(dim=num_tokens)

        # attention outputs
        # A = Aw x v
        self.flops_attention += self.batch_size * self.num_heads * mm_flops(m=num_tokens,
                                                                            k=num_tokens,
                                                                            n=latent_vector_size_per_head)

        # linear layer
        # print(self.batch_size * mm_flops(m=self.num_tokens, k=self.latent_vector_size, n=self.latent_vector_size))
        # self.flops_attention += self.batch_size * mm_flops(m=self.num_tokens,
        #                                                    k=self.latent_vector_size,
        #                                                    n=self.latent_vector_size)

    def set_flops_of_mlp(self, num_tokens):
        # 1st linear layer
        self.flops_ffn += self.batch_size * mm_flops(m=num_tokens,
                                                     k=self.latent_vector_size,
                                                     n=self.mlp_hidden_size)

        # gelu
        self.flops_ffn += self.batch_size * num_tokens * self.mlp_hidden_size

        # 2nd
        # TODO: previous submission
        self.flops_ffn += self.batch_size * mm_flops(m=num_tokens,
                                                     k=self.latent_vector_size,
                                                     n=self.mlp_hidden_size)


def get_orginal_vit_flops(model_size, batch_size=256):
    assert model_size in ["base", "large"], "model size must be either base or large"
    model_config = ViTConfig(model_size)
    naive_vit_flops_logger = NaiveVitFlopsLogger(model_config=model_config, batch_size=batch_size) 
    return naive_vit_flops_logger.get_flops()