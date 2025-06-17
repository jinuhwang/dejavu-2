from .naive import NaiveVitFlopsLogger
from .config import ViTConfig
from .ops import *
from ..paths import get_path_manager



class DiffRateFlopsLogger(NaiveVitFlopsLogger):
    def __init__(self, model_config: ViTConfig, batch_size, prune_kept_nums, merge_kept_nums):
        super().__init__(model_config, batch_size)

        # number of tokens to reuse
        self.prune_kept_nums = prune_kept_nums
        self.merge_kept_nums = merge_kept_nums

        self.merge_flops = 0
        self.prune_flops = 0

    def reset_flops(self):
        super().reset_flops()
        self.merge_flops = 0
        self.prune_flops = 0

    def sum_flops(self):
        self.flops_total = (
            self.flops_qkv_projection
            + self.flops_attention 
            + self.flops_ffn
            + self.prune_flops
            + self.merge_flops
        )
        
    def set_flops_of_encoder(self, i=None):
        # Calculate number of tokens after pruning and merging in previous layers
        if i == 0:
            num_tokens = self.num_tokens
        else:
            num_tokens = min(self.prune_kept_nums[i - 1], self.merge_kept_nums[i - 1])

        # Number of tokens pruned and merged in this layer
        prune_kept_num = self.prune_kept_nums[i]
        merge_kept_num = self.merge_kept_nums[i]


        # 1st layer norm
        self.flops_qkv_projection += self.batch_size * num_tokens * layer_norm_1d_flops(dim=self.latent_vector_size)

        # MHA, output dimension: [Batch, N, D]
        self.set_flops_of_multi_head_attention(num_tokens=num_tokens)

        self.set_flops_of_prune_and_merge(num_tokens=num_tokens, prune_kept_num=prune_kept_num, merge_kept_num=merge_kept_num)

        num_tokens = min(prune_kept_num, merge_kept_num)

        self.flops_ffn += self.batch_size * mm_flops(m=num_tokens,
                                                     k=self.latent_vector_size,
                                                     n=self.latent_vector_size)

        # 1st residual
        # self.flops_ffn += self.batch_size * num_tokens * self.latent_vector_size
        self.flops_ffn += self.batch_size * num_tokens * self.latent_vector_size
        # self.flops_ffn += self.batch_size * self.r * self.latent_vector_size

        # 2nd layer norm
        self.flops_ffn += self.batch_size * num_tokens * layer_norm_1d_flops(dim=self.latent_vector_size)
        # self.flops_ffn += self.batch_size * self.r * layer_norm_1d_flops(dim=self.latent_vector_size)
        
        # MLP
        self.set_flops_of_mlp(num_tokens)
        
        # 2nd residual
        self.flops_ffn += self.batch_size * num_tokens * self.latent_vector_size
        # self.flops_ffn += self.batch_size * self.r * self.latent_vector_size

    def set_flops_of_prune_and_merge(self, num_tokens, prune_kept_num, merge_kept_num):
        # Prune 
        # select r tokens
        self.prune_flops += int(math.ceil(num_tokens * math.log2(num_tokens) + prune_kept_num))

        # Merge
        important_num = merge_kept_num
        unimportant_num = prune_kept_num - merge_kept_num

        # select important tokens
        self.merge_flops += int(math.ceil(num_tokens * math.log2(num_tokens) + important_num))
        # select unimportant tokens
        self.merge_flops += int(math.ceil(num_tokens * math.log2(num_tokens) + unimportant_num))

        # Calculate similarity matrix
        self.merge_flops += self.batch_size * mm_flops(
            m=unimportant_num,
            k=self.latent_vector_size,
            n=important_num
        )

        # Calculate merging
        self.merge_flops += self.batch_size * unimportant_num * self.latent_vector_size



def get_diffrate_flops(model_size, target_flops, batch_size=256):
    assert model_size in ["base", "large"], "model size must be either base or large"
    if model_size == "base":
        dataset = "msrvtt"
    elif model_size == "large":
        dataset = "how2qa"

    path_manager = get_path_manager()
    prune_kept_nums, merge_kept_nums = path_manager.get_diffrate_prune_merge(dataset, target_flops)

    model_config = ViTConfig(model_size)
    diff_rate_transformer_flops_logger = DiffRateFlopsLogger(model_config=model_config,
                                                                   batch_size=batch_size,
                                                                   prune_kept_nums=prune_kept_nums,
                                                                   merge_kept_nums=merge_kept_nums)
    return diff_rate_transformer_flops_logger.get_flops()

