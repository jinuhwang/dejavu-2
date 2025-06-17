from .config import ViTConfig, RunConfig
from .ops import *

class MemoryLogger:
    def __init__(self, model_config: ViTConfig, run_config: RunConfig):
        self.mcfg = model_config
        self.rcfg= run_config
        
        self.init_params()
        self.model_size = self.mcfg.nparams()

        # this is for cache
        self.layers = [[
            'stage_states.1',
            'stage_states.2',
            'stage_states.3',
            'stage_states.4',
        ]
        for i in range(self.mcfg.nlayer)]

        # this is for flops calculation
        self.layers = [[
            'ln1', # done
            'qkvgen', # done
            'restoration' if i > 0 else None, # todo: it is restoration now
            'mha', # done

            # 'cls_attn_sum',
            
            # - stage states
            'cls_sum' if i < self.mcfg.nlayer - 1 else None,
            'vnorm1' if i < self.mcfg.nlayer - 1 else None,
            'vnorm2' if i < self.mcfg.nlayer - 1 else None,
            # -- iter 4
            'stage_states.1',
            'stage_states.2',
            'stage_states.3',
            'stage_states.4',

            'proj', # done
            'res1', # done
            'ln2', # done
            'fc1', # done
            'gelu', # done        
            'fc2', # done
            'res2', # done
        ]
        for i in range(self.mcfg.nlayer)]
    
    def log_filtered(self, name_filter: callable):
        self.init_params()
        logs = []
        for layer_idx, layer_names in enumerate(self.layers):
            for layer_name in layer_names:
                if layer_name is None:
                    continue
                self.update_params(layer_idx, layer_name)
                if name_filter(layer_name):
                    logs.append(self.log(f"layer{layer_idx}.{layer_name}"))
        
        return logs

    def log_cache_size(self):
        def logic(layer_name:str):
            return layer_name.startswith('stage_states')
        return self.log_filtered(logic)
    
    def init_params(self):
        # total 1 cache track. only consider ref_norm for 11 layers
        # hqkv is tracked as activation.
        self.cache_size = self.rcfg.batch_size * (self.mcfg.nlayer-1) * (self.mcfg.npatch-1) * self.mcfg.hidden_size
        # self.cache_size = self.rcfg.batch_size * (self.mcfg.npatch-1) * self.mcfg.hidden_size
        # 4 frames each
        self.activation_size = 4 * self.rcfg.batch_size * self.mcfg.npatch * self.mcfg.hidden_size 
        self.flops = 0
        self.flops_qkv_projection = 0
        self.flops_attention = 0
        self.flops_ffn = 0
        self.flops_overhead = 0
    
    def log_all(self, is_flops_per_layer: bool = False):
        logs = []
        for layer_idx, layer_names in enumerate(self.layers):
            for layer_name in layer_names:
                if layer_name is None:
                    continue
                self.update_params(layer_idx, layer_name)
                logs.append(self.log(f"layer{layer_idx}.{layer_name}"))
                if is_flops_per_layer:
                    self.flops = 0
        
        return logs
    
    def get_flops(self):
        return self.flops
    
    def log(self, label):
        return {
            "label": label,
            "cache_size": self.cache_size * precision / 1024 / 1024 / 1024, # precision & to megabytes
            "activation_size": self.activation_size * precision / 1024 / 1024 / 1024,
            "model_size": self.model_size * precision / 1024 / 1024 / 1024,
            "flops": self.flops,
        }
        
    # layer_idx is 0 ~ nlayer - 1
    def update_params(self, layer_idx:int, layer_name:str):
        
        if layer_idx < self.mcfg.nlayer - 1:
            reuse_rate_sum = sum(self.rcfg.reuse_rates[layer_idx])
        if layer_idx > 0:
            qkv_reuse_rate_sum = sum(self.rcfg.reuse_rates[layer_idx-1])
        
        if layer_name in ['ln1', 'ln2']:
            flops = ln_flops(self.rcfg.batch_size * self.mcfg.npatch, self.mcfg.hidden_size)

            if layer_name == 'ln1':
                self.flops_qkv_projection += flops
            else:
                self.flops_ffn += flops

            self.flops += flops
        elif layer_name in ['qkvgen', 'proj', 'fc1', 'fc2']:
            if layer_name == 'qkvgen':
                if layer_idx == 0:
                    bs = 4 * self.rcfg.batch_size * self.mcfg.npatch
                elif layer_idx > 0:
                    bs = (4 - qkv_reuse_rate_sum) * self.rcfg.batch_size * self.mcfg.npatch
            elif layer_name in ['proj', 'fc1', 'fc2']:
                if layer_idx < self.mcfg.nlayer - 1:
                    bs = (4 - reuse_rate_sum) * self.rcfg.batch_size * self.mcfg.npatch
                elif layer_idx == self.mcfg.nlayer - 1:
                    bs = 4 * self.rcfg.batch_size

            if layer_name in ['qkvgen', 'proj', 'fc1']:
                dim1 = self.mcfg.hidden_size
            elif layer_name == 'fc2':
                dim1 = self.mcfg.mlp_size

            if layer_name == 'qkvgen':
                dim2 = 3 * self.mcfg.hidden_size 
            elif layer_name == 'fc1':
                dim2 = self.mcfg.mlp_size
            elif layer_name in ['proj', 'fc2']:
                dim2 = self.mcfg.hidden_size

            flops = mm_flops(bs, dim1, dim2)

            if layer_name == 'qkvgen':
                self.flops_qkv_projection += flops
            else:
                self.flops_ffn += flops
            self.flops += flops

            self.activation_size = bs * dim2
        
        elif layer_name in ['res1', 'res2']:
            self.flops_ffn += self.activation_size
            self.flops += self.activation_size
        
        elif layer_name == 'gelu':
            flops = gelu_flops(self.activation_size)
            self.flops_ffn += flops
            self.flops += flops
        
        elif layer_name == 'mha':
            bs = self.rcfg.batch_size
            nh = self.mcfg.nhead
            hph = self.mcfg.hidden_size // nh # hidden per head
            flops = 0

            # mm1 batchsize*numhead 개의 (npatch, hph) @ (hph, npatch)
            flops += mm_flops(bs * nh * self.mcfg.npatch, hph, self.mcfg.npatch)
            # (bs, nh, npatch, npatch) scale, softmax
            flops += bs * nh * self.mcfg.npatch ** 2
            flops += softmax_flops(bs * nh * self.mcfg.npatch, self.mcfg.npatch)
            # mm2
            flops += mm_flops(bs * nh * self.mcfg.npatch, self.mcfg.npatch, hph)

            flops = 4 * flops

            self.flops_attention += flops
            self.flops += flops

        elif layer_name == 'restoration':
            if self.mcfg.nlayer == 0:
                return
            # mlp * 4 / for parameter, please refer to vgenie/model/train/restoration.py#43
            inner_dim = 64
            bs = 4 * self.rcfg.batch_size * self.mcfg.npatch
            flops = 4 * (mm_flops(bs, self.mcfg.hidden_size, inner_dim) + bs * inner_dim + mm_flops(bs, inner_dim, self.mcfg.hidden_size)) # todo         
            self.flops_overhead += flops
            self.flops += flops
        
        elif layer_name == 'cls_sum':
            if layer_idx == self.mcfg.nlayer - 1:
                return
            flops = (self.mcfg.nhead - 1) * 4 * self.rcfg.batch_size * (self.mcfg.npatch - 1)
            self.flops_overhead += flops
            self.flops += flops
        elif layer_name in ['vnorm1', 'vnorm2']:
            if layer_idx == self.mcfg.nlayer - 1:
                return
            # vgenie/model/reuse/opt_attention/model.py #316,317
            flops = vnorm_flops(9 * self.rcfg.batch_size * self.mcfg.npatch, self.mcfg.hidden_size)
            self.flops_overhead += flops
            self.flops += flops

        elif layer_name in ['stage_states.1', 'stage_states.2', 'stage_states.3', 'stage_states.4']:
            if layer_idx == self.mcfg.nlayer - 1:
                return
            iter_idx = int(layer_name[-1:])-1
            reuse_rate = self.rcfg.reuse_rates[layer_idx]

            flops = 0

            flops += self.rcfg.batch_size * self.mcfg.npatch * mm_flops(1, self.mcfg.hidden_size, 1)
            if iter_idx >= 1:
                flops += self.rcfg.batch_size * self.mcfg.npatch * mm_flops(1, self.mcfg.hidden_size, 1)
                flops += self.rcfg.batch_size * self.mcfg.npatch

            #329 forward inner / refer to variables at vgenie/model/train/decision.py#68
            # assume using compressed info
            input_dim = 3
            inner_dim = 64
            # bn(input_dim) --> bn can be fused with future matmul
            # mlp
            flops += self.rcfg.batch_size * (mm_flops(self.mcfg.npatch-1, input_dim, inner_dim) + (self.mcfg.npatch-1) * inner_dim + mm_flops(self.mcfg.npatch-1, inner_dim, 1) )
            
            #335 decision > 0
            flops += self.rcfg.batch_size * self.mcfg.npatch

            #348 stage_states_per_batch
            # ~reuse_map
            flops += self.mcfg.npatch * self.rcfg.batch_size
            # torch.cumsum(reuse_map)
            flops += cumsum_flops(self.rcfg.batch_size, self.mcfg.npatch)

            # compute_indices = cumsum - 1
            flops += self.rcfg.batch_size * self.mcfg.npatch
            # torch.cumsum(compute_cnts)
            flops += cumsum_flops(1, self.rcfg.batch_size)

            # pre_proj - most_similar_pre_proj
            flops += self.rcfg.batch_size * self.mcfg.npatch * self.mcfg.hidden_size
            # reference_cache_len += compute_cnts
            flops += 1
            # compute_cache_len += compute_total
            flops += 1

            self.flops_overhead += flops
            self.flops += flops
            
            if iter_idx < 2:
                self.cache_size += self.rcfg.batch_size * (self.mcfg.npatch - 1) * self.mcfg.hidden_size
            else:
                self.cache_size -= self.rcfg.batch_size * (self.mcfg.npatch - 1) * self.mcfg.hidden_size

        else:
            raise ValueError(f"unknown layer name {layer_name}")

def get_reusevit_flops(model_size, reuse_rate, batch_size=256):
    assert model_size in ["base", "large"], "model size must be either base or large"
    assert len(reuse_rate) == 4, "reuse rate must be a list of 4 elements"

    model_config = ViTConfig(model_size)
    run_config = RunConfig(model_config, batch_size, lambda model_config: [reuse_rate] * (model_config.nlayer -1))
    memory_logger = MemoryLogger(model_config, run_config)
    logs_ours = memory_logger.log_cache_size()
    # print(logs_ours)
    flops = memory_logger.get_flops()

    return int(flops / 4)
