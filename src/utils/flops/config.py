# from vit paper https://arxiv.org/pdf/2010.11929
# create a dataclass
from dataclasses import dataclass, field

precision = 4

@dataclass
class ViTConfig:
    model_type: str
    hidden_size: int = field(init=False)
    mlp_size: int = field(init=False)
    nlayer: int = field(init=False)
    nhead: int = field(init=False)
    npatch: int = field(init=False)

    def __post_init__(self):
        if self.model_type == "base":
            self.hidden_size = 768
            self.mlp_size = 3072
            self.nlayer = 12
            self.nhead = 12
            self.npatch = 14 ** 2 + 1
        elif self.model_type == "large":
            self.hidden_size = 1024
            self.mlp_size = 4096
            self.nlayer = 24
            self.nhead = 16
            self.npatch = 16 ** 2 + 1
        else:
            raise ValueError("model_type must be one of [base, large]")
    
    def nparams(self):
        return self.nlayer * (
            2 * 2 * (self.hidden_size)
            + (1 + self.hidden_size) * self.hidden_size * 3
            + (1 + self.hidden_size) * self.hidden_size
            + (1 + self.hidden_size) * self.mlp_size
            + (1 + self.mlp_size) * self.hidden_size
        )

@dataclass
class RunConfig:
    model_config: ViTConfig
    batch_size: int
    reuse_rates_gen: callable
    # [layer_idx][iter_idx]
    reuse_rates: list = field(init=False)
    
    def __post_init__(self):
        self.reuse_rates = self.reuse_rates_gen(self.model_config)
