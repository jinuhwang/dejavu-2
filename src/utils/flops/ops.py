import math

gelu_flop = 1
rsquare_flop = 1
exp_flop = 1
precision = 4

def layer_norm_1d_flops(dim: int):
    flops = 0
    # mean calculation
    # -- sum an divide
    flops += dim-1
    flops += 1
    # -- subtract mean
    flops += dim
    # std calculation
    # -- square
    flops += dim
    # -- sum squares
    flops += dim-1
    # -- mean square. mean calculated above / subtract (mean-square) - (square-mean) / root square / add eps
    flops += 2 + rsquare_flop + 1
    # -- divide std
    flops += dim
    # scale and shift --> this can be integrated with the next matmul
    # flops += 2 * dim
    return flops

def ln_flops(bs:int, dim: int):
    flops = 0
    # mean calculation
    # -- sum an divide
    flops += dim-1
    flops += 1
    # -- subtract mean
    flops += dim
    # std calculation
    # -- square
    flops += dim
    # -- sum squares
    flops += dim-1
    # -- mean square. mean calculated above / subtract (mean-square) - (square-mean) / root square / add eps
    flops += 2 + rsquare_flop + 1
    # -- divide std
    flops += dim
    # scale and shift --> this can be integrated with the next matmul
    # flops += 2 * dim
    return bs * flops

def mm_flops(m:int, k:int, n:int):
    # flops of dot-product for single output element
    flops = k + k-1
    
    # scale it to flops of entire output matrix
    flops *= (m * n)

    return flops

def softmax_1d_flops(dim:int):
    flops = 0
    # numerical stability
    # -- get max
    flops += dim-1
    # -- subtract max value
    flops += dim
    # exponentiation
    flops += exp_flop * dim
    # sum exponentiation
    flops += dim-1
    # divide
    flops += dim
    return flops

def softmax_flops(bs:int, dim:int):
    flops = 0
    # numerical stability
    # -- get max
    flops += dim-1
    # -- subtract max value
    flops += dim
    # exponentiation
    flops += exp_flop * dim
    # sum exponentiation
    flops += dim-1
    # divide
    flops += dim
    return bs * flops

def gelu_flops(bs:int):
    return bs * gelu_flop

def vnorm_flops(bs:int, dim:int):
    flops = 0
    # square / sum / root square / eps
    flops += dim + dim - 1 + rsquare_flop + 1
    # divide all
    flops += dim
    return bs * flops

def cumsum_flops(bs:int, dim:int):
    flops = 0
    # refer to https://en.wikipedia.org/wiki/Prefix_sum, Shorter span, more parallel
    return bs * (dim * (math.log2(dim) - 1) + 1)
