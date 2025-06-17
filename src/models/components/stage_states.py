import torch

import triton
import triton.language as tl

@triton.jit
def stage_states_local_kernel(
    # pointers
    reuse_map_ptr, # [B, N]
    pre_proj_ptr, # [B, N, dim]
    pre_proj_norm_ptr, # [B, N, dim]
    hidden_states_ptr , # [B, N, dim]  
    ref_ptr, # [B, N, dim]
    ref_norm_ptr, # [B, N, dim]
    ref_gather_idx_ptr, # [B, N]
    diff_pre_proj_ptr, # [B, N, dim]
    compute_cache_ptr, # [4*N, dim]
    hidden_cache_ptr, # [4*N, dim]
    compute_cache_len_ptr, # [1]
    gather_idxs_ptr, # [B, N]
    compute_indices_ptr, # [B, N]
    compute_bases_ptr, # [B]
    # strides
    map_batch_stride,
    states_batch_stride,
    states_row_stride,
    cache_row_stride,
    # constants
    B, N, dim,
    BLOCK_COL: tl.constexpr  # dim
):
    work_idx = tl.program_id(0)
    batch_idx = tl.program_id(1)
    row_idx = tl.program_id(2)
    col_offsets = tl.arange(0, BLOCK_COL)

    col_boundary_mask = col_offsets < dim

    # Load reuse map for each token
    reuse_map_ptr = reuse_map_ptr + batch_idx * map_batch_stride + row_idx
    reuse_map = tl.load(reuse_map_ptr)

    # If current token is reused
    if reuse_map:
        if work_idx == 0:
            # Calculate diff_pre_proj
            ref_row_ptr = ref_ptr + batch_idx * states_batch_stride + row_idx * states_row_stride + col_offsets
            ref_row = tl.load(ref_row_ptr, mask=col_boundary_mask)

            pre_proj_row_ptr = pre_proj_ptr + batch_idx * states_batch_stride + row_idx * states_row_stride + col_offsets
            pre_proj_row = tl.load(pre_proj_row_ptr, mask=col_boundary_mask)

            diff_pre_proj = pre_proj_row - ref_row

            # Save diff_pre_proj
            diff_pre_proj_row_ptr = diff_pre_proj_ptr + batch_idx * states_batch_stride + row_idx * states_row_stride + col_offsets
            tl.store(diff_pre_proj_row_ptr, diff_pre_proj, mask=col_boundary_mask)

            # Overwrite pre_proj with ref
            tl.store(pre_proj_row_ptr, ref_row, mask=col_boundary_mask)
        else:
            # Overwrite pre_proj_norm with ref_norm
            ref_norm_row_ptr = ref_norm_ptr + batch_idx * states_batch_stride + row_idx * states_row_stride + col_offsets
            ref_norm_row = tl.load(ref_norm_row_ptr, mask=col_boundary_mask)

            pre_proj_norm_row_ptr = pre_proj_norm_ptr + batch_idx * states_batch_stride + row_idx * states_row_stride + col_offsets
            tl.store(pre_proj_norm_row_ptr, ref_norm_row, mask=col_boundary_mask)

            ref_gather_idx_ptr = ref_gather_idx_ptr + batch_idx * map_batch_stride + row_idx
            gather_idx = tl.load(ref_gather_idx_ptr)

            # Update gather_idx
            gather_idx_ptr = gather_idxs_ptr + batch_idx * map_batch_stride + row_idx
            tl.store(gather_idx_ptr, gather_idx)
    # If current otken should be recomputed
    else:
        # Calculate where to copy the value
        if batch_idx == 0:
            compute_base = 0
        else:
            compute_base = tl.load(compute_bases_ptr + batch_idx).to(tl.int32)

        compute_indice_ptr = compute_indices_ptr + batch_idx * map_batch_stride + row_idx
        compute_indice = tl.load(compute_indice_ptr)

        compute_cache_len = tl.load(compute_cache_len_ptr)
        recompute_row_offset = batch_idx * states_batch_stride + row_idx * states_row_stride + col_offsets

        compute_row_idx = compute_cache_len + compute_indice + compute_base
        cache_row_offset = compute_row_idx * cache_row_stride + col_offsets

        if work_idx == 0:
            # Copy pre_proj to compute_cache
            pre_proj_row = tl.load(pre_proj_ptr + recompute_row_offset, mask=col_boundary_mask)
            tl.store(compute_cache_ptr + cache_row_offset, pre_proj_row, mask=col_boundary_mask)

            # Calculate gather_idx
            gather_idx = compute_row_idx + B * N
            # Update gather_idx
            gather_idx_ptr = gather_idxs_ptr + batch_idx * map_batch_stride + row_idx
            tl.store(gather_idx_ptr, gather_idx)
        else:
            # Collect hidden_states
            hidden_row = tl.load(hidden_states_ptr + recompute_row_offset, mask=col_boundary_mask)
            tl.store(hidden_cache_ptr + cache_row_offset, hidden_row, mask=col_boundary_mask)


def stage_states_local(
    reuse_map, # [B, N]
    pre_proj, # [B, N, dim]
    pre_proj_norm, # [B, N, dim]
    hidden_states, # [B, N, dim]
    ref, # [B, N, dim]
    ref_norm, # [B, N, dim]
    ref_gather_idx, # [B, N]
    diff_pre_proj, # [B, N, dim]
    compute_cache, # [B*4*N, dim]
    hidden_cache, # [B*4*N, dim]
    compute_cache_len,
    gather_idxs, # [B, N]
):
    B, N, dim = pre_proj.shape
    assert reuse_map.shape == (B, N)
    assert pre_proj.shape == (B, N, dim)
    assert pre_proj_norm.shape == (B, N, dim)
    assert hidden_states.shape == (B, N, dim)
    assert ref.shape == (B, N, dim)
    assert ref_norm.shape == (B, N, dim)
    assert diff_pre_proj.shape == (B, N, dim)
    assert compute_cache.shape == (B*4*N, dim)
    assert hidden_cache.shape == (B*4*N, dim)

    BLOCK_COL = triton.next_power_of_2(dim)

    # Example
    # reuse_map: [[1, 0, 0, 1], [1, 0, 0, 0]]
    # cumsum: [[0, 1, 2, 2], [0, 1, 2, 3]]
    # compute_indices: used to locate where to copy value for compaction
    # [[-1, 0, 1, 1], [-1, 0, 1, 2,]]
    # compute_cnts: number of computed tokens per batch
    # [2, 3]
    # compute_base: Offset for each batch
    # [2, 5] => [5, 2] (compute_total, batch 1 start, ...)
    cumsum = torch.cumsum(~reuse_map, dim=1, dtype=torch.int64) # [B, N]
    compute_indices = cumsum -1 # [B, N]
    compute_cnts = cumsum[:, -1] # [B]
    compute_bases = torch.roll(torch.cumsum(compute_cnts, axis=0), shifts=1, dims=0) # [B]
    compute_total = compute_bases[0]

    stage_states_local_kernel[(2, B, N)](
        # pointers
        reuse_map_ptr=reuse_map,
        pre_proj_ptr=pre_proj,
        pre_proj_norm_ptr=pre_proj_norm,
        hidden_states_ptr=hidden_states,
        ref_ptr=ref,
        ref_norm_ptr=ref_norm,
        ref_gather_idx_ptr=ref_gather_idx,
        diff_pre_proj_ptr=diff_pre_proj,
        compute_cache_ptr=compute_cache,
        hidden_cache_ptr=hidden_cache,
        compute_cache_len_ptr=compute_cache_len,
        gather_idxs_ptr=gather_idxs,
        compute_indices_ptr=compute_indices, # [B, N]
        compute_bases_ptr=compute_bases, # [B]
        # strides
        map_batch_stride=reuse_map.stride(0),
        states_batch_stride=pre_proj.stride(0),
        states_row_stride=pre_proj.stride(1),
        cache_row_stride=compute_cache.stride(0),
        # constants
        B=B, N=N, dim=dim,
        BLOCK_COL=BLOCK_COL
    )
    
    # Update the cache length
    compute_cache_len += compute_total