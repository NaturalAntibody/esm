# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import biotite.structure
import numpy as np
from esm.inverse_folding.util import (
    load_structure,
    extract_coords_from_structure,
    get_sequence_loss,
    get_encoder_output,
    ScoringResult,
)


def extract_coords_from_complex(structure: biotite.structure.AtomArray):
    """
    Args:
        structure: biotite AtomArray
    Returns:
        Tuple (coords_list, seq_list)
        - coords: Dictionary mapping chain ids to L x 3 x 3 array for N, CA, C
          coordinates representing the backbone of each chain
        - seqs: Dictionary mapping chain ids to native sequences of each chain
    """
    coords = {}
    seqs = {}
    all_chains = biotite.structure.get_chains(structure)
    for chain_id in all_chains:
        chain = structure[structure.chain_id == chain_id]
        coords[chain_id], seqs[chain_id] = extract_coords_from_structure(chain)
    return coords, seqs


def load_complex_coords(fpath, chains):
    """
    Args:
        fpath: filepath to either pdb or cif file
        chains: the chain ids (the order matters for autoregressive model)
    Returns:
        Tuple (coords_list, seq_list)
        - coords: Dictionary mapping chain ids to L x 3 x 3 array for N, CA, C
          coordinates representing the backbone of each chain
        - seqs: Dictionary mapping chain ids to native sequences of each chain
    """
    structure = load_structure(fpath, chains)
    return extract_coords_from_complex(structure)


def _concatenate_coords(coords, target_chain_id, padding_length=10):
    """
    Args:
        coords: Dictionary mapping chain ids to L x 3 x 3 array for N, CA, C
            coordinates representing the backbone of each chain
        target_chain_id: The chain id to sample sequences for
        padding_length: Length of padding between concatenated chains
    Returns:
        Tuple (coords, seq)
            - coords is an L x 3 x 3 array for N, CA, C coordinates, a
              concatenation of the chains with padding in between
            - seq is the extracted sequence, with padding tokens inserted
              between the concatenated chains
    """
    pad_coords = np.full((padding_length, 3, 3), np.nan, dtype=np.float32)
    # For best performance, put the target chain first in concatenation.
    coords_list = [coords[target_chain_id]]
    for chain_id in coords:
        if chain_id == target_chain_id:
            continue
        coords_list.append(pad_coords)
        coords_list.append(coords[chain_id])
    coords_concatenated = np.concatenate(coords_list, axis=0)
    return coords_concatenated


def sample_sequence_in_complex(
    model,
    coords,
    target_chain_id,
    sequence: str,
    temperature=1.0,
    padding_length=10,
    positions_to_sample: list[int] | None = None,
):
    """
    Samples sequence for one chain in a complex.
    Args:
        model: An instance of the GVPTransformer model
        coords: Dictionary mapping chain ids to L x 3 x 3 array for N, CA, C
            coordinates representing the backbone of each chain
        target_chain_id: The chain id to sample sequences for
        padding_length: padding length in between chains
    Returns:
        Sampled sequence for the target chain
    """
    target_chain_len = coords[target_chain_id].shape[0]
    all_coords = _concatenate_coords(coords, target_chain_id)
    device = next(model.parameters()).device

    # Supply padding tokens for other chains to avoid unused sampling for speed
    padding_pattern = list(sequence) + ["<pad>"] * (all_coords.shape[0] - len(sequence))

    if positions_to_sample is None:
        positions_to_sample = list(range(target_chain_len))

    for i in positions_to_sample:
        padding_pattern[i] = "<mask>"

    sampled = model.sample(
        all_coords, partial_seq=padding_pattern, temperature=temperature, device=device
    )
    sampled = sampled[:target_chain_len]
    return sampled


def score_sequence_in_complex(
    model,
    alphabet,
    coords,
    target_chain_id,
    target_seq,
    padding_length=10,
    positions_to_score: list[int] | None = None,
) -> ScoringResult:
    """
    Scores sequence for one chain in a complex.
    Args:
        model: An instance of the GVPTransformer model
        alphabet: Alphabet for the model
        coords: Dictionary mapping chain ids to L x 3 x 3 array for N, CA, C
            coordinates representing the backbone of each chain
        target_chain_id: The chain id to sample sequences for
        target_seq: Target sequence for the target chain for scoring.
        padding_length: padding length in between chains
        positions_to_score: List of positions to calculate loss for.
    Returns:
        ScoringResult containing:
        - logits: Raw logits over the vocabulary for each position in the target chain
        - ll_fullseq: Average log-likelihood over the full target chain
        - ll_withcoord: Average log-likelihood in target chain excluding those
            residues without coordinates
    """
    all_coords = _concatenate_coords(coords, target_chain_id, padding_length)

    logits, loss, target_padding_mask = get_sequence_loss(
        model, alphabet, all_coords, target_seq
    )

    if positions_to_score is not None:
        loss = loss[positions_to_score]
        target_padding_mask = target_padding_mask[positions_to_score]

    ll_fullseq = -np.sum(loss * ~target_padding_mask) / np.sum(~target_padding_mask)

    # Also calculate average when excluding masked portions
    coord_mask = np.all(np.isfinite(coords[target_chain_id]), axis=(-1, -2))
    if positions_to_score is not None:
        coord_mask = coord_mask[positions_to_score]
    ll_withcoord = -np.sum(loss * coord_mask) / np.sum(coord_mask)

    return ScoringResult(logits=logits, ll_fullseq=ll_fullseq, ll_withcoord=ll_withcoord)


def score_sequence_in_complex_batch(
    model,
    alphabet,
    coords_batch: list[dict],
    target_chain_id_batch: list[str],
    target_seq_batch: list[str],
    padding_length: int | list[int] = 10,
    positions_to_score_batch: list[list[int] | None] | None = None,
) -> list[ScoringResult]:
    """
    Scores sequences for multiple chains in complexes (batched version).
    
    Args:
        model: An instance of the GVPTransformer model
        alphabet: Alphabet for the model
        coords_batch: List of dictionaries, each mapping chain ids to L x 3 x 3 array 
            for N, CA, C coordinates representing the backbone of each chain
        target_chain_id_batch: List of chain ids to score sequences for
        target_seq_batch: List of target sequences for the target chains for scoring
        padding_length: Padding length in between chains (single int or list of ints)
        positions_to_score_batch: List of position lists to calculate loss for, or None
    
    Returns:
        List of ScoringResult objects, each containing:
        - logits: Raw logits over the vocabulary for each position in the target chain
        - ll_fullseq: Average log-likelihood over the full target chain
        - ll_withcoord: Average log-likelihood in target chain excluding those
            residues without coordinates
    """
    from esm.inverse_folding.util import CoordBatchConverter
    import torch.nn.functional as F
    
    batch_size = len(coords_batch)
    
    # Handle padding_length as either single value or list
    if isinstance(padding_length, int):
        padding_lengths = [padding_length] * batch_size
    else:
        padding_lengths = padding_length
    
    # Handle positions_to_score_batch
    if positions_to_score_batch is None:
        positions_to_score_batch = [None] * batch_size
    
    # Prepare all coordinates by concatenating chains
    all_coords_list = []
    target_chain_lens = []
    
    for i in range(batch_size):
        coords = coords_batch[i]
        target_chain_id = target_chain_id_batch[i]
        pad_len = padding_lengths[i]
        
        all_coords = _concatenate_coords(coords, target_chain_id, pad_len)
        all_coords_list.append(all_coords)
        target_chain_lens.append(coords[target_chain_id].shape[0])
    
    # Prepare batch for model
    device = next(model.parameters()).device
    batch_converter = CoordBatchConverter(alphabet)
    batch = [(all_coords_list[i], None, target_seq_batch[i]) for i in range(batch_size)]
    coords, confidence, strs, tokens, padding_mask = batch_converter(batch, device=device)
    
    # Forward pass through model
    prev_output_tokens = tokens[:, :-1].to(device)
    target = tokens[:, 1:]
    target_padding_mask = target == alphabet.padding_idx
    logits, _ = model.forward(coords, padding_mask, confidence, prev_output_tokens)
    loss = F.cross_entropy(logits, target, reduction="none")
    
    # Process results for each item in batch
    results = []
    for i in range(batch_size):
        target_chain_len = target_chain_lens[i]
        
        # Extract loss and masks for this sequence (only up to target chain length)
        item_loss = loss[i, :target_chain_len].cpu().detach().numpy()
        item_padding_mask = target_padding_mask[i, :target_chain_len].cpu().numpy()
        item_logits = logits[i, :target_chain_len]
        
        # Apply positions_to_score filter if provided
        positions_to_score = positions_to_score_batch[i]
        if positions_to_score is not None:
            item_loss = item_loss[positions_to_score]
            item_padding_mask = item_padding_mask[positions_to_score]
        
        # Calculate log-likelihood for full sequence
        ll_fullseq = -np.sum(item_loss * ~item_padding_mask) / np.sum(~item_padding_mask)
        
        # Calculate log-likelihood excluding positions without coordinates
        coords_dict = coords_batch[i]
        target_chain_id = target_chain_id_batch[i]
        coord_mask = np.all(np.isfinite(coords_dict[target_chain_id]), axis=(-1, -2))
        
        if positions_to_score is not None:
            coord_mask = coord_mask[positions_to_score]
        
        ll_withcoord = -np.sum(item_loss * coord_mask) / np.sum(coord_mask)
        
        results.append(
            ScoringResult(logits=item_logits, ll_fullseq=ll_fullseq, ll_withcoord=ll_withcoord)
        )
    
    return results


def get_encoder_output_for_complex(model, alphabet, coords, target_chain_id):
    """
    Args:
        model: An instance of the GVPTransformer model
        alphabet: Alphabet for the model
        coords: Dictionary mapping chain ids to L x 3 x 3 array for N, CA, C
            coordinates representing the backbone of each chain
        target_chain_id: The chain id to sample sequences for
    Returns:
        Dictionary mapping chain id to encoder output for each chain
    """
    all_coords = _concatenate_coords(coords, target_chain_id)
    all_rep = get_encoder_output(model, alphabet, all_coords)
    target_chain_len = coords[target_chain_id].shape[0]
    return all_rep[:target_chain_len]
