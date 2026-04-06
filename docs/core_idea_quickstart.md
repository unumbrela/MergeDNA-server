# MergeDNA Core Idea Quickstart

This note is the shortest path to understanding MergeDNA from code.

## 1. One-sentence mental model

MergeDNA is a hierarchical autoencoder for DNA: it first learns a dynamic tokenizer that merges nearby bases into variable-length tokens, then models global context on the merged sequence, and finally reconstructs the original bases while using the merging outcome to drive adaptive masking during pre-training.

## 2. Read in this order

1. `scripts/core_idea_walkthrough.py`
2. `mergedna/model/mergedna.py`
3. `mergedna/model/local_encoder.py`
4. `mergedna/model/token_merging.py`
5. `mergedna/model/latent_encoder.py`
6. `mergedna/model/local_decoder.py`
7. `mergedna/training/losses.py`

## 3. Paper concept -> code location

- Local Encoder / dynamic tokenization:
  `mergedna/model/local_encoder.py`
- Local-window differentiable token merging:
  `mergedna/model/token_merging.py`
- Latent Encoder / global context modeling:
  `mergedna/model/latent_encoder.py`
- Latent Decoder + Local Decoder / reconstruction:
  `mergedna/model/latent_encoder.py`, `mergedna/model/local_decoder.py`
- MTR + latent MTR + AMTM:
  `mergedna/model/mergedna.py`, `mergedna/training/losses.py`

## 4. Core tensor flow

Assume input DNA has shape `[B, N]`.

1. Character tokenization:
   `DNACharTokenizer` maps `A/T/C/G/N` to ids.

2. Local Encoder:
   `input_ids [B, N] -> z_L [B, L, D]`
   `source [B, N, L]`

3. Latent Encoder:
   `z_L [B, L, D] -> z_L_prime [B, L, D]`

4. Latent Decoder:
   `z_L_prime [B, L, D] -> z_hat_L [B, L, D]`

5. Local Decoder:
   `token_unmerge(z_hat_L, source) -> z_N [B, N, D]`
   `output_head(z_N) -> logits [B, N, vocab]`

6. Extra pre-training branch:
   `z_L_prime -> global token merging -> z_K_prime [B, K, D]`
   `source_prime [B, K, L] -> adaptive mask -> mask_N [B, N]`

## 5. The four files that matter most

### A. `mergedna/model/mergedna.py`

This is the whole architecture assembly and the cleanest place to understand the big picture.

- `MergeDNA.__init__`: wires together Local Encoder, Latent Encoder, Latent Decoder, Local Decoder.
- `encode`: local tokenization + global context modeling.
- `decode`: latent decode + token unmerge + base reconstruction.
- `forward_pretrain`: the full paper idea in code.

The most important function is `forward_pretrain`, because it contains the three pre-training passes:

1. standard reconstruction `L_MTR`
2. latent reconstruction with Local Encoder frozen
3. adaptive masked token modeling `L_AMTM`

## 6. Where the dynamic tokenizer actually happens

### B. `mergedna/model/local_encoder.py`

The Local Encoder is not a static tokenizer. It is a stack of blocks:

- local-window self-attention
- token merging

The key logic is:

- initialize `source` as identity, meaning every original base starts as its own token
- after each layer, merge some tokens and update `source`
- after several layers, sequence length shrinks from `N` to `L`

This is the code realization of "dynamic tokenization".

## 7. Where merging is decided

### C. `mergedna/model/token_merging.py`

This file is the heart of the paper.

`LocalWindowTokenMerging.forward` does four things:

1. split the sequence into local windows
2. project tokens with `group_proj` to get a lightweight merge metric
3. score token similarity and pick merge pairs with bipartite matching
4. merge token representations and update `source`

Two details matter:

- merging is local, so token boundaries depend on local context
- `source` preserves which original bases were absorbed into each merged token

`GlobalTokenMerging.forward` is the pre-training-only global selection stage:

- it reduces `L` local tokens to `K` salient tokens
- it returns `source_prime`, which records how local tokens were grouped at the latent level
- `source_prime` later drives adaptive masking

## 8. Where long-range modeling happens

### D. `mergedna/model/latent_encoder.py`

This is the global context model.

- `LatentEncoder.forward`: standard full-attention Transformer over merged tokens
- `forward_with_selection`: same encoder output, then apply global token merging to keep only salient latent tokens
- `LatentDecoder`: lightweight decoder used during pre-training reconstruction

So the structure is:

`dynamic local segmentation -> global Transformer context -> compressed salient latent view`

## 9. How tokens return to base resolution

### E. `mergedna/model/local_decoder.py`

`token_unmerge` is the exact bridge back to nucleotide space.

It uses `source [B, N, L]` to map each merged token back to the original base positions:

`z_N = source @ z_hat_L`

Then `LocalDecoder.forward`:

1. unmerges token features
2. refines them with local attention
3. projects to nucleotide logits

This is why `source` is the most important bookkeeping tensor in the project.

## 10. Why AMTM is not ordinary MLM

### F. `mergedna/training/losses.py`

The paper's adaptive idea is implemented here.

- `compute_mtr_loss`: ordinary reconstruction loss
- `compute_adaptive_mask`: convert `source_prime` into an importance-aware mask
- `compute_amtm_loss`: compute loss only on masked important positions

The key mechanism is:

- large merged groups are treated as lower-information regions
- small groups or singleton groups get higher masking probability
- masking is pushed back from local token space to base space through `source`

So the model is not masking uniformly. It masks what the merge structure says is worth predicting.

## 11. Fastest way to run the whole idea

### Minimal walkthrough

```bash
python scripts/core_idea_walkthrough.py --sequence ATATATATATATATATACGTGCTAAGTCGGTAGGGGCCCCGGCCGCGC
```

This prints:

- local token lengths after dynamic merging
- latent token selection result
- adaptive mask positions
- the three pre-training losses

### Smoke test

```bash
python test_model.py
```

This verifies:

- tokenizer works
- pretrain forward works
- encode/decode modes work
- gradients flow through all major components

## 12. If you want to re-implement the paper yourself

Implement in this order:

1. char-level DNA tokenizer
2. local-window attention block
3. local token merging with a `source` matrix
4. Local Encoder as repeated `attention + merging`
5. full-attention Latent Encoder
6. token unmerge with `source`
7. Local Decoder reconstruction head
8. global token merging for latent selection
9. AMTM masking from `source_prime`
10. three-pass pre-training loop

Do not start from the full training runner. First get the following invariant correct:

- you can trace one sequence from `[B, N] -> [B, L, D] -> [B, N, vocab]`
- `source` always correctly maps original positions to merged tokens

Once that works, the rest is mostly training engineering.

## 13. The single most important insight

MergeDNA is not "a Transformer with a better tokenizer". The tokenizer itself is inside the model and optimized by reconstruction and masking objectives, so tokenization, compression, importance selection, and context modeling are learned as one system.
