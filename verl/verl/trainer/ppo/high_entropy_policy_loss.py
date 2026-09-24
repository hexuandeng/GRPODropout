# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""High-entropy minority-token policy loss.

Reproduction of the policy-gradient masking proposed in
"Beyond the 80/20 Rule: High-Entropy Minority Tokens Drive Effective
Reinforcement Learning for LLM Reasoning" (arXiv:2506.01939, NeurIPS 2025).

Idea: only the small fraction of high-entropy "forking" tokens drive
effective RLVR. We keep the standard (dual-clip) PPO/GRPO objective, but
restrict the policy-gradient update to the top ``top_entropy_quantile``
fraction of tokens by per-token entropy. Low-entropy tokens receive no
gradient.

This module ONLY ADDS a new registered policy loss (``high_entropy``). It does
not modify any existing loss. Importing it is enough to register the function.
"""

from typing import Optional

import torch

import verl.utils.torch_functional as verl_F
from verl.trainer.ppo.core_algos import register_policy_loss


@register_policy_loss("high_entropy")
def compute_policy_loss_high_entropy(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    entropy: Optional[torch.Tensor] = None,
    loss_agg_mode: str = "token-mean",
    config=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Dual-clip PPO objective restricted to high-entropy minority tokens.

    Mirrors ``compute_policy_loss`` (the ``vanilla`` loss) exactly, except the
    final aggregation mask keeps only the top ``top_entropy_quantile`` fraction
    of tokens by per-token entropy.

    Args:
        old_log_prob (torch.Tensor): log-prob under the old policy, (bsz, resp_len).
        log_prob (torch.Tensor): log-prob under the current policy, (bsz, resp_len).
        advantages (torch.Tensor): advantage estimates, (bsz, resp_len).
        response_mask (torch.Tensor): valid-token mask, (bsz, resp_len).
        entropy (torch.Tensor): per-token entropy, (bsz, resp_len). Required;
            when None the loss falls back to the full ``response_mask``.
        loss_agg_mode (str): aggregation mode passed to ``agg_loss``.
        config: actor config; reads ``config.policy_loss.top_entropy_quantile``
            (fraction of highest-entropy tokens to keep, default 0.2) and the
            clip-ratio fields (mirrors the vanilla loss).

    Returns:
        (pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower): identical 4-tuple
        contract as every other registered policy loss, so the downstream
        metric-logging code in ``dp_actor`` needs no change.
    """
    # --- clip-ratio config, mirroring the vanilla compute_policy_loss path ---
    cliprange = config.clip_ratio
    cliprange_low = config.clip_ratio_low if config.clip_ratio_low is not None else cliprange
    cliprange_high = config.clip_ratio_high if config.clip_ratio_high is not None else cliprange
    clip_ratio_c = config.get("clip_ratio_c", 3.0)
    top_entropy_quantile = config.policy_loss.get("top_entropy_quantile", 0.2)

    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    # --- standard dual-clip PPO surrogate (identical to vanilla loss) ---
    negative_approx_kl = log_prob - old_log_prob
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_pg_losses1 = torch.maximum(pg_losses1, pg_losses2)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    # --- high-entropy minority-token mask (the only addition) ---
    # NOTE: top-k selection is computed per micro-batch (local), pooled over all
    # valid tokens, exactly matching the official get_global_entropy_top_mask.
    # This avoids any cross-rank communication.
    loss_mask = _high_entropy_mask(entropy, response_mask, top_entropy_quantile)

    # pg_clipfrac / ppo_kl are reported over the full response_mask (same
    # semantics as the vanilla loss) so they stay comparable across runs; only
    # the optimized objective is restricted to high-entropy tokens.
    #
    # Aggregation safety: with the high-entropy mask, an entire sequence (row)
    # may contain zero kept tokens. The shared agg_loss() divides by the
    # per-row mask sum under the "seq-mean-token-mean" mode, which would be a
    # 0/0 -> NaN for such rows. We therefore aggregate locally here (a faithful
    # re-implementation of agg_loss) with a guarded denominator, instead of
    # modifying the shared agg_loss() used by every other loss.
    pg_loss = _agg_loss_high_entropy(pg_losses, loss_mask, loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


def _agg_loss_high_entropy(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_agg_mode: str,
) -> torch.Tensor:
    """Aggregate the masked loss, mirroring core_algos.agg_loss but guarding
    against empty rows produced by the high-entropy mask.

    This is intentionally a local copy (not a modification of the shared
    agg_loss) so the high-entropy path cannot affect any other loss. The only
    behavioural difference vs agg_loss is that per-row token-mean denominators
    are clamped to >= 1, turning would-be NaN rows (no kept tokens) into a 0
    contribution instead.
    """
    if loss_agg_mode == "token-mean":
        # global masked-mean; safe as long as the whole batch has >=1 kept token
        return verl_F.masked_mean(loss_mat, loss_mask)
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        return torch.mean(seq_losses)
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_token_sum = torch.sum(loss_mat * loss_mask, dim=-1)
        seq_token_cnt = torch.sum(loss_mask, dim=-1)
        # guard: rows with no kept high-entropy tokens contribute 0, not NaN
        seq_losses = seq_token_sum / seq_token_cnt.clamp(min=1.0)
        return torch.mean(seq_losses)
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        return torch.sum(seq_losses) / loss_mask.shape[-1]
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")


def _high_entropy_mask(
    entropy: Optional[torch.Tensor],
    response_mask: torch.Tensor,
    top_entropy_quantile: float,
) -> torch.Tensor:
    """Return a binary mask selecting the top-``top_ratio`` fraction of tokens by
    entropy, pooled over all valid tokens of the (local micro-)batch.

    Faithful re-implementation of the official
    ``get_global_entropy_top_mask`` from
    ``Shenzhi-Wang/Beyond-the-80-20-Rule-RLVR`` (verl recipe
    ``rlvr_with_high_entropy_tokens_only``):

      - flatten entropy + response_mask, keep only valid tokens
      - count = max(1, ceil(N * top_ratio))  (exact top-k count, not a quantile)
      - select via ``torch.topk`` over the pooled valid tokens
      - scatter the selected positions back into a [B, S] mask

    "global" in the upstream name means "across all tokens of the batch"
    (NOT a cross-rank all_gather); the threshold is computed locally on the
    micro-batch, exactly as upstream does (no distributed reduction).

    Falls back to ``response_mask`` when entropy is unavailable or the ratio is
    >= 1.0 (degenerates to plain GRPO over all valid tokens).
    """
    if entropy is None or top_entropy_quantile >= 1.0:
        return response_mask

    flat_entropy = entropy.flatten()
    flat_mask = response_mask.flatten().bool()
    response_entropy = flat_entropy[flat_mask]

    out = torch.zeros_like(entropy)
    if response_entropy.numel() == 0:
        return out

    # exact top-k count with a ceiling, mirroring upstream:
    #   top_k = max(1, int(N * top_ratio + 0.9999))
    n_valid = response_entropy.numel()
    top_k = max(1, int(n_valid * top_entropy_quantile + 0.9999))
    top_k = min(top_k, n_valid)

    _, topk_idx = torch.topk(response_entropy, k=top_k)

    # map the selected (within-valid) indices back to flat [B*S] positions
    response_positions = flat_mask.nonzero(as_tuple=False).squeeze(1)
    top_positions = response_positions[topk_idx]

    flat_out = out.flatten()
    flat_out[top_positions] = 1
    return flat_out.view_as(entropy)
