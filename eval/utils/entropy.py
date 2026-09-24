"""Training-consistent actor/entropy computation for evaluation.

This is the single source of truth for the "actor/entropy" metric so that both
the online eval path (eval.py) and the offline recompute path
(recompute_entropy.py) use an identical definition, matching the training-time
`actor/entropy` metric and the held-out probe (heldout/eval_heldout_entropy.py).

Definition (identical to verl dp_actor):
  - encode prompt and response separately (add_special_tokens=False) and concat;
  - HF forward (use_cache=False) to get full-vocab logits;
  - take the logits that predict the response tokens;
  - if temperature > 0, divide logits by temperature (as dp_actor does);
  - per-token entropy = logsumexp(logits) - sum(softmax(logits) * logits);
  - response-level entropy = token-mean over the response tokens.

Dataset-level `actor_entropy_token_mean` = sum of all response-token entropies
divided by the total number of response tokens (a token-mean over all tokens,
equivalent to the training token-mean).
"""

from __future__ import annotations

import torch


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Identical to verl.utils.torch_functional.entropy_from_logits."""
    pd = torch.nn.functional.softmax(logits, dim=-1)
    return torch.logsumexp(logits, dim=-1) - torch.sum(pd * logits, dim=-1)


@torch.no_grad()
def response_token_entropy(model, tokenizer, prompt_text, response_text, temperature, device,
                           chunk_size=1024):
    """Recompute training-consistent per-token actor/entropy for one response.

    Returns (entropy_tensor, response_len). entropy_tensor is None and
    response_len is 0 when the response encodes to zero tokens.

    ``chunk_size`` bounds peak memory: the response-token logits are turned into
    per-token entropy in slices of at most ``chunk_size`` positions, so we never
    materialize the full (resp_len, vocab) float tensor at once. This matters for
    long responses (max_tokens up to 8192) over a large vocabulary.
    """
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids
    resp_ids = tokenizer(response_text, return_tensors="pt", add_special_tokens=False).input_ids
    if resp_ids.shape[1] == 0:
        return None, 0
    input_ids = torch.cat([prompt_ids, resp_ids], dim=1).to(device)
    prompt_len = prompt_ids.shape[1]
    resp_len = resp_ids.shape[1]

    out = model(input_ids=input_ids, use_cache=False)
    logits = out.logits  # (1, seq, vocab)
    # logits at [prompt_len-1 : prompt_len-1+resp_len] predict the response tokens
    resp_logits = logits[:, prompt_len - 1 : prompt_len - 1 + resp_len, :].squeeze(0)  # (resp_len, vocab)

    # compute entropy in chunks to bound peak memory (fp32 upcast per chunk only)
    ent_chunks = []
    for start in range(0, resp_len, chunk_size):
        chunk = resp_logits[start : start + chunk_size]
        if temperature and temperature > 0:
            chunk = chunk / temperature
        ent_chunks.append(entropy_from_logits(chunk.float()).to("cpu"))
    ent = torch.cat(ent_chunks, dim=0) if ent_chunks else None  # (resp_len,) on CPU
    # free the big logits tensor before returning
    del out, logits, resp_logits
    return ent, resp_len


def compute_entropy_for_file_outputs(model, tokenizer, file_outputs, temperature, device,
                                     prompt_key="prompt_text", progress_desc=None):
    """Fill per-response actor_entropy_mean / response_len in-place on file_outputs.

    Each item must provide ``prompt_key`` (the rendered prompt string) and
    ``generated_responses`` (list of response strings). For every response a
    parallel list ``actor_entropy_mean`` and ``response_len`` is written back on
    the item.

    Returns (total_ent_sum, total_tok) so callers can compute the dataset-level
    token-mean entropy: total_ent_sum / total_tok.
    """
    iterable = file_outputs
    if progress_desc is not None:
        try:
            from tqdm import tqdm
            iterable = tqdm(file_outputs, desc=progress_desc)
        except Exception:
            iterable = file_outputs

    total_ent_sum = 0.0
    total_tok = 0
    for item in iterable:
        prompt_text = item.get(prompt_key)
        responses = item.get("generated_responses", []) or []
        entropy_means = []
        response_lens = []
        for response in responses:
            if prompt_text is None:
                entropy_means.append(None)
                response_lens.append(0)
                continue
            ent, rlen = response_token_entropy(
                model, tokenizer, prompt_text, response, temperature, device
            )
            if ent is not None and rlen > 0:
                entropy_means.append(float(ent.mean().item()))
                response_lens.append(int(rlen))
                total_ent_sum += float(ent.sum().item())
                total_tok += int(rlen)
            else:
                entropy_means.append(None)
                response_lens.append(0)
        item["actor_entropy_mean"] = entropy_means
        item["response_len"] = response_lens

    return total_ent_sum, total_tok
