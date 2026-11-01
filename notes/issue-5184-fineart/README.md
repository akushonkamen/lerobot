# FineART — one-frame state/action lag (lerobot#5184)

Dataset: [`lerobot/FineART`](https://huggingface.co/datasets/lerobot/FineART), revision `135d9b0377a629058541c37ad1503fd4c2597848`.

## Finding

`observation.state[t]` is **exactly equal** to `action[t-1]` in every frame, all 14 dims.
The two streams are not independent measurements: state appears to be back-filled from the
previous action/command (or action is `state[t+1]`), so state-conditioned training on this
revision has a one-frame action leakage.

Verified on `data/chunk-000/file-000.parquet` (115,974,783 bytes; 920,430 rows, 242 episodes):

```
$ python notes/issue-5184-fineart/verify_state_action_lag.py fineart_f0.parquet
rows=920430 episodes=242 state_dim=14 action_dim=14
episodes_checked=242 max|state[t]-action[t-1]|=0.0
```

Same-frame contrast (rules out duplicated columns): `max|state[t]-action[t]| = 0.204`.

## Proposed dataset-card note (for the Hub dataset card, not the lerobot codebase)

> **⚠️ Known issue (one-frame state/action lag):** in this revision, `observation.state[t]`
> is exactly equal to `action[t-1]` at every step (verified on `chunk-000/file-000.parquet`,
> 242/242 episodes, max|Δ|=0.0). The state and action streams are not independent
> measurements. If you train state-conditioned policies on this data, be aware that the
> state observation carries no information beyond the previous action — effectively a
> one-frame leakage of the command history. A re-export fixing the recording script's
> state write path is needed.

## AI assistance disclosure

Per [AI_POLICY.md](../../AI_POLICY.md), we disclose that this investigation and the
verification script / dataset-card note here were prepared with significant assistance
from an AI coding agent, reviewed by the human contributor
(@akushonkamen, who ran the reproduction and reviewed every file). This disclosure is
also in the issue comment and must be carried into the PR description when the
Hub-side dataset-card PR is opened.

## Status

- Code-side fix: none applicable — this is dataset content, not lerobot code.
- Hub-side action: dataset-card note above is ready to open as a community dataset PR;
  not yet submitted because huggingface.co was unreachable from this machine during the
  session (mirror `hf-mirror.com` was used for download). Ref: comment
  https://github.com/huggingface/lerobot/issues/5184#issuecomment-6561410088
