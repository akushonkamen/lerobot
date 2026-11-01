# Verification for huggingface/lerobot#5184 (lerobot/FineART dataset, revision 135d9b0)
#
# Checks whether observation.state[t] equals action[t-1] exactly per episode.
#
# Usage:
#   curl -sL --max-time 240 -o fineart_f0.parquet \
#     'https://hf-mirror.com/datasets/lerobot/FineART/resolve/135d9b0377a629058541c37ad1503fd4c2597848/data/chunk-000/file-000.parquet'
#   python notes/issue-5184-fineart/verify_state_action_lag.py fineart_f0.parquet
#
# Exit code 0 = lag confirmed (state[t] == action[t-1] everywhere), 1 = not confirmed, 2 = error.
import sys

import numpy as np
import pandas as pd


def main(path: str) -> int:
    df = pd.read_parquet(path, columns=["episode_index", "observation.state", "action"])
    s = np.stack(df["observation.state"].values)
    a = np.stack(df["action"].values)
    ep = df["episode_index"].values
    print(f"rows={len(df)} episodes={len(set(ep))} state_dim={s.shape[1]} action_dim={a.shape[1]}")

    max_lag = 0.0
    n_checked = 0
    for e in sorted(set(ep)):
        m = ep == e
        se, ae = s[m], a[m]
        if len(se) < 2:
            continue
        d = np.abs(se[1:] - ae[:-1]).max()
        max_lag = max(max_lag, float(d))
        n_checked += 1
    print(f"episodes_checked={n_checked} max|state[t]-action[t-1]|={max_lag}")
    return 0 if max_lag == 0.0 else 1


if __name__ == "__main__":
    try:
        sys.exit(main(sys.argv[1]))
    except Exception as exc:  # noqa: BLE001
        print(f"error: {exc}", file=sys.stderr)
        sys.exit(2)
