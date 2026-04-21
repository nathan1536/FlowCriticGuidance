## 📋 Table of Contents
- [Installation](#️-installation)


## 🛠️ Installation
Please follow the detailed instructions in [INSTALL.md](INSTALL.md) to set up the environment and install dependencies.

## Code file I used for Adroit Door critic guidance

- scripts/rollout_sac_adroit_to_zarr.py

- scripts/train_sac_adroit.py

- scripts/train_chunked_critic.py

- ManiFlow/maniflow/workspace/train_maniflow_dex_workspace.py

- ManiFlow/maniflow/policy/maniflow_image_policy.py

- ManiFlow/maniflow/env/adroit

- ManiFlow/maniflow/config/maniflow_image_timm_policy_dex.yaml

## Code scripts for mcr-metaworld-flowql

1. Train the SAC agent for metaworld

- scripts/train_sac_metaworld.py

- ManiFlow/maniflow/env/metaworld/sb3_metaworld_state_env.py

2. MCR Precomputing

- scripts/precompute_mcr_embeddings.py

3. Downstream Training

- ManiFlow/maniflow/workspace/train_flowql_metaworld_workspace.py

- ManiFlow/maniflow/policy/mcr_flow_policy.py

- ManiFlow/maniflow/policy/maniflow_state_policy.py