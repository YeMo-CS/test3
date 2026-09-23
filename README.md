# ReAlloc Code

ReAlloc operates at the prompt, rollout and token levels and supports both Group Relative Policy Optimization (GRPO) and On-Policy Distillation (OPD). Each level can be enabled independently or combined with either of the other levels.

## Environment Setup

Our code is mainly based on verl (v0.7.0). To prepare the environment used for OPD and RL:

```bash
conda create -n verl python==3.12
conda activate verl
cd verl/
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
pip install math-verify
```

## Method

### Prompt-level weighting

For every prompt, we measure the variance of its binary outcome rewards across sampled rollouts. Within each PPO mini-batch, prompts with the lowest reward variance are assigned zero optimization weight, while the retained prompts are renormalized to preserve the original loss scale. 

### Rollout-level weighting

We divide each response into eight relative-position blocks and mean-pool its hidden states to construct a rollout representation. A global centroid is formed from correct rollouts, and incorrect rollouts are weighted according to their cosine distance from this centroid. The weights are normalized among the eligible incorrect rollouts of each prompt. Correct rollouts keep unit weight, and prompts removed by prompt-level filtering are excluded from rollout-level weight estimation.

### Token-level weighting

We use current-policy token entropy to identify local uncertainty peaks. For each valid response token, the method compares its entropy with neighboring tokens on both sides and assigns larger weights to bidirectional local peaks. Token weights are normalized within each response and converted to mean-one loss multipliers, preserving the scale of the original token-mean objective.

### Joint use

The three components are compatible and can be enabled separately, in pairs, or together. Prompt filtering determines which prompt groups contribute to optimization, rollout weighting adjusts the relative contribution of individual responses, and token weighting redistributes the loss within each response.

## Code

The main implementation is located in:

- `verl/verl/trainer/ppo/prompt_filtering.py`
- `verl/verl/trainer/ppo/rollout_level_weighting.py`
- `verl/verl/trainer/ppo/token_entropy_weighting.py`
- `verl/verl/trainer/ppo/ray_trainer.py`
- `verl/verl/workers/actor/dp_actor.py`

The default configuration is defined in

- `verl/verl/trainer/config/ppo_trainer.yaml`.

Reference launch scripts are provided for both supported training algorithms:

```bash
bash grpo-3L.sh
bash opd-3L.sh
```

## Acknowledgements

This codebase is built upon [VERL](https://github.com/verl-project/verl), an open-source reinforcement learning framework for large language models, and the public [OPD](https://github.com/thunlp/OPD) implementation for on-policy distillation. 
We sincerely thank the authors and contributors of these projects for releasing their valuable code. Their work provided the training infrastructure on which this implementation is developed.
