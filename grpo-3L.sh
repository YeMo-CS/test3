#!/usr/bin/env bash
set -x
ray stop --force

NNODES=1
GPUS=8

export PROMPT_FILTERING_ENABLE=True
export PROMPT_FILTERING_DROP_FRACTION=0.5
export ROLLOUT_LEVEL_WEIGHTING_ENABLE=True
export ROLLOUT_LEVEL_WEIGHTING_TEMPERATURE=1.5
export TOKEN_ENTROPY_WEIGHTING_ENABLE=True
export TOKEN_ENTROPY_WEIGHTING_WINDOW_SIZE=4

export PROJECT_NAME='grpo' 
export EXPERIMENT_NAME=grpo-3L-retain${PROMPT_FILTERING_DROP_FRACTION}-t${ROLLOUT_LEVEL_WEIGHTING_TEMPERATURE}-win${TOKEN_ENTROPY_WEIGHTING_WINDOW_SIZE}-$(date +%Y-%m-%d_%H-%M-%S)
export MAX_VAL_RESP_LENGTH=8192 
export PROJECT_PATH=/mnt/workspace/weight/mysh
export CKPT_PATH=${PROJECT_PATH}/${PROJECT_NAME}/${EXPERIMENT_NAME}

cd /mnt/workspace/OPD/verl
export WANDB_MODE=offline
export WANDB_DIR="/mnt/workspace/weight/wandb/${PROJECT_NAME}/${EXPERIMENT_NAME}/wandb"
export WANDB_CACHE_DIR="/mnt/workspace/weight/wandb/${PROJECT_NAME}/${EXPERIMENT_NAME}/wandb-cache"
export WANDB_CONFIG_DIR="/mnt/workspace/weight/wandb/${PROJECT_NAME}/${EXPERIMENT_NAME}/wandb-config"
export WANDB_DATA_DIR="/mnt/workspace/weight/wandb/${PROJECT_NAME}/${EXPERIMENT_NAME}/wandb-data"
export WANDB_ARTIFACT_DIR="/mnt/workspace/weight/wandb/${PROJECT_NAME}/${EXPERIMENT_NAME}/wandb-artifacts"

mkdir -p \
  "${WANDB_DIR}" \
  "${WANDB_CACHE_DIR}" \
  "${WANDB_CONFIG_DIR}" \
  "${WANDB_DATA_DIR}" \
  "${CKPT_PATH}" 

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=./datasets/dapo-math-17k-processed.parquet \
    data.val_files=./datasets/test_data/AIME24/test.parquet \
    data.train_batch_size=64 \
    data.max_prompt_length=1024 \
    data.max_response_length=8192 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=/mnt/workspace/weight/qwen3_4b_base \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=1e-4 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.token_entropy_weighting_enable="${TOKEN_ENTROPY_WEIGHTING_ENABLE}" \
    actor_rollout_ref.actor.token_entropy_weighting_window_size="${TOKEN_ENTROPY_WEIGHTING_WINDOW_SIZE}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.loss_agg_mode=seq-mean-token-mean \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    +actor_rollout_ref.rollout.val_kwargs.max_tokens=$MAX_VAL_RESP_LENGTH \
    custom_reward_function.path="./verl/verl/utils/reward_score/ttrl_math/__init__.py" \
    custom_reward_function.name=reward_func \
    algorithm.use_kl_in_reward=False \
    trainer.prompt_filtering.enable="${PROMPT_FILTERING_ENABLE}" \
    trainer.prompt_filtering.drop_fraction="${PROMPT_FILTERING_DROP_FRACTION}" \
    trainer.rollout_level_weighting.enable="${ROLLOUT_LEVEL_WEIGHTING_ENABLE}" \
    trainer.rollout_level_weighting.temperature="${ROLLOUT_LEVEL_WEIGHTING_TEMPERATURE}" \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=${PROJECT_NAME} \
    trainer.experiment_name=${EXPERIMENT_NAME} \
    trainer.default_local_dir=${PROJECT_PATH}/${PROJECT_NAME}/${EXPERIMENT_NAME} \
    trainer.val_before_train=False \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.n_gpus_per_node=${GPUS} \
    trainer.nnodes=${NNODES} \
    trainer.save_freq=10 \
    trainer.test_freq=10 \
    trainer.log_val_generations=2 \
    trainer.validation_data_dir=/mnt/workspace/weight/validation_log/${PROJECT_NAME}/$EXPERIMENT_NAME \
    trainer.total_epochs=1
