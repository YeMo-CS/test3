#!/usr/bin/env bash

set -euo pipefail

ray stop --force

export NUM_GPUS=16
export RAY_memory_usage_threshold=0.99
export PYTHONUNBUFFERED=1
export TORCH_NCCL_BLOCKING_WAIT=1
export NCCL_TIMEOUT=7200
export TORCH_DISTRIBUTED_DEBUG=INFO
export TOKENIZERS_PARALLELISM=true
export HYDRA_FULL_ERROR=1
export NCCL_DEBUG=WARN

export PROJECT_PATH=/mnt/workspace/weight/mysh
export PROJECT_NAME="opd"

export ADV_ESTIMATOR="token_reward_direct"
export MAX_PROMPT_LENGTH=1024
export MAX_RESP_LENGTH=8192
export MAX_VAL_RESP_LENGTH=32768
export MINI_BATCH_SIZE=128
export DATA_SHUFFLE=True
export TEMPERATURE=1.0
export TEACHER_TEMPERATURE=1.0
export REPETITION_PENALTY=1.0
export N_RESPONSES=8
export LOG_PROB_TOP_K=16
export TOP_K_STRATEGY="only_stu"
export REWARD_WEIGHT_MODE="student_p"
export USE_KL=False
export ENABLE_FORMAT_REWARD=False
export MODEL_DTYPE=bfloat16
export IS_PLOT=False
export LOSS_AGG_MODE="seq-mean-token-mean"
export PARALLEL_SIZE=1
export TEST_FREQ=10
export SAVE_FREQ=10
export VAL_BEFORE_TRAIN=False

export PROMPT_FILTERING_ENABLE=True
export PROMPT_FILTERING_DROP_FRACTION=0.5
export ROLLOUT_LEVEL_WEIGHTING_ENABLE=True
export ROLLOUT_LEVEL_WEIGHTING_TEMPERATURE=1.5
export TOKEN_ENTROPY_WEIGHTING_ENABLE=True
export TOKEN_ENTROPY_WEIGHTING_WINDOW_SIZE=4

export TRAIN_DATASET=datasets/dapo-math-17k-processed.parquet
export TRAIN_DATASET_NAME="DAPO-Math-17k"
export TEST_DATASET=datasets/test_data/AIME24/test.parquet

export ACTOR_MODEL_PATH=/mnt/workspace/weight/qwen3_4b_base
export ACTOR_MODEL_NAME="$(basename "${ACTOR_MODEL_PATH%/}")"

export REWARD_MODEL_ENABLE=True
export REWARD_MODEL_PATH=/mnt/workspace/weight/qwen3_4b_grpo
export REWARD_MODEL_NAME="$(basename "${REWARD_MODEL_PATH%/}")"

export MAX_MODEL_LEN=$(( \
    MAX_RESP_LENGTH + MAX_PROMPT_LENGTH > MAX_VAL_RESP_LENGTH + MAX_PROMPT_LENGTH \
    ? MAX_RESP_LENGTH + MAX_PROMPT_LENGTH \
    : MAX_VAL_RESP_LENGTH + MAX_PROMPT_LENGTH \
))

export PPO_MAX_TOKEN_LEN_PER_GPU=$(( \
    MAX_PROMPT_LENGTH + MAX_RESP_LENGTH > 32768 \
    ? MAX_PROMPT_LENGTH + MAX_RESP_LENGTH \
    : 32768 \
))


export EXPERIMENT_NAME=opd-3L-retain${PROMPT_FILTERING_DROP_FRACTION}-t${ROLLOUT_LEVEL_WEIGHTING_TEMPERATURE}-win${TOKEN_ENTROPY_WEIGHTING_WINDOW_SIZE}-$(date +%Y-%m-%d_%H-%M-%S)
export CKPT_PATH=${PROJECT_PATH}/${PROJECT_NAME}/${EXPERIMENT_NAME}

export WANDB_MODE=offline
export WANDB_DIR="/mnt/workspace/weight/wandb/${PROJECT_NAME}/${EXPERIMENT_NAME}/wandb"
export WANDB_CACHE_DIR="/mnt/workspace/weight/wandb/${PROJECT_NAME}/${EXPERIMENT_NAME}/wandb-cache"
export WANDB_CONFIG_DIR="/mnt/workspace/weight/wandb/${PROJECT_NAME}/${EXPERIMENT_NAME}/wandb-config"
export WANDB_DATA_DIR="/mnt/workspace/weight/wandb/${PROJECT_NAME}/${EXPERIMENT_NAME}/wandb-data"


mkdir -p \
  "${WANDB_DIR}" \
  "${WANDB_CACHE_DIR}" \
  "${WANDB_CONFIG_DIR}" \
  "${WANDB_DATA_DIR}" \
  "${CKPT_PATH}" 


KL_ARGS=""
if [[ "${USE_KL}" == "True" ]]; then
    KL_ARGS="actor_rollout_ref.actor.use_kl_loss=True \
actor_rollout_ref.actor.kl_loss_coef=0.02 \
actor_rollout_ref.actor.kl_loss_type=low_var_kl"
else
    KL_ARGS="actor_rollout_ref.actor.use_kl_loss=False"
fi


ray start --head
sleep 5

cd /mnt/workspace/OPD/verl
python3 -m verl.trainer.main_ppo \
    +data.apply_chat_template_kwargs.enable_thinking=False \
    actor_rollout_ref.rollout.repetition_penalty="${REPETITION_PENALTY}" \
    actor_rollout_ref.rollout.temperature="${TEMPERATURE}" \
    +actor_rollout_ref.rollout.teacher_temperature="${TEACHER_TEMPERATURE}" \
    algorithm.adv_estimator="${ADV_ESTIMATOR}" \
    data.shuffle="${DATA_SHUFFLE}" \
    data.train_files="${TRAIN_DATASET}" \
    data.val_files="${TEST_DATASET}" \
    data.train_batch_size=$((MINI_BATCH_SIZE * PARALLEL_SIZE)) \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length=32768 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    data.return_raw_chat=True \
    actor_rollout_ref.model.path="${ACTOR_MODEL_PATH}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_activation_offload=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${MINI_BATCH_SIZE}" \
    actor_rollout_ref.actor.use_dynamic_bsz=True \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=1 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="${PARALLEL_SIZE}" \
    ${KL_ARGS} \
    actor_rollout_ref.actor.loss_agg_mode="${LOSS_AGG_MODE}" \
    actor_rollout_ref.actor.token_entropy_weighting_enable="${TOKEN_ENTROPY_WEIGHTING_ENABLE}" \
    actor_rollout_ref.actor.token_entropy_weighting_window_size="${TOKEN_ENTROPY_WEIGHTING_WINDOW_SIZE}" \
    trainer.rollout_level_weighting.enable="${ROLLOUT_LEVEL_WEIGHTING_ENABLE}" \
    trainer.rollout_level_weighting.temperature="${ROLLOUT_LEVEL_WEIGHTING_TEMPERATURE}" \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    actor_rollout_ref.actor.fsdp_config.model_dtype="${MODEL_DTYPE}" \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.model_dtype="${MODEL_DTYPE}" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=True \
    actor_rollout_ref.rollout.max_num_batched_tokens="${PPO_MAX_TOKEN_LEN_PER_GPU}" \
    +actor_rollout_ref.rollout.log_prob_top_k="${LOG_PROB_TOP_K}" \
    +actor_rollout_ref.rollout.top_k_strategy="${TOP_K_STRATEGY}" \
    +actor_rollout_ref.rollout.reward_weight_mode="${REWARD_WEIGHT_MODE}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${PARALLEL_SIZE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.max_model_len="${MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.n="${N_RESPONSES}" \
    actor_rollout_ref.rollout.val_kwargs.n=8 \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=1 \
    +actor_rollout_ref.rollout.val_kwargs.max_tokens="${MAX_VAL_RESP_LENGTH}" \
    +reward_model.reward_kwargs.enable_format_reward=$ENABLE_FORMAT_REWARD \
    reward_model.enable="${REWARD_MODEL_ENABLE}" \
    reward_model.model.path="${REWARD_MODEL_PATH}" \
    reward_model.model.input_tokenizer=null \
    reward_model.model.use_remove_padding=True \
    reward_model.model.fsdp_config.param_offload=False \
    +reward_model.model.dtype="${MODEL_DTYPE}" \
    reward_model.micro_batch_size_per_gpu=24 \
    custom_reward_function.path="./verl/verl/utils/reward_score/ttrl_math/__init__.py" \
    custom_reward_function.name=reward_func \
    trainer.prompt_filtering.enable="${PROMPT_FILTERING_ENABLE}" \
    trainer.prompt_filtering.drop_fraction="${PROMPT_FILTERING_DROP_FRACTION}" \
    trainer.val_before_train="${VAL_BEFORE_TRAIN}" \
    trainer.log_val_generations=2 \
    trainer.logger=['console','wandb'] \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.validation_data_dir=/mnt/workspace/weight/validation_log/${PROJECT_NAME}/$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=$NUM_GPUS \
    trainer.nnodes=1 \
    trainer.max_actor_ckpt_to_keep=1 \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs=1 \
    trainer.default_local_dir="${CKPT_PATH}" \
    trainer.is_plot="${IS_PLOT}"
