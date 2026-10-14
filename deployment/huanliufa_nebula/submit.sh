#!/usr/bin/env bash
set -euo pipefail
set +x  # 不输出包含密钥的 shell trace
cd "$(dirname "$0")"

source ./config.sh
export WANDB_ENTITY WANDB_PROJECT WANDB_MODE WANDB_BASE_URL
case "$WANDB_MODE" in online|offline|disabled) ;; *) echo '无效的 WANDB_MODE' >&2; exit 1 ;; esac
OSS_MOUNT=/data/oss_bucket_0
if [[ -z ${FMC_AK:-} || -z ${FMC_SK:-} || -z ${FMC_ENDPOINT:-} ]]; then
  if [[ ! -r "$OSS_CREDS_FILE" ]]; then
    echo "无法读取 $OSS_CREDS_FILE；请设置 OSS_CREDS_FILE 或 FMC_AK、FMC_SK、FMC_ENDPOINT。" >&2
    exit 1
  fi
  source "$OSS_CREDS_FILE"
fi
: "${FMC_AK:?缺少 FMC_AK}"
: "${FMC_SK:?缺少 FMC_SK}"
: "${FMC_ENDPOINT:?缺少 FMC_ENDPOINT}"
# 平台要求生产 endpoint 域名，不带协议前缀。
MOUNT_ENDPOINT=${FMC_ENDPOINT#https://}
MOUNT_ENDPOINT=${MOUNT_ENDPOINT#http://}
MOUNT_ENDPOINT=${MOUNT_ENDPOINT%/}
if [[ -z "$MOUNT_ENDPOINT" || "$MOUNT_ENDPOINT" == *[/:,[:space:]]* || "$FMC_AK" == *,* || "$FMC_SK" == *,* ]]; then
  echo "请提供单个生产 endpoint 域名及对应凭据；本脚本只挂载 fmc。" >&2
  exit 1
fi
for flag in "$ENABLE_OSS_CACHE" "$DISABLE_OSS_META_CACHE" "$OSS_APPENDABLE"; do
  [[ "$flag" == true || "$flag" == false ]] || { echo '缓存开关必须为 true 或 false' >&2; exit 1; }
done
ENVS="PYTHONUNBUFFERED=1,OSS_MOUNT=$OSS_MOUNT,PROGRESS_SECONDS=$PROGRESS_SECONDS,ENV_ARCHIVE_REL=$ENV_ARCHIVE_REL,WANDB_ENTITY=$WANDB_ENTITY,WANDB_PROJECT=$WANDB_PROJECT,WANDB_MODE=$WANDB_MODE,WANDB_BASE_URL=$WANDB_BASE_URL"
for name in DATA_PATH GS_PYTHON; do
  value=${!name:-}
  if [[ -n $value ]]; then
    [[ $value == /* && $value != *[,:[:space:]]* ]] || { echo "$name 必须是 worker 可访问的绝对路径，不能含空白、逗号或冒号" >&2; exit 1; }
    ENVS+=",$name=$value"
  fi
done
ENTRY=train_node.py
USER_PARAMS="--run-name=$JOB_NAME"
if [[ -n ${EXPERIMENT_ID:-} ]]; then
  [[ $EXPERIMENT_ID == E01 || $EXPERIMENT_ID == E09 ]] || { echo '仅保留 E01 和 E09'; exit 1; }
  ENTRY=experiment_node.py
  if [[ ${DYNAMIC_UPDATE:-0} == 1 ]]; then
    ENTRY=dynamic_update_node.py
  fi
  USER_PARAMS="--experiment=$EXPERIMENT_ID --stop-after=${STOP_AFTER:-665000}"
  [[ -z ${RESUME_CHECKPOINT:-} ]] || USER_PARAMS+=" --resume=$RESUME_CHECKPOINT"
  [[ -z ${EXPERIMENT_OUTPUT:-} ]] || USER_PARAMS+=" --output=$EXPERIMENT_OUTPUT"
fi

ARGS=(nebulactl run mdl
  "--nebula_project=$NEBULA_PROJECT"
  "--queue=$QUEUE"
  "--entry=$ENTRY"
  "--algo_name=$ALGO_NAME"
  --worker_count=1
  "--user_params=$USER_PARAMS"
  "--file.cluster_file=$CLUSTER_FILE"
  "--job_name=$JOB_NAME"
  '--ignore=__pycache__/*,.git/*,tools/*'
  "--env=$ENVS"
  "--oss_access_id=$FMC_AK"
  "--oss_access_key=$FMC_SK"
  --oss_bucket=fmc
  "--oss_endpoint=$MOUNT_ENDPOINT"
  "--enable_oss_cache=$ENABLE_OSS_CACHE"
  "--disable_oss_meta_cache=$DISABLE_OSS_META_CACHE"
  "--oss_appendable=$OSS_APPENDABLE"
  --max_failover_times=0)

echo "队列: $QUEUE; 项目: $NEBULA_PROJECT; 单 worker / 单 GPU"
echo "任务挂载: oss://fmc/ → $OSS_MOUNT；读取缓存: $ENABLE_OSS_CACHE"
if [[ -n ${EXPERIMENT_ID:-} ]]; then
  echo "实验: $EXPERIMENT_ID；本阶段停止: ${STOP_AFTER:-665000}；输出由实验 JSON 的 output_relative 指定"
else
  echo "输出: oss://fmc/beifen/data/3dgs_bushu/$JOB_NAME/"
fi
if [[ ${1:-} == --dry-run ]]; then
  for arg in "${ARGS[@]}"; do
    case "$arg" in
      --oss_access_id=*) printf '%s ' '--oss_access_id=<redacted>' ;;
      --oss_access_key=*) printf '%s ' '--oss_access_key=<redacted>' ;;
      *) printf '%q ' "$arg" ;;
    esac
  done
  printf '\n'
  exit 0
fi
command -v nebulactl >/dev/null
python3 ./submit_with_auth.py "${ARGS[@]}"
