#!/usr/bin/env bash
set -euo pipefail
set +x
cd "$(dirname "$0")"
source ./config.sh
source "$OSS_CREDS_FILE"
: "${FMC_AK:?}" "${FMC_SK:?}" "${FMC_ENDPOINT:?}"
endpoint=${FMC_ENDPOINT#https://}
endpoint=${endpoint#http://}
endpoint=${endpoint%/}
run_name=${RUN_NAME:-sam3_full_$(date +%Y%m%d_%H%M%S)}
status_dir=/data/oss_bucket_0/beifen/data/3dgs_bushu/preprocessing/$run_name
if [[ ${1:-} == --dry-run ]]; then
  echo "项目=$NEBULA_PROJECT 队列=$QUEUE；1 个任务，$SHARDS 个 worker，每 worker 1 卡"
  echo "每卡 batch size=$BATCH_SIZE"
  echo "保存: OSS SDK，每 worker $UPLOAD_WORKERS 个上传线程，队列上限 $UPLOAD_QUEUE 张"
  echo "模型=$SAM3_CHECKPOINT"
  echo "数据=$DATA_PATH；写回 mask_sam3/ 和 semantic_weights/，跳过已有文件"
  echo "状态=$status_dir；分片 i 处理排序图像列表的 [i::$SHARDS]"
  exit 0
fi
[[ -f "$SAM3_SOURCE/model_builder.py" ]] || { echo "缺少 SAM3 源码: $SAM3_SOURCE"; exit 1; }
stage=$(mktemp -d /tmp/sam3_submit_XXXXXX)
trap 'rm -rf "$stage"' EXIT
cp worker.py generate.py output_store.py cluster.json "$stage/"
cp -r "$SAM3_SOURCE" "$stage/sam3"
mkdir -p logs
logs_dir=$PWD/logs
cd "$stage"
  echo "提交 1 个任务，$SHARDS 个单卡 worker"
  nebulactl run mdl --nebula_project="$NEBULA_PROJECT" --queue="$QUEUE" \
    --entry=worker.py --algo_name="$ALGO_NAME" --worker_count="$SHARDS" \
    --user_params="--shards=$SHARDS --batch-size=$BATCH_SIZE --output-backend=oss --upload-workers=$UPLOAD_WORKERS --upload-queue=$UPLOAD_QUEUE" \
    --file.cluster_file=./cluster.json --job_name="$run_name" \
    --env="PYTHONUNBUFFERED=1,DATA_PATH=$DATA_PATH,SAM3_CHECKPOINT=$SAM3_CHECKPOINT,STATUS_DIR=$status_dir,OSS_CREDS_FILE=$OSS_CREDS_FILE" \
    --oss_access_id="$FMC_AK" --oss_access_key="$FMC_SK" --oss_bucket=fmc \
    --oss_endpoint="$endpoint" --enable_oss_cache=true --disable_oss_meta_cache=true \
    --max_failover_times=0 '--ignore=**/__pycache__/*,**/*.pyc' \
    2>&1 | tee "$logs_dir/${run_name}.log"
echo "1 个任务已提交，共 $SHARDS 个分片。状态目录: $status_dir"
