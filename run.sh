# #!/bin/bash

MODE=${1}
EXPERIMENT=${2}
DATASET=${3}
MISSING_PERC=${4}

DIR=./
. ${DIR}venv/bin/activate

if [[ "${EXPERIMENT}" == "MISSING_CLICKS" ]]; then
  if [[ "${DATASET}" != "SB20" ]]; then
    echo "ERROR: MISSING CLICKS EXPERIMENTS NOT IMPLEMENTED FOR CN20. QUITTING..."
    exit
  fi
  echo "Running experiment with missing input clicks..."
  CONFIG=${DIR}configs/SB20_missing_clicks_${MISSING_PERC}.yaml
  if [[ "${MODE}" == "EVAL" ]]; then
    MODEL=${DIR}results/paper_models/SB20_missing_clicks_${MISSING_PERC}_0499.ckpt
  fi
elif [[ "${EXPERIMENT}" == "BASIC" ]]; then
  echo "Running experiment on basic panoptic one-click segmentation..."
  CONFIG=${DIR}configs/${DATASET}_basic.yaml
  if [[ "${MODE}" == "EVAL" ]]; then
    MODEL=${DIR}results/paper_models/${DATASET}_basic_0499.ckpt
  fi
else
  echo "ERROR: NEED TO CHOOSE VALID EXPERIMENT. QUITTING..."
  exit
fi

if [[ "${MODE}" == "TRAIN" ]]; then
  echo "Running training..."
  python run_train.py --config ${CONFIG}
elif [[ "${MODE}" == "EVAL" ]]; then
  echo "Running evaluation..."
  python run_eval.py --config ${CONFIG} --model ${MODEL}
else
  echo "ERROR: NEED TO CHOOSE VALID MODE. QUITTING..."
  exit
fi

