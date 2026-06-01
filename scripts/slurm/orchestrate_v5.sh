#!/bin/bash
# Wait for the v4 + data2 tokenize jobs to finish, then merge+split and submit
# training. Designed to run on the SCITAS login node via nohup; survives SSH
# disconnects. Logs everything to /path/to/home/logs/orchestrate_v5.log.
set -uo pipefail

JOB_V4=3450149
JOB_D2=3450155
NANO_DIR=/path/to/nano4M-Audio
NANOFM_DIR=/path/to/nanofm
LOG=/path/to/home/logs/orchestrate_v5.log
> "$LOG"

log() { echo "$(date +%H:%M:%S) $*" | tee -a "$LOG"; }

log "== orchestrator start =="
log "waiting for v4($JOB_V4) + data2($JOB_D2) ..."

# 1. Poll until both jobs are out of the queue
while true; do
  in_queue=$(squeue -h -j $JOB_V4,$JOB_D2 2>/dev/null | wc -l)
  if [ "$in_queue" -eq 0 ]; then break; fi
  sleep 60
done
log "both jobs left the queue"

# 2. Verify both completed cleanly
ST_V4=$(sacct -j $JOB_V4 -X --noheader -o State 2>/dev/null | head -1 | tr -d ' ')
ST_D2=$(sacct -j $JOB_D2 -X --noheader -o State 2>/dev/null | head -1 | tr -d ' ')
log "v4 state=$ST_V4   data2 state=$ST_D2"
if [ "$ST_V4" != "COMPLETED" ] || [ "$ST_D2" != "COMPLETED" ]; then
  log "ABORT: one or both tokenize jobs did not complete cleanly"
  exit 2
fi

# 3. Sanity counts before merge
log "v4 audio files: $(find $NANO_DIR/data/tokenized_v4 -path '*tok_audio@512*' -name '*.npy' 2>/dev/null | wc -l)"
log "data2 audio files: $(find $NANO_DIR/data/tokenized_data2 -path '*tok_audio@512*' -name '*.npy' 2>/dev/null | wc -l)"

# 4. Merge + split + audio cb2 offset
source /work/com-304/new_environment/anaconda3/etc/profile.d/conda.sh
conda activate nanofm
log "== merge + split =="
mkdir -p $NANO_DIR/data/tokenized_v5
python $NANO_DIR/scripts_v4/v5_merge_and_split.py \
  --sources $NANO_DIR/data/tokenized_v4 $NANO_DIR/data/tokenized_data2 \
  --out_root $NANO_DIR/data/tokenized_v5 \
  --report $NANO_DIR/data/tokenized_v5/splits.json \
  >> "$LOG" 2>&1
MERGE_RC=$?
log "merge exit=$MERGE_RC"
if [ $MERGE_RC -ne 0 ]; then log "ABORT: merge failed"; exit 3; fi

# 5. Post-merge counts
for sp in train val test; do
  log "[$sp] stems = $(ls $NANO_DIR/data/tokenized_v5/$sp/tok_audio@512 2>/dev/null | wc -l)"
done

# 6. Submit training
log "== sbatch training =="
cd $NANOFM_DIR
SBOUT=$(sbatch sbatch_train_v5.sh 2>&1)
echo "$SBOUT" >> "$LOG"
TRAIN_JOB=$(echo "$SBOUT" | grep -oE 'Submitted batch job [0-9]+' | grep -oE '[0-9]+')
log "training job submitted: $TRAIN_JOB"
echo $TRAIN_JOB > /path/to/home/logs/train_jobid.txt
log "== orchestrator done =="
