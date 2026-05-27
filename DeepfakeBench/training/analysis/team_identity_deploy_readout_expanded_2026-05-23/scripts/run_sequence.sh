#!/bin/bash
# After Slot A v2 CLS finishes, run the other 4 ckpts sequentially.
# Each ckpt logs to its own file.

cd /Users/roeedar/Documents/repos/Effort-AIGI-Detection-DtectVision/DeepfakeBench/training

OUT=analysis/team_identity_deploy_readout_expanded_2026-05-23
SCRIPT=$OUT/scripts/score_all_ckpts.py

# Order matters: face-pool depends on the CLS ckpt being loaded once; T5C, P8A, E2B are independent.
CKPTS=(
    "SLOT_A_ANCHOR_AWARE_STEP3500_FACE_POOL"
    "T5C_PERIODIC_STEP3500"
    "P8A_REFERENCE_STEP5000"
    "E2B_TOP_N_STEP3200"
)

for CK in "${CKPTS[@]}"; do
    LOGFILE=$OUT/_score_${CK}.log
    echo "[seq] Starting ${CK} at $(date)" | tee $LOGFILE
    python $SCRIPT --only $CK >> $LOGFILE 2>&1
    echo "[seq] Finished ${CK} at $(date)" | tee -a $LOGFILE
done
