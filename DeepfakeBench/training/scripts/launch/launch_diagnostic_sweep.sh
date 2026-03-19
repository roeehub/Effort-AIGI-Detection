#!/bin/bash
# =============================================================================
# DIAGNOSTIC SWEEP LAUNCHER
# =============================================================================
# Launch all diagnostic experiments to identify the bottleneck in SVD training:
#
# Set A: LR Confound Elimination (isolate LR vs capacity effect)
#   A1: k=1 @ LR=1e-4 (same LR as k>=4)
#   A2: k=8 @ LR=2e-4 (same LR as k=1,2)
#
# Set B: Constraint Dominance (test if orthogonal loss is too restrictive)
#   B1: k=8 @ lambda_reg=0.01 (nearly relaxed)
#   B2: k=8 @ lambda_reg=0 (fully relaxed)
#
# Set C: Control Surface (test if we're adapting the right modules)
#   C1: k=4 late 3 blocks + unfreeze proj+LN
#   C2: k=4 late 3 blocks + SVD on MLP
#   C3: k=4 late 3 blocks + BOTH (combined)
#
# Expected outcomes:
# - If A1 drops to ~0.81: LR was causing k=1 to appear better
# - If A2 rises to ~0.82: LR was causing k>=4 to appear worse
# - If B1/B2 breaks plateau: Constraint is the bottleneck
# - If C1/C2/C3 breaks plateau: Control surface is the bottleneck
#
# Usage:
#   ./launch_diagnostic_sweep.sh all <region> <wandb_project>
#   ./launch_diagnostic_sweep.sh A <region> <wandb_project>
#   ./launch_diagnostic_sweep.sh B <region> <wandb_project>
#   ./launch_diagnostic_sweep.sh C <region> <wandb_project>
#   ./launch_diagnostic_sweep.sh A1 <region> <wandb_project>
# =============================================================================

set -e

EXPERIMENT_SET="${1:-all}"
REGION="${2:-asia-southeast1}"
WANDB_PROJECT="${3:-B16-diagnostic}"

EXPERIMENT_DIR="experiments/B16_diagnostic_sweep"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== DIAGNOSTIC SWEEP LAUNCHER ===${NC}"
echo -e "Experiment Set: ${YELLOW}${EXPERIMENT_SET}${NC}"
echo -e "Region: ${YELLOW}${REGION}${NC}"
echo -e "W&B Project: ${YELLOW}${WANDB_PROJECT}${NC}"
echo ""

launch_experiment() {
    local config_file="$1"
    local name="$2"
    
    if [ ! -f "$config_file" ]; then
        echo -e "${RED}ERROR: Config file not found: ${config_file}${NC}"
        return 1
    fi
    
    echo -e "${GREEN}Launching: ${name}${NC}"
    echo "  Config: ${config_file}"
    
    ./launch_experiment.sh "${WANDB_PROJECT}" "${REGION}" "${config_file}"
    
    echo -e "${GREEN}✓ Launched ${name}${NC}"
    echo ""
}

case "${EXPERIMENT_SET}" in
    "all")
        echo -e "${YELLOW}Launching ALL 7 diagnostic experiments...${NC}"
        echo ""
        
        # Set A: LR Confounds
        launch_experiment "${EXPERIMENT_DIR}/A1_k1_lr1e4.yaml" "A1: k=1 @ LR=1e-4"
        launch_experiment "${EXPERIMENT_DIR}/A2_k8_lr2e4.yaml" "A2: k=8 @ LR=2e-4"
        
        # Set B: Constraint Dominance
        launch_experiment "${EXPERIMENT_DIR}/B1_k8_lambda0.01.yaml" "B1: k=8 @ lambda=0.01"
        launch_experiment "${EXPERIMENT_DIR}/B2_k8_lambda0.yaml" "B2: k=8 @ lambda=0"
        
        # Set C: Control Surface
        launch_experiment "${EXPERIMENT_DIR}/C1_k4_late3_unfreeze_proj_ln.yaml" "C1: k=4 + unfreeze proj/LN"
        launch_experiment "${EXPERIMENT_DIR}/C2_k4_late3_svd_mlp.yaml" "C2: k=4 + SVD MLP"
        launch_experiment "${EXPERIMENT_DIR}/C3_k4_late3_both.yaml" "C3: k=4 + BOTH"
        ;;
    
    "A"|"a")
        echo -e "${YELLOW}Launching Set A (LR Confound Elimination)...${NC}"
        launch_experiment "${EXPERIMENT_DIR}/A1_k1_lr1e4.yaml" "A1: k=1 @ LR=1e-4"
        launch_experiment "${EXPERIMENT_DIR}/A2_k8_lr2e4.yaml" "A2: k=8 @ LR=2e-4"
        ;;
    
    "B"|"b")
        echo -e "${YELLOW}Launching Set B (Constraint Dominance)...${NC}"
        launch_experiment "${EXPERIMENT_DIR}/B1_k8_lambda0.01.yaml" "B1: k=8 @ lambda=0.01"
        launch_experiment "${EXPERIMENT_DIR}/B2_k8_lambda0.yaml" "B2: k=8 @ lambda=0"
        ;;
    
    "C"|"c")
        echo -e "${YELLOW}Launching Set C (Control Surface)...${NC}"
        launch_experiment "${EXPERIMENT_DIR}/C1_k4_late3_unfreeze_proj_ln.yaml" "C1: k=4 + unfreeze proj/LN"
        launch_experiment "${EXPERIMENT_DIR}/C2_k4_late3_svd_mlp.yaml" "C2: k=4 + SVD MLP"
        launch_experiment "${EXPERIMENT_DIR}/C3_k4_late3_both.yaml" "C3: k=4 + BOTH"
        ;;
    
    "A1"|"a1")
        launch_experiment "${EXPERIMENT_DIR}/A1_k1_lr1e4.yaml" "A1: k=1 @ LR=1e-4"
        ;;
    
    "A2"|"a2")
        launch_experiment "${EXPERIMENT_DIR}/A2_k8_lr2e4.yaml" "A2: k=8 @ LR=2e-4"
        ;;
    
    "B1"|"b1")
        launch_experiment "${EXPERIMENT_DIR}/B1_k8_lambda0.01.yaml" "B1: k=8 @ lambda=0.01"
        ;;
    
    "B2"|"b2")
        launch_experiment "${EXPERIMENT_DIR}/B2_k8_lambda0.yaml" "B2: k=8 @ lambda=0"
        ;;
    
    "C1"|"c1")
        launch_experiment "${EXPERIMENT_DIR}/C1_k4_late3_unfreeze_proj_ln.yaml" "C1: k=4 + unfreeze proj/LN"
        ;;
    
    "C2"|"c2")
        launch_experiment "${EXPERIMENT_DIR}/C2_k4_late3_svd_mlp.yaml" "C2: k=4 + SVD MLP"
        ;;
    
    "C3"|"c3")
        launch_experiment "${EXPERIMENT_DIR}/C3_k4_late3_both.yaml" "C3: k=4 + BOTH"
        ;;
    
    *)
        echo -e "${RED}Unknown experiment set: ${EXPERIMENT_SET}${NC}"
        echo ""
        echo "Usage: $0 <set> <region> <wandb_project>"
        echo ""
        echo "Sets:"
        echo "  all  - Launch all 7 experiments"
        echo "  A    - Set A: LR Confound (A1, A2)"
        echo "  B    - Set B: Constraint Dominance (B1, B2)"
        echo "  C    - Set C: Control Surface (C1, C2, C3)"
        echo "  A1   - Individual: k=1 @ LR=1e-4"
        echo "  A2   - Individual: k=8 @ LR=2e-4"
        echo "  B1   - Individual: k=8 @ lambda=0.01"
        echo "  B2   - Individual: k=8 @ lambda=0"
        echo "  C1   - Individual: k=4 + unfreeze proj/LN"
        echo "  C2   - Individual: k=4 + SVD MLP"
        echo "  C3   - Individual: k=4 + BOTH"
        exit 1
        ;;
esac

echo -e "${GREEN}=== DIAGNOSTIC SWEEP LAUNCHED ===${NC}"
echo ""
echo "Monitor progress at: https://wandb.ai/dtect-vision/${WANDB_PROJECT}"
echo ""
echo -e "${YELLOW}Expected decision tree:${NC}"
echo "1. If A1 ≈ 0.81 and A2 ≈ 0.82 → LR was the confound, not capacity"
echo "2. If B1/B2 > 0.85 → Constraint is the bottleneck"
echo "3. If C1/C2/C3 > 0.85 → Control surface is the bottleneck"
echo "4. If all still ~0.82 → Need more data or different approach (distillation?)"
