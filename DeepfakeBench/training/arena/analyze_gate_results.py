"""Analyze extremity gate results: best configs, gate value comparison."""
import pandas as pd

rdf = pd.read_csv("strategy_results/extremity_gate_full.csv")
zero = rdf[rdf["teams_fp"] == 0]

# Baseline
ungated = zero[(zero["T"]==0.30) & (zero["K"]==18) & (zero["gate"]=="none")]
b = ungated.iloc[0]
print(f"BASELINE: T=0.30 K=18 -> TP={b.tp:.0f} FP={b.fp:.0f} TPR={b.tpr:.4f} TgtTPR={b.tgt_tpr:.3f} BalAcc={b.bal_acc:.4f}")
print()

# Best gate configs for T=0.30, K=18
gated = zero[(zero["T"]==0.30) & (zero["K"]==18) & (zero["gate"]=="extremity")]
g = gated.sort_values(["fp","tpr"], ascending=[True,False]).drop_duplicates(["fp"]).head(15)
print("Best gates at T=0.30, K=18:")
print(f"  E     M   H    TP  FP    TPR  TgtTPR  BalAcc FP_saved")
print("  " + "-"*58)
for r in g.itertuples():
    saved = int(b.fp) - int(r.fp)
    print(f"  {r.E:.2f}  {r.M:>2.0f}  {r.H:>2.0f}  {r.tp:>4.0f}  {r.fp:>2.0f}  {r.tpr:.4f}  {r.tgt_tpr:.3f}   {r.bal_acc:.4f}  {saved:+d}")

# GLOBAL: best gate value (FP saved vs TPR cost)
print()
print("=" * 85)
print("GLOBAL: best gate value across all T,K combos (0 Teams FP)")
print("=" * 85)

ungated_all = zero[zero["gate"]=="none"].copy()
gated_all = zero[zero["gate"]=="extremity"].copy()

# For each (T,K), best ungated
ub = ungated_all.groupby(["T","K"]).agg(
    base_fp=("fp","min"), base_tpr=("tpr","max"), base_tgt=("tgt_tpr","max"), base_bal=("bal_acc","max")
).reset_index()

# For each (T,K), best gated by bal_acc
gb_idx = gated_all.groupby(["T","K"])["bal_acc"].idxmax()
gb = gated_all.loc[gb_idx][["T","K","E","M","H","fp","tpr","tgt_tpr","bal_acc"]].copy()
gb.columns = ["T","K","E","M","H","gate_fp","gate_tpr","gate_tgt","gate_bal"]

merged = ub.merge(gb, on=["T","K"], how="inner")
merged["fp_saved"] = merged["base_fp"] - merged["gate_fp"]
merged["tpr_cost"] = merged["base_tpr"] - merged["gate_tpr"]
top = merged[merged["fp_saved"] > 0].sort_values(["fp_saved","tpr_cost"], ascending=[False,True]).head(25)

print(f"  T     K  | BaseFP GateFP Saved | BasTPR  GatTPR  Cost   | E     M   H")
print("  " + "-"*78)
for r in top.itertuples():
    print(f"  {r.T:.2f}  {r.K:>2.0f} | {r.base_fp:>5.0f}  {r.gate_fp:>5.0f}  {r.fp_saved:>+4.0f} | "
          f"{r.base_tpr:.4f}  {r.gate_tpr:.4f}  {r.tpr_cost:>+.4f} | {r.E:.2f}  {r.M:>2.0f}  {r.H:>2.0f}")

# Dig deeper: for our baseline T=0.30 K=18, show ALL unique (FP, TPR) outcomes
print()
print("=" * 85)
print("ALL UNIQUE OUTCOMES for T=0.30, K=18 + gate (0 Teams FP)")
print("=" * 85)
outcomes = gated.groupby(["fp"]).agg(
    max_tpr=("tpr","max"), max_tgt=("tgt_tpr","max"), max_bal=("bal_acc","max"), count=("tpr","count")
).reset_index()
print(f"  FP  max_TPR  max_TgtTPR  max_BalAcc  configs")
print("  " + "-"*50)
for r in outcomes.itertuples():
    marker = " <-- BASELINE" if int(r.fp) == int(b.fp) else ""
    print(f"  {r.fp:>2.0f}  {r.max_tpr:.4f}  {r.max_tgt:.3f}       {r.max_bal:.4f}     {r.count:>5}{marker}")

# The sweet spot analysis: what's the best we can do with only modest TPR cost?
print()
print("=" * 85)
print("SWEET SPOT: best bal_acc with FP < baseline (0 Teams FP, any T/K)")
print("=" * 85)
candidates = zero[(zero["fp"] < int(b.fp))].copy()
candidates = candidates.sort_values("bal_acc", ascending=False).head(30)
print(f"  gate       T     K   E     M   H   TP   FP   TPR    TgtTPR  BalAcc")
print("  " + "-"*68)
seen = set()
for r in candidates.itertuples():
    key = (int(r.fp), round(r.tpr, 3))
    if key in seen:
        continue
    seen.add(key)
    if len(seen) > 20:
        break
    print(f"  {r.gate:<9} {r.T:.2f}  {r.K:>2.0f}  {r.E:.2f}  {r.M:>2.0f}  {r.H:>2.0f}  {r.tp:>4.0f}  {r.fp:>2.0f}  {r.tpr:.4f}  {r.tgt_tpr:.3f}   {r.bal_acc:.4f}")
