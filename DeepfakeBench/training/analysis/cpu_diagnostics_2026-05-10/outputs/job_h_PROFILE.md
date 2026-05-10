# Job H — From-scratch (E2B) vs FT-from-P8A representation comparison

## 1. Per-layer cosine similarity P8A vs E2B

| layer | n | cos p25 | cos p50 | cos p75 | cos min |
|---:|---:|---:|---:|---:|---:|
| 0 | 800 | 0.998 | 0.998 | 0.998 | 0.998 |
| 3 | 800 | 0.992 | 0.992 | 0.993 | 0.989 |
| 6 | 800 | 0.942 | 0.949 | 0.958 | 0.909 |
| 9 | 800 | 0.899 | 0.920 | 0.936 | 0.796 |
| 11 | 800 | 0.087 | 0.168 | 0.288 | -0.029 |

## 2. Forgery vs shortcut AUC per layer

### P8A trajectory

| layer | forgery_auc | is_dor | is_chronic_6 | lap_var_high | min_dim_high | invariance_max |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.725 | 0.965 | 0.692 | 0.905 | 0.762 | -0.240 |
| 3 | 0.922 | 0.972 | 0.888 | 0.896 | 0.946 | -0.050 |
| 6 | 0.992 | 0.995 | 0.953 | 0.946 | 0.981 | -0.003 |
| 9 | 0.979 | 0.996 | 0.914 | 0.939 | 0.971 | -0.017 |
| 11 | 0.976 | 0.995 | 0.909 | 0.923 | 0.971 | -0.019 |

### E2B trajectory

| layer | forgery_auc | is_dor | is_chronic_6 | lap_var_high | min_dim_high | invariance_max |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0.724 | 0.965 | 0.692 | 0.905 | 0.762 | -0.241 |
| 3 | 0.937 | 0.977 | 0.903 | 0.900 | 0.957 | -0.039 |
| 6 | 0.992 | 0.995 | 0.955 | 0.936 | 0.977 | -0.003 |
| 9 | 0.984 | 0.998 | 0.952 | 0.934 | 0.969 | -0.014 |
| 11 | 0.995 | 0.999 | 0.975 | 0.956 | 0.974 | -0.004 |

## 3. Reading

**Invariance score** (forgery_auc - max_shortcut_auc) at each layer:
- Higher = more 'forgery-only' signal at that layer
- Lower = forgery signal is co-mingled with shortcuts
- A 'good' representation has high forgery_auc AND low shortcut_auc (= high invariance)

**Cosine similarity** (P8A vs E2B same frame, same layer):
- Higher = both ckpts produce similar representations at that layer
- Lower = ckpts have diverged at that layer
- Interpretation: shows where P8A's FT chain pulled away from E2B's scratch path