# Resource Tuning & Misc Notes

## Prefetch & Workers by VM Type

| GCP VM type      | vCPUs | Suggested `num_workers` | `.prefetch(k)` |
|------------------|------:|------------------------:|---------------:|
| n1‑standard‑4    |   4   |            2            |        5       |
| n1‑standard‑8    |   8   |            4            |       10       |
| n1‑standard‑16   |  16   |            8            |       20       |
| a2‑highgpu‑1g    |  12   |            6            |       20       |
| a2‑highgpu‑8g    |  96   |           32            |       50       |

> **Tip:** If you run out‑of‑memory (OOM) on **CPU RAM**, lower the **prefetch** value first—then reduce the number of workers.

---

## Tiny‑Method Handling

A “tiny” method with very few frames (≈ 200) often **hurts convergence** more than it helps.  
If you want to exclude such methods:

1. Open `config.yaml`.
2. Navigate to `methods → use_fake_methods`.
3. Comment‑out the method you’d like to skip.

_No code changes are required._

---

## Tensor Shape Reminder

Frame tensors are provided as **`(C, T, H, W)`**.  
Swap to **`(T, C, H, W)`** if your model expects a time‑first layout.

---

## Cross‑Entropy Loss Note

For `torch.nn.CrossEntropyLoss`, ensure your label tensor’s **`dtype`** is `torch.long` with **class indices** `{0, 1}` (see comment in code).