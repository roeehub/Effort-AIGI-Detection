# Pre-test 3 — CLS attention vs face IoU

For each ckpt, compute L11 CLS attention to patches on first 100 triptych frames; convert to 14x14 grid; compare to face bbox.

If attention concentrates on face → model uses local face features.
If attention scatters / on background → forgery-localization aux loss could redirect.


| ckpt | n_real | n_fake | real IoU | fake IoU | real face_attn_mass | fake face_attn_mass |
|---|---:|---:|---:|---:|---:|---:|
| P8A | 54 | 45 | 0.544 | 0.631 | 0.610 | 0.712 |
| T3_S1_step1500 | 54 | 45 | 0.524 | 0.591 | 0.588 | 0.682 |
| T3_S1_step2500 | 54 | 45 | 0.511 | 0.594 | 0.567 | 0.669 |
| E2B | 54 | 45 | 0.496 | 0.525 | 0.584 | 0.639 |