# Augmentation Comparison

| Run | Test Accuracy | Test Macro F1 |
|-----|--------------|---------------|
| No augmentation | — | — |
| With augmentation | 1.0000 | 1.0000 |

## Notes
- Primary metric: **Macro F1** (equal weight per class regardless of sample count).
- Augmentation strategies: Gaussian noise (std=0.005), time shift (±10%), SpecAugment.
- noaug: best val_loss = 1.3106
- aug: best val_loss = 0.0036
