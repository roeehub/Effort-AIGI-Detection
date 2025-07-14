# main.py
import random

from dataloaders import (
    create_naive_dataloader,
    create_method_aware_dataloaders,
    temperatured_weights,
)
from prepare_splits import prepare_video_splits

CONFIG_PATH = "config.yaml"


def main():
    # ------------------------------------------------------------------ #
    # 1 Prepare data splits                                              #
    # ------------------------------------------------------------------ #
    train_videos, val_videos, config = prepare_video_splits(CONFIG_PATH)

    # ------------------------------------------------------------------ #
    # 2 Select a dataloader flavour                                      #
    # ------------------------------------------------------------------ #
    # Example C: Method-Aware loaders
    method_loaders = create_method_aware_dataloaders(train_videos, config)

    # Build weighting scheme (temperature 0.5 by default)
    method_names  = list(method_loaders.keys())
    method_sizes  = [method_loaders[m].dataset_size for m in method_names]
    weights       = temperatured_weights(method_sizes, T=0.5)

    print(f"Method order example: {method_names[:5]}")
    print("Weights (inv-freq^0.5):",
          {m: f"{w:.3f}" for m, w in zip(method_names, weights)})

    # Training-loop sketch
    iters = {m: iter(dl) for m, dl in method_loaders.items()}
    for step in range(5):
        chosen = random.choices(method_names, weights=weights, k=1)[0]
        try:
            vids, lbls, _ = next(iters[chosen])
        except StopIteration:
            iters[chosen] = iter(method_loaders[chosen])
            vids, lbls, _ = next(iters[chosen])
        print(f"Step {step} – batch from '{chosen}': {vids.shape}")


if __name__ == "__main__":
    main()
