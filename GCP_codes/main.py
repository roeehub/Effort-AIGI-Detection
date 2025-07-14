# main.py
import random
import numpy as np

from prepare_splits import prepare_video_splits
from dataloaders import (
    create_naive_dataloader,
    create_method_aware_dataloaders,
    create_simple_dataloader
)

CONFIG_PATH = 'config.yaml'

def main():
    # 1. Prepare the data splits based on the config file
    # This is done once at the start of training.
    train_videos, val_videos, config = prepare_video_splits(CONFIG_PATH)

    # --- CHOOSE YOUR DATALOADER TYPE ---

    # # Option A: Naive 50/50 Real/Fake Dataloader
    # print("\n--- Example: Naive Dataloader ---")
    # naive_loader = create_naive_dataloader(train_videos, config)
    # for i, (video_batch, label_batch, _) in enumerate(naive_loader):
    #     print(f"Batch {i}: Video shape={video_batch.shape}, Labels={label_batch.tolist()}")
    #     if i >= 2: break # Demo a few batches
    #
    # # Option B: Simple Shuffled Dataloader (My Suggestion)
    # print("\n--- Example: Simple Shuffled Dataloader (Baseline) ---")
    # simple_loader = create_simple_dataloader(train_videos, config)
    # for i, (video_batch, label_batch, method_batch) in enumerate(simple_loader):
    #     print(f"Batch {i}: Video shape={video_batch.shape}, Methods in batch={method_batch}")
    #     if i >= 2: break

    # Option C: Method-Aware Dataloaders (Separate loader for each method)
    print("\n--- Example: Method-Aware Dataloaders ---")
    method_loaders = create_method_aware_dataloaders(train_videos, config)
    
    # Training Loop for Method-Aware
    # To cycle through methods:
    method_names = list(method_loaders.keys())
    random.shuffle(method_names)
    print(f"Training on methods in this order: {method_names[:5]}...")
    
    # For a balanced approach, you can create a weighted sampler over `method_names`
    method_sizes = {m: len(dl.dataset) for m, dl in method_loaders.items()}
    total_videos = sum(method_sizes.values())
    weights = [method_sizes[m] / total_videos for m in method_names]
    
    print("\n--- Example: Method-Aware BALANCED Training Loop ---")
    # In each step, you'd pick a method based on weights and get a batch from it
    # This requires creating iterators for each dataloader.
    method_iters = {name: iter(loader) for name, loader in method_loaders.items()}
    
    for step in range(5): # Simulate 5 training steps
        # Pick a method based on its data abundance
        chosen_method = random.choices(method_names, weights=weights, k=1)[0]
        
        try:
            video_batch, label_batch, _ = next(method_iters[chosen_method])
            print(f"Step {step}: Training on batch from balanced choice '{chosen_method}'. Shape={video_batch.shape}")
        except StopIteration:
            # Refill the iterator if it's exhausted
            print(f"Method '{chosen_method}' exhausted. Resetting iterator.")
            method_iters[chosen_method] = iter(method_loaders[chosen_method])
            video_batch, label_batch, _ = next(method_iters[chosen_method])
            print(f"Step {step}: Training on batch from balanced choice '{chosen_method}'. Shape={video_batch.shape}")


if __name__ == '__main__':
    # You'll need to install these packages
    # pip install torchdata torch torch-vision pyyaml gcsfs Pillow
    main()