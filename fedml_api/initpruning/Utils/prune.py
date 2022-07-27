from tqdm import tqdm
import torch
import logging
import numpy as np

def prune_loop(model, loss, pruner, dataloader, device, sparsity, schedule, scope, epochs,
               reinitialize=False, train_mode=True, shuffle=False, invert=False):
    r"""Applies score mask loop iteratively to a final sparsity level.
    """
    # Set model to train or eval mode
    model.train()
    if not train_mode:
        model.eval()

    # Prune model
    for epoch in tqdm(range(epochs)):
        pruner.score(model, loss, dataloader, device)
        if schedule == 'exponential':
            sparse = sparsity**((epoch + 1) / epochs)
        elif schedule == 'linear':
            sparse = 1.0 - (1.0 - sparsity)*((epoch + 1) / epochs)
        elif schedule == 'direct' or (epoch + 1) == len(epochs):
            sparse = sparsity
        # Invert scores
        if invert:
            pruner.invert()
        pruner.mask(sparse, scope)



    # Reainitialize weights
    if reinitialize:
        model._initialize_weights()

    # Shuffle masks
    if shuffle:
        pruner.shuffle()

    # Confirm sparsity level
    remaining_params, total_params = pruner.stats()
    logging.info(f"Remain params {remaining_params}, total params {total_params}, expect density {sparsity}, real density {remaining_params/total_params}")
    if np.abs(remaining_params - total_params*sparsity) >= 20:
        print("ERROR: {} prunable parameters remaining, expected {}".format(remaining_params, total_params*sparsity))
        quit()
