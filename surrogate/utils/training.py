import numpy as np
import random
# import matplotlib.pyplot as plt
import torch
import calendar
import time
import math
import os
from tqdm import tqdm

import time
import math
import os
import calendar
import torch

# if started from jupter use tqdm.notebook else use nromal tqdm
try:
    shell = get_ipython().__class__.__name__
    if shell == 'ZMQInteractiveShell':
        # Jupyter notebook or JupyterLab
        from tqdm.notebook import tqdm
    else:
        # Terminal or other environments
        from tqdm import tqdm
except NameError:
    # Standard Python interpreter
    from tqdm import tqdm


def train_nn(nn, train_loader, valid_loader, lossfunction, optimizer, UUID='default', 
             max_epochs=1000, patience_limit=25, min_delta=0.0001, relative=True):
    """ 
    If relative=True, min_delta=0.01 means the validation loss must improve by at least 1%.
    """
    training_ID = int(calendar.timegm(time.gmtime()))
    if UUID != 'default':
        UUID = f'{hash(UUID)}'
    print(f'The ID for this training is {UUID}_{training_ID}.')

    train_loss = []
    valid_loss = []
    best_valid_loss = math.inf
    patience = 0

    os.makedirs('./.temp', exist_ok=True)
    temp_path = f'./.temp/NN_{UUID}_{training_ID}.pt'

    pbar = tqdm(total=max_epochs, desc="Training epochs")

    for epoch in range(max_epochs):
        time_start = time.time()
        
        # --- Training ---
        total_loss = 0.0
        total_samples = 0
        nn.train()
        for x_train, y_train in train_loader:
            prediction_train = nn(x_train)
            L_train = lossfunction(prediction_train, y_train)
            total_loss += L_train.item() * x_train.size(0)
            total_samples += x_train.size(0)

            optimizer.zero_grad()
            L_train.backward()
            optimizer.step()

        weighted_mean_loss = total_loss / total_samples
        train_loss.append(weighted_mean_loss)

        # --- Validation ---
        total_valid_loss = 0.0
        total_valid_samples = 0
        nn.eval()
        with torch.no_grad():
            for x_valid, y_valid in valid_loader:
                prediction_valid = nn(x_valid)
                L_valid = lossfunction(prediction_valid, y_valid)
                total_valid_loss += L_valid.item() * x_valid.size(0)
                total_valid_samples += x_valid.size(0)

        weighted_mean_valid_loss = total_valid_loss / total_valid_samples
        valid_loss.append(weighted_mean_valid_loss)

        # --- Early stopping ---
        if relative:
            improvement_needed = best_valid_loss * min_delta
        else:
            improvement_needed = min_delta

        if weighted_mean_valid_loss < best_valid_loss - improvement_needed:
            best_valid_loss = weighted_mean_valid_loss
            torch.save(nn, temp_path)
            patience = 0
        else:
            patience += 1

        if patience > patience_limit:
            tqdm.write("Early stop triggered.")
            break

        time_end = time.time()
        pbar.set_postfix({
            'Train Loss': f'{weighted_mean_loss:.5f}',
            'Valid Loss': f'{weighted_mean_valid_loss:.5f}',
            'Patience': patience,
            'Epoch Time': f'{time_end - time_start:.2f}s'
        })
        pbar.update(1)

    pbar.close()

    # ensure model exists
    if not os.path.exists(temp_path):
        torch.save(nn, temp_path)

    resulted_nn = torch.load(temp_path, weights_only=False)
    os.remove(temp_path)
    
    print('Finished training.')
    return resulted_nn, train_loss, valid_loss