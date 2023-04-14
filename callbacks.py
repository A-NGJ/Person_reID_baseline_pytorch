from pathlib import Path

import numpy as np
import torch


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""

    def __init__(self, patience=5, verbose=False, delta=0):
        """
        Parameters
        ----------
        patience int:
            How long to wait after last time validation loss improved.
            Default: 7
        verbose bool:
            If True, prints a message for each validation loss improvement.
            Default: False
        delta float:
            Minimum change in the monitored quantity to qualify as an improvement.
            Default: 0
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.early_stop = False
        self.delta = delta
        self.best_loss = np.inf

    def __call__(self, val_loss):
        if self.best_loss - val_loss < self.delta:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            # self.save_checkpoint(val_loss, model)
            self.counter = 0

    # def save_checkpoint(self, val_loss, model):
    #     """Saves model when validation loss decrease."""
    #     if self.verbose:
    #         print(
    #             f"Validation loss decreased ({self.best_loss:.6f} --> {val_loss:.6f}).  Saving model ..."
    #         )
    #     torch.save(model.state_dict(), self.checkpoint_path)
