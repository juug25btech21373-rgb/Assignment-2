import os

import numpy as np
import torch


class EarlyStopping:
    """Save the best validation-loss checkpoint."""

    def __init__(self, dataset, p, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

        os.makedirs("saved_model", exist_ok=True)
        self.model_path = os.path.join(
            "saved_model", "ckpt_{}_{}.model".format(dataset, p)
        )

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print("training process", self.counter)
            if self.verbose:
                print(
                    "EarlyStopping counter: {} out of {}".format(
                        self.counter, self.patience
                    )
                )
            if self.counter >= self.patience:
                self.early_stop = True
                return True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0
            print("training process:", self.counter)

        return False

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(
                "Validation loss decreased ({} --> {}). Saving model ...".format(
                    self.val_loss_min, val_loss
                )
            )
        torch.save(model, self.model_path)
        self.val_loss_min = val_loss

    def load_model(self):
        # This project stores a full model object. Explicitly opt into loading
        # that local checkpoint under PyTorch 2.6+.
        return torch.load(
            self.model_path,
            map_location="cpu",
            weights_only=False
        )
