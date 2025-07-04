import os
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report, roc_auc_score
from tqdm import tqdm

from utils import logger

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")


class GNNTrainer:
    """Trainer for GNN models"""

    def __init__(
        self,
        model,
        data,
        optimizer,
        criterion,
        device=None,
        batch_size=32,
        checkpoint_dir="checkpoints",
    ):
        self.model = model
        self.data = data
        self.optimizer = optimizer
        self.criterion = criterion
        self.batch_size = batch_size
        self.device = (
            device
            if device is not None
            else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)

        self.model.to(self.device)
        self.data.to(self.device)

        # Add learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",  # We want to maximize validation accuracy
            factor=0.5,  # Multiply LR by this factor on plateau
            patience=5,  # Number of epochs to wait before reducing LR
            verbose=True,
            min_lr=1e-8,  # Don't reduce LR below this value
            threshold=1e-4,  # Minimum change to qualify as an improvement
            threshold_mode="abs",  # Use absolute change in validation accuracy
        )

    def _get_batches(self, mask):
        """Get batches of node indices"""
        node_idx = torch.where(mask)[0]
        num_nodes = len(node_idx)

        # Shuffle node indices
        perm = torch.randperm(num_nodes)
        node_idx = node_idx[perm]

        # Create batches
        batches = []
        for i in range(0, num_nodes, self.batch_size):
            batch_idx = node_idx[i : i + self.batch_size]
            batches.append(batch_idx)
        return batches

    def train_epoch(self):
        """Train for one epoch using mini-batches"""
        self.model.train()
        total_loss = 0
        num_nodes = int(self.data.train_mask.sum())
        batches = self._get_batches(self.data.train_mask)

        # Use tqdm for progress bar
        for batch_idx in tqdm(batches, desc="Training", leave=False):
            self.optimizer.zero_grad()

            # Forward pass
            out = self.model(
                self.data.x, self.data.edge_index, self.data.edge_attr
            )
            batch_out = out[batch_idx]
            batch_labels = self.data.y[batch_idx]

            # Compute loss and backward
            loss = self.criterion(batch_out, batch_labels)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * len(batch_idx)

        return total_loss / num_nodes

    @torch.no_grad()
    def evaluate(self, mask):
        """Evaluate the model on the given mask"""
        self.model.eval()
        out = self.model(self.data.x, self.data.edge_index, self.data.edge_attr)
        pred = out[mask].argmax(dim=1)
        correct = pred == self.data.y[mask]
        return float(correct.sum()) / int(mask.sum())

    def save_checkpoint(self, val_acc, epoch):
        """Save model checkpoint"""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "val_acc": val_acc,
        }
        checkpoint_path = os.path.join(
            self.checkpoint_dir, f"model_checkpoint_{timestamp}.pt"
        )
        torch.save(checkpoint, checkpoint_path)
        logger.success(
            f"Saved checkpoint to {checkpoint_path} (val_acc: {val_acc:.4f})"
        )

    def load_checkpoint(self, checkpoint_path=None):
        """Load model checkpoint"""
        if checkpoint_path is None:
            # Find the best checkpoint
            checkpoints = [
                f for f in os.listdir(self.checkpoint_dir) if f.endswith(".pt")
            ]
            if not checkpoints:
                logger.warning("No checkpoints found")
                return None
            checkpoint_path = os.path.join(
                self.checkpoint_dir, sorted(checkpoints)[-1]
            )

        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
            logger.success(f"Loaded checkpoint from {checkpoint_path}")
            return checkpoint["epoch"], checkpoint["val_acc"]
        else:
            logger.warning(f"No checkpoint found at {checkpoint_path}")
            return None

    def train(self, epochs=200, patience=100, verbose=True):
        """Train the model with early stopping"""
        start_epoch = 1
        best_val_acc = 0

        patience_counter = 0
        best_model_state = None

        # Use tqdm for epoch progress
        pbar = tqdm(range(start_epoch, epochs + 1), desc="Epochs")
        for epoch in pbar:
            loss = self.train_epoch()
            train_acc = self.evaluate(self.data.train_mask)
            val_acc = self.evaluate(self.data.val_mask)

            # Update learning rate based on validation accuracy
            self.scheduler.step(val_acc)
            current_lr = self.optimizer.param_groups[0]["lr"]

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{loss:.4f}",
                    "train_acc": f"{train_acc:.4f}",
                    "val_acc": f"{val_acc:.4f}",
                    "lr": f"{current_lr:.2e}",
                }
            )

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                # Save checkpoint
                self.save_checkpoint(val_acc, epoch)
                # Save best model weights
                best_model_state = {
                    k: v.cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
            else:
                patience_counter += 1

            if patience_counter >= patience:
                if verbose:
                    logger.warning(f"Early stopping at epoch {epoch}")
                break

            if current_lr <= self.scheduler.min_lrs[0]:
                if verbose:
                    logger.warning(
                        f"Stopping - Learning rate {current_lr:.2e} below minimum"
                    )
                break

        # Restore best model weights
        if best_model_state is not None:
            self.model.load_state_dict(best_model_state)

        # Final evaluation
        test_acc = self.evaluate(self.data.test_mask)
        if verbose:
            logger.success(f"Test Accuracy: {test_acc:.4f}")

        return test_acc

    def get_predictions(self, places_df, threshold=0.5):
        """Get predictions for all nodes using a probability threshold"""
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index)
            prob = F.softmax(out, dim=1)[:, 1]
            pred = (prob >= threshold).int()

        # Create a DataFrame with predictions
        results_df = places_df.copy()
        results_df["predicted_favorite"] = pred.cpu().numpy()
        results_df["favorite_probability"] = prob.cpu().numpy()

        return results_df

    def evaluate_predictions(self, results_df, verbose=True):
        """Evaluate predictions using classification metrics"""
        y_true = results_df["is_favorite"].values
        y_pred = results_df["predicted_favorite"].values
        y_prob = results_df["favorite_probability"].values

        if verbose:
            logger.highlight("\nModel Performance:")
            logger.success(classification_report(y_true, y_pred))
            logger.success(
                f"ROC AUC Score: {roc_auc_score(y_true, y_prob):.3f}",
            )

            # Add probability distribution information
            logger.highlight("\nProbability Distribution:")
            logger.success(f"Mean probability: {y_prob.mean():.3f}")
            logger.success(f"Median probability: {np.median(y_prob):.3f}")
            logger.success(f"Std probability: {y_prob.std():.3f}")
            logger.success(f"Min probability: {y_prob.min():.3f}")
            logger.success(f"Max probability: {y_prob.max():.3f}")

        return {
            "classification_report": classification_report(
                y_true, y_pred, output_dict=True
            ),
            "roc_auc": roc_auc_score(y_true, y_prob),
            "prob_stats": {
                "mean": y_prob.mean(),
                "median": np.median(y_prob),
                "std": y_prob.std(),
                "min": y_prob.min(),
                "max": y_prob.max(),
            },
        }

    def show_top_predictions(self, results_df, top_k=20, verbose=True):
        """Show top places predicted as favorites"""
        cols = ["title", "favorite_probability", "is_favorite"]
        top_predicted = (
            results_df[results_df["predicted_favorite"] == 1]
            .sort_values("favorite_probability", ascending=False)
            .head(top_k)
        )

        top_new_items = (
            results_df[
                (results_df["predicted_favorite"] == 1)
                & (results_df["is_favorite"] == 0)
            ]
            .sort_values("favorite_probability", ascending=False)
            .head(top_k)
        )

        if verbose:
            logger.highlight(f"\nTop {top_k} places predicted as favorites:")
            logger.info(top_predicted[cols].to_string(index=False))
            logger.highlight(
                f"\nTop {top_k} places predicted as new favorites:"
            )
            logger.success(top_new_items[cols].to_string(index=False))

        return top_predicted
