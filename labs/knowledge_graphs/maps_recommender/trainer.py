import torch
import torch.nn.functional as F
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score
import click


class GNNTrainer:
    """Trainer for GNN models"""

    def __init__(self, model, data, optimizer, criterion, device=None):
        self.model = model
        self.data = data
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.data.to(self.device)

    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        self.optimizer.zero_grad()
        out = self.model(self.data.x, self.data.edge_index)
        loss = self.criterion(out[self.data.train_mask], self.data.y[self.data.train_mask])
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def evaluate(self, mask):
        """Evaluate the model on the given mask"""
        self.model.eval()
        out = self.model(self.data.x, self.data.edge_index)
        pred = out.argmax(dim=1)
        correct = pred[mask] == self.data.y[mask]
        acc = int(correct.sum()) / int(mask.sum())
        return acc

    def train(self, epochs=200, patience=100, verbose=True):
        """Train the model with early stopping"""
        best_val_acc = 0
        patience_counter = 0

        for epoch in range(1, epochs + 1):
            loss = self.train_epoch()
            train_acc = self.evaluate(self.data.train_mask)
            val_acc = self.evaluate(self.data.val_mask)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                if verbose:
                    click.secho(f"Early stopping at epoch {epoch}", fg="yellow")
                break

            if verbose and epoch % 10 == 0:
                click.secho(
                    f"Epoch {epoch:03d}, Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Val Acc: {val_acc:.4f}",
                    fg="blue",
                )

        # Final evaluation
        test_acc = self.evaluate(self.data.test_mask)
        if verbose:
            click.secho(f"Test Accuracy: {test_acc:.4f}", fg="green")

        return test_acc

    def get_predictions(self, places_df):
        """Get predictions for all nodes"""
        self.model.eval()
        with torch.no_grad():
            out = self.model(self.data.x, self.data.edge_index)
            pred = out.argmax(dim=1)
            prob = F.softmax(out, dim=1)[:, 1]

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
            click.secho("\nModel Performance:", fg="green")
            click.secho(classification_report(y_true, y_pred), fg="green")
            click.secho(
                f"ROC AUC Score: {roc_auc_score(y_true, y_prob):.3f}",
                fg="green",
            )

        return {
            "classification_report": classification_report(y_true, y_pred, output_dict=True),
            "roc_auc": roc_auc_score(y_true, y_prob),
        }

    def show_top_predictions(self, results_df, top_k=10, verbose=True):
        """Show top places predicted as favorites"""
        top_predicted = (
            results_df[results_df["predicted_favorite"] == 1]
            .sort_values("favorite_probability", ascending=False)
            .head(top_k)
        )

        if verbose:
            click.secho(f"\nTop {top_k} places predicted as favorites:", fg="blue")
            click.secho(
                top_predicted[["title", "favorite_probability", "is_favorite"]],
                fg="blue",
            )

        return top_predicted
