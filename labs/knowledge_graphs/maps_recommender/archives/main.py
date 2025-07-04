import click
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from api import GooglePlacesAPI
from archives.graph import GraphGenerator
from archives.trainer import GNNTrainer
from dataloader import PlacesDataLoader
from dotenv import load_dotenv
from gnn import GNNModel
from tqdm import tqdm

from utils import logger

load_dotenv()
tqdm.pandas()


@click.command()
@click.option(
    "--distance",
    type=float,
    default=2,
    help="Distance threshold in km for graph edges",
)
@click.option("--hidden", type=int, default=64, help="Hidden channels for GNN")
@click.option(
    "--epochs", type=int, default=20, help="Number of training epochs"
)
@click.option(
    "--patience", type=int, default=5, help="Patience for early stopping"
)
@click.option(
    "--batch_size", type=int, default=16, help="Batch size for training"
)
@click.option(
    "--checkpoint_dir",
    type=str,
    default="data/checkpoints",
    help="Directory to save model checkpoints",
)
def main(distance, hidden, epochs, patience, batch_size, checkpoint_dir):
    gp = GooglePlacesAPI()
    gp.ping()
    pdl = PlacesDataLoader()
    datasets = pdl.load()

    for file, df in datasets.items():
        logger.info(f"Dataset {file} has {len(df)} records.")

    # Prepare graph data
    logger.info("Preparing graph data")
    graph_preparator = GraphGenerator(distance_threshold_km=distance)

    # Get favorite places
    favorite_places = set(datasets["favorite_places"]["place_id"])

    # Build graph
    G = graph_preparator.build(datasets["unique_places"], favorite_places)
    logger.info(
        f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges"
    )

    # Convert to PyTorch Geometric format
    data = graph_preparator.prepare_pytorch_geometric_data(G)
    logger.info(
        f"Prepared PyTorch Geometric data with {data.num_features} features"
    )

    # Create train/val/test masks
    data = graph_preparator.create_train_val_test_masks(data)

    # Initialize model
    model = GNNModel(
        num_features=data.num_features, hidden_channels=hidden, num_classes=2
    )
    optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

    # Calculate class weights for imbalanced data
    labels = data.y.numpy()
    num_positive = np.sum(labels == 1)
    num_negative = np.sum(labels == 0)
    logger.highlight(f"Positive: {num_positive}, Negative: {num_negative}")
    pos_weight = torch.tensor(
        num_negative / num_positive
    ).sqrt()  # Square root to soften the weight
    class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float)

    # Use weighted CrossEntropyLoss
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Train model
    logger.info("Training GNN model")
    trainer = GNNTrainer(
        model,
        data,
        optimizer,
        criterion,
        batch_size=batch_size,
        checkpoint_dir=checkpoint_dir,
    )
    trainer.train(epochs=epochs, patience=patience)

    # Create evaluation dataframe with only test set places
    eval_df = datasets["unique_places"].copy()
    eval_df["is_favorite"] = eval_df["place_id"].isin(favorite_places)
    logger.highlight(f"{eval_df.is_favorite.value_counts()}")
    logger.highlight(f"{eval_df[eval_df.is_favorite].place_id.nunique()}")
    results_df = trainer.get_predictions(eval_df)

    # Evaluate predictions
    trainer.evaluate_predictions(results_df)

    # Show top predictions
    trainer.show_top_predictions(results_df)


if __name__ == "__main__":
    main()
