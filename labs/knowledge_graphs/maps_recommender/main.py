import os
import click
from dotenv import load_dotenv
import requests
import json
import pandas as pd
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from api import GooglePlacesAPI
from dataloader import GraphDataProcessor, PlacesDataLoader
from gnn import GNNModel
from trainer import GNNTrainer
from utils import logger

load_dotenv()
tqdm.pandas()


@click.command()
@click.option("--location", type=str, help="Location to search for")
@click.option(
    "--distance",
    type=float,
    default=1.0,
    help="Distance threshold in km for graph edges",
)
@click.option("--hidden", type=int, default=64, help="Hidden channels for GNN")
@click.option("--epochs", type=int, default=400, help="Number of training epochs")
@click.option("--patience", type=int, default=200, help="Patience for early stopping")
def main(location, distance, hidden, epochs, patience):
    gp = GooglePlacesAPI()
    gp.ping()
    pdl = PlacesDataLoader()
    datasets = pdl.load()

    for file, df in datasets.items():
        logger.info(f"Dataset {file} has {len(df)} records.")

    # # Prepare graph data
    # logger.info("Preparing graph data", fg="blue")
    # graph_preparator = GraphDataProcessor(distance_threshold_km=distance)

    # # Get favorite places
    # favorite_places = set(datasets["favorite_places"]["title"].str.lower())

    # # Build graph
    # G, places_df = graph_preparator.build_graph(
    #     datasets["unique_places"], favorite_places
    # )
    # logger.info(
    #     f"Built graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges",
    #     fg="blue",
    # )

    # # Convert to PyTorch Geometric format
    # data = graph_preparator.prepare_pytorch_geometric_data(G, places_df)
    # logger.info(
    #     f"Prepared PyTorch Geometric data with {data.num_features} features",
    #     fg="blue",
    # )

    # # Create train/val/test masks
    # data = graph_preparator.create_train_val_test_masks(data)

    # # Initialize model
    # model = GNNModel(
    #     num_features=data.num_features, hidden_channels=hidden, num_classes=2
    # )
    # optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=5e-4)

    # # Calculate class weights for imbalanced data
    # labels = data.y.numpy()
    # num_positive = np.sum(labels == 1)
    # num_negative = np.sum(labels == 0)
    # pos_weight = num_negative / num_positive
    # class_weights = torch.tensor([1.0, pos_weight], dtype=torch.float)

    # # Use weighted CrossEntropyLoss
    # criterion = nn.CrossEntropyLoss(weight=class_weights)

    # # Train model
    # logger.info("Training GNN model", fg="green")
    # trainer = GNNTrainer(model, data, optimizer, criterion)
    # trainer.train(epochs=epochs, patience=patience)

    # # Get predictions
    # results_df = trainer.get_predictions(places_df)

    # # Evaluate predictions
    # trainer.evaluate_predictions(results_df)

    # # Show top predictions
    # trainer.show_top_predictions(results_df)


if __name__ == "__main__":
    main()
