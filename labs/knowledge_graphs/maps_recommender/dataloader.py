import networkx as nx
import pandas as pd
import torch
from torch_geometric.data import Data
import math
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import pickle


class GraphDataProcessor:
    def __init__(self, distance_threshold_km=1.0):
        self.distance_threshold_km = distance_threshold_km
        self.text_encoder = SentenceTransformer("all-MiniLM-L6-v2")
        self.data_dir = "labs/knowledge_graphs/maps_recommender/data/processed"
        os.makedirs(self.data_dir, exist_ok=True)

    def _get_graph_path(self):
        """Get the path for the saved graph file"""
        return os.path.join(
            self.data_dir, f"graph_threshold_{self.distance_threshold_km}.pkl"
        )

    def save_graph(self, G, places_df):
        """Save the graph and places DataFrame to disk"""
        graph_path = self._get_graph_path()
        data = {"graph": G, "places_df": places_df}
        with open(graph_path, "wb") as f:
            pickle.dump(data, f)

    def load_graph(self):
        """Load the graph and places DataFrame from disk if it exists"""
        graph_path = self._get_graph_path()
        if os.path.exists(graph_path):
            with open(graph_path, "rb") as f:
                data = pickle.load(f)
            return data["graph"], data["places_df"]
        return None, None

    def create_text_description(self, row):
        """Create a markdown-style text description from categorical features"""
        description = f"# {row.title}\n\n"

        # Add place types
        if not pd.isna(row.types):
            description += "## Place Types\n"
            for place_type in row.types:
                if place_type not in [
                    "point_of_interest",
                    "establishment",
                    "store",
                ]:
                    description += f"- {place_type}\n"
            description += "\n"

        # Add numerical features as text
        description += "## Features\n"
        if not pd.isna(row.price_level):
            description += f"- Price Level: {row.price_level}\n"
        if not pd.isna(row.rating):
            description += f"- Rating: {row.rating}\n"
        if not pd.isna(row.user_ratings_total):
            description += f"- Total Ratings: {row.user_ratings_total}\n"

        return description

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate the distance between two points on Earth in kilometers"""
        R = 6371

        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = (
            math.sin(dlat / 2) ** 2
            + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        )
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c

        return distance

    def build_graph(self, places_df, favorite_places):
        """Build a graph where nodes are places and edges connect places within the distance threshold"""
        # Try to load existing graph first
        G, saved_places_df = self.load_graph()
        if G is not None and saved_places_df is not None:
            print("Loaded existing graph from disk")
            return G, saved_places_df

        print("Building new graph...")
        places_df = places_df.copy()
        # Filter to only include "want to go" places
        # places_df = places_df[places_df["relation"] == "want_to_go"]
        places_df["is_favorite"] = (
            places_df["title"].str.lower().isin(favorite_places).astype(int)
        )

        # Create text descriptions and embeddings for all places
        text_descriptions = [
            self.create_text_description(row) for _, row in places_df.iterrows()
        ]
        text_embeddings = self.text_encoder.encode(text_descriptions)

        # Create a graph
        G = nx.Graph()
        for idx, (row, embedding) in enumerate(
            zip(places_df.iterrows(), text_embeddings)
        ):
            _, row = row
            # Combine text embedding with numerical features
            numerical_features = np.array(
                [
                    row.price_level if not pd.isna(row.price_level) else 0,
                    row.rating if not pd.isna(row.rating) else 0,
                    (
                        row.user_ratings_total
                        if not pd.isna(row.user_ratings_total)
                        else 0
                    ),
                ]
            )

            # Concatenate text embedding with numerical features
            node_features = np.concatenate([embedding, numerical_features])

            G.add_node(idx, features=node_features)
            G.nodes[idx]["label"] = row.is_favorite

        for i in range(len(places_df)):
            for j in range(i + 1, len(places_df)):
                if (
                    pd.isna(places_df.iloc[i].lat)
                    or pd.isna(places_df.iloc[i].lon)
                    or pd.isna(places_df.iloc[j].lat)
                    or pd.isna(places_df.iloc[j].lon)
                ):
                    continue

                distance = self.haversine_distance(
                    places_df.iloc[i].lat,
                    places_df.iloc[i].lon,
                    places_df.iloc[j].lat,
                    places_df.iloc[j].lon,
                )

                if distance <= self.distance_threshold_km:
                    G.add_edge(i, j, weight=1.0 / (1 + distance))

        # Save the constructed graph
        self.save_graph(G, places_df)

        return G, places_df

    def prepare_pytorch_geometric_data(self, G, places_df):
        """Convert NetworkX graph to PyTorch Geometric format"""
        edge_index = []
        edge_attr = []

        for u, v, data in G.edges(data=True):
            edge_index.append([u, v])
            edge_index.append([v, u])
            edge_attr.append(data.get("weight", 1.0))
            edge_attr.append(data.get("weight", 1.0))

        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float).view(-1, 1)

        # Get node features
        x = torch.tensor(
            [G.nodes[node]["features"] for node in G.nodes()], dtype=torch.float
        )

        # Prepare node labels
        y = torch.tensor(
            [G.nodes[node]["label"] for node in G.nodes()], dtype=torch.long
        )

        # Create PyTorch Geometric Data object
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

        return data

    def create_train_val_test_masks(
        self, data, train_ratio=0.7, val_ratio=0.15
    ):
        """Create masks for training, validation, and test sets with balanced class distribution"""
        num_nodes = data.num_nodes
        labels = data.y.numpy()

        # Get indices for each class
        class_0_indices = np.where(labels == 0)[0]
        class_1_indices = np.where(labels == 1)[0]

        # Calculate sizes for each class in each split
        train_size_0 = int(train_ratio * len(class_0_indices))
        train_size_1 = int(train_ratio * len(class_1_indices))
        val_size_0 = int(val_ratio * len(class_0_indices))
        val_size_1 = int(val_ratio * len(class_1_indices))

        # Shuffle indices
        np.random.shuffle(class_0_indices)
        np.random.shuffle(class_1_indices)

        # Create splits for each class
        train_indices_0 = class_0_indices[:train_size_0]
        val_indices_0 = class_0_indices[
            train_size_0 : train_size_0 + val_size_0
        ]
        test_indices_0 = class_0_indices[train_size_0 + val_size_0 :]

        train_indices_1 = class_1_indices[:train_size_1]
        val_indices_1 = class_1_indices[
            train_size_1 : train_size_1 + val_size_1
        ]
        test_indices_1 = class_1_indices[train_size_1 + val_size_1 :]

        # Combine indices
        train_indices = np.concatenate([train_indices_0, train_indices_1])
        val_indices = np.concatenate([val_indices_0, val_indices_1])
        test_indices = np.concatenate([test_indices_0, test_indices_1])

        # Create masks
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        train_mask[train_indices] = True
        val_mask[val_indices] = True
        test_mask[test_indices] = True

        data.train_mask = train_mask
        data.val_mask = val_mask
        data.test_mask = test_mask

        return data
