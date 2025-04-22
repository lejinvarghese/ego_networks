import networkx as nx
import pandas as pd
import torch
from torch_geometric.data import Data
import math
from sentence_transformers import SentenceTransformer
import numpy as np
import os
import pickle
from api import GooglePlacesAPI
from utils import logger


class PlacesDataLoader:
    def __init__(self):
        self.api = GooglePlacesAPI()
        self.datasets = {}

    def load(self):
        for file in [
            "want_to_go",
            "favorite_places",
            "saved_places",
            "reviews",
            "unique_places",
        ]:
            file_path = f"data/processed/{file}.csv"
            if os.path.exists(file_path):
                logger.info(f"Reading {file}")
                self.datasets[file] = pd.read_csv(file_path)
            else:
                logger.info(f"Processing {file}")
                df = self._process_dataset(file)
                self.datasets[file] = df
        return self.datasets

    def _process_dataset(self, file: str):
        file_path = f"data/processed/{file}.csv"
        if file == "saved_places":
            df = pd.read_json(f"data/{file}.json")
            df = pd.DataFrame(list(df.features))
            df["Title"] = df["properties"].map(lambda x: x.get("location", {}).get("name"))
            df = df[["Title"]].dropna()
        elif file == "reviews":
            df = pd.read_json(f"data/{file}.json")
            df = pd.DataFrame(list(df.features))
            df["rating"] = df["properties"].map(lambda x: x.get("five_star_rating_published"))
            df["Title"] = df["properties"].map(lambda x: x.get("location", {}).get("name"))
            df = df[["Title", "rating"]].dropna()
        elif file == "unique_places":
            df = self._extract_place_details()
        else:
            df = pd.read_csv(f"data/{file}.csv")[["Title"]]

        if file != "unique_places":
            df["place_id"] = df["Title"].progress_map(lambda x: self.api.find_place(x))
        df.columns = [f"{str.lower(col)}" for col in df.columns]
        df.to_csv(file_path, index=False)
        return df

    def _extract_place_details(self):
        """Extract place details from the datasets"""
        unique_places = pd.concat([v for _, v in self.datasets.items()])[["place_id"]].dropna().drop_duplicates()
        unique_places["place_details"] = unique_places.place_id.progress_map(lambda x: self.api.get_place_details(x).get("result", {}))
        unique_places["title"] = unique_places.place_details.map(lambda x: x.get("name", ""))
        unique_places["price_level"] = unique_places.place_details.map(lambda x: x.get("price_level", 0))
        unique_places["rating"] = unique_places.place_details.map(lambda x: x.get("rating", 0))
        unique_places["user_ratings_total"] = unique_places.place_details.map(lambda x: x.get("user_ratings_total", 0))
        unique_places["types"] = unique_places.place_details.map(lambda x: x.get("types", []))

        unique_places["overview"] = unique_places.place_details.map(lambda x: x.get("editorial_summary", {}).get("overview", ""))
        unique_places["serves_beer"] = unique_places.place_details.map(lambda x: x.get("serves_beer", False))
        unique_places["serves_wine"] = unique_places.place_details.map(lambda x: x.get("serves_wine", False))
        unique_places["serves_dinner"] = unique_places.place_details.map(lambda x: x.get("serves_dinner", False))
        unique_places["serves_lunch"] = unique_places.place_details.map(lambda x: x.get("serves_lunch", False))
        unique_places["serves_vegetarian_food"] = unique_places.place_details.map(lambda x: x.get("serves_vegetarian_food", False))
        unique_places["address"] = unique_places.place_details.map(lambda x: x.get("formatted_address", False))

        unique_places["url"] = unique_places.place_details.map(lambda x: x.get("url", ""))
        unique_places["lat"] = unique_places.place_details.map(lambda x: x.get("geometry", {}).get("location", {}).get("lat"))
        unique_places["lon"] = unique_places.place_details.map(lambda x: x.get("geometry", {}).get("location", {}).get("lng"))
        unique_places["document"] = unique_places.progress_apply(self._create_document, axis=1)
        return unique_places

    def _create_document(self, row):
        """Create a markdown style document from categorical features"""
        description = f"## Location \nName: {row.title}\n"
        if not pd.isna(row.overview):
            description += f"Overview: {row.overview}\n"
        if not pd.isna(row.address):
            description += f"Address: {row.address}\n"

        if not pd.isna(row.types):
            types = eval(row.types) if isinstance(row.types, str) and row.types.startswith("[") else row.types
            if len(types) > 0:
                description += "\n## Categories\n"
                for place_type in types:
                    if place_type not in [
                        "point_of_interest",
                        "establishment",
                        "store",
                    ]:
                        description += f"- {place_type.replace('_', ' ').title()}\n"
                description += "\n"

        description += "## Features\n"
        services = []
        if row.serves_beer:
            services.append("Beer")
        if row.serves_wine:
            services.append("Wine")
        if row.serves_dinner:
            services.append("Dinner")
        if row.serves_lunch:
            services.append("Lunch")
        if row.serves_vegetarian_food:
            services.append("Vegetarian Food")

        if services:
            description += f"- Serves: {', '.join(services)}.\n"

        price_level_map = {
            0: "Free",
            1: "Inexpensive",
            2: "Moderate",
            3: "Expensive",
            4: "Very Expensive",
        }
        if not pd.isna(row.price_level):
            description += f"- Price Level: {price_level_map[row.price_level]}\n"
        if not pd.isna(row.rating):
            description += f"- Rating: {row.rating}\n"
        if not pd.isna(row.user_ratings_total):
            description += f"- Total Ratings: {row.user_ratings_total}\n"

        return description


class GraphDataLoader:
    def __init__(self, distance_threshold_km: int = 2):
        self.distance_threshold_km = distance_threshold_km
        self.text_encoder = SentenceTransformer("Alibaba-NLP/gte-modernbert-base")
        self.file_path = self._get_path()

    def _get_path(self):
        """Get the path for the saved graph file"""
        data_dir = "data/processed"
        os.makedirs(data_dir, exist_ok=True)
        return os.path.join(data_dir, f"graph_threshold_{self.distance_threshold_km}.pkl")

    def save(self, G):
        """Save the graph and places DataFrame to disk"""
        data = {"graph": G}
        with open(self.file_path, "wb") as f:
            pickle.dump(data, f)

    def load(self):
        """Load the graph and places DataFrame from disk if it exists"""
        if os.path.exists(self.file_path):
            with open(self.file_path, "rb") as f:
                data = pickle.load(f)
            return data["graph"]
        return None

    def haversine_distance(self, lat1, lon1, lat2, lon2):
        """Calculate the distance between two points on Earth in kilometers"""
        R = 6371

        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        distance = R * c

        return distance

    def build(self, nodes, labels):
        """Build a graph where nodes are places and edges connect places within the distance threshold"""
        G = self.load()
        if G is not None:
            logger.success("Loaded existing graph from disk.")
            return G

        logger.info("Building new graph.")
        nodes = nodes.copy()
        logger.highlight(f"Initial nodes count: {len(nodes)}")
        logger.highlight(f"Initial labels count: {len(labels)}")

        # Convert labels to lowercase for case-insensitive comparison
        labels_lower = {str(l).lower() for l in labels}
        nodes["is_favorite"] = nodes["place_id"].str.lower().isin(labels_lower).astype(int)

        logger.highlight(f"Nodes with is_favorite=1: {nodes['is_favorite'].sum()}")
        logger.highlight(f"Unique place_ids in nodes: {nodes['place_id'].nunique()}")
        logger.highlight(f"Nodes with missing lat/lon: {nodes[pd.isna(nodes['lat']) | pd.isna(nodes['lon'])].shape[0]}")

        text_descriptions = [row.document for _, row in nodes.iterrows()]
        text_embeddings = self.text_encoder.encode(text_descriptions, show_progress_bar=True)
        logger.highlight(f"Generated embeddings count: {len(text_embeddings)}")

        G = nx.Graph()
        for idx, (row, embedding) in enumerate(zip(nodes.iterrows(), text_embeddings)):
            _, row = row
            node_features = np.concatenate([embedding])
            G.add_node(idx, features=node_features)
            G.nodes[idx]["label"] = row.is_favorite

        logger.highlight(f"Graph nodes count: {G.number_of_nodes()}")
        logger.highlight(f"Graph nodes with label=1: {sum(1 for _, d in G.nodes(data=True) if d['label'] == 1)}")

        edge_count = 0
        skipped_edges = 0
        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                if pd.isna(nodes.iloc[i].lat) or pd.isna(nodes.iloc[i].lon) or pd.isna(nodes.iloc[j].lat) or pd.isna(nodes.iloc[j].lon):
                    skipped_edges += 1
                    continue

                distance = self.haversine_distance(
                    nodes.iloc[i].lat,
                    nodes.iloc[i].lon,
                    nodes.iloc[j].lat,
                    nodes.iloc[j].lon,
                )

                if distance <= self.distance_threshold_km:
                    G.add_edge(i, j, weight=1.0 / (1 + distance))
                    edge_count += 1

        logger.highlight(f"Added {edge_count} edges")
        self.save(G)
        return G

    def prepare_pytorch_geometric_data(self, G):
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

        x = torch.tensor([G.nodes[node]["features"] for node in G.nodes()], dtype=torch.float)
        y = torch.tensor([G.nodes[node]["label"] for node in G.nodes()], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)

        return data

    def create_train_val_test_masks(self, data, train_ratio=0.7, val_ratio=0.15):
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
        val_indices_0 = class_0_indices[train_size_0 : train_size_0 + val_size_0]
        test_indices_0 = class_0_indices[train_size_0 + val_size_0 :]

        train_indices_1 = class_1_indices[:train_size_1]
        val_indices_1 = class_1_indices[train_size_1 : train_size_1 + val_size_1]
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
