import os

import pandas as pd
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
            df["Title"] = df["properties"].map(
                lambda x: x.get("location", {}).get("name")
            )
            df = df[["Title"]].dropna()
        elif file == "reviews":
            df = pd.read_json(f"data/{file}.json")
            df = pd.DataFrame(list(df.features))
            df["rating"] = df["properties"].map(
                lambda x: x.get("five_star_rating_published")
            )
            df["Title"] = df["properties"].map(
                lambda x: x.get("location", {}).get("name")
            )
            df = df[["Title", "rating"]].dropna()
        elif file == "unique_places":
            df = self._extract_place_details()
        else:
            df = pd.read_csv(f"data/{file}.csv")[["Title"]]

        if file != "unique_places":
            df["place_id"] = df["Title"].progress_map(
                lambda x: self.api.find_place(x)
            )
        df.columns = [f"{str.lower(col)}" for col in df.columns]
        df.to_csv(file_path, index=False)
        return df

    def _extract_place_details(self):
        """Extract place details from the datasets"""
        unique_places = (
            pd.concat([v for _, v in self.datasets.items()])[["place_id"]]
            .dropna()
            .drop_duplicates()
        )
        unique_places["place_details"] = unique_places.place_id.progress_map(
            lambda x: self.api.get_place_details(x).get("result", {})
        )
        unique_places["title"] = unique_places.place_details.map(
            lambda x: x.get("name", "")
        )
        unique_places["price_level"] = unique_places.place_details.map(
            lambda x: x.get("price_level", 0)
        )
        unique_places["rating"] = unique_places.place_details.map(
            lambda x: x.get("rating", 0)
        )
        unique_places["user_ratings_total"] = unique_places.place_details.map(
            lambda x: x.get("user_ratings_total", 0)
        )
        unique_places["types"] = unique_places.place_details.map(
            lambda x: x.get("types", [])
        )

        unique_places["overview"] = unique_places.place_details.map(
            lambda x: x.get("editorial_summary", {}).get("overview", "")
        )
        unique_places["serves_beer"] = unique_places.place_details.map(
            lambda x: x.get("serves_beer", False)
        )
        unique_places["serves_wine"] = unique_places.place_details.map(
            lambda x: x.get("serves_wine", False)
        )
        unique_places["serves_dinner"] = unique_places.place_details.map(
            lambda x: x.get("serves_dinner", False)
        )
        unique_places["serves_lunch"] = unique_places.place_details.map(
            lambda x: x.get("serves_lunch", False)
        )
        unique_places["serves_vegetarian_food"] = (
            unique_places.place_details.map(
                lambda x: x.get("serves_vegetarian_food", False)
            )
        )
        unique_places["address"] = unique_places.place_details.map(
            lambda x: x.get("formatted_address", False)
        )

        unique_places["url"] = unique_places.place_details.map(
            lambda x: x.get("url", "")
        )
        unique_places["lat"] = unique_places.place_details.map(
            lambda x: x.get("geometry", {}).get("location", {}).get("lat")
        )
        unique_places["lon"] = unique_places.place_details.map(
            lambda x: x.get("geometry", {}).get("location", {}).get("lng")
        )
        unique_places["document"] = unique_places.progress_apply(
            self._create_document, axis=1
        )
        return unique_places

    def _create_document(self, row):
        """Create a markdown style document from categorical features"""
        description = f"## Location \nName: {row.title}\n"
        if not pd.isna(row.overview):
            description += f"Overview: {row.overview}\n"
        if not pd.isna(row.address):
            description += f"Address: {row.address}\n"

        if not pd.isna(row.types):
            types = (
                eval(row.types)
                if isinstance(row.types, str) and row.types.startswith("[")
                else row.types
            )
            if len(types) > 0:
                description += "\n## Categories\n"
                for place_type in types:
                    if place_type not in [
                        "point_of_interest",
                        "establishment",
                        "store",
                    ]:
                        description += (
                            f"- {place_type.replace('_', ' ').title()}\n"
                        )
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
            description += (
                f"- Price Level: {price_level_map[row.price_level]}\n"
            )
        if not pd.isna(row.rating):
            description += f"- Rating: {row.rating}\n"
        if not pd.isna(row.user_ratings_total):
            description += f"- Total Ratings: {row.user_ratings_total}\n"

        return description
