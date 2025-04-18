import os
import click
from dotenv import load_dotenv
import requests
import json
import pandas as pd
from urllib.parse import quote_plus
from tqdm import tqdm

import ampligraph

load_dotenv()
tqdm.pandas()

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")


class GooglePlacesAPI:
    def __init__(self, api_key: str = GOOGLE_MAPS_API_KEY):
        self.api_key = api_key
        self.base_url = "https://maps.googleapis.com/maps/api/place"
        # self.base_url = "https://places.googleapis.com/v1/places"

    def find_place(self, place_name: str, location: str = "Toronto") -> str:
        try:
            place = quote_plus(place_name)
            url = f"{self.base_url}/findplacefromtext/json?input={place}+{location}&inputtype=textquery&key={self.api_key}"
            response = requests.get(url)
            return json.loads(response.text)["candidates"][0]["place_id"]
        except Exception as e:
            click.secho(f"Error finding place: {place_name} {e}", fg="red")
            return None

    def get_place_details(self, place_id: str) -> dict:
        try:
            url = f"{self.base_url}/details/json?place_id={place_id}&key={self.api_key}"
            response = requests.get(url)
            return json.loads(response.text)
        except Exception as e:
            click.secho(f"Error getting place details: {place_id} {e}", fg="red")
            return None

    def ping(self, location: str = "la palette"):
        place_id = self.find_place(location)
        if place_id:
            details = self.get_place_details(place_id)
            click.secho(f"Place details: {json.dumps(details, indent=4)}", fg="blue")


class PersonalPlacesProcessor:
    def __init__(self):
        self.places = GooglePlacesAPI()
        self.datasets = {}

    def process(self):
        for file in [
            "want_to_go",
            "favorite_places",
            "saved_places",
            "reviews",
        ]:
            output_file = f"data/processed/{file}.csv"
            if os.path.exists(output_file):
                click.secho(f"Reading {file} because it exists.", fg="yellow")
                self.datasets[file] = pd.read_csv(output_file)
            else:
                click.secho(f"Processing {file}", fg="green")
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
                else:
                    df = pd.read_csv(f"data/{file}.csv")[["Title"]]

                df["place_id"] = df["Title"].progress_map(lambda x: self.places.find_place(x))
                df.columns = [f"{str.lower(col)}" for col in df.columns]
                df.to_csv(output_file, index=False)
                self.datasets[file] = df

        output_file = f"data/processed/unique_places.csv"
        if os.path.exists(output_file):
            click.secho(f"Reading {file} because it exists.", fg="yellow")
            unique_places = pd.read_csv(output_file)
        else:
            unique_places = self.get_unique_place_details()
            unique_places.to_csv(output_file, index=False)
        self.datasets["unique_places"] = unique_places
        return self.datasets

    def get_unique_place_details(self):
        unique_places = pd.concat([v for _, v in self.datasets.items()])[["place_id"]].dropna().drop_duplicates()
        unique_places["place_details"] = unique_places.place_id.progress_map(
            lambda x: self.places.get_place_details(x).get("result", {})
        )

        unique_places["price_level"] = unique_places.place_details.map(lambda x: x.get("price_level", 0))
        unique_places["rating"] = unique_places.place_details.map(lambda x: x.get("rating", 0))
        unique_places["types"] = unique_places.place_details.map(lambda x: x.get("types", []))

        unique_places["overview"] = unique_places.place_details.map(
            lambda x: x.get("editorial_summary", {}).get("overview", "")
        )
        unique_places["serves_beer"] = unique_places.place_details.map(lambda x: x.get("serves_beer", False))
        unique_places["serves_wine"] = unique_places.place_details.map(lambda x: x.get("serves_wine", False))
        unique_places["serves_dinner"] = unique_places.place_details.map(lambda x: x.get("serves_dinner", False))
        unique_places["serves_lunch"] = unique_places.place_details.map(lambda x: x.get("serves_lunch", False))
        unique_places["serves_vegetarian_food"] = unique_places.place_details.map(
            lambda x: x.get("serves_vegetarian_food", False)
        )
        unique_places["address"] = unique_places.place_details.map(lambda x: x.get("formatted_address", False))

        unique_places["url"] = unique_places.place_details.map(lambda x: x.get("url", ""))
        unique_places["lat"] = unique_places.place_details.map(
            lambda x: x.get("geometry", {}).get("location", {}).get("lat")
        )
        unique_places["lon"] = unique_places.place_details.map(
            lambda x: x.get("geometry", {}).get("location", {}).get("lng")
        )
        unique_places["user_ratings_total"] = unique_places.place_details.map(lambda x: x.get("user_ratings_total", 0))

        return unique_places


@click.command()
@click.option("--location", type=str, help="Location to search for")
def main(location):
    gp = GooglePlacesAPI()
    gp.ping()
    processor = PersonalPlacesProcessor()
    datasets = processor.process()

    for file, df in datasets.items():
        click.secho(f"{file} has {len(df)} records", fg="yellow")
        click.secho(df.head(), fg="green")


if __name__ == "__main__":
    main()
