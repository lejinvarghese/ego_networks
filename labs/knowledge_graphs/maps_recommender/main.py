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

    def ping(self, location: str = "cn tower"):
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
                    df["Title"] = df["properties"].progress_apply(lambda x: x.get("location", {}).get("name"))
                    df = df[["Title"]].dropna()
                elif file == "reviews":
                    df = pd.read_json(f"data/{file}.json")
                    df = pd.DataFrame(list(df.features))
                    df["rating"] = df["properties"].progress_map(lambda x: x.get("five_star_rating_published"))
                    df["Title"] = df["properties"].progress_apply(lambda x: x.get("location", {}).get("name"))
                    df = df[["Title", "rating"]].dropna()
                else:
                    df = pd.read_csv(f"data/{file}.csv")[["Title"]]

                df["place_id"] = df["Title"].progress_map(lambda x: self.places.find_place(x))
                df.columns = [f"{str.lower(col)}" for col in df.columns]
                df.to_csv(output_file, index=False)
                self.datasets[file] = df
            self.datasets["unique_places"] = (
                pd.concat([v for _, v in self.datasets.items()]).place_id.dropna().drop_duplicates()
            )
        return self.datasets


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
