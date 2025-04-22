import os
import requests
import json
from urllib.parse import quote_plus
from utils import logger

from dotenv import load_dotenv

load_dotenv()

GOOGLE_MAPS_API_KEY = os.getenv("GOOGLE_MAPS_API_KEY")


class GooglePlacesAPI:
    def __init__(self, api_key: str = GOOGLE_MAPS_API_KEY):
        self.api_key = api_key
        self.base_url = "https://maps.googleapis.com/maps/api/place"

    def find_place(self, place_name: str, location: str = "Toronto") -> str:
        """Find a place by name and location"""
        try:
            place = quote_plus(place_name)
            url = f"{self.base_url}/findplacefromtext/json?input={place}+{location}&inputtype=textquery&key={self.api_key}"
            response = requests.get(url)
            return json.loads(response.text)["candidates"][0]["place_id"]
        except Exception as e:
            logger.error(f"Error finding place: {place_name} {e}")
            return None

    def get_place_details(self, place_id: str) -> dict:
        """Get details for a place by place ID"""
        try:
            url = f"{self.base_url}/details/json?place_id={place_id}&key={self.api_key}"
            response = requests.get(url)
            return json.loads(response.text)
        except Exception as e:
            logger.error(f"Error getting place details: {place_id} {e}")
            return None

    def ping(self, location: str = "la palette"):
        """Health check to use the API to get place details"""
        place_id = self.find_place(location)
        if place_id:
            details = self.get_place_details(place_id)
            logger.highlight(f"Place details: {json.dumps(details, indent=4)}")
