import os
from warnings import filterwarnings

from dotenv import load_dotenv

try:
    from src.neighborhoods.twitter import TwitterEgoNeighborhood
    from src.network import HomogenousEgoNetwork
    from src.recommender import EgoNetworkRecommender
    from utils.custom_logger import CustomLogger
except ModuleNotFoundError:
    from ego_networks.src.neighborhoods.twitter import TwitterEgoNeighborhood
    from ego_networks.src.network import HomogenousEgoNetwork
    from ego_networks.src.recommender import EgoNetworkRecommender
    from ego_networks.utils.custom_logger import CustomLogger

load_dotenv()
filterwarnings("ignore")
logger = CustomLogger(__name__)

TWITTER_USERNAME = os.getenv("TWITTER_USERNAME")
TWITTER_API_BEARER_TOKEN = os.getenv("TWITTER_API_BEARER_TOKEN")
TWITTER_FOCAL_NODE_ID = os.getenv("TWITTER_FOCAL_NODE_ID")
CLOUD_STORAGE_BUCKET = os.getenv("CLOUD_STORAGE_BUCKET")
MAX_RADIUS = 2


def main(
    k: int = 10,
    update_neighborhood: bool = False,
    update_measures: bool = False,
    update_recommendations: bool = True,
    recommendation_strategy: str = "diverse",
):
    if update_neighborhood:
        twitter_hood = TwitterEgoNeighborhood(
            focal_node=TWITTER_USERNAME,
            max_radius=2,
            api_bearer_token=TWITTER_API_BEARER_TOKEN,
        )
        twitter_hood.update_neighborhood(mode="append")

    network = HomogenousEgoNetwork(
        focal_node_id=TWITTER_FOCAL_NODE_ID,
        radius=MAX_RADIUS,
    )

    if update_measures:
        measures = network.create_measures(
            calculate_nodes=True, calculate_edges=True
        )
        recommender = EgoNetworkRecommender(
            network_measures=measures.node_measures
        )
    else:
        recommender = EgoNetworkRecommender(use_cache=True)

    if update_recommendations:
        targets = network.get_ego_user_attributes(
            radius=1, attribute="username"
        )
        labels = network.get_ego_user_attributes(radius=2, attribute="username")
        profile_images = network.get_ego_user_attributes(
            radius=2, attribute="profile_image_url"
        )
        recommender.train(recommendation_strategy)
        recommender.test(targets)
        (
            recommended_profiles,
            recommended_profile_images,
        ) = recommender.get_recommendations(
            targets, labels, profile_images, k=k
        )
        logger.info(recommended_profiles)
        return recommended_profiles, recommended_profile_images
    else:
        logger.info("No recommendations updated during this run.")


if __name__ == "__main__":
    main(
        k=20,
        update_neighborhood=False,
        update_measures=False,
        update_recommendations=True,
        recommendation_strategy="diverse",
    )
