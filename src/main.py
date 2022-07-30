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
INTEGRATED_FOCAL_NODE_ID = os.getenv("INTEGRATED_FOCAL_NODE_ID")
CLOUD_STORAGE_BUCKET = os.getenv("CLOUD_STORAGE_BUCKET")
MAX_RADIUS = 2


def main():
    # twitter_hood = TwitterEgoNeighborhood(
    #     focal_node=TWITTER_USERNAME,
    #     max_radius=2,
    #     api_bearer_token=TWITTER_API_BEARER_TOKEN,
    #     storage_bucket=CLOUD_STORAGE_BUCKET,
    # )
    # twitter_hood.update_neighborhood()

    network = HomogenousEgoNetwork(
        focal_node_id=INTEGRATED_FOCAL_NODE_ID,
        radius=MAX_RADIUS,
        storage_bucket=CLOUD_STORAGE_BUCKET,
    )
    targets = network.get_ego_user_attributes(radius=1, attribute="username")
    labels = network.get_ego_user_attributes(radius=2, attribute="username")
    measures = None
    # measures = network.create_measures(
    #     calculate_nodes=True, calculate_edges=True
    # )

    if measures:
        recommender = EgoNetworkRecommender(
            network_measures=measures.node_measures
        )
    else:
        recommender = EgoNetworkRecommender(storage_bucket=CLOUD_STORAGE_BUCKET)
    recommender.train()
    recommender.test(targets)
    recommendations = recommender.get_recommendations(targets, labels, k=25)
    logger.info(recommendations)


if __name__ == "__main__":
    main()
