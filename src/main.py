import os
from warnings import filterwarnings

from dotenv import load_dotenv

try:
    from src.neighborhoods.twitter import TwitterEgoNeighborhood
    from src.network import HomogenousEgoNetwork
except ModuleNotFoundError:
    from ego_networks.src.neighborhoods.twitter import TwitterEgoNeighborhood
    from ego_networks.src.network import HomogenousEgoNetwork

load_dotenv()
filterwarnings("ignore")
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
    measures = network.create_measures(
        calculate_nodes=True, calculate_edges=True
    )

    measures.summary_measures.to_csv(
        f"{CLOUD_STORAGE_BUCKET}/data/processed/measures/{MAX_RADIUS}/summary_measures.csv",
        index=False,
    )

    measures.node_measures.to_csv(
        f"{CLOUD_STORAGE_BUCKET}/data/processed/measures/{MAX_RADIUS}/node_measures.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
