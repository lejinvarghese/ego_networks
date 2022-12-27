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


class Configuration:
    logger = CustomLogger(__name__)
    twitter_username = os.getenv("TWITTER_USERNAME")
    twitter_api_bearer_token = os.getenv("TWITTER_API_BEARER_TOKEN")
    twitter_focal_node_id = os.getenv("TWITTER_FOCAL_NODE_ID")
    cloud_storage_bucket = os.getenv("CLOUD_STORAGE_BUCKET")
    max_radius = 2


class Controller(Configuration):
    def __init__(self):
        self.network = HomogenousEgoNetwork(
            focal_node_id=self.twitter_focal_node_id,
            radius=self.max_radius,
        )
        self.recommender = EgoNetworkRecommender(use_cache=True)

    def update_neighborhood(self, mode: str = "append"):
        _neighborhoods = [
            TwitterEgoNeighborhood(
                focal_node=self.twitter_username,
                max_radius=self.max_radius,
                api_bearer_token=self.twitter_api_bearer_token,
            )
        ]
        for n in _neighborhoods:
            n.update_neighborhood(mode=mode)

        self.network = HomogenousEgoNetwork(
            focal_node_id=self.twitter_focal_node_id,
            radius=self.max_radius,
        )

    def update_measures(
        self, calculate_nodes: bool = True, calculate_edges: bool = True
    ):
        _measures = self.network.create_measures(
            calculate_nodes=calculate_nodes, calculate_edges=calculate_edges
        )
        self.recommender = EgoNetworkRecommender(
            network_measures=_measures.node_measures
        )

    def update_recommendations(
        self,
        recommendation_strategy: str = "connectors",
        evaluate: bool = False,
        n_recommendations: int = 10,
    ):
        self.recommender.train(recommendation_strategy)
        _targets = list(
            self.network.get_ego_user_attributes(
                radius=1, attribute="username"
            ).keys()
        )
        if evaluate:
            self.recommender.test(_targets)

        _rec_profile_ids = self.recommender.get_recommendations(
            _targets, k=n_recommendations
        )
        (
            rec_profile_names,
            rec_profile_images,
        ) = self.__get_profile_information(_rec_profile_ids)
        self.logger.info(f"Recommendations: {rec_profile_names}")
        return rec_profile_names, rec_profile_images

    def __get_profile_information(self, profile_ids: list):
        _names = self.network.get_ego_user_attributes(
            radius=2, attribute="username"
        )
        _images = self.network.get_ego_user_attributes(
            radius=2, attribute="profile_image_url"
        )
        return [_names.get(i) for i in profile_ids], [
            _images.get(i) for i in profile_ids
        ]
