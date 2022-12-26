from warnings import filterwarnings

try:
    from src.controller import Controller
except ModuleNotFoundError:
    from ego_networks.src.controller import Controller

filterwarnings("ignore")

if __name__ == "__main__":
    c = Controller()
    c.update_recommendations()
