python3 -m venv .venv
source .venv/bin/activate

pip3 install -r requirements.txt
pip3 install torch==1.11.0+cpu torchvision==0.12.0+cpu torchaudio==0.11.0+cpu -f https://download.pytorch.org/whl/cpu/torch_stable.html
pip3 install torch-scatter -f https://data.pyg.org/whl/torch-1.11.0+cpu.html
pip3 install torch-sparse -f https://data.pyg.org/whl/torch-1.11.0+cpu.html
pip3 install torch-geometric
ipython kernel install --name "twitter_network" --user
