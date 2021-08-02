# conda create --name ampligraph python=3.7
conda activate ampligraph
pip install ampligraph jupyterlab
conda install --file requirements.txt
ipython kernel install --name "ampligraph" --user
jupyter lab --notebook-dir=. --port 8080 --ip 0.0.0.0 --ContentsManager.allow_hidden=True