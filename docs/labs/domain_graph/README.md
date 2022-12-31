# Instructions

## Setup

```bash
cd docs/labs/domain_graph
CONDA_ENV_NAME=conda-env-domains
CONDA_ENV_FILE=env.yml
conda env create --name ${CONDA_ENV_NAME} --file=${CONDA_ENV_FILE} 
```

#### Optional

A local `.env` file with the following variables:

```bash
GOODREADS_FOCAL_NODE_ID=<PROFILE_ID>
```

if not, modify the `get_books` function in the `docs/labs/domain_graph/nodes.py` to return a list of book descriptions.

## Run

```bash
conda activate ${CONDA_ENV_NAME}
python3 -m docs.labs.domain_graph.main
```

## Output

![x](./figure.png)

#### Legend

```markdown
`node color`: community affiliation
`node size`: node betweenness centrality
`edge color`: community affiliation
  - `within community`: same as community nodes
  - `between community`: grey
`node line thickness`: recent relative focus
```