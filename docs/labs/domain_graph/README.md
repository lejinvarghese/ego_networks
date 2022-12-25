# Instructions

## Setup

```bash
cd docs/labs/domain_graph
CONDA_ENV_NAME=conda-env-domains
CONDA_ENV_FILE=env.yml
conda env create --name ${CONDA_ENV_NAME} --file=${CONDA_ENV_FILE} 
```

## Run

```bash
conda activate ${CONDA_ENV_NAME}
python3 -m docs.labs.domain_graph.main
```

## Output

![x](./figure.png)