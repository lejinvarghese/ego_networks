# Ego Networks

[![codecov](https://codecov.io/gh/lejinvarghese/ego_networks/branch/master/graph/badge.svg?token=248C9C6ZHK)](https://codecov.io/gh/lejinvarghese/ego_networks)

<p align="center">
    <img src="./assets/sample.png" alt="sample" width="500"/>
</p>


## Objectives

This project is a broad effort to give an individual control over what information they consume, what sub communities they're connected, and how information diffusion over networks might affect their perspective. We want to study information flow and belief propogation through complex networks, help people find `highly personalized communities` from their immediate ego network, but also `avoid echo chambers, filter bubbles`.

-   We start by creating the two step neighborhood network for `Twitter`.
    -   We only consider the `out neighbors`, i.e. who the user follows (or friends?), the intent being that it's who the user follows matter more than who follows the user.
    -   However, the `information flow` is inward.
-    The framework is designed to be extensible to other social networks and other types of content and idea networks.
     -   Work is in progress to extend this to a `heterogenous network` of multiple entities such as `people, content, communities, ideas` and differing relationships between them.
     -   This could also extend to a complex [multiplex network](https://cosnet.bifi.es/network-theory/multiplex-networks/), in which information would flow through multiple layers of the network (imagine real and virtual multi layered networks) with differing diffusion patterns.

<p align="center">
    <img src="https://cosnet.bifi.es/wp-content/uploads/2014/06/multiplex_networks_2a.jpg" alt="ego" width="500"/>
</p>

-   Primary goal is to study (through observation and simulation) information diffusion, learning and it's effects on users, potentially through [Degroot Learning](https://en.wikipedia.org/wiki/DeGroot_learning) or through [Bayesian Learning](https://en.wikipedia.org/wiki/Mathematical_models_of_social_learning) models.
    -   [Stanford](https://github.com/lejinvarghese/graph_data_science/blob/master/docs/social_economic_networks/w6-learning.pdf)
    -   [MIT](https://economics.mit.edu/files/4902)

<p align="center">
    <img src="https://bldavies.com/blog/degroot-learning-social-networks/figures/example-1.svg" alt="ego" width="500"/>
</p>


## Run

### Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Source Code

```bash
source .venv/bin/activate
python3 -m src.main
```

### Streamlit Application

```bash
source .venv/bin/activate
streamlit run 0_🏠_home.py
```

## Twitter User Recommendations

<p float="left" align="middle">
        <img src="./assets/recs_strategy_diverse.png" alt="sample" width="220"/>
        <img src="./assets/recs_strategy_connectors.png" alt="sample" width="220"/>
        <img src="./assets/recs_strategy_influencers.png" alt="sample" width="220"/>
</p>

### Observations

- The `diverse` recommendation algorithms are tuned to use network measures to surface a list spread across ideological diversity, which can be seen in these top three recommendations.
- The `connector` algorithms are tuned to surface users who are likely to be network integrators, which in this case have a higher proportion of scientific and government institutions.
- The `influencer` algorithms are tuned to surface users who are likely to be popular ideological influencers.
- These will in future evolve to include more nuanced strategies and measures.

## Inspiration

- [Gobo](https://www.media.mit.edu/projects/gobo/overview/) by Ethan Zuckerman et al., MIT Media Lab
  - [GitHub: inactive](https://github.com/mitmedialab/gobo)
- [Information Centrality](https://www.researchgate.net/publication/329414133_Understanding_Information_Centrality_Metric_A_Simulation_Approach)
- [Centrality Measures in Social Networks](https://thesai.org/Publications/ViewPaper?Volume=10&Issue=1&Code=IJACSA&SerialNo=13)