## Unlocking the Musical Map of LastFM Asia: Evidence from Graph Neural Network

We dive into a social network in [LastFM](https://www.last.fm/), an online music database and music social network platform, and explore user music tastes using [Graph Neural Network](https://ieeexplore.ieee.org/document/4700287) (GNN). We are interested in building a user-user and user-artist recommendation system, and explaining our recommendations with a GNN interpretation tool, [GNNExplainer](https://proceedings.neurips.cc/paper/2019/file/d80b7040b773199015de6d3b4293c8ff-Paper.pdf).

---

**Data** [[Link](https://github.com/benedekrozemberczki/datasets#lastfm-asia-social-network)]: A social network of LastFM users collected from the public API in March 2020. 

**Graph dataset overview**:
* Directed graph without self-loops
* 7,624 nodes (users)
* 27,806 edges (user-user connections)
* 7,842 features (binary entries indicating whether following an artist or not)
* 18 communities (user regions)

<p align='center'>
  <img width='500' src='img/graph_viz.png'>
</p>

**Question**:
* Q1. Friend recommendation / user-user recommendation:
  * **Can we predict a user's region based on the user feature and connections?**
  * EDA shows that users tend to connect with those from the same region. This provides supportive evidence on recommending users within the same region, at the same time showing potential opportunities connecting users from different regions.
* Q2. Artist recommendation / user-artist recommendation:
  * **Can we predict a user's music taste (artist preference) based on the user's connections?**
  * Connections indicate similarity in graphs. Learning a user's connections' followed artists help undertand the user's taste.

**Model architecture**:
* General architecture: `input` -> `graph encoder` -> `node embedding` -> `classifier` -> `output`   
  * Q1. Region predtion: GNN-based multi-class classification.   
    <img width='400' src='img/multi_class_model.png'>

  * Q2. Feature prediction: GNN-based multi-label classification.   
    <img width='400' src='img/multi_label_model.png'>

**Results (as of April 5, 2023)**:
* Multi-class classification accuracy: 80.85%
* Multi-label classification accuracy (with [Hamming Distance](https://torchmetrics.readthedocs.io/en/stable/classification/hamming_distance.html)): 94.91%

**To-do**:
* Fine-tune two GNN models for better model performance
* Apply GNNExplainer to interpret model predictions
