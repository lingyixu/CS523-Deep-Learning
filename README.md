# Unlocking the Musical Map of LastFM Asia: Evidence from Graph Neural Network

We dive into a social network of [LastFM](https://www.last.fm/), an online music database and music social network platform, and explore user music tastes using [Graph Neural Network](https://ieeexplore.ieee.org/document/4700287) (GNN). We are interested in building a user-user and user-artist recommendation system, and explaining our recommendations with GNN interpretation tools, [GAT](https://arxiv.org/abs/1710.10903) and [_GNNExplainer_](https://proceedings.neurips.cc/paper/2019/file/d80b7040b773199015de6d3b4293c8ff-Paper.pdf).

<img width='900' src='img/flowchart.png'>

# 

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

#

**Question**:
* Q1. Friend recommendation (user-user recommendation):
  * **Can we predict a user's region based on the user feature and connections?**
  * EDA [[notebook]](https://github.com/lingyixu/CS523-Deep-Learning/blob/main/last-FM_dataset/523Dateset.ipynb) shows that users tend to connect with those from the same region. This provides supportive evidence on recommending users within the same region, at the same time showing potential opportunities connecting users from different regions.
* Q2. Content recommendation (user-artist recommendation):
  * **Can we predict a user's music taste (artist preference) based on the user's connections?**
  * Connections indicate similarity in graphs. Learning a user's connections' followed artists help undertand the user's taste.

#

**Model architecture**:
* General architecture: `input` -> `graph encoder` -> `node embedding` -> `classifier` -> `output`   
  * Q1. Region predtion: GNN-based multi-class classification (left), Benchmark: MLP (right)   
    
    <img width='200' src='img/multi_class_model_new.png'> &nbsp;
    <img width='200' src='img/MLP.png'>

  * Q2. Artist prediction: GNN-based multi-label classification.   
    
    <img width='200' src='img/multi_label_model_new.png'>

#

**Training strategy**:
* **Cross validation**: randomly divide the nodes into 5 non-overlapping folds. Train:test = 4:1 in each fold.
* **Regularization**: use L2 regularization (weight decay) to compress large weights in case of overfitting.
* **Hyperparameter tuning**: implement grid-search for combinations of learning rate, weight decay, number of epochs, and number of nodes in each graph and fully connected layer of the models.
* **Masking**: prevent information leakage during message passing while making the best use of all available information. See a visualized explanation below.   
  <p align='center'>
  <img width='800' src='img/masking.png'>
  </p>
  
#

**Model interpretation**:
* **Edge importance**: self-attention scores learned by GAT.    
  See a visualization for node (user) #1005 below, where arrows indicate edge directions and edge thinkness indicate level of importance.   
  <p align='center'>
  <img width='400' src='img/GAT_edge_contribution.png'>
  </p>
* **Feature importance**: a subset of features that contribute most to model predictions learned by _GNNExplainer_.    
  See feature importance visualization for the overall dataset (left) and randomly picked individual users (right) below.   
  
  <p align='center'>
  <img width='450' src='img/feature_importance_overall.png'>
  <img width='370' src='img/feature_importance_individual.png'>
  </p>

#

**Progress**:
* As of April 18, 2023:
  * Update with cross validation: [notebook](https://github.com/lingyixu/CS523-Deep-Learning/blob/main/GNN/GCN_Multilabel_Classification_5Fold_CV.ipynb)
  * Update with other graph convolutaional methods: [GAT](https://github.com/lingyixu/CS523-Deep-Learning/blob/main/GNN/GAT_Multilabel_Classification_5Fold_CV.ipynb)
  * Update with benchmark graph model interpretation approach: edge weights with GAT
* As of April 5, 2023:
  * Multi-class classification accuracy [[notebook]](https://github.com/lingyixu/CS523-Deep-Learning/blob/main/GNN/GCN_Multiclass_Classification.ipynb): 80.85% 
  * Multi-label classification accuracy (with [Hamming Distance](https://torchmetrics.readthedocs.io/en/stable/classification/hamming_distance.html)) [[notebook]](https://github.com/lingyixu/CS523-Deep-Learning/blob/main/GNN/GCN_Multilabel_Classification.ipynb): 94.91%


# 

### References:
* Scarselli, F., Gori, M., Tsoi, A. C., Hagenbuchner, M., & Monfardini, G. (2008). The graph neural network model. _IEEE transactions on neural networks_, 20(1), 61-80.
* Veličković, P., Cucurull, G., Casanova, A., Romero, A., Lio, P., & Bengio, Y. (2017). Graph attention networks. _arXiv preprint_ arXiv:1710.10903.
* Ying, Z., Bourgeois, D., You, J., Zitnik, M., & Leskovec, J. (2019). Gnnexplainer: Generating explanations for graph neural networks. _Advances in neural information processing systems_, 32.
* Rozemberczki, B., & Sarkar, R. (2020, October). Characteristic functions on graphs: Birds of a feather, from statistical descriptors to parametric models. In _Proceedings of the 29th ACM international conference on information & knowledge management_ (pp. 1325-1334).
