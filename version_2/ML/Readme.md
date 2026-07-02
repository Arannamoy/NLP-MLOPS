### Types of ML
#### Supervised: Labeled Data 
- kNN
- Logistic Regression
- Linear Regression
- Support Vector Machines
- Decision Tree and Random Forest
- Neural Networks
#### Unsupervised: Unlabeled data
- Clustering
    - K means
    - DBSCAN
    - Hierachical Cluster Analysis
- Anomaly detection and novelty detection
    - One Class SVM
    - Isolation Forest
- Visualization and dimensionality reduction
    - Principal Component Analysis (PCA)
    - Kernel PCA
    - Locally Linear Embedding (LLE)
    - t-Distributed Stochastic Neighbor Embedding (t-SNE)
- Association rule learning
   - Apriori
   - Eclat
#### Semisupervised: Labeled data and unlabeled data
#### Reinforcement: Rewards and pelnalties

#### Batch Learning(Offline Learning) vs Online Learning
|Batch Learning|Online Learning|
|--------------|---------------|
|Trains on the entire dataset at once|Trains on data instances sequentially (one by one or in small groups)|
|Requires full retraining to incorporate new data|Updates weights instantly as new data arrives|
|Does not learn from new live data|Continuously learns and adapts in real-time|
|High (Requires lots of RAM/Computing power)|Low (Uses small chunks of data, then discards them)|
|Train, then deploy|Deploy and keep training|

#### Instance-Based Learning vs Model-Based Learning
|Feature|Instance-Based Learning|Model-Based Learning|
|-------|-----------------------|--------------------|
|Data Storage|Must keep all (or most) training data|Can discard data after training; only keep the model|
|Training Speed|Practically instant (just saving data)|Slower (requires optimization/gradient descent)|
|Prediction Speed|Slower (must compare to all data)|Very fast (just applying a formula)|
|Generalization|Uses similarity (e.g. Euclidean distance)|Uses a predictive model (e.g., y=mx+c)|
|Complexity|Becomes very slow as data grows|Complexity depends on the model architecture, not data size|

#### NFL
- If you make absolutely no assumption about the data, then there is no reason to prefer one model over any
other.

#### Pipeline
- A sequence of data processing components is called a data pipeline.