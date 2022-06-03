# MA-BDA - Anomaly Detection in Network Traffic with K-means Clustering

###### Alt Thibaud, Bueche Lucas | Deadline on Friday 10.06.2022 | [github.com/weevood/MA-BDA_Project](https://github.com/weevood/MA-BDA_Project)

## Summary

The goal is to **detect anomalous behavior in the network traffic of an organization**. Anomalous behavior can point to things like intrusion attempts, denial-of-service attacks, port scanning, etc.

The dataset was generated for a data analysis competition from raw TCP dump data for a local-area network (LAN) simulating a typical U.S. Air Force LAN. The LAN was operated as if it were a true Air Force environment, but peppered with multiple attacks. Feature extraction was already run on the data, the dataset contains a list of connections, and for each connection 38 features, like the number of bytes sent, login attempts, TCP errors, and so on.

As the data is not labeled, an unsupervised learning algorithm is applied, more specifically *K-means clustering*. The idea is to let the clustering discover normal behavior. Connections that fall outside of clusters are potentially anomalous.

## Documentation

### 1. Description of the dataset (size, information it contains)

The KDD Cup dataset comes from Lincoln Laboratories, who set up an environment to simulating a typical US Air Force LAN and acquire nine weeks of raw TCP dump data. The Laboratories operated the LAN as if it were a real Air Force environment, but subjected it to multiple attacks.

This dataset is very large, containing approximately 743 MB of data and about **5 million individual network connections**. A connection is a sequence of TCP packets starting and ending at defined times, between which data flows to and from a source IP address to a target IP address, according to a well-defined protocol.  Each connection is labelled either as normal or as an attack, with exactly one specific type of attack. 

Each connection is a row in a CSV file as follows: 

```csv
0,tcp,http,SF,215,45076,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0.00,0.00,0.00,0.00,1.00,0.00,0.00,0,0,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,normal.
```

#### Network connection characteristics

For each network connection, the data set contains the **42** following characteristics: 

| # | Feature | Data type | Example |
|:-:|:--------|:----------|:--------|
| 1 | duration | continuous | 0 |
| 2 | protocol\_type | symbolic | tcp |
| 3 | service | symbolic | http |
| 4 | flag | symbolic | SF |
| 5 | src\_bytes | continuous | 215 |
| 6 | dst\_bytes | continuous | 45076 |
| 7 | land | symbolic | 0 |
| 8 | wrong\_fragment | continuous | 0 |
| 9 | urgent | continuous | 0 |
| 10 | hot | continuous | 0 |
| 11 | num\_failed\_logins | continuous | 0 |
| 12 | logged\_in | symbolic | 1 |
| 13 | num\_compromised | continuous | 0 |
| 14 | root\_shell | continuous | 0 |
| 15 | su\_attempted | continuous | 0 |
| 16 | num\_root | continuous | 0 |
| 17 | num\_file\_creations | continuous | 0 |
| 18 | num\_shells | continuous | 0 |
| 19 | num\_access\_files | continuous | 0 |
| 20 | num\_outbound\_cmds | continuous | 0 |
| 21 | is\_host\_login | symbolic | 0 |
| 22 | is\_guest\_login | symbolic | 0 |
| 23 | count | continuous | 1 |
| 24 | srv\_count | continuous | 1 |
| 25 | serror\_rate | continuous | 0.00 |
| 26 | srv\_serror\_rate | continuous | 0.00 |
| 27 | rerror\_rate | continuous | 0.00 |
| 28 | srv\_rerror\_rate | continuous | 0.00 |
| 29 | same\_srv\_rate | continuous | 1.00 |
| 30 | diff\_srv\_rate | continuous | 0.00 |
| 31 | srv\_diff\_host\_rate | continuous | 0.00 |
| 32 | dst\_host\_count | continuous | 0 |
| 33 | dst\_host\_srv\_count | continuous | 0 |
| 34 | dst\_host\_same\_srv\_rate | continuous | 0.00 |
| 35 | dst\_host\_diff\_srv\_rate | continuous | 0.00 |
| 36 | dst\_host\_same\_src\_port\_rate | continuous | 0.00 |
| 37 | dst\_host\_srv\_diff\_host\_rate | continuous | 0.00 |
| 38 | dst\_host\_serror\_rate | continuous | 0.00 |
| 39 | dst\_host\_srv\_serror\_rate | continuous | 0.00 |
| 40 | dst\_host\_rerror\_rate | continuous | 0.00 |
| 41 | dst\_host\_srv\_rerror\_rate | continuous | 0.00 |
| 42 | label | string | normal |

The label value, given in the last field, can be one of the following (23 possibilities) : *back, buffer\_overflow, ftp_write, guess_passwd, imap, ipsweep, land, loadmodule, multihop, neptune, nmap, normal, perl, phf, pod, portsweep, rootkit, satan, smurf, spy, teardrop, warezclient, warezmaster*. In the dataset, most connections are labeled *normal*.

#### Attacks main categories

The various attacks can be divided into four main categories:

1. **DOS**: Denial-of-service attack, like *SYN flood*
2. **R2L**: Unauthorized access from a remote machine like *guessing password*
3. **U2R**: Unauthorized access to local superuser (root) privileges like *various buffer overflow attacks*
4. **Probing**: Surveillance and other probing like *port scanning*

### 2. Description of the features used and any pre-processing to extract additional features

On this dataset, the features extraction was already done for us. We know that there are already 38 numeric features that we can exploit.

#### Labels distribution 

First, we want to know the distribution of labels on the data and how many data there are for each label. To do this with *Spark*, you simply group the data by "label", order them and display them. In addition, we calculate the percentage distribution of each label on the dataset.

```spark
data.select("label").groupBy("label").count().orderBy($"count".desc)
	 .withColumn("percentage", round(($"count" / data.count()) * 100, 2))
	 .show(100)

+----------------+-------+----------+
|           label|  count|percentage|
+----------------+-------+----------+
|          smurf.|2807886|     57.32|
|        neptune.|1072017|     21.88|
|         normal.| 972781|     19.86|
|          satan.|  15892|      0.32|
|        ipsweep.|  12481|      0.25|
|      portsweep.|  10413|      0.21|
|           nmap.|   2316|      0.05|
|           back.|   2203|      0.04|
|    warezclient.|   1020|      0.02|
|       teardrop.|    979|      0.02|
|            pod.|    264|      0.01|
|   guess_passwd.|     53|       0.0|
|buffer_overflow.|     30|       0.0|
|           land.|     21|       0.0|
|    warezmaster.|     20|       0.0|
|           imap.|     12|       0.0|
|        rootkit.|     10|       0.0|
|     loadmodule.|      9|       0.0|
|      ftp_write.|      8|       0.0|
|       multihop.|      7|       0.0|
|            phf.|      4|       0.0|
|           perl.|      3|       0.0|
|            spy.|      2|       0.0|
+----------------+-------+----------+
```

We can indeed confirm that there are 23 possible labels. The most frequent labels on our data, by far, are : _smurf (~57%)_ and _neptune (~22%)_. Interestingly, the connections identified as *normal*, which are not anomalies, represent just under 20% of our dataset. All other labels are very poorly represented (less than 1%).

#### Non-numeric features

We have three columns (in addition to the labels) that contains non-numeric values : *protocol_type*, *service*, and *flag*. In the initial stages, these features will not be explored. These non-numeric features can be represented using "one- hot encoding", which can then be considered as numeric dimensions.

**Example for the *protocol_type* column**

As there is three possible values for this column (*TCP*, *UDP* or *ICMP*), these three dimension can be represent by one value (x, x, x). Of these three values only one can be set to 
1, the other values are set to 0. 

We can see as a table of *protocol_type* with only one enable dimension. Example for *UDP* :

| TCP | UDP | ICMP |
|:---:|:---:|:----:|
|  0  |  1  |  0   |

If we want to represent the *TCP* value of the characteristic, we encode: 1,0,0. If you want to represent the *UDP* value of the characteristic, you will encode: 0,1,0. And if you want to represent the *ICMP* value of the characteristic, you should encode: 0,0,1.

This one-hot encoding transformation can also be used for the features *service*, and *flag*.

### 3. Questions for which you hope to get an answer from the analysis

After brainstorming and validation, we have decided to develop the following analytical questions and answer them using *Spark*.

 - **What are the characteristics and features that define an anomaly ?**
 - **How to find the optimal value of the hyperparameter K of the K-means clustering ?**
 - **What is the distribution of attacks on each protocol (*TCP, UDP, ICMP*...), by which service (port) were they carried out, what type of attacks are they and what was the final purpose of the attack ?**

###### Analytical question ideas

- _What defines an anomaly ?_
- _Why use an unsupervised algorithm such as K-Means ?_
- _How to find the correct K ?_
- _Which types of anomaly can we detect on a network ?_
- _Which patterns are often used by attackers to exploit flaws ?_
- _What is the distribution of attacks on each protocol (TCP, UDP, ICMP...) ?_
	- _By which service (port) were they carried out ?_
	- _What type of attacks are they?_
	- _What was the final purpose of the attack ?_
	- _..._

### 4. Algorithms you applied

### 5. Optimisations you performed

### 6. Your approach to testing and evaluation

### 7. Results you obtained

The three analytical questions results, which we had to develop and answer using the Spark, are available below.

#### a) What are the characteristics and features that define an anomaly ?
##### Result
##### Development

#### b) How to find the optimal value of the hyperparameter K of the K-means clustering ?
##### Result
##### Development

For this question, we want to know how many clusters are appropriate for this data set. As there are 23 possible distinct label values for classification, it seems that k must be at least 23. If the value of k chosen is equal to the number of data points, each point will be its own cluster. The value of k must therefore be between 23 and 4.9 million, this leaves us with a considerable choice of values !

A clustering is considered good if each data point is close to its nearest centroid. The Euclidean distance can be used to define this distance.
KMeansModel provides a *computeCost* method that calculates the sum of the squared distances and can be used to define the score of a cluster.

###### The beginning - `clusteringScore0`

For this first grouping, we ignore non-numerical features and try to place each data point near its nearest centroid with Euclidean distance. To do this, *KMeansModel* provides us with the `computeCost()` method which can be used to compute the mean squared distance.

As a first test, we chose to make the value of k evolve between 20 and 300 with jumps of 10 (i.e. we tested the clustering with k=20, then k=30, k=40, etc).

| *k from 20 to 300, jumps of 10*<br> ![clusteringScore0](images/Qb-clusteringScore0-1.png) |
|:---:|

With these parameters, we can see that a good value for k seems to be between 200 and 280. We decide to run the process again with k evolving between 200 and 280 with jumps of 5. As we can fall into a local minimum, we decide to run the tests twice.

| *k from 200 to 280, jumps of 5*<br> ![clusteringScore0](images/Qb-clusteringScore0-2.png) |
|:---:|

As we see again, the score decreases as k increases and the best score value for k is two times when k = 280. This makes sense because the more clusters you add, the closer the data points can be to a centroid. To be sure, we decide to restart the training by increasing the range to 320.

| *k from 200 to 320, jumps of 5*<br> ![clusteringScore0](images/Qb-clusteringScore0-3.png) |
|:---:|

Not surprisingly, the best score is now obtained with k = 320. In view of the results, this method will not be sufficient to determine the best value of k. It is therefore necessary to make some improvements.

###### Max iteration and tolerance - `clusteringScore1`

To improve the classification, we can run the iteration for a longer time via the `setTol()` method. This method controls the minimum significant amount of centroid movement. It is also possible to increase the maximum number of iterations with the `setMaxIter()` method which will prevent the algorithm from stopping too soon.

Here again, we chose to make the value of k evolve between 20 and 300 with jumps of 10 to see first results. We increase the max iteration from default value 20 to 40 and decrease the tolerance value from default 1.0e-4 to 1.0e-5.

The aim is to find the "elbow" in the graph beyond which the increase in k stops reducing the score significantly.

| *k from 20 to 300, jumps of 10*<br> ![clusteringScore1](images/Qb-clusteringScore1-1.png) |
|:---:|

| *k from 35 to 175, jumps of 5*<br> ![clusteringScore1](images/Qb-clusteringScore1-2.png) |
|:---:|

With this max iteration and tolerance values, the elbow seems to be around 120.

| *k from 200 to 280, jumps of 5*<br> ![clusteringScore1](images/Qb-clusteringScore1-3.png) |
|:---:|

If we try with largers k-values, we see a kind of anarchy and no "elbow" at all!

###### Normalization - `clusteringScore2`

To further improve our results and find the right value of k we can normalise the different characteristics. MLlib provides the `StandardScale` class which allows you to easily perform these standarisations.

| *k from 20 to 300, jumps of 10*<br> ![clusteringScore2](images/Qb-clusteringScore2-1.png) |
|:---:|

With this normalisation, the elbow seems to be around 160.

| *k from 20 to 200, jumps of 5*<br> ![clusteringScore2](images/Qb-clusteringScore2-2.png) |
|:---:|

If we decrease the jumps, the elbow seems now to be between 120 and 140.

| *k from 220 to 320, jumps of 5*<br> ![clusteringScore2](images/Qb-clusteringScore2-3.png) |
|:---:|

With larger k-values, we do not see any other "elbow" emerging.

###### Non-numeric features - `clusteringScore3`

After this standardisation, it would be nice to add the non-numerical characteristics to improve the clustering. The MLlib library provides a `OneHotEncoder` class allowing easily the implementation of one-hot encoding. Therefore, the clustering will use all available input features.

| *k from 20 to 300, jumps of 10*<br> ![clusteringScore3](images/Qb-clusteringScore3-1.png) |
|:---:|

| *k from 100 to 240, jumps of 5*<br> ![clusteringScore3](images/Qb-clusteringScore3-2.png) |
|:---:|

###### Entropy - `clusteringScore4`

Finally, to achieve good clustering, we would like to have clusters whose label collections are homogeneous and therefore have low entropy. A weighted average of the entropy can therefore be **used as a new score** to define the optimal value of k. 

To do so, the following 4 steps are applied:

1. Predict the cluster for each data item
2. Extract collections of labels, by cluster
3. Count the labels in the collections
4. Average entropy weighted by cluster size

In this way, it is possible to search for a local minimum value for k because the entropy does not necessarily decrease when k increases.

| *k from 20 to 300, jumps of 10*<br> ![clusteringScore4](images/Qb-clusteringScore4-1.png) |
|:---:|

| *k from 140 to 220, jumps of 5*<br> ![clusteringScore4](images/Qb-clusteringScore4-2.png) |
|:---:|

| *k from 160 to 200, jumps of 1*<br> ![clusteringScore4](images/Qb-clusteringScore4-2.png) |
|:---:|

###### Clustering - `fitPipeline4`

*k = 180*

```scala
```

##### Improvements

###### Cosine distance measure - `clusteringScore5`

As we have seen, the K-means algorithm requires a notion of distance between data points. So far, we have used the simple Euclidean distance to measure the distance between data points. There is another distance function supported by Spark MLlib: The Cosine distance measure.
 
Cosine similarity is a measure of similarity between two non-zero vectors in an inner product space that measures the cosine of the angle between them. **It is therefore a judgment on orientation and not on magnitude**. Two vectors with the same orientation have a cosine similarity of 1, two vectors oriented at 90° to each other have a similarity of 0, and two diametrically opposed vectors have a similarity of -1; and this independent of their magnitude.

| *k from 20 to 300, jumps of 10*<br> ![clusteringScore4](images/Qb-clusteringScore5-1.png) |
|:---:|

###### Silhouette coefficient - `clusteringScore6`

Another measure attempting to assess not only the proximity of points within a cluster, but also the proximity of points to other clusters can be implemented. The *Silhouette coefficient* is one such measure, and we have carried out tests with it.

#### c) What is the distribution of attacks on each protocol (*TCP, UDP, ICMP*...), by which service (port) were they carried out, what type of attacks are they and what was the final purpose of the attack ?
##### Result
##### Development

### 8. Possible future enhancements

To improve clustering and obtain even better results on the classification of these anomalies, various other models could be applied instead of simple *K-means clustering*. For example, a *Gaussian mixture model* or *DBSCAN* could capture more subtle relationships between data points and cluster centres. Or a neural network with different hidden layers.

## Sources

- [Chapter 5 (Anomaly Detection in Network Trafc with K-means Clustering) of Advanced Analytics with Spark by Sean Owen](Documentation/Anomaly-Detection-in-Network-Traffic-with-K-means-Clustering.pdf)
- [kdd.ics.uci.edu/databases/kddcup99/kddcup99.html](http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html)
- [kaggle.com/code/abhaymudgal/intrusion-detection-system](https://www.kaggle.com/code/abhaymudgal/intrusion-detection-system)
- [spark.apache.org/docs/latest/ml-clustering.html#k-means](https://spark.apache.org/docs/latest/ml-clustering.html#k-means)
- [spark.apache.org/docs/latest/api/scala/org/apache/spark/ml/clustering/KMeans.html](https://spark.apache.org/docs/latest/api/scala/org/apache/spark/ml/clustering/KMeans.html)
