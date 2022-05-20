# MA-BDA - Anomaly Detection in Network Traffic with K-means Clustering

###### Alt Thibaud, Bueche Lucas | Deadline on Friday 10.06.2022 | [github.com/weevood/MA-BDA_Project](https://github.com/weevood/MA-BDA_Project)

## Summary

The goal is to **detect anomalous behavior in the network traffic of an organization**. Anomalous behavior can point to things like intrusion attempts, denial-of-service attacks, port scanning, etc.

The dataset was generated for a data analysis competition from raw TCP dump data for a local-area network (LAN) simulating a typical U.S. Air Force LAN. The LAN was operated as if it were a true Air Force environment, but peppered with multiple attacks. Feature extraction was already run on the data, the dataset contains a list of connections, and for each connection 38 features, like the number of bytes sent, login attempts, TCP errors, and so on.

As the data is not labeled, an unsupervised learning algorithm is applied, more specifically *K-means clustering*. The idea is to let the clustering discover normal behavior. Connections that fall outside of clusters are potentially anomalous.

**Analytical question ideas**

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

## Documentation

### 1. Description of the dataset (size, information it contains)

The KDD Cup dataset is very large, containing approximately 743 MB of data and about 4.9 million individual network connections.

For each network connection, the data set contains the **38** following characteristics: 

- The number of bytes sent
- The login attempts
- TCP errors
- ... (**!TODO!**)

Each connection is a row in a CSV file as follows: 

```csv
```

### 2. Description of the features used and any pre-processing to extract additional features

### 3. Questions for which you hope to get an answer from the analysis

We have decided to develop the following analytical questions and answer them using *Spark*.

#### a) What are the characteristics and features that define an anomaly ?
#### b) How to find the optimal value of the hyperparameter K of the K-means clustering ?
#### c) What is the distribution of attacks on each protocol (*TCP, UDP, ICMP*...), by which service (port) were they carried out, what type of attacks are they and what was the final purpose of the attack ?

### 4. Algorithms you applied

### 5. Optimisations you performed

### 6. Your approach to testing and evaluation

### 7. Results you obtained

The three analytical questions, which we had to develop and answer using the Spark results, are available below.

#### a) 
##### Result
##### Development

#### b)  
##### Result
##### Development

#### c)  
##### Result
##### Development

### 8. Possible future enhancements

## Sources

- http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
- https://datahub.io/machine-learning/kddcup99
- https://www.kaggle.com/code/abhaymudgal/intrusion-detection-system