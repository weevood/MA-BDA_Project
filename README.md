# MA-BDA - Anomaly Detection in Network Traffic with K-means Clustering

###### Alt Thibaud, Bueche Lucas | Deadline on Friday 10.06.2022 | [github.com/weevood/MA-BDA_Project](https://github.com/weevood/MA-BDA_Project)

## Summary

The goal is to detect anomalous behavior in the network traffic of an organization. Anomalous behavior can point to things like intrusion attempts, denial-of-service attacks, port scanning, etc.

The dataset was generated for a data analysis competition from raw TCP dump data for a local-area network (LAN) simulating a typical U.S. Air Force LAN. The LAN was operated as if it were a true Air Force environment, but peppered with multiple attacks. Feature extraction was already run on the data, the dataset contains a list of connections, and for each connection 38 features, like the number of bytes sent, login attempts, TCP errors, and so on.

As the data is not labeled, an unsupervised learning algorithm is applied, more specifically K-means clustering. The idea is to let the clustering discover normal behavior. Connections that fall outside of clusters are potentially anomalous.

**Proposed analytical questions**

1. What defines an anomaly ?
2. Why use an unsupervised algorithm such as K-Means ?
3. How to find the correct K ?
4. Which types of anomaly can we detect on a network ?

## Documentation

### 1. Description of the dataset (size, information it contains)

### 2. Description of the features used and any pre-processing to extract additional features

### 3. Questions for which you hope to get an answer from the analysis

We have decided to develop the following analytical questions and answer them using *Spark*.

#### a) 
#### b) 
#### c) 

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