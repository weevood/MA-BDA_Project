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

The KDD Cup dataset comes from Lincoln Laboratories, who set up an environment to simulating a typical US Air Force LAN and acquire nine weeks of raw TCP dump data. The Laboratories operated the LAN as if it were a real Air Force environment, but subjected it to multiple attacks.

This dataset is very large, containing approximately 743 MB of data and about **5 million individual network connections**. A connection is a sequence of TCP packets starting and ending at defined times, between which data flows to and from a source IP address to a target IP address, according to a well-defined protocol.  Each connection is labelled either as normal or as an attack, with exactly one specific type of attack. 

Each connection is a row in a CSV file as follows: 

```csv
0,tcp,http,SF,215,45076,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,1,1,0.00,0.00,0.00,0.00,1.00,0.00,0.00,0,0,0.00,0.00,0.00,0.00,0.00,0.00,0.00,0.00,normal.
```

#### Network connection characteristics

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

The label value, given in the last field, can be one of the following : *back, buffer\_overflow, ftp_write, guess_passwd, imap, ipsweep, land, loadmodule, multihop, neptune, nmap, normal, perl, phf, pod, portsweep, rootkit, satan, smurf, spy, teardrop, warezclient, warezmaster*. In the dataset, most connections are labeled *normal*.

#### Attacks main categories

The various attacks can be divided into four main categories:

1. **DOS**: Denial-of-service attack, like *SYN flood*
2. **R2L**: Unauthorized access from a remote machine like *guessing password*
3. **U2R**: Unauthorized access to local superuser (root) privileges like *various buffer overflow attacks*
4. **Probing**: Surveillance and other probing like *port scanning*

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

#### a) What are the characteristics and features that define an anomaly ?
##### Result
##### Development

#### b) How to find the optimal value of the hyperparameter K of the K-means clustering ?
##### Result
##### Development

#### c) What is the distribution of attacks on each protocol (*TCP, UDP, ICMP*...), by which service (port) were they carried out, what type of attacks are they and what was the final purpose of the attack ?
##### Result
##### Development

### 8. Possible future enhancements

## Sources

- Chapter 5 (Anomaly Detection in Network Trafc with K-means Clustering) of Advanced Analytics with Spark by Sean Owen
- http://kdd.ics.uci.edu/databases/kddcup99/kddcup99.html
- https://www.kaggle.com/code/abhaymudgal/intrusion-detection-system