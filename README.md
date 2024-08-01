# Attention-guided Counterfactual User Contrastive Modeling for Sequential Recommendation
This is the code for our paper ***A***ttention-guided ***C***ounterfactual ***U***ser Contrastive Modeling
for Sequential ***Rec***ommendation (ACURec).
## Requirements
We use the following environment:
* Python 3.8
* Pytorch 2.3.1
* numpy 1.24.3
* tqdm 4.64.4
## Datasets
We use Amazon (Sports, Grocery, CD, Home), LastFM, and Yelp datasets for experiments, which can be found in following links:
* Amazon: http://jmcauley.ucsd.edu/data/amazon/
* LastFM: https://grouplens.org/datasets/hetrec-2011/
* Yelp: https://www.yelp.com/dataset
## Quick Start
You can run ACURec with the following code or use the script run.sh. The training log of Yelp is provided in /log/yelp for reproduce.
```
python runACURec.py --dataset yelp --seed 2024 --train_batch 2048 --essential_ratio 0.5 --lamda 0.1
```


