# Sentiment-Analyzer
Also a sarcasm detector now

Different models to train
- IMDB movie review based sentiment analysis (either positive or negative) 
    - Run with option flag '--type imdb'
- Twitter Sentiment140 based sentiment analysis (either positive or negative) 
    - Run with option flag '--type twitter'
- Twitter sarcasm detection (either sarcastic or not)
    - Run with option flag '--type sarcasm' 
- Run Sentiment and Sarcasm models in tandem fine tuning both models
    - Run with option flag '--type together'

Ways to run repl
- Pure sarcasm detection 
    - Run with option flag '--use_sarcasm'
- Sentiment analysis without sarcasm detection
    - Specify dataset to be run with using '--type [imdb or twitter]'
- Sentiment analysis with sarcasm detection (If sarcasm is detected, sentiment is flipped)
    - Run with option flag '--use_sarcasm' and '--type [imdb or twitter]'

## Datasets
* Elaborate more on these 
Below are the datasets used 
IMDB
Sentiment140, 1.6 million tweets each tagged with a sentiment 0-4
Sarcasm on Reddit -
https://www.kaggle.com/danofer/sarcasm?select=test-balanced.csv
Just download train-balanced-sarcasm.csv

IDK if we need to setup a download.sh for this one lol
If we do we will need to upload the file itself to some cloud service 

## Setup

## How to run