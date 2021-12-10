# Sentiment-Analyzer
Also a sarcasm detector now

Different models to train
- IMDB movie review based sentiment analysis (either positive or negative) 
    - Run with option flag '--type imdb'
- Twitter Sentiment140 based sentiment analysis (either positive or negative) 
    - Run with option flag '--type twitter'
- Twitter sarcasm detection (either sarcastic or not)
    - Run with option flag '--type sarcasm'

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

## Setup

## How to run