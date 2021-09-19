import numpy as np
import string
import pandas as pd
import nltk
from transformers import BertTokenizer

reminder_train = 'name.tsv'
reminder_eval = 'name2.tsv'
reminder_test = 'name3.tsv'

weather_train = 'name.tsv'
weather_eval = 'name.tsv'
weather_test = 'name.tsv'

reminder_train_df = pd.read_csv(reminder_train, sep='\t')
reminder_eval_df = pd.read_csv(reminder_eval, sep='\t')
reminder_eval_df = pd.read_csv(reminder_test, sep='\t')

weather_train_df = pd.read_csv(weather_train, sep='\t')
weather_eval_df = pd.read_csv(weather_eval, sep='\t')
weather_test_df = pd.read_csv(weather_test, sep='\t')

# lowercasing

reminder_train_df = reminder_train_df.str.lower()
weather_train_df = weather_train_df.str.lower()

# tokenization
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True)


tokenized_weather_train_df = weather_train_df.map(tk_function, batched=True)
tokenized_ reminder_train_df = reminder_train_df.map(tk_function, bathced=True)

# BPE tokenization??
...

# Conical form semantic tree
...
