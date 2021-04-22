import time

import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm
import os
from tensorflow.keras import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Bidirectional, Flatten, Dropout, BatchNormalization, Embedding, Input, \
    TimeDistributed
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import plot_model

import numpy as np
# print(np.unique(sections))

# from datasets import load_dataset
# raw_datasets = load_dataset('csv', data_files=['./tmp3/b_dataset_2_train.csv'])
# print(raw_datasets)
contraction_mapping = {"ain't": "is not", "aren't": "are not", "can't": "cannot", "'cause": "because",
                       "could've": "could have", "couldn't": "could not",

                       "didn't": "did not", "doesn't": "does not", "don't": "do not", "hadn't": "had not",
                       "hasn't": "has not", "haven't": "have not",

                       "he'd": "he would", "he'll": "he will", "he's": "he is", "how'd": "how did",
                       "how'd'y": "how do you", "how'll": "how will", "how's": "how is",

                       "I'd": "I would", "I'd've": "I would have", "I'll": "I will", "I'll've": "I will have",
                       "I'm": "I am", "I've": "I have", "i'd": "i would",

                       "i'd've": "i would have", "i'll": "i will", "i'll've": "i will have", "i'm": "i am",
                       "i've": "i have", "isn't": "is not", "it'd": "it would",

                       "it'd've": "it would have", "it'll": "it will", "it'll've": "it will have", "it's": "it is",
                       "let's": "let us", "ma'am": "madam",

                       "mayn't": "may not", "might've": "might have", "mightn't": "might not",
                       "mightn't've": "might not have", "must've": "must have",

                       "mustn't": "must not", "mustn't've": "must not have", "needn't": "need not",
                       "needn't've": "need not have", "o'clock": "of the clock",

                       "oughtn't": "ought not", "oughtn't've": "ought not have", "shan't": "shall not",
                       "sha'n't": "shall not", "shan't've": "shall not have",

                       "she'd": "she would", "she'd've": "she would have", "she'll": "she will",
                       "she'll've": "she will have", "she's": "she is",

                       "should've": "should have", "shouldn't": "should not", "shouldn't've": "should not have",
                       "so've": "so have", "so's": "so as",

                       "this's": "this is", "that'd": "that would", "that'd've": "that would have", "that's": "that is",
                       "there'd": "there would",

                       "there'd've": "there would have", "there's": "there is", "here's": "here is",
                       "they'd": "they would", "they'd've": "they would have",

                       "they'll": "they will", "they'll've": "they will have", "they're": "they are",
                       "they've": "they have", "to've": "to have",

                       "wasn't": "was not", "we'd": "we would", "we'd've": "we would have", "we'll": "we will",
                       "we'll've": "we will have", "we're": "we are",

                       "we've": "we have", "weren't": "were not", "what'll": "what will",
                       "what'll've": "what will have", "what're": "what are",

                       "what's": "what is", "what've": "what have", "when's": "when is", "when've": "when have",
                       "where'd": "where did", "where's": "where is",

                       "where've": "where have", "who'll": "who will", "who'll've": "who will have", "who's": "who is",
                       "who've": "who have",

                       "why's": "why is", "why've": "why have", "will've": "will have", "won't": "will not",
                       "won't've": "will not have",

                       "would've": "would have", "wouldn't": "would not", "wouldn't've": "would not have",
                       "y'all": "you all",

                       "y'all'd": "you all would", "y'all'd've": "you all would have", "y'all're": "you all are",
                       "y'all've": "you all have",

                       "you'd": "you would", "you'd've": "you would have", "you'll": "you will",
                       "you'll've": "you will have",

                       "you're": "you are", "you've": "you have"}

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

Introductions = []
Introductions_without_stopwords = []
Abstracts = []

T_I_FILE = "./article_3_train_{}.txt"
T_A_FILE = "./summary_3_train_{}.txt"
for year in range(1,5):
    with open(T_I_FILE.format(year), "rt", encoding='utf-8') as TEXT_I_FILE:
        for i in TEXT_I_FILE.readlines():
            Introduction_Text = ''.join(i.rstrip(''))
            Introductions.append(Introduction_Text)
            Introductions_without_stopwords.append(
                ' '.join([w for w in Introduction_Text.split() if w not in stop_words]))
        TEXT_I_FILE.close()
    with open(T_A_FILE.format(year), "rt", encoding='utf-8', errors='ignore') as TEXT_A_FILE:
        for i in TEXT_A_FILE.readlines():
            Abstract_Text = ''.join(i.rstrip())
            Abstracts.append(Abstract_Text)
        TEXT_A_FILE.close()
print(len(Abstracts), len(Introductions), len(Introductions_without_stopwords))
data = pd.DataFrame(
        {'article': Introductions,
         'summary': Abstracts})
data.to_csv('b_dataset_3_train.csv', index=False)
'''
python  run_summarization_longformer.py --num_workers 12  --save_prefix eval_long16k_nooverlap_large --model_path facebook/bart-large-xsum --max_input_len 16368 --batch_size 2 --grad_accum 4 --grad_ckpt   --attention_mode sliding_chunks_no_overlap --attention_window 340 --val_every 0.333333333  --debug --resume summarization/run_long16k_nooverlap_large/_ckpt_epoch_3_v1.ckpt  --val_percent_check 1.0 --disable_checkpointing

python run_summarization.py --model_name_or_path facebook/bart-large-xsum --do_train --do_eval --train_file ./a_dataset_1_train.csv --validation_file ./a_dataset_1_test.csv --text_column Article --summary_column Summary --source_prefix "summarize: " --output_dir ./tmp/tst-summarization --per_device_train_batch_size=2 --per_device_eval_batch_size=2 --overwrite_output_dir --predict_with_generate

python run_summarization.py --model_name_or_path t5-small --do_train --do_eval --train_file ./dataset3/b_dataset_1_train.csv --validation_file ./dataset3/b_dataset_1_test.csv --text_column document --summary_column summary --source_prefix "summarize: " --output_dir ./tmp/t5-small-dataset-1 --per_device_train_batch_size=8 --per_device_eval_batch_size=8 --overwrite_output_dir --predict_with_generate

python run_summarization.py --model_name_or_path t5-small --do_train --do_eval --train_file ./dataset3/b_dataset_1_train.csv --validation_file ./dataset3/b_dataset_1_test.csv --source_prefix "summarize: " --output_dir ./tmp/t5-small-dataset-1 --per_device_train_batch_size=6 --per_device_eval_batch_size=6 --overwrite_output_dir --predict_with_generate

python run_summarization.py --model_name_or_path t5-large --do_train --do_eval --train_file ./dataset3/b_dataset_1_train.csv --validation_file ./dataset3/b_dataset_1_test.csv --source_prefix "summarize: " --output_dir ./tmp/t5-large-dataset-1 --per_device_train_batch_size=6 --per_device_eval_batch_size=6 --overwrite_output_dir --predict_with_generate

python run_summarization.py --model_name_or_path nsi319/legal-led-base-16384 --do_train --do_eval --train_file ./dataset3/b_dataset_1_train.csv --validation_file ./dataset3/b_dataset_1_test.csv --source_prefix "" --output_dir ./tmp/led-dataset-1 --per_device_train_batch_size=4 --per_device_eval_batch_size=4 --overwrite_output_dir --predict_with_generate


'''
