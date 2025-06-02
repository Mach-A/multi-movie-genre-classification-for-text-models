# multi-movie-genre-classification-for-text-models
Tasked with implementing 2 text-centric models(LSTM and BERT) for the classic movie-genre classification
# for BERT Model
import torch

from torch import nn

from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import MultiLabelBinarizer

from transformers import BertTokenizer, BertForSequenceClassification

from sklearn.metrics import accuracy_score, f1_score

import pandas as pd

import numpy as np

from tqdm import tqdm
