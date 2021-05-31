import pandas as pd
import os
import numpy as np
import json
from collections import Counter
import os
import pickle
import json
import tensorflow as tf
import tensorflow.keras.layers as L
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
import transformers as trfm
from transformers import AutoTokenizer, TFAutoModel, TFElectraModel, ElectraTokenizer
from tokenizers import BertWordPieceTokenizer



# Here I used the electra tokenizer as electra model was trained using this tokenizer and it conains more words.
def encode_electra(texts, tokenizer, chunk_size=256, maxlen=512, enable_padding=False):
    tokenizer.enable_truncation(max_length=maxlen)
    if enable_padding:
        tokenizer.enable_padding(max_length=maxlen)

    all_ids = []

    for i in range(0, len(texts), chunk_size):
        text_chunk = texts[i:i + chunk_size].tolist()
        encs = tokenizer.encode_batch(text_chunk)
        all_ids.extend([enc.ids for enc in encs])

    return np.array(all_ids)


# Here I used Bert tokenizer as it has a special id called 'cls_token_id' which will tell electra model where the question
# and where the answer starts.
def combine_qa_ids(q_ids, a_ids, tokenizer, maxlen=512):
    combined_ids = []

    for i in range(q_ids.shape[0]):
        ids = []
        ids.append(tokenizer.cls_token_id)
        ids.extend(q_ids[i])
        ids.append(tokenizer.sep_token_id)
        ids.extend(a_ids[i])
        ids.append(tokenizer.sep_token_id)
        ids.extend([tokenizer.pad_token_id] * (maxlen - len(ids)))

        combined_ids.append(ids)

    return np.array(combined_ids)

def make_model():
    transformer = TFElectraModel.from_pretrained(r'models\electra')
    input_ids = L.Input(shape=(512,), dtype=tf.int32)
    x = transformer(input_ids)[0]
    x = x[:, 0, :]
    x = L.Dense(1, activation='sigmoid', name='sigmoid')(x)

    model = Model(inputs=input_ids, outputs=x)
    model.compile(
        loss='binary_crossentropy',
        metrics=['accuracy'],
        optimizer=Adam(lr=1e-5)
    )
    last_layer = pickle.load(open('models\sigmoid.pickle', 'rb'))
    model.get_layer('sigmoid').set_weights(last_layer)

    return model

electra_tokenizer = trfm.ElectraTokenizer.from_pretrained("google/electra-small-discriminator")
bert_tokenizer = BertWordPieceTokenizer('vocab.txt', lowercase=True)

def make_final_df(input_value):
    community = pd.read_csv('community.csv')
    multilingual = pd.read_csv('multilingual.csv')
    news = pd.read_csv('news.csv')
    covid_biomedical = pd.read_csv(r'covid/biomedical.csv')
    covid_expert = pd.read_csv('covid\expert.csv')
    covid_general = pd.read_csv('covid\general.csv')
    unilingual = multilingual[multilingual['language'] == 'english']
    df = pd.concat([news, community, unilingual, covid_biomedical, covid_expert, covid_general], axis=0, ignore_index=True)
    df = df[['question', 'answer', 'wrong_answer']]
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            df.dropna(inplace=True)
            df.reset_index(inplace=True)
            df.drop(['index'], axis=1, inplace=True)
    df = df[(df['answer'].apply(lambda x: len(x))<400) & (df['question'].apply(lambda x: len(x))<50)]
    df.reset_index(inplace=True)
    df.drop(['index'], axis=1, inplace=True)
    df['Input'] = input_value
    final_df = df[['answer', 'wrong_answer', 'Input']]
    return final_df

def make_pred(Input):
    Input = str(Input)
    final_df = make_final_df(Input)
    MAX_LEN = 512
    q_ids = encode_electra(final_df.Input.values, bert_tokenizer, maxlen=MAX_LEN//2 - 2)
    a_ids = encode_electra(final_df.answer.values, bert_tokenizer, maxlen=MAX_LEN//2 - 2)
    wa_ids = encode_electra(final_df.wrong_answer.values, bert_tokenizer, maxlen=MAX_LEN//2 - 2)
    correct_ids = combine_qa_ids(q_ids, a_ids, electra_tokenizer, maxlen=MAX_LEN)
    wrong_ids = combine_qa_ids(q_ids, wa_ids, electra_tokenizer, maxlen=MAX_LEN)
    input_ids = np.concatenate([correct_ids, wrong_ids])
    model = make_model()
    y_score = model.predict(input_ids, verbose=1, batch_size=64)
    most_likely_ans = np.argmax(y_score)
    output = final_df.loc[most_likely_ans, 'answer']
    #y_pred = y_score.round().astype(int)
    return output
