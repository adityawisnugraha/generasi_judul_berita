import numpy as np
import pandas as pd
import nltk
import re
import tensorflow as tf
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, LSTM, Embedding, Dense, Attention, Concatenate, MultiHeadAttention, Dropout, LayerNormalization
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import matplotlib.pyplot as plt
import seaborn as sns

# Download NLTK data
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Text cleansing function
def text_cleansing(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
    tokens = word_tokenize(text)
    return ' '.join(tokens)

# Dummy data (replace with your actual dataset loading)
data = {'content': ['This is an example article content about technology.'],
        'title': ['Technology Advances']}
df = pd.DataFrame(data)

# Preprocessing
df['content'] = df['content'].apply(text_cleansing)
df['title'] = df['title'].apply(text_cleansing)

# Tokenization
tokenizer = Tokenizer()
tokenizer.fit_on_texts(df['content'].tolist() + df['title'].tolist())

vocab_size = len(tokenizer.word_index) + 1

encoder_input = tokenizer.texts_to_sequences(df['content'])
decoder_input = tokenizer.texts_to_sequences(df['title'])

encoder_input = pad_sequences(encoder_input, padding='post')
decoder_input = pad_sequences(decoder_input, padding='post')

X_train, X_test, y_train, y_test = train_test_split(encoder_input, decoder_input, test_size=0.2)

# Build simple LSTM Encoder-Decoder model
embedding_dim = 64
latent_dim = 128

encoder_inputs = Input(shape=(None,))
enc_emb = Embedding(vocab_size, embedding_dim)(encoder_inputs)
encoder_lstm = LSTM(latent_dim, return_state=True)
_, state_h, state_c = encoder_lstm(enc_emb)

encoder_states = [state_h, state_c]

decoder_inputs = Input(shape=(None,))
dec_emb_layer = Embedding(vocab_size, embedding_dim)
dec_emb = dec_emb_layer(decoder_inputs)

decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(dec_emb, initial_state=encoder_states)

decoder_dense = Dense(vocab_size, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy')

# Dummy training (replace X_test, y_test with real validation set)
model.fit([X_train, y_train], np.expand_dims(y_train, -1), batch_size=32, epochs=1, validation_data=([X_test, y_test], np.expand_dims(y_test, -1)))

print("Model training complete.")