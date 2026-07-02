# %%
import nltk
nltk.download("gutenberg")
from nltk.corpus import gutenberg
import pandas as pd
import numpy as np
data=gutenberg.raw('shakespeare-hamlet.txt')
with open("hamlet.txt","w") as fl:
    fl.write(data)

# %%
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split

# %%
with open("hamlet.txt","r") as fl:
    text=fl.read().lower()

tokenizer=Tokenizer()
tokenizer.fit_on_texts([text])
total_words=len(tokenizer.word_index)+1
total_words

# %%
# create input sequences
input_sequences=[]
for line in text.split("\n"):
    token_list=tokenizer.texts_to_sequences([line])[0]
    for i in range(1,len(token_list)):
        n_gram_sequence=token_list[:i+1]
        input_sequences.append(n_gram_sequence)

# %%
max_sequence_len=max([len(x)for x in input_sequences])

# %%
max_sequence_len

# %%
input_sequences=np.array(pad_sequences(input_sequences,maxlen=max_sequence_len,padding='pre'))

# %%
input_sequences

# %%
# create predictors and label
import tensorflow as tf
x,y=input_sequences[:,:-1],input_sequences[:,-1]

# %%
x,y

# %%
y=tf.keras.utils.to_categorical(y,num_classes=total_words)

# %%
X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=42)

# %%
# Train LSTM RNN
model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Input(shape=(X_train.shape[1],)))
model.add(tf.keras.layers.Embedding(total_words,100))
model.add(tf.keras.layers.LSTM(150,return_sequences=True))
model.add(tf.keras.layers.Dropout(0.2))
# tf.keras.layers.
model.add(tf.keras.layers.LSTM(50))
model.add(tf.keras.layers.Dense(total_words,activation='softmax'))

# Compile
model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
model.summary()

# %%
early_stopping=tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,restore_best_weights=True)

# %%
history=model.fit(X_train,y_train,epochs=1000,validation_data=(X_test,y_test),verbose=1)

# %%
def predict_next_word(model,tokenizer,text,max_sequence_len):
    token_list=tokenizer.texts_to_sequences([text])[0]
    if len(token_list)>=max_sequence_len:
        token_list=token_list[-(max_sequence_len-1):]
    token_list=pad_sequences([token_list],maxlen=max_sequence_len-1,padding="pre")
    predicted=model.predict(token_list,verbose=0)
    predicted_word_index=np.argmax(predicted,axis=1)
    for word,index in tokenizer.word_index.items():
        if index==predicted_word_index:
            return word
    return None

# %%
input_text="To be or not to be"
print(f"Input text:{input_text}")
max_sequence_len=model.input_shape[1]+1
next_word=predict_next_word(model,tokenizer,input_text,max_sequence_len)
print(f"next word prediction:{next_word}")

# %%
model.save("next_word_lstm.h5")

import pickle
with open("tokenizer.pkl","wb") as fl:
    pickle.dump(tokenizer,fl,protocol=pickle.HIGHEST_PROTOCOL)


