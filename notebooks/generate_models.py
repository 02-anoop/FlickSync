import os
import json
from zipfile import ZipFile

import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

kaggle_ditionary = json.load(open('kaggle.json'))
os.environ['KAGGLE_USERNAME'] = kaggle_ditionary['username']
os.environ['KAGGLE_KEY'] = kaggle_ditionary['key']
data = pd.read_csv('imdb-dataset-of-50k-movie-reviews\\IMDB Dataset.csv')
data['sentiment'].value_counts()
data.replace({"sentiment" : {"positive" : 1, "negative" : 0}}, inplace=True)
train_data,test_data = train_test_split(data, test_size=0.2, random_state=42)
tokenizer = Tokenizer(num_words = 5000)
tokenizer.fit_on_texts(train_data["review"])
X_train = pad_sequences(tokenizer.texts_to_sequences(train_data["review"]), maxlen = 200)
X_test = pad_sequences(tokenizer.texts_to_sequences(test_data["review"]), maxlen = 200)
Y_train = train_data["sentiment"]
Y_test = test_data["sentiment"]
model = Sequential()
model.add(Embedding(input_dim=5000, output_dim=128, input_length = 200))
model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1,activation='sigmoid'))
model.build(input_shape=(None, 200))
model.summary()

model.compile(optimizer='adam', loss="binary_crossentropy", metrics=["accuracy"])
model.fit(X_train, Y_train, epochs=5, batch_size=64, validation_split=0.2)

loss, accuracy = model.evaluate(X_test,Y_test)
print(f"Test loss : {loss} Test accuracy : {accuracy}")
def predict_sentiment(review):
    sequence = tokenizer.texts_to_sequences([review])
    padded_sequence = pad_sequences(sequence, maxlen = 200)
    prediction = model.predict(padded_sequence)
    sentiment = "positive" if prediction[0][0] > 0.5 else "negative"
    return sentiment
new_review = "This movie was fantastic. I loved it."
sentiment = predict_sentiment(new_review)
print(f"The sentiment of the review is : {sentiment}")
new_review = "This movie was not that good"
sentiment = predict_sentiment(new_review)
print(f"The sentiment of the review is: {sentiment}")
import pickle

# Save the trained model to a file
with open('model.pkl', 'wb') as file:
    pickle.dump(model, file)
with open('tokenizer.pkl','wb') as file:
    pickle.dump(tokenizer, file)

