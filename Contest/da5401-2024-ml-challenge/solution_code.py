import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer


# Load the .npy file
################

array1 = np.load('embeddings_1.npy')
array2 = np.load('embeddings_2.npy')
array = np.concatenate((array1, array2))


#Load the label
################


def parse_label_file(filename, delimiter=';'):
    with open(filename, 'r') as f:
        lines = f.readlines()

    labels = []
    for line in lines:
        labels.append(line.strip().split(delimiter))

    return labels


def create_label_to_index(labels):
  unique_labels = set(labels)
  label_to_index = {label: i for i, label in enumerate(unique_labels)}
  return label_to_index


def to_multi_hot(labels, label_to_index):
  vocab_size = len(label_to_index)
  multi_hot = np.zeros(vocab_size)

  for label in labels:
    index = label_to_index[label]
    multi_hot[index] = 1

  return multi_hot


labelsfile1 = "icd_codes_1.txt"
labels1 = parse_label_file(labelsfile1)
labelsfile2 = "icd_codes_2.txt"
labels2 = parse_label_file(labelsfile2)
labels = labels1 + labels2

all_labels = []
for label in labels:
    all_labels += label
label_to_index = create_label_to_index(all_labels)

multi_hot = []
for label in labels:
    multi_hot.append(to_multi_hot(label, label_to_index))
multi_hot = np.array(multi_hot)


# Sample data
################

X_train = array
y_train = multi_hot
print(y_train.shape)

#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

X_test = array = np.load('test_data.npy')

print(X_train.shape)
print(y_train.shape)
print(X_test.shape)
#RandomForestClassifier (Not as accurate as neural networks (presented below) though.)
################
'''
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report


# Create a Random Forest Classifier
rf_clf = RandomForestClassifier(n_jobs = -1)

# Define the parameter grid
param_grid = {
    'n_estimators': [64, 128, 256, 512],
    'max_depth': [2, 4, 8, 16, 32],
    'min_samples_split': [2, 4, 8, 16, 32],
    'min_samples_leaf': [1, 2, 4, 8, 16, 32]
}

# Perform grid search with cross-validation
grid_search = GridSearchCV(rf_clf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Print the best parameters and the best score
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)

# Use the best model to make predictions
best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)

# Evaluate the model
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
'''

#Neural Networks
################

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation


# Create the model
model = Sequential([
    #tf.keras.layers.LeakyReLU(alpha=0.2),
    Dense(1024*16, activation='relu', input_dim=1024),
    Dropout(0.2),
    #Dense(256, activation='relu'),
    tf.keras.layers.LeakyReLU(alpha=0.2),
    Dropout(0.2),
    Dense(1400, activation='sigmoid')
])

# Compile the model
optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)
model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=8, batch_size=64, validation_split=0.2)

y_pred_prob = model.predict(X_test)

threshold = 0.5
y_pred = (y_pred_prob > threshold).astype(int)


"""
#Heirarchical Neural Networks
################

import tensorflow as tf
from tensorflow.keras.layers import Input, LSTM, Dense, Attention
from tensorflow.keras.models import Model

def create_han_model(max_sequence_length=198982, input_dim=1024):
    # Input layer
    input_layer = Input(shape=(max_sequence_length, input_dim))
    # Sentence-level LSTM and attention
    sentence_lstm = LSTM(units=1024, return_sequences=True)(input_layer)
    sentence_attention = Attention()([sentence_lstm, sentence_lstm])
    sentence_output = Dense(1024, activation='relu')(sentence_attention)

    # Document-level LSTM and attention
    document_lstm = LSTM(units=1024)(sentence_output)
    document_attention = Attention()([document_lstm, document_lstm])
    document_output = Dense(1024, activation='relu')(document_attention)

    # Output layer
    output_layer = Dense(1040, activation='sigmoid')(document_output)

    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model


model = create_han_model()


# Train the model
X_train = np.expand_dims(X_train, axis=0)
y_train = np.expand_dims(y_train, axis=0)
X_test = np.expand_dims(X_test, axis=0)

model.fit(X_train, y_train, epochs=8, batch_size=32, validation_split = 0.2)

y_pred_prob = model.predict(X_test)

threshold = 0.5
y_pred = (y_pred_prob > threshold).astype(int)
"""
"""
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense, Dropout, Attention



# Define the model
def create_model():
    input_layer = Input(shape=(198982, 1024,))
    # Consider using embedding layer if features are categorical
    # embedding_layer = Embedding(input_dim=vocab_size, output_dim=embedding_dim)(input_layer)

    # Word-level attention
    word_encoder = LSTM(units=64, return_sequences=True)(input_layer)
    word_attention = Attention()([word_encoder, word_encoder])
    word_context = LSTM(units=64)(word_attention)

    # Sentence-level attention
    sentence_encoder = LSTM(units=64, return_sequences=True)(word_context)
    sentence_attention = Attention()([sentence_encoder, sentence_encoder])
    sentence_context = LSTM(units=64)(sentence_attention)

    output_layer = Dense(1400, activation='sigmoid')(sentence_context)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model

# Create and compile the model
model = create_model()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model

X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
"""
"""
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, TransformerEncoder

# ... (rest of the code)

# Define the model
def create_model():
    input_layer = Input(shape=(1024,))

    # Transformer Encoder
    x = TransformerEncoder(num_layers=2, num_heads=4, hidden_size=64)(input_layer[:, None, :])
    x = tf.squeeze(x, axis=1)  # Remove the unnecessary time dimension

    # Output Layer
    output_layer = Dense(1400, activation='sigmoid')(x)

    model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
    return model

# Create and compile the model
model = create_model()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model

#X_train = X_train.reshape(X_train.shape[0], 1, X_train.shape[1])
#X_test = X_test.reshape(X_test.shape[0], 1, X_test.shape[1])
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2)
"""
#Write Outputfile
################

def create_index_to_label(label_to_index):
  index_to_label = {v: k for k, v in label_to_index.items()}
  return index_to_label

index_to_label = create_index_to_label(label_to_index)

def create_txt_file(y_pred, index_to_label, filename="predics.csv"):

  with open(filename, 'w') as f:
    f.write("id,labels\n")
    for i, prediction in enumerate(y_pred):
      labels = []
      for j, value in enumerate(prediction):
        if value == 1:
          labels.append(index_to_label[j])
      labels = sorted(labels)  
      f.write(f"{i+1},{';'.join(labels)}\n")


create_txt_file(y_pred, index_to_label)
