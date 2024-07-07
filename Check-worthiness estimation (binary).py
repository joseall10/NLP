import pandas as pd
import numpy as np
import nltk
import spacy
import re
import tensorflow as tf
from sklearn.svm import SVC
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import GridSearchCV

# Download necessary resources from nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Initialize spacy for lemmatization
nlp = spacy.load('en_core_web_sm')

# Text preprocessing function
def preprocess_text(text):
    text = text.lower()  # Convert text to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Remove non-alphabetic characters
    tokens = word_tokenize(text)  # Tokenize the text
    stop_words = set(stopwords.words('english'))  # Define stopwords in English
    tokens = [word for word in tokens if word not in stop_words]  # Remove stopwords
    lemmatizer = WordNetLemmatizer()  # Initialize lemmatizer
    tokens = [lemmatizer.lemmatize(word) for word in tokens]  # Lemmatize tokens
    return ' '.join(tokens)  # Join tokens back into a single string

# Load the dataset
df = pd.read_csv('CT24_checkworthy_english_train.tsv', sep='\t')

# Apply text preprocessing
df['processed_text'] = df['Text'].apply(preprocess_text)

# Map labels to binary values
label_map = {'Yes': 1, 'No': 0}
df['class_label'] = df['class_label'].map(label_map)

# Save the preprocessed dataset to a new CSV file
df.to_csv('CT24_checkworthy_english_train_preprocessed.csv', index=False)

# Display the first few rows of the preprocessed dataset
print(df.head())

from gensim.models import Word2Vec

# Train the Word2Vec model
sentences = [text.split() for text in df['processed_text']]
word2vec_model = Word2Vec(sentences=sentences, vector_size=100, window=5, min_count=1, workers=4)
word2vec_model.save("word2vec_model.bin")

# Function to get the average vector of a text using the Word2Vec model
def get_average_vector(text, model, vector_size):
    average_vector = np.zeros((vector_size,), dtype="float32")
    num_words = 0
    for word in text.split():
        if word in model.wv:
            average_vector += model.wv[word]
            num_words += 1
    if num_words != 0:
        average_vector /= num_words
    return average_vector

# Apply the function to get the average vectors of each text in 'processed_text'
vector_size = 100
df['processed_text_vector'] = df['processed_text'].apply(lambda x: get_average_vector(x, word2vec_model, vector_size))

# Drop unnecessary columns
df_train = df.drop(columns=['Sentence_id','Text', 'processed_text'])

# Display the first few rows of the training dataset
print(df_train.head())

# Load the test dataset
df_test = pd.read_csv('CT24_checkworthy_english_dev-test.tsv', sep='\t')
df_test['class_label'] = df_test['class_label'].map(label_map)
df_test['processed_text'] = df_test['Text'].apply(preprocess_text)
df_test['processed_text_vector'] = df_test['processed_text'].apply(lambda x: get_average_vector(x, word2vec_model, vector_size))

# Drop unnecessary columns from the test dataset
df_test = df_test.drop(columns=['Sentence_id','Text', 'processed_text'])

# Display the first few rows of the test dataset
print(df_test.head())

# Verify the data
print("Class distribution in the training set:")
print(df_train['class_label'].value_counts())

print("Class distribution in the test set:")
print(df_test['class_label'].value_counts())

print("Example of average text vector:")
print(df_train['processed_text_vector'].iloc[0])

# Define the RNN model creation function
def create_model(optimizer='adam', init='glorot_uniform'):
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.LSTM(128, input_shape=(1, vector_size), return_sequences=True, kernel_initializer=init))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.LSTM(128, kernel_initializer=init))
    model.add(tf.keras.layers.Dropout(0.2))
    model.add(tf.keras.layers.Dense(1, activation='sigmoid', kernel_initializer=init))
    model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
    return model

# Prepare the training and test data for the RNN
X_train = np.array(df['processed_text_vector'].tolist()).reshape(-1, 1, vector_size)
y_train = np.array(df_train['class_label'])
X_test = np.array(df_test['processed_text_vector'].tolist()).reshape(-1, 1, vector_size)
y_test = np.array(df_test['class_label'])

# Create the model with manually optimized parameters
nn_model = create_model(optimizer='adam', init='glorot_uniform')
nn_model.fit(X_train, y_train, epochs=10, batch_size=20, verbose=1)

# Make predictions with the RNN
y_pred_proba_nn = nn_model.predict(X_test)
y_pred_nn = (y_pred_proba_nn > 0.5).astype(int).flatten()

# Calculate the F1-score for the RNN
f1_nn = f1_score(y_test, y_pred_nn)
print(f'F1-score for the neural network: {f1_nn}')

# Verify the labels and predictions of the RNN
print("Example of test labels:")
print(y_test[:10])

print("Example of neural network predictions:")
print(y_pred_nn[:10])

# Prepare the training and test data for the SVM
X_train = np.array(df['processed_text_vector'].tolist())
X_test = np.array(df_test['processed_text_vector'].tolist())

# Define the SVM model
svm_model = SVC()

# Define the parameter grid for GridSearchCV
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf', 'linear']
}

# Configure GridSearchCV
grid = GridSearchCV(estimator=svm_model, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)

# Fit the SVM model
grid_result = grid.fit(X_train, y_train)

# Display the best parameters found
print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")

# Train the model with the best parameters
best_model = grid_result.best_estimator_

# Make predictions on the test set
y_pred_svm = best_model.predict(X_test)
f1_svm = f1_score(y_test, y_pred_svm)
print(f'F1-score for the SVM: {f1_svm}')

# Verify the labels and predictions of the SVM
print("Example of SVM predictions:")
print(y_pred_svm[:10])

# Display the classification report for both models
print("Classification report for the neural network:\n", classification_report(y_test, y_pred_nn))
print("Classification report for the SVM:\n", classification_report(y_test, y_pred_svm))
