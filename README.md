## Check-worthiness estimation (binary)

### Description
We are going to compare two models that have the mission to determine whether a claim in a tweet and/or transcriptions is worth fact-checking. For that we are going to use a recurrent neuronal network and a SVM.

### Dataset
We are going to use a dataset that is partitioned in train and test. The train dataset is formed of 22502 examples of tweets in English and the test dataset is formed of 318 examples.

### Code 
The code is divided in parts that are the following:

  # Library import
  In the first lines, we are start importing the neccesary libraries for the project:spacy, tensorflow...
 
  # Necesary recourses for NLKT
  We are going to download a set of recourses necesaries for the code.

  # Initialization of spacy

  # Function for data preprocessingç
  We create a function for preprocessing the dataset that do the following steps:
  - Convert the text to lowercase
  - Remove the non alfabethic words
  - Tokenization of the text
  - Remove the stopwords
  - Lemmatization of the text
  - Return the preprocessed text

  # Load the dataset
  In this part, we also do the preprocessing of the data and we change the codificate the class label. Finally, we save the dataset in a CSV format.

  # Word2Vec trainment
  We are going to train a Word2Vec model with the train dataset and later we save the model.

  # Function for obtain average vector
  We create a function that obtain the average vector of a text.

  # Apply previous function
  We applied the function to the train dataset and finally we delete the innecesary columns for the test.

  # Load and preprocessing of the test dataset
  We load the test dataset and we apply the preprocessing function to it like we do with the train dataset.

  # Verify the data
  In this lines we verify which is the distribution of the classes in both datasets.

  # Define RNN model
  We define a RNN model with 64 recurrential layers, a dropout of 0.5 and a sigmoid activation to do the binary classification.

  # Trainment and evaluation of the RNN model
  We train the RNN model with the best hyperparameters and we evaluate it with the test dataset.

  # GridSearchSVM 
  We do a grid search to search the best hyperparameters for the SVM model. 

  # Trainment and evaluation of the SVM model
  We train the SVM model with the best hyperparameters and we evaluate it with the test dataset.

### Results
In the two, we are going to obtain a good accuracy but we are going to tell the results of the F1-score:
- RNN: 0.43
- SVM: 0.4

For that we execute 5 experiments for echa model and the results are the following with a aproximation to the second decimal:
- RNN: 0.4, 0.43, 0.45, 0.38, 0.48
- SVM: 0.3, 0.41, 0.48, 0.35, 0.49

### Conclusions
We can conclude several things about the project:
- The RNN model is better than the SVM model in general, but it´s not a big difference.
- The big oscilations on the results of the F1_score is because if we see the distribution on classes on trainment and test we can see that the positive class has far fewer examples than the negative class. This is due to the recall is good because the model can classificate good the negative examples but in the positive class, classification is more complicated.
- To improve, we have to augment the number of examples of the positive classes or use a model where aplly weights to the classes to focus on getting the positive class right. 
