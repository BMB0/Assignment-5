# Neural Networks Song Genre Classification Problem
## Introduction

For the present work we seek to generate a song genre classifier using artificial neural networks with Tensorflow and Keras. A dataset was assigned with songs that had different attributes that described the songs with numerical parameters: 'track_id', 'artists', 'album_name', 'track_name', 'popularity', 'duration_ms', 'explicit', 'danceability', 'energy', 'key', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'time_signature', 'track_genre'.

## Data Analysis

To start working with the information, the first thing to do was to recognize those attributes that are representative to describe the genre of each song, so it was decided to discard from the dataset those that are not relevant or do not seem to be directly related to the genre to which each song belongs. These attributes are: 'track_id', 'artists', 'album_name', 'track_name', 'explicit', 'key', 'time_signature'. Those attributes that described songs through categorical data were replaced by numerical ones so that they can be worked on the network.

## Data Cleaning and Normalization

Then, the corresponding cleaning was performed, removing those data that were damaged or had missing information. With those useful data filtered out, data normalization was performed, since certain activation functions would have problems when performing the categories due to the difference in scales between the data. 

## Separate Data

The data were separated by training data and test data with 80% to 20% correspondingly. The labels are separated to enter the data into the model so that the data can be compared. 

## Formulating Hypothesis and Model Building

Moving on to the hypothesis of the model, several tests were performed in order to find a model that was able to train and predict the genres of the songs in the test data.

Models were defined with hidden layers of 1 to 3 layers with Relu activation functions, as well as Tanh and Sigmoid, which gave similar results except for Tanh, which gave lower accuracy. In the case of the output layer, Softmax was found to perform similarly to Sigmoid. Linear was also tested but it gave very poor results because it was not an activation function oriented to categorical problems. 

As for the number of neurons used per layer, powers of 2 were used, varying from the size of dimensions of our problem (in this case 12 dimensions) to 1024 in hidden layers. As for the output layer, we always used a number of neurons equivalent to the number of genera we had (114 genera). 

As for the optimizer, "Adam" was used, with different learning rates, but the one that gave the best performance was the default one with a value of 0.001. A sparse_categorical_crossentropy loss and metrics showing the accuracy. 
## Testing Models

In the different tests performed, it was observed that the accuracy of the training data increased, but the accuracy of the test data had a tendency to stagnate at approximately 30% and then overfitting was performed. 

Analyzing the combinations performed and the accuracy achieved, it is believed that this is due to the number of categories, which become unsustainable for the different activation functions, causing them to have poor predictions.

## Saving Models

The model saving was not performed by pkl, according to the Keras documentation it is not recommended to perform this action, so it was decided to use a Keras function to save the model.
