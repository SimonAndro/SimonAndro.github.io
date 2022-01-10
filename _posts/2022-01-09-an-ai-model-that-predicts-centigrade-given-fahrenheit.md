---
layout: post
title: Get the feel - Building a dummy AI model that predicts temperature in Centigrade given temperature in Fahrenheit
subtitle: Learning the temperature conversion formular from data values
cover-img: /assets/img/path.jpg
thumbnail-img: /assets/img/temp-conv-dummy-ai-model/thumb.png
share-img: /assets/img/path.jpg
gh-repo: simonandro/predict-centigrade-given-fahrenheit
tags: [AI model, model training, keras, machine learning, deep learning, temparature conversion]
---

  In this article, we will be building a dummy AI model that learns the temperature conversion formular from data values generated using the actual formular.

  The code can be found on [github](https://github.com/SimonAndro/predict-centigrade-given-fahrenheit)

### Motivation for building this dummy model:
  To Show how a model can be built and evaluated using the keras framework

### Background
  The temperature conversion formular form Degrees Fahrenheit to Degrees Centigrade is given below in python:

  ```
  centigrade = (fahrenheit - 32)*(5/9)
  ```

  Given some data values generated using the above formular, we want to train a simple AI model to learn the formular and be able to predict temperature in Centigrade given unseen values of temperature in Fahrenheit.

### Implementation in Python and Kera
We shall begin by importing the necessary python modules

{% highlight python linenos %}
"""
A dummy AI model for predicting temperature in degrees centigrade
given degrees fahrenheit 
"""

#
#import necessary modules
#
from sklearn.model_selection import train_test_split
from keras import models
from keras import layers

import numpy as np
import matplotlib.pyplot as plt
{% endhighlight %}

We then go ahead to generate a dummy dataset using the temperature conversion formular
{% highlight python linenos %}
#
# create  a dummy dataset
#
#centigrade = (fahrenheit - 32)*(5/9) # conversion formula
sample_size = 1000
test_size = 0.25

# train set
x_train = np.random.uniform(1,100,(int(sample_size*(1-test_size)),1)) # generate random Fahrenheit values between 1 and 100
y_train = np.array([(fahrenheit - 32)*(5/9) for fahrenheit in x_train]) # calculate corresponding Centigrade values

# test set
x_test = np.random.uniform(101,200,(int(sample_size*test_size),1)) # generate random Fahrenheit values between 1 and 100
y_test = np.array([(fahrenheit - 32)*(5/9) for fahrenheit in x_test]) # calculate corresponding Centigrade values
{% endhighlight %}

Note how we specify different value  ranges for the train and test set using the uniform function in the random package of numpy, what the parameters mean is shown below:

``` 
np.random.uniform(lower value, higher value, size) 
```

The different ranges are specified so as to ensure we don't leak test values into the train data set.

We go ahead to build the model, specifying the layers and the number of units :

{% highlight python linenos %}
#
# build the model
#
def build_model():
    """
    build a sequential dense neural network
    """
    model = models.Sequential()
    model.add(layers.Dense(64, activation='relu',
                           input_shape=(1,)))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    return model
{% endhighlight %}

### Explanation of the model
The model is made of 3 densely connected neural network layers stacked together sequentially.
The number of units in each layer has been chosen out of try and error, the stop condition being until when satisfying results are achieved. These units can be tweaked more to achieve better results.
During model compilation, the chosen loss function is mean squared error since we want to know how far the model is diverging from the actual values we want to predict. We also chose to monitor the mean absolute error in the metrics, for a similar reason as with the mean squared error.

After building the model, we go ahead to fit the training data onto the built model. 
{% highlight python linenos %}
#
# model training
#
num_epochs = 20
model = build_model()
history = model.fit(x_train, y_train,
                        epochs=num_epochs, batch_size=8, verbose=1)
loss_values = history.history['loss']

mae_values = history.history['mae']

epochs = range(1, len(loss_values)+1)
{% endhighlight %}

We chose to train the model for 20 epochs and  a batch size of 8. 
Using pyplot from matplotlib, we can visualize the training process as below.
{% highlight python linenos %}
# training visualization
plt.figure()
plt.plot(epochs, loss_values, 'bo', label='Training loss')
plt.plot(epochs, mae_values, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()    
plt.show()   
{% endhighlight %}

![Model Training visualization](https://raw.githubusercontent.com/SimonAndro/predict-centigrade-given-fahrenheit/main/20epochs_training.png){: .mx-auto.d-block :}


We also go ahead to evaluate the model on the test data set, the results obtained are shown below:
{% highlight python linenos %}
#
# model evaluation
#
eval_mse, eval_mae = model.evaluate(x_test, y_test, verbose=1)
print("MSE=%s, MAE=%s" % (eval_mse, eval_mae))
{% endhighlight %}
```
[MSE=53.75002670288086, MAE=6.751595497131348]
```

The Results from the training process show that the model has learnt to fit the input values to the output values by around the 10th epoch. Further training maintains the loss and the mean absoulte error values almost constant. At this point, we shall make a decision of stopping the training at the 10th epoch so as to improve the model generalization on unseen data. 
The results of the evaulation, taking MAE as the bench mark, imply that we are off by almost 7 degrees centigrade which is quite large in actual sense.

Training the model again with an epoch count of 10, the new results from the model evalaution are shown below:
```
[MSE=18.294105529785156, MAE=4.252650260925293]
```
The above results show that the model has improved abit from the first MAE by about 2.5 degrees centigrade.

Let us go ahead to generate new data values in an unseen range so as to see how well the model performs on unseen data.
{% highlight python linenos %}
#
# application of the trained model, predicting values
#
x_new = np.random.uniform(201,500,(50,1)) # generate random Fahrenheit values between 1 and 100
y_new = np.array([(fahrenheit - 32)*(5/9) for fahrenheit in x_new]) # calculate corresponding Centigrade values

y_new_predicted = model.predict(x_new)
# plot predicted and actual
plt.figure()
plt.plot(y_new_predicted, 'b', label='Predicted Values')
plt.plot(y_new, 'g', label='Actual Values')
# plt.plot(y_new_predicted, y_new, 'b', label='Predicted Values')
plt.title('Predicted and Actual Values')
plt.legend()
plt.show()
{% endhighlight %}

The results of the plot of predicted values and the expected values are shown below:

![Model Training visualization](https://raw.githubusercontent.com/SimonAndro/predict-centigrade-given-fahrenheit/main/mae_4_predicted_actual.png){: .mx-auto.d-block :}

The results in the plot above show that the model has been able to learn the temperature conversion formular to some extent, the model is able to predict centigrade values with an average absolute error of about 4.2 degrees centigrade.

### Improvement
Exposing the model to more data values and adjusting the number of layers and input units can help to improve on how well the model performs in predicting the values of the temperature conversion formular.
Normalizing the temperature values before feeding them into the model could be another way of improving the performance.
