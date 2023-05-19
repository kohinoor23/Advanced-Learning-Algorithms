## Neural Network Training
```python
import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense
model = Sequential([
  Dense(units=25, activation = "sigmoid").
  Dense(units=15, activation = "sigmoid")
  Dense(units=1, activation = "sigmoid")], name = "mymodel")

#Specify the loss function - like BinaryCrossentropy 
from tensorflow.keras.losses import BinaryCrossentropy
model.compile(loss = BinaryCrossentropy())
model.fit(X,Y, epochs = 100)
```
- Gradient Descent works as discussed earlier. Consider all the parameters of the network as one set of parameters while appyling cost reduction.
- Alternatives to sigmoid
   1. ReLU : $g(z) = max(0,z)$
   1. Linear : $g(z) = z$
   1. Softmax : ?
   1. tanh
   1. Leaky ReLU
   1. Switch

- Choose activation function on basis of type of output label. For hidden layers, ReLU is prominent.
## MultiClass Classification
- Example  
<p align = "center">
<img width="471" alt="image" src="https://github.com/atul2602/Advanced-Learning-Algorithms/assets/61497490/260fb64b-f388-4397-94fb-5c2621d62f30">
</p>

## Softmax Regression
$$z_i = \vec{w_i}.\vec{x} + b_i$$
$$a_i = \frac{e^{z_i}}{\sum_{j=1}^{n}e^{z^i}}$$

- Optimised Implementation for RoundOff errors: 
-   Skip calculation of intermediate $a_i$ 
-   Makes the loss function directly $-\log g\left(z \right)$
-   Final layer becomes linear, followed by one additional code line.
```python
...
...
Dense(units = 10, activation = 'linear')])

model.compile(loss=SparseCategoricalCrossEntropy(from_logits = True)
#note that logits argument is new.

logits = model(X)  #output is linear
f_x = tf.nn.softmax(logits) #make it softmax
```


