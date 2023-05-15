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

