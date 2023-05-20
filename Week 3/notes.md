## Advice on building ML systems
Eg. Debugging a learning algorithm making large errors. Choices are many, so spend time judiciously.
  - Get more data
  - Add/remove features/polynomial features
  - Modify regularisation constant
---
### ML Diagnostics ($Evaluate \rightarrow Improve$)
#### Evaluation
- For 1D features, we can look at the regression curve and tell about overfitting/underfitting through wiggly nature of curve.
- For higher dimensional, **split the data into two parts**:
    - Training Set (~70%)
    - Testing Set (~30%) - to get idea of generalisation error
- First, fit the data into parameters using training data ($m_{train}$)
- Then, calculate cost $J_{test}(\vec{w}, b)$ using test data ($m_{test}$), ignore regularisation here.
- Compare $J_{train}(\vec{w}, b)$ and $J_{test}(\vec{w}, b)$
- For logistic regression, we can also **compare fraction of misclassified set** for test and train.

### Model Selection 
- A flawed approach: Fitting the best d-degree polynomial, by changing d and looking at test error. You have contaminated the test data too!
- **Split the data into three parts**:
    - Training (~60%)
    - Cross Validation/Development(Dev) (~20%)
    - Testing (~20%)
- Choose d using dev set, $J_{CV}(\vec{w}, b)$
- Test the dth order model using test data.
#### Choosing Neural Net Architecture
- Similar procedure as above, instead of d, you have all models you think can work.

### Diagnosing Bias and Variance
<p align = "center">
<img width="475" alt="image" src="https://github.com/atul2602/Advanced-Learning-Algorithms/assets/61497490/ee104278-09e7-4e02-bec6-3792bccbc12e"> 
</p>

- In high bias, $J_{train}$ and $J_{CV}$, both are high.
- In high variance, $J_{CV}$ >> $J_{train}$ ~ 0.
- For a good model, $J_{train}$ and $J_{CV}$, both are low.
<p align = "center">
<img width="485" alt="image" src="https://github.com/atul2602/Advanced-Learning-Algorithms/assets/61497490/9a56ad8e-323f-4d91-98fb-b096f8d2e456"> 
</p>

- High bias and variance case may occur in Neural networks. It involved underfitting and overfitting parts of training data.

#### Regularisation and bias/variance
- High $\lambda$ lead to high bias, wherease low $\lambda$ leads to high variance.
- Vary $\lambda$ from 0 to 10 and choose one which gives least $J_{CV}$.
<p align = "center">
<img width="386" alt="image" src="https://github.com/atul2602/Advanced-Learning-Algorithms/assets/61497490/f1a2ac87-e672-49fd-9e3d-d365040e7854">
</p>

#### Establishing baseline performance
1. Compare $J_{train}$ with human level performance.
<p align = "center">
<img width="401" alt="image" src="https://github.com/atul2602/Advanced-Learning-Algorithms/assets/61497490/246f7a99-3cce-483c-bb6d-55f3ec76255c">
</p>

- In above Speech Recognition example, what appears to be a _high bias_ problem, is actually a _high variance_ problem !  
2. Competing algorithms performance
3. Guess based on experience
- It also depends on quality of data avaiable. For eg, in speech recognition, most of the recordings are noisy.
- If $J_{CV}$ >> $J_{train}$ >> $J_{baseline}$, then high bias and high variance exists.

### Learning Curves
- Performance vs Experience (Training data)
<p align = "center">
<img width="314" alt="image" src="https://github.com/atul2602/Advanced-Learning-Algorithms/assets/61497490/0161458a-ed0e-4aa7-95bc-bbd5ec7a7e6a"> 

<img width="510" alt="image" src="https://github.com/atul2602/Advanced-Learning-Algorithms/assets/61497490/90e8da16-2b07-4ffb-ac77-5a9545f45a34"> 
</p>

 - So in high bias model, getting more data will not help us improve, as error tends to constancy. It does help in high variance case.
<p align = "center">
<img width="512" alt="image" src="https://github.com/atul2602/Advanced-Learning-Algorithms/assets/61497490/d3104f07-5f28-4334-bb1c-8a6abd9ac342">
</p>

- Good, but expensive way of gauging bias and variance.
---
### Review
  - Get more data $\rightarrow$ fixes high variance
  - Add/remove features  $\rightarrow$ fixes high bias/high variance
  - Increase/decrease regularisation constant $\rightarrow$ fixes high variance/high bias

### Bias/Variance and Neural Networks
- Large neural nets are _low bias_ machines. 
- Recipe: 
<p align = "center">
  <img width="442" alt="image" src="https://github.com/atul2602/Advanced-Learning-Algorithms/assets/61497490/9f6e5dd7-f44d-4d50-9377-a825da2075d1">
</p> 

- Regularised large NNs usually perform better than small NNs on variance, although computationally expensive.
```python
#Regularised MNIST model
layer_1 = Dense(units = 25, activation = "relu", kernel_regularizer = L2(0.01))  #note the regularizer
layer_2 = ...
model = Sequential([layer_1, layer_2])
```

