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


 
