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
## Iterative Loop of ML development
<p align = "center">
<img width="413" alt="image" src="https://github.com/atul2602/Advanced-Learning-Algorithms/assets/61497490/6812c8c1-494a-499e-8285-78d478dc4798">
</p> 

### Error Analysis
- The thing to check after Bias and Variance
- Eg, Email spam detection:
    - Look at misclassified examples (ramdom sample of ~100 misclassified emails)
    - Categorise them into common groups like Pharma, mispelled, fishy routing
    - Work on top groups eg, add more relevant data, ...
    - Best way is to combine bias/variance analysis with error analysis. Eg, if BV analysis tells to add more features, you now know with features are relevant. 
- Challenges: Difficult to analyse if even humans arent good at that task

### Tips for adding data
- Add data for types indicated by error analysis
- Data Augmentation: Create new training example from existing ones. Changes should be representative of real life
    - change orientation/size/contrast of image/ random warping
    - add different noises to audio clips
- Data Synthesis, usually for CV tasks
> Data Centric approach is also an efficient way as model-centric appproach

### Transfer Learning
- Learning using data from a different task
- Useful when data is less
- Eg. Transfer learning from object classification to image classfication
    - Train an object classification model : _Supervised Pretraining_ 
    - replace last layer with 0-9 digit classification layer. Use all other layers as it is
    - Train the model : _Fine Tuning_
- We can use readymade NNs for step 1 $\rightarrow$ faster development
- note: type of inputs must be same in both steps, like image-image.
- Why does this work?
<p align = "center">
<img width="230" alt="image" src="https://github.com/atul2602/Advanced-Learning-Algorithms/assets/61497490/e1316506-85a2-48a4-8295-a27cd98e1b3e">
</p>

## Full Cycle of ML Project
1. Scope of the project : Define the project
2. Data: Define and collect data
3. Train Model: Train, analyse error, develop interatively  (may go back to 2.)
4. Deploy in production : Monitor performance and maintain  (may go back to 3., 2.)
### Deployment (MLOps)
- deploy on a inference server, get API calls from apps, make prediction
- needs software engineering 

## Ethics in ML
### Bias in society
### Adverse Usecases (toxic/deep fake)
#### Tips:
- Get a diverse team to think on what all can go wrong
- Carry literature search on industry laws/standards
- audit the systems to possible harm _before deployment_
- Develop mitigation system for possible harms in deployment

## Skewed Datasets
- Checking accuracy is difficult in such applications, eg. detect rare disease
- New error metric: **Precision/Recall**
    - Use 2\*2 matrix to compute true +ve, true -ve, ...
    - $$Precision = \frac{True Positives}{True Positives + False Positives}$$
    - $$Recall = \frac{True Positives}{Actual Positives} = \frac{True Positives}{True Positives + False Negatives}$$
    - Both should be decently high (definetely $\ne 0$ )
- TradeOff
    - Precision means "if a person is diagnosed +ve, a high chance that they truly are"
    - Recall means "if a person is truly +ve, high chance that they are diagnosed +ve"
    - Increase 0.5 threshold in Logistic Regression : Increases Precision, decreases recall
    - Decrease 0.5 threshold in logistic regression : Decreases precision, increases recall
 - Plot to pick threshold
 <p align = "center">
 <img width="167" alt="image" src="https://github.com/atul2602/Advanced-Learning-Algorithms/assets/61497490/df1e8cf0-44a8-4735-afba-6c5c5e41572e">
 </p>
 
 - **F1 Score**
     - To compare precision/recall all together
     - Closer to lower value
     - $$F1 score = \frac{2.P.R}{P+R}$$

 
    
