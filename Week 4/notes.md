## Decision Trees
<p align = "center">
  <img width="550" alt="image" src="https://github.com/atul2602/Advanced-Learning-Algorithms/assets/61497490/eb86fcfe-8c79-4d4c-9dc2-8bda46953518">
</p>

- Above is an example of classification task : cat or not cat, given features which take finite discrete values.

### Learning
- **Decision 1**: How to choose what feature to split on at each node? _Maximise purity_ of batches
- **Decision 2**: When do you stop splitting?
    - When a node is 100% one class
    - When a node exceeds maxm. allowed depth (prevents overfitting)
    - Improvements in purity are less than a defined threshold 
    - When number of examples in a node is below a threshold

#### Measuring Purity
- **Entropy**: $H(p1) = -p_{1} \log_{2}{p_1} - p_{0}\log_{2}{p_0}$ , $p_0 = 1 - p_1$,
where $p_1$ is the fraction of positive class at the node
<p align = "center">
<img width="225" alt="image" src="https://github.com/atul2602/Advanced-Learning-Algorithms/assets/61497490/b4eb8e2e-4d54-4803-9e44-2a5a47c8f836">
</p>

NOTE : Take $0.\log{0} = 0$
- For more than 2 features, convert k-categorical feature into k-binary features (one hot encoding)
- (By this pt. you know how will entropy function look like)

#### Choosing a split 
- **Information Gain** : $$H({p_1}^{root})- \frac{W^{left}.H({p_1}^{left}) + W^{right}.H({p_1}^{right})}{W^{root}}$$ <p align = "center">, $W$ = No. of examples</p>
    - Provides info. regarding reduction in entropy
    - Also used for stopping criteria

#### Learning Algorithm: 
Start with all the examples at the root $\rightarrow$ Calculate I.G. for all possible features $\rightarrow$ Split the node $\rightarrow$ Check for stopping $\rightarrow$ Recurse

#### Continuous Valued features
Eg. Weight of cats/dogs
Choose a cutoff for the features like $Weight > x$, by trying values at consecutive sorted mid-point and select one with best Information Gain.

### Generalization to Regression Trees
Eg. Predict weight from features : Create decision tree on features, and define weight for a leaf as average of examples in the leaf.
- Choosing a split : Replace $H$ in classification to `Variance` of the predicted output feature
 
## Tree Ensembles
- Need: Trees are highly sensitive to small changes in data
- Prediction : Voting by trees
- Build : **Sample with replacement** `n` examples from the set of examples, `m` times to create m bagged decision trees
#### Random Forest Algorithm
- For b = 1 to B (64-100):
    - Use sampling with replacement to create new training set of size `m`
    - Train a decision tree on the new dataset
- **Randomizing the feature choice**: At each split, choose subset k of n available features and take the k features ahead, $k \approx \sqrt{n}$ 
- Testing : Take votes from all trees in ensemble (forest)

### XG Boost : Idea of delicate practice
- While sampling with replacement, skew the probability towards examples missclassified by previous trees
- eXtreme Gradient Boosting
    - Open-source
    - Fast, efficient
    - Good default splitting, stopping criteria
    - Built in regularization
    - good for comps.

```python
from xgboost import XGBClassifier
model = XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
```
### When to use Decision Trees (vs NNs)?
- Work well on tabular(structured) data, spreadsheet
- Not recommended for unstructured data: Images, videos, audio, text   (NNs recommended)
- Fast (training), whereas NNs are slow
- Small trees are human-interpretable
- NNs work with transfer learning, and easier to string together multiple NNs



