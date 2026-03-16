# Online Shoppers Purchasing Intention Prediction
Course: AAI-501 – Foundations of Artificial Intelligence
Institution: University of San Diego
Team Members: Nathan Butcher, Paola Marsal, Cory Parker, Glen Salazar
Date: March 2026

# Project Overview
This project investigates the prediction of *online purchasing intention* using machine learning methods applied to behavorial session data from an e-commerce platform.

The dataset used is the *Online Shoppers Purchasing Intention Dataset* from the *UCL Machine Learning Repository* (Sakar & Kastro, 2018).  The dataset contains *12,330 user sessions collected over a year*, where each record represents a single visitor interaction with an online shopping website.  

Each session contains *10 numerical features and 8 categorical features* describing user behavior which includes:
    - Number of pages visited
    - Duration of visits
    - Bounce rate (percentage of visitors who land on website and leave without interacting with the site)
    - Exit rate
    - Page value
    - Traffic Source
    - Visitor type
    - Returning visitor status

*Binary Classfication Problem*
The target variable is *Revenue ∈ {0,1}*, where
    0 = Session didn't result in a purchse
    1 = Session did result in a purchase

## Business Motivation
Predicting purchase intention is important in for e-commerce platforms.
Accurate prediction enables organizations to:
    - Personalize user experiences in real time
    - Optimize marketing and advertising spend
    - Reduce rate which the cart is abandoned
    - Target high-value customers
    - Improve conversion rates
   
## Algorithms Used
1. __K-Means Clustering (Unsupervised Learning)__
K-Means will be used to locate **behavioral segments of shoppers** based on browsing patterns.

2. __Logistic Regression__
Logistic Regression provides an *interpretable baseline classifier*.
The model estimates the probability of a purchase:
> $$P(y = 1|x) = \frac{1}{1 + e^{-(\beta_{0} + \beta^{T}x)}}$$

**Advantages:**
- Interpretable
- Fast training
- Strong baseline for binary classification

3. __Random Forest__
An ensemble learning method, which aggregates multiple decision trees.

**Advantages:**
- Handles nonlinear relationships
- Robust to overfitting
- Provides feature importance metric
    
4. __Support Vector Machine (SVM)__
Attempts to find the maximum-margin hyperplane separating the classes.

Optimization Objective:
> $$min_{_{w,b}}\frac{1}{2}||w||^{2}$$
> $$y_{i}(w \dot x_{i} + b) \leq 1$$

## Dataset

**Source:** UCI Machine Learning Repository  
**Dataset:** Online Shoppers Purchasing Intention Dataset  

**Observations:**  
12,330 sessions

**Features:**  
18 attributes (10 numerical, 8 categorical)

**Target Variable:**  
`Revenue`

**Class Distribution**

- Non-purchase sessions: ~84.5%
- Purchase sessions: ~15.5%

Because the dataset is highly imbalanced, additional techniques will be explored to ensure reliable model performance.

---

## Key Challenges

### Class Imbalance

The dataset is significantly skewed toward non-purchase sessions.

Methods that will be investigated include:

- SMOTE (Synthetic Minority Over-sampling Technique)
- Class weighting
- Stratified sampling

These approaches aim to improve the model’s ability to correctly identify purchasing sessions.

---

### Feature Selection

With 18 available features, determining which variables contribute most to predictive performance is important.

Methods considered include:

- Random Forest feature importance
- Correlation analysis
- Recursive feature elimination

This analysis may allow the model to maintain similar predictive accuracy using a reduced feature set.

---

### Hyperparameter Tuning

Model performance will be optimized through hyperparameter tuning.

Examples include:

**Random Forest**

- `n_estimators`
- `max_depth`
- `min_samples_split`

**Support Vector Machine**

- `kernel`
- `C` (regularization parameter)
- `gamma`

Grid search or randomized search will be used to identify well-performing parameter combinations.

---

### Optimal Number of Clusters

For K-Means clustering, determining the appropriate number of clusters is necessary.

Methods used:

- Elbow Method
- Silhouette Score

These metrics help identify meaningful shopper behavioral segments.

---

## Model Evaluation

Models will be evaluated using common classification metrics:

- **Accuracy** – overall prediction correctness  
- **Precision** – proportion of predicted purchases that were correct  
- **Recall** – proportion of actual purchases correctly identified  
- **F1 Score** – harmonic mean of precision and recall  
- **AUROC** – area under the receiver operating characteristic curve  

Visualization techniques will include:

- Confusion matrices
- ROC curves
- Feature importance plots

---

## Expected System Behavior

The system developed in this project will demonstrate several capabilities.

### Shopper Segmentation

The K-Means clustering model will group sessions into behavioral clusters such as:

- High-engagement browsers
- Quick-exit visitors
- High-intent buyers

This allows exploratory understanding of user behavior.

---

### Purchase Prediction

The classification models will predict whether a given session will result in a purchase based on session-level behavioral metrics.

Such predictions could support real-time actions such as:

- personalized offers
- targeted marketing messages
- dynamic product recommendations

---

### Algorithm Comparison

The system will produce a comparative analysis of:

- Logistic Regression
- Random Forest
- Support Vector Machine

Performance comparisons will be supported with quantitative metrics and visualizations.

---

### Actionable Insights

By combining clustering and classification results, the system will help identify which shopper behaviors are most strongly associated with purchasing outcomes.

These insights can support decision-making in digital commerce environments.

---

## References

Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321-357.

Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273-297.

Geron, A. (2022). *Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow* (3rd ed.). O’Reilly Media.

Poole, D. L., & Mackworth, A. K. (2023). *Artificial intelligence: Foundations of computational agents* (3rd ed.). Cambridge University Press.

Sakar, C., & Kastro, Y. (2018). Online shoppers purchasing intention dataset. UCI Machine Learning Repository.

Sakar, C. O., Polat, S., Katircioglu, M., & Kastro, Y. (2019). Real-time prediction of online shoppers’ purchasing intention using multilayer perceptron and LSTM recurrent neural networks. *Neural Computing and Applications*, 31(10), 6893-6908.

Scikit-learn developers. (2024). Support vector machines. https://scikit-learn.org/stable/modules/svm.html








