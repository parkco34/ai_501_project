# Online Shoppers Purchasing Intention Prediction

This project is a part of the AAI-501 course in the Applied Artificial Intelligence Program at the University of San Diego (USD).

**-- Project Status: Completed**

## Installation

To use this project, first clone the repo on your device using the command below:

```bash
git init
git clone https://github.com/parkco34/ai_501_project.git
```

Then install the required packages:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn
```

To run the consolidated notebook:

```bash
cd notebooks
jupyter notebook team4_consolidated.ipynb
```

## Project Intro/Objective

The purpose of this project is to predict whether an online shopping session will result in a purchase, using machine learning methods applied to behavioral session data from an e-commerce platform. The system combines unsupervised clustering with supervised classification to segment shoppers by browsing behavior and predict purchase intent, generating actionable insights for e-commerce conversion optimization.

## Contributors

- Paola Marsal (Team Lead)
- Nathan Butcher
- Glen Salazar
- Cory Parker

## Methods Used

- Exploratory Data Analysis
- K-Means Clustering
- Logistic Regression
- Support Vector Machines (SVM)
- Random Forest
- SMOTE Oversampling
- Hyperparameter Tuning (GridSearchCV)
- Feature Importance Analysis
- Data Visualization

## Technologies

- Python 3.x
- pandas
- NumPy
- scikit-learn
- imbalanced-learn
- Matplotlib
- Seaborn

## Project Description

**Dataset:** Online Shoppers Purchasing Intention Dataset (Sakar & Kastro, 2018), sourced from the UCI Machine Learning Repository.

- **Size:** 12,330 sessions collected over one year
- **Features:** 18 total attributes (10 numeric, 8 categorical) describing browsing behavior such as page visit counts, session durations, bounce rates, exit rates, page values, traffic source, and visitor type
- **Target Variable:** `Revenue` (binary — purchase or no purchase)
- **Class Distribution:** ~84.5% non-purchase, ~15.5% purchase

The severe class imbalance was addressed using SMOTE on the training set. Four algorithms were implemented and compared: K-Means Clustering segmented shoppers into behavioral groups using the Elbow Method and Silhouette Score. Logistic Regression served as an interpretable baseline classifier. Support Vector Machine (RBF kernel) was tuned via GridSearchCV over C and gamma. Random Forest provided ensemble predictions and built-in feature importance rankings. All supervised models were evaluated using accuracy, precision, recall, F1-score, and AUROC.

## Repository Structure

```
ai_501_project/
├── notebooks/
│   ├── team4_consolidated.ipynb    # Final consolidated notebook (all models)
│   ├── parker_eda.ipynb            # EDA development notebook
│   ├── parker_svm.ipynb            # SVM development notebook
│   ├── logistic_regression.ipynb   # Logistic Regression development notebook
│   ├── kmeans.ipynb                # K-Means development notebook
│   └── random_forest.ipynb         # Random Forest development notebook
├── online_shoppers_intention.csv   # Dataset
└── README.md
```

## License

This project is for academic purposes as part of the AAI-501 course at the University of San Diego.

## Acknowledgments

We thank Professor Andrew for guidance throughout the course.

## References

Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5-32.

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321-357.

Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273-297.

Sakar, C., & Kastro, Y. (2018). Online shoppers purchasing intention dataset. UCI Machine Learning Repository.

Sakar, C. O., Polat, S., Katircioglu, M., & Kastro, Y. (2019). Real-time prediction of online shoppers' purchasing intention using multilayer perceptron and LSTM recurrent neural networks. *Neural Computing and Applications*, 31(10), 6893-6908.
