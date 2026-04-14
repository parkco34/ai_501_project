# Online Shoppers Purchasing Intention Prediction

This project was completed for AAI-501 at the University of San Diego.

**Project Status: Completed**

## Installation

Clone the repository:

```bash
git clone https://github.com/parkco34/ai_501_project.git
cd ai_501_project
```

Install the required packages:

```bash
pip install pandas numpy seaborn matplotlib scikit-learn imbalanced-learn
```

Run the consolidated notebook:

```bash
cd notebooks
jupyter notebook team4_consolidated.ipynb
```

## Project Objective

The goal of this project is to predict whether an online shopping session will result in a purchase using machine learning methods applied to behavioral e-commerce data.

## Dataset

**Source:** UCI Machine Learning Repository
**Dataset:** Online Shoppers Purchasing Intention Dataset

- **Size:** 12,330 sessions
- **Features:** 18 total attributes
- **Target Variable:** `Revenue`
- **Class Distribution:** ~84.5% non-purchase, ~15.5% purchase

Class imbalance was addressed using SMOTE on the training data.

## Methods Used

- Exploratory Data Analysis
- K-Means Clustering
- Logistic Regression
- Support Vector Machine (SVM)
- Random Forest
- SMOTE Oversampling
- Hyperparameter Tuning (GridSearchCV)

## Results

| Model | Accuracy | Precision | Recall | F1 Score | AUROC |
|-------|----------|-----------|--------|----------|-------|
| Logistic Regression | 0.8621 | 0.5430 | 0.6937 | 0.6092 | 0.8594 |
| Random Forest | 0.8808 | 0.5961 | 0.7147 | 0.6500 | 0.9148 |
| SVM | 0.8760 | 0.5770 | 0.7380 | 0.6480 | 0.9090 |
| K-Means | N/A | N/A | N/A | N/A | Silhouette: 0.66 / 0.71 |

Random Forest achieved the highest AUROC, while SVM achieved the highest recall.

## Repository Structure

```
ai_501_project/
├── notebooks/
│   ├── team4_consolidated.ipynb
│   ├── parker_eda.ipynb
│   ├── parker_svm.ipynb
│   ├── logistic_regression.ipynb
│   ├── kmeans.ipynb
│   └── random_forest.ipynb
├── online_shoppers_intention.csv
└── README.md
```

## Contributors

- Paola Marsal
- Nathan Butcher
- Glen Salazar
- Cory Parker

## License

This project is for academic purposes as part of the AAI-501 course at the University of San Diego.

## References

Breiman, L. (2001). Random forests. *Machine Learning*, 45(1), 5–32.

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE: Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research*, 16, 321–357.

Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning*, 20(3), 273–297.

Sakar, C., & Kastro, Y. (2018). Online shoppers purchasing intention dataset. UCI Machine Learning Repository.

Sakar, C. O., Polat, S., Katircioglu, M., & Kastro, Y. (2019). Real-time prediction of online shoppers' purchasing intention. *Neural Computing and Applications*, 31(10), 6893–6908.
