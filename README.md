# IMDb Movie Rating Prediction Project

## Team Members
- Liad Malachi (ID: 212452882)
- Shirel Elmaleh (ID: 345653984)

---

## Project Overview

This project aims to predict IMDb movie ratings (`averageRating`) using Machine Learning techniques based on movie metadata.

The workflow includes:
- Data cleaning and preprocessing
- Feature engineering
- Model training and evaluation
- Fairness and error analysis
- Prevention of data leakage

Two regression models were evaluated using **10-Fold Cross Validation**:

| Model | RMSE |
|---------|---------|
| Elastic Net Regressor | 1.141 |
| Random Forest Regressor | 1.118 |

Based on the evaluation results, the **Random Forest Regressor** was selected as the final model due to its superior predictive performance and its ability to capture non-linear relationships within the data.

---

## Project Files

| File | Description |
|--------|--------|
| `project-part2.ipynb` | Complete Jupyter Notebook containing preprocessing, feature engineering, model training, and evaluation |
| `dataset.csv` | Original IMDb dataset used for training |
| `model.pkl` | Final trained and serialized Random Forest Pipeline |
| `README.md` | Project documentation and instructions |

---

## Final Model

The final model was retrained on the full available dataset in order to maximize its predictive capabilities for test-time.

The saved model consists of a complete **scikit-learn Pipeline** that embeds:
- Missing value imputation (Median for continuous, Most Frequent for binary features)
- Feature scaling (`StandardScaler`)
- The trained **Random Forest Regressor**

The entire pipeline was serialized and saved using **joblib** as `'model.pkl'`.

---

## Requirements

The following Python packages are required to run the project and load the model:

```bash
pip install pandas numpy joblib scikit-learn