# Dry Bean Classification using Machine Learning

## a. Problem statement

This project implements multiple supervised machine learning classification models to predict the type of dry bean based on geometric and shape-based features.

The dataset contains physical and structural measurements of beans, and the goal is to correctly classify them into their respective bean varieties.


## b. Dataset description

The dataset used is the Dry Bean Dataset, which contains:

- 13,000+ samples
- 16 numerical feature columns
- 1 target column (`Class`)
- 7 bean varieties:
  - SEKER
  - BARBUNYA
  - BOMBAY
  - CALI
  - DERMASON
  - HOROZ
  - SIRA

### Features:
- Area
- Perimeter
- MajorAxisLength
- MinorAxisLength
- AspectRation
- Eccentricity
- ConvexArea
- EquivDiameter
- Extent
- Solidity
- Roundness
- Compactness
- ShapeFactor1
- ShapeFactor2
- ShapeFactor3
- ShapeFactor4


The target variable (`Class`) represents the bean type.


##  Model Performance Comparison

| Model                   | Accuracy     | AUC          | Precision    | Recall       | F1           | MCC          |
| ----------------------- | ------------ | ------------ | ------------ | ------------ | ------------ | ------------ |
| **Logistic Regression** | 0.934264     | 0.995727     | 0.943665     | 0.942003     | 0.942752     | 0.920563     |
| **Decision Tree**       | 0.905252     | 0.951610     | 0.917029     | 0.919824     | 0.918349     | 0.885547     |
| **KNN**                 | 0.936467     | 0.988076     | 0.948475     | 0.945753     | 0.947041     | 0.923164     |
| **Naive Bayes**         | 0.907088     | 0.992574     | 0.911884     | 0.915233     | 0.913199     | 0.888100     |
| **Random Forest**       | 0.938671     | 0.994846     | 0.948116     | 0.945717     | 0.946870     | 0.925833     |
| **XGBoost**             | 0.943445     | 0.996501     | 0.953234     | 0.951431     | 0.952316     | 0.931604     |

##  Observations

| Model                   | Observations                                                                                          |
| ----------------------- | ----------------------------------------------------------------------------------------------------- |
| **Logistic Regression** | Good performance, indicating good linear separability within the dataset.                             |
| **Decision Tree**       | Low accurancy and MCC score. Produced comparatively lower performance than other models.              |
| **KNN**                 | Performed competitively after feature scaling, showing effective class clustering in feature space.   |
| **Naive Bayes**         | Achieved moderate performance with strong probabilistic separation but lower classification accuracy. |
| **Random Forest**       | Delivered high and stable performance across evaluation metrics.                                      |
| **XGBoost**             | Top performer. Achieved the best overall performance across all evaluated models.                     |
