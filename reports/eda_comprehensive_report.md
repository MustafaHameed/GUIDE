# Exploratory Data Analysis Summary

## Dataset Overview
- **Total samples**: 395
- **Total features**: 33
- **Categorical features**: 17
- **Numerical features**: 16
- **Average final grade (G3)**: 10.4
- **Pass rate (≥10)**: 67.1%

## Key Categorical Features
The most important categorical features for predicting student performance:

- **Mjob**: Importance score 0.044
  - Most common: other (141 students)
- **higher**: Importance score 0.033
- **sex**: Importance score 0.029
  - Gender distribution: {'F': np.int64(208), 'M': np.int64(187)}
- **reason**: Importance score 0.024
- **paid**: Importance score 0.018

## Categorical Variable Relationships
- Strongest categorical association: **famsup** ↔ **paid** (Cramer's V = 0.284)
- Number of strong categorical associations (Cramer's V > 0.2): 8

## Key Insights and Recommendations
- **Mjob** shows the strongest relationship with academic performance
- Students with no previous failures average 11.3 points
- Students with failures average 6.7 points
- Students with low absences average 10.4 points vs 10.5 for high absences