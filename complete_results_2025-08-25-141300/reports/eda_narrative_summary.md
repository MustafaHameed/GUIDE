# Exploratory Data Analysis Report: Student Performance Dataset

## Executive Summary

This report presents key findings from the exploratory data analysis of the student performance dataset containing **395 students** across various demographic and academic variables.

## Key Findings

### Overall Performance
- **Pass Rate**: 67.1% of students achieved a passing grade (≥10)
- **Average Final Grade**: 10.42 ± 4.58
- **Grade Range**: 0 to 20

### Strongest Numerical Correlations
The following variable pairs show the strongest linear relationships:

1. **G2 ↔ G3**: 0.905 (strong positive correlation)
2. **G1 ↔ G2**: 0.852 (strong positive correlation)
3. **G1 ↔ G3**: 0.801 (strong positive correlation)
4. **Dalc ↔ Walc**: 0.648 (moderate positive correlation)
5. **Medu ↔ Fedu**: 0.623 (moderate positive correlation)


### Most Influential Categorical Variables
Based on mutual information analysis, the following categorical variables have the strongest association with final grades:

- **Nursery**: Mutual Information = 0.062, Association strength (Cramér's V) = 0.000 (not significant)
- **Schoolsup**: Mutual Information = 0.041, Association strength (Cramér's V) = 0.158 (significant)
- **Romantic**: Mutual Information = 0.041, Association strength (Cramér's V) = 0.075 (not significant)


### Detailed Insights

#### Academic Performance Patterns
- **Grade Progression**: The correlation between G1→G2→G3 shows 0.852 (G1-G2) and 0.905 (G2-G3)
- **Study Time Impact**: Weekly study time shows a correlation of 0.098 with final grades
- **Attendance Effect**: School absences correlate 0.034 with final performance

#### Demographic Factors
- **Gender Performance**: Male students perform slightly better (average difference: 0.95 points)
- **School Performance**: School 'GP' shows higher average performance (difference: 0.64 points)


#### Risk Factors
- **High Absence Risk**: Students with >10 absences have an average grade of 10.15
- **Previous Failures**: Students with past failures average 7.27 vs 11.25 for those without

## Recommendations for Further Analysis

1. **Intervention Targeting**: Focus on students with high absences and previous failures
2. **Early Warning System**: Use G1 and G2 grades as strong predictors for final performance
3. **Support Programs**: Consider targeted support for underperforming demographic groups
4. **Study Habits**: Investigate the relationship between study time and effective learning strategies

---
*Report generated automatically from EDA analysis*
