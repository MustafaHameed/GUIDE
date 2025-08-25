# Data Card: Student Performance Dataset

## Dataset Overview

**Dataset Name:** Student Performance Dataset  
**Domain:** Educational Data Mining, Student Academic Performance Prediction  
**Original Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Student+Performance)  
**License:** CC BY 4.0 (Creative Commons Attribution 4.0 International)  
**Data Collection Period:** 2005-2006 Academic Year  
**Geographic Coverage:** Portugal (Alentejo region)  

## Description

This dataset contains information about student achievement in secondary education of two Portuguese schools. The data attributes include student grades, demographic, social and school related features, and it was collected by using school reports and questionnaires.

### Key Characteristics
- **Instances:** 649 students (395 Mathematics, 649 Portuguese language courses)
- **Features:** 33 attributes (30 categorical/ordinal + 3 numeric grades)
- **Target Variables:** 
  - G1: First period grade (numeric: 0-20)
  - G2: Second period grade (numeric: 0-20) 
  - G3: Final grade (numeric: 0-20, 0=fail)
- **Missing Values:** None in the cleaned dataset
- **File Format:** CSV with semicolon delimiters

## Data Collection Methodology

### Data Sources
1. **School Reports:** Official academic records from two secondary schools
2. **Student Questionnaires:** Self-reported demographic and social information
3. **Administrative Records:** School-related features and family background

### Collection Process
- Data collected during the 2005-2006 school year
- Two distinct subjects: Mathematics (student-mat.csv) and Portuguese (student-por.csv)
- Students consented to data collection for research purposes
- Some students appear in both subject datasets

### Quality Assurance
- Cross-validation between school records and questionnaires
- Removal of incomplete records
- Standardization of categorical responses
- Grade validation against school policies (0-20 scale)

## Schema and Features

### Demographic Features
- **sex:** Student gender (binary: 'F', 'M')
- **age:** Student age (numeric: 15-22)
- **address:** Home address type (binary: 'U' urban, 'R' rural)
- **famsize:** Family size (binary: 'LE3' ≤3, 'GT3' >3)
- **Pstatus:** Parent cohabitation status (binary: 'T' together, 'A' apart)

### Family Background
- **Medu:** Mother's education (ordinal: 0-4, higher = more education)
- **Fedu:** Father's education (ordinal: 0-4, higher = more education)
- **Mjob:** Mother's job (categorical: 'teacher', 'health', 'civil_services', 'at_home', 'other')
- **Fjob:** Father's job (categorical: 'teacher', 'health', 'civil_services', 'at_home', 'other')
- **reason:** Reason for choosing school (categorical: 'close_home', 'school_reputation', 'course_preference', 'other')
- **guardian:** Student's guardian (categorical: 'mother', 'father', 'other')

### School-Related Features
- **school:** Student's school (binary: 'GP' Gabriel Pereira, 'MS' Mousinho da Silveira)
- **traveltime:** Travel time to school (ordinal: 1-4, 1: <15min, 4: >1hour)
- **studytime:** Weekly study time (ordinal: 1-4, 1: <2hours, 4: >10hours)
- **failures:** Number of past class failures (numeric: 0-4)
- **schoolsup:** Extra educational support (binary)
- **famsup:** Family educational support (binary)
- **paid:** Extra paid classes (binary)
- **activities:** Extra-curricular activities (binary)
- **nursery:** Attended nursery school (binary)
- **higher:** Wants higher education (binary)
- **internet:** Internet access at home (binary)
- **romantic:** In a romantic relationship (binary)

### Social and Health Features
- **famrel:** Quality of family relationships (ordinal: 1-5, 5=excellent)
- **freetime:** Free time after school (ordinal: 1-5, 5=very high)
- **goout:** Going out with friends (ordinal: 1-5, 5=very high)
- **Dalc:** Workday alcohol consumption (ordinal: 1-5, 5=very high)
- **Walc:** Weekend alcohol consumption (ordinal: 1-5, 5=very high)
- **health:** Current health status (ordinal: 1-5, 5=very good)
- **absences:** Number of school absences (numeric: 0-93)

### Target Variables (Academic Performance)
- **G1:** First period grade (numeric: 0-20)
- **G2:** Second period grade (numeric: 0-20)
- **G3:** Final grade (numeric: 0-20, primary target)

## Sensitive Attributes and Bias Considerations

### Protected Characteristics
- **sex:** Gender-based analysis for educational equity
- **age:** Age discrimination in academic assessment
- **address:** Urban vs. rural educational opportunities
- **Medu/Fedu:** Socioeconomic status via parental education
- **school:** School-based institutional bias

### Known Biases and Limitations
1. **Geographic Bias:** Data from only two Portuguese schools, may not generalize globally
2. **Temporal Bias:** Collected in 2005-2006, educational systems have evolved
3. **Cultural Bias:** Portuguese educational context and grading system
4. **Selection Bias:** Students who agreed to participate in data collection
5. **Reporting Bias:** Self-reported social features may contain inaccuracies

### Fairness Considerations
- **Gender Equity:** Historical differences in STEM performance by gender
- **Socioeconomic Status:** Parental education strongly correlates with student outcomes
- **Geographic Disparities:** Urban vs. rural resource availability
- **School Effects:** Institutional differences between the two schools

## Data Splits and Evaluation

### Recommended Splits
- **Standard:** 70% training, 15% validation, 15% test (stratified by G3 grade bins)
- **Temporal:** If modeling sequential grades, ensure no leakage from G3 to G1/G2 predictions
- **Cross-School:** Use one school for training, other for testing (distribution shift analysis)

### Evaluation Metrics
- **Primary:** Classification accuracy (pass/fail with threshold=10)
- **Regression:** MAE, RMSE for grade prediction
- **Fairness:** Demographic parity, equalized odds, calibration across sensitive groups
- **Stability:** Performance consistency across data splits and random seeds

## Ethical Considerations

### Privacy and Consent
- Student data anonymized with no personally identifiable information
- Consent obtained for research use during original collection
- School identities preserved for institutional analysis while maintaining student anonymity

### Educational Impact
- **Positive Use Cases:**
  - Early intervention systems for at-risk students
  - Resource allocation optimization
  - Understanding factors affecting academic success

- **Potential Misuse:**
  - Individual student profiling or discrimination
  - Reinforcing existing educational inequalities
  - Automated decision-making without human oversight

### Recommendations for Responsible Use
1. **Aggregate Analysis:** Focus on population-level insights, not individual predictions
2. **Bias Monitoring:** Regular fairness audits across demographic groups
3. **Human Oversight:** Algorithmic recommendations should support, not replace, educator judgment
4. **Transparency:** Model decisions should be explainable to educators and students
5. **Continuous Validation:** Regular revalidation with contemporary data

## Usage Guidelines

### Appropriate Applications
- Academic research on educational data mining
- Development of early warning systems for student support
- Understanding factors influencing academic achievement
- Fairness and bias analysis in educational ML systems
- Methodological research on predictive modeling

### Inappropriate Applications
- High-stakes individual student decisions without human review
- Commercial student ranking or labeling systems
- Discrimination against protected characteristics
- Applications outside educational contexts without domain validation

### Citation Requirements
When using this dataset, please cite:
```
P. Cortez and A. Silva. Using Data Mining to Predict Secondary School Student Performance. 
In A. Brito and J. Teixeira Eds., Proceedings of 5th FUture BUsiness TEChnology Conference 
(FUBUTEC 2008) pp. 5-12, Porto, Portugal, April, 2008, EUROSIS, ISBN 978-90-9023441-7.
```

## Quality and Maintenance

### Data Quality Score: B+ (Good)
- **Completeness:** Excellent (no missing values)
- **Accuracy:** Good (validated against school records)
- **Consistency:** Good (standardized collection process)
- **Timeliness:** Limited (15+ years old)
- **Representativeness:** Moderate (limited geographic scope)

### Known Issues
- Dataset age may limit contemporary relevance
- Limited to Portuguese educational system
- Self-reported social features may contain noise
- Some students appear in both subject datasets (linkage considerations)

### Recommended Updates
- Contemporary data collection with similar methodology
- Expansion to additional schools and regions
- Longitudinal tracking of student outcomes
- Integration with modern educational technology usage patterns

## Technical Specifications

### File Information
- **student-mat.csv:** 395 students, Mathematics course
- **student-por.csv:** 649 students, Portuguese course  
- **Encoding:** UTF-8
- **Delimiter:** Semicolon (;)
- **Header:** First row contains feature names
- **Format:** CSV (Comma-Separated Values, semicolon-delimited)

### Loading Instructions
```python
import pandas as pd

# Load mathematics dataset
df_math = pd.read_csv('student-mat.csv', sep=';')

# Load Portuguese dataset  
df_port = pd.read_csv('student-por.csv', sep=';')

# Basic preprocessing for binary classification
df_math['pass'] = (df_math['G3'] >= 10).astype(int)
```

---

**Data Card Version:** 1.0  
**Last Updated:** August 2025  
**Contact:** See repository maintainers for questions or concerns  
**License:** This data card is provided under CC BY 4.0 license