# Student Performance Dataset

This repository contains the `student-mat.csv` dataset with 395 records of Portuguese secondary school students and 33 features. The dataset includes demographic information, study habits and the final grade columns `G1`, `G2` and `G3`.

## Dataset Description

Each row in the CSV describes one student with the following columns:

- `school`: student's school (`GP` or `MS`)
- `sex`: student's sex (`F` or `M`)
- `age`: age in years (15–22)
- `address`: home address type (`U` for urban, `R` for rural)
- `famsize`: family size (`LE3` ≤ 3 or `GT3` > 3)
- `Pstatus`: parents' cohabitation status (`T` together or `A` apart)
- `Medu`: mother's education level (0 none to 4 higher education)
- `Fedu`: father's education level (0 none to 4 higher education)
- `Mjob`: mother's job (`teacher`, `health`, `services`, `at_home`, `other`)
- `Fjob`: father's job (`teacher`, `health`, `services`, `at_home`, `other`)
- `reason`: reason to choose this school (`home`, `reputation`, `course`, `other`)
- `guardian`: student's guardian (`mother`, `father`, `other`)
- `traveltime`: home to school travel time (1 <15 min, 2 15–30 min, 3 30–60 min, 4 >1 h)
- `studytime`: weekly study time (1 <2 h, 2 2–5 h, 3 5–10 h, 4 >10 h)
- `failures`: number of past class failures (n if 1≤n<3, else 4)
- `schoolsup`: extra educational support (`yes` or `no`)
- `famsup`: family educational support (`yes` or `no`)
- `paid`: extra paid classes (`yes` or `no`)
- `activities`: extra-curricular activities (`yes` or `no`)
- `nursery`: attended nursery school (`yes` or `no`)
- `higher`: wants to take higher education (`yes` or `no`)
- `internet`: Internet access at home (`yes` or `no`)
- `romantic`: with a romantic relationship (`yes` or `no`)
- `famrel`: quality of family relationships (1 very bad to 5 excellent)
- `freetime`: free time after school (1 very low to 5 very high)
- `goout`: going out with friends (1 very low to 5 very high)
- `Dalc`: workday alcohol consumption (1 very low to 5 very high)
- `Walc`: weekend alcohol consumption (1 very low to 5 very high)
- `health`: current health status (1 very bad to 5 very good)
- `absences`: number of school absences (0–93)
- `G1`: first period grade (0–20)
- `G2`: second period grade (0–20)
- `G3`: final grade (0–20)

## Usage

Load the data with [pandas](https://pandas.pydata.org/):

```python
import pandas as pd

df = pd.read_csv('student-mat.csv')
print(df.head())
```

## File

- `student-mat.csv` – raw dataset sourced from the UCI Machine Learning Repository.
