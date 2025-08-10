# Student Performance Dataset

This repository contains the `student-mat.csv` dataset with 395 records of Portuguese secondary school students and 33 features. The dataset includes demographic information, study habits and final grade columns `G1`, `G2` and `G3`.

## Usage

Load the data with [pandas](https://pandas.pydata.org/):

```python
import pandas as pd

df = pd.read_csv('student-mat.csv')
print(df.head())
```

## File

- `student-mat.csv` – raw dataset sourced from the UCI Machine Learning Repository.
