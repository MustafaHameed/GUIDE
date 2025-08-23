# Open University Learning Analytics Dataset (OULAD)

## Dataset Overview

The Open University Learning Analytics Dataset (OULAD) contains data about courses, students and their interactions with Virtual Learning Environment (VLE) for seven selected courses and their presentations. The dataset includes demographic data, assessment results, and detailed logs of student interactions with course materials.

## Download

Run the helper script to fetch and extract the raw CSV files:

```bash
python scripts/oulad_download.py
```

The data will be stored in `data/oulad/raw/`.

## Key Characteristics

- **Temporal Coverage**: Multiple course presentations labeled 'B' (February start) and 'J' (October start) from 2013-2014
- **Student Population**: 32,593 students across multiple modules
- **Course Modules**: Seven different course modules with varying difficulty and subject areas
- **Data Granularity**: Daily interaction logs, assessment submissions, demographic attributes

## Data Structure

OULAD consists of multiple interconnected tables linked by student IDs, course presentations, and module codes:

### Core Tables
- **studentInfo**: Student demographics (age, gender, education level, socioeconomic indicators)
- **studentVle**: Daily student interactions with VLE resources (clicks, views, downloads)
- **vle**: VLE object metadata (activity types, week associations)
- **studentRegistration**: Student module registration details and final outcomes
- **studentAssessment**: Assessment submission records and scores
- **assessments**: Assessment metadata (type, weight, due dates)

### Key Variables
- **Presentations**: 'B' (February start) and 'J' (October start) indicate timing
- **Final Results**: Pass, Fail, Withdrawn, Distinction outcomes
- **Sensitive Attributes**: Gender, age bands, education level, deprivation indices
- **Temporal Features**: Week-by-week engagement patterns, assessment timing

## Data Processing Pipeline

The OULAD data requires careful preprocessing to create a unified machine learning dataset:

1. **Feature Engineering**: Aggregate VLE interactions by week, compute assessment summaries
2. **Temporal Alignment**: Handle variable course durations and presentation timing
3. **Missing Data**: Address dropout patterns and incomplete engagement records
4. **Label Creation**: Binary pass/fail classification from final results

### Feature Definitions

The dataset builder in `src/oulad/build_dataset.py` generates a set of
aggregated features for each student-module-presentation combination:

- **VLE interaction features**
  - `vle_total_clicks`: total number of clicks across all VLE materials.
  - `vle_mean_clicks`: average number of clicks per active day.
  - `vle_max_clicks`: maximum clicks recorded on any single day.
  - `vle_first4_clicks`: clicks accumulated in the first four weeks.
  - `vle_last4_clicks`: clicks during the final four weeks of the course.
  - `vle_cumulative_clicks`: cumulative clicks across the presentation.
  - `vle_days_active`: number of distinct days with activity.
- **Assessment features**
  - `assessment_count`: number of submitted assessments.
  - `assessment_mean_score`: average score across assessments.
  - `assessment_last_score`: score on the most recent assessment.
  - `assessment_ontime_rate`: proportion of assessments submitted on or before their due date.
- **Existing registration attributes** such as `studied_credits` and
  `num_of_prev_attempts` are carried over for modeling.

### Label Creation

Labels are derived from the `studentRegistration` table:

- `label_pass` is 1 when the `final_result` is "Pass" and 0 otherwise.
- `label_fail_or_withdraw` is 1 for students who either "Fail" or
  "Withdrawn" and 0 for all other outcomes.
These binary targets enable downstream classification tasks and fairness
evaluation.

## Fairness Considerations

OULAD contains sensitive demographic attributes that enable fairness analysis:
- Gender (Male/Female)
- Age bands (0-35, 35-55, 55+)
- Highest education level
- Index of Multiple Deprivation (IMD) bands
- Regional indicators

## References

- [OU Analyse Dataset Portal](https://analyse.kmi.open.ac.uk/open-dataset)
- [OULAD Scientific Data Paper](https://www.nature.com/articles/sdata2017171)
- [Technical Documentation](https://pmc.ncbi.nlm.nih.gov/articles/PMC5704676/)