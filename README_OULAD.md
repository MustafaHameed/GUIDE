# Open University Learning Analytics Dataset (OULAD)

## Dataset Overview

The Open University Learning Analytics Dataset (OULAD) contains data about courses, students and their interactions with Virtual Learning Environment (VLE) for seven selected courses and their presentations. The dataset includes demographic data, assessment results, and detailed logs of student interactions with course materials.

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