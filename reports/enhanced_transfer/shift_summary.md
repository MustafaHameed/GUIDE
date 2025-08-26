# Domain Shift Analysis: OULAD → UCI

## Overall Domain Similarity
- **Proxy A-distance**: 2.000 (0=identical, 2=completely different)
- **Domain Classifier AUC**: 1.000 (0.5=indistinguishable domains)

⚠️ **High domain shift detected** - transfer learning may be challenging

## Feature Shift Analysis
- **Features analyzed**: 5
- **Mean PSI**: 9.747
- **Max PSI**: 17.147

**Top shifting features (PSI)**: academic_proxy, engagement_proxy, ses_proxy, age_band, sex

## Label Shift Analysis
- **Source class proportions**: ['0.598', '0.402']
- **Estimated target proportions**: ['1.000', '0.000']
⚠️ **Significant label shift detected** - consider label shift correction

## Recommendations
- Use importance weighting and domain adaptation
- Consider CORAL feature alignment
- Apply DANN for deep learning models
- Apply label shift correction (Saerens-Decock method)
