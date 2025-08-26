# Transfer Learning Report: OULAD → UCI

## Dataset Information
- **OULAD Dataset**: 5000 samples, 6 features
- **UCI Dataset**: 395 samples, 6 features
- **Shared Features**: gender, age_group, education_level, socioeconomic_status, prior_attempts, study_load

## Model Performance

### OULAD (Source Domain) Performance
- **logistic**: Accuracy = 0.5980, ROC AUC = 0.5132
- **random_forest**: Accuracy = 0.5240, ROC AUC = 0.4849
- **mlp**: Accuracy = 0.5880, ROC AUC = 0.5069

### UCI (Target Domain) Transfer Performance
- **Baseline (Majority Class)**: 0.6709

- **logistic**: Accuracy = 0.3291 (Δ = -0.3418), ROC AUC = 0.6948
- **random_forest**: Accuracy = 0.6253 (Δ = -0.0456), ROC AUC = 0.5433
- **mlp**: Accuracy = 0.4608 (Δ = -0.2101), ROC AUC = 0.5518

## Key Findings
1. **Best Transfer Model**: random_forest (improvement: -0.0456)
2. **Transfer Success**: No (models outperform baseline)
3. **Domain Gap**: Transfer learning effectiveness varies by model type

## Feature Mapping
The transfer learning uses these shared conceptual features:
- **Gender**: Direct mapping between datasets
- **Age Group**: Age ranges mapped to categories  
- **Education Level**: Educational background indicators
- **Socioeconomic Status**: Family and social context proxies
- **Prior Attempts**: Academic history indicators
- **Study Load**: Academic engagement measures

Generated on: 2025-08-26 11:35:56
