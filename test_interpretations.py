#!/usr/bin/env python3
"""Test script for interpretations functionality."""

import sys
import pandas as pd
from pathlib import Path
import re

# Set the results directory to the existing results
RESULTS_DIR = Path('./complete_results_2025-08-25-141300')

def interpret_model_performance():
    """Interpret model performance results."""
    interpretations = []
    
    try:
        # Load model performance data
        perf_path = RESULTS_DIR / "tables" / "model_performance.csv"
        if perf_path.exists():
            df = pd.read_csv(perf_path)
            
            # Find best performing model
            best_model = df.loc[df['accuracy_mean'].idxmax()]
            worst_model = df.loc[df['accuracy_mean'].idxmin()]
            
            interpretations.append(f"🏆 **Best Model**: {best_model['model_type']} achieved {best_model['accuracy_mean']:.3f} accuracy with low variance (±{best_model['accuracy_std']:.3f})")
            interpretations.append(f"📉 **Lowest Performance**: {worst_model['model_type']} with {worst_model['accuracy_mean']:.3f} accuracy")
            
            # Analyze performance spread
            acc_range = df['accuracy_mean'].max() - df['accuracy_mean'].min()
            interpretations.append(f"📊 **Performance Range**: {acc_range:.3f} accuracy spread indicates {'significant' if acc_range > 0.1 else 'moderate'} model differences")
            
            # Identify robust models (low std)
            robust_models = df[df['accuracy_std'] < df['accuracy_std'].median()]['model_type'].tolist()
            interpretations.append(f"🎯 **Most Robust Models**: {', '.join(robust_models[:3])} show consistent performance across folds")
            
    except Exception as e:
        interpretations.append(f"⚠️ Could not interpret model performance: {e}")
    
    return interpretations

def interpret_statistical_tests():
    """Interpret statistical test results."""
    interpretations = []
    
    try:
        stats_path = RESULTS_DIR / "tables" / "statistical_tests.csv"
        if stats_path.exists():
            df = pd.read_csv(stats_path)
            
            # Analyze significance
            significant = df[df['p_value'] < 0.05]
            interpretations.append(f"📈 **Significant Differences**: {len(significant)}/{len(df)} models show statistically significant performance differences (p < 0.05)")
            
            # Effect sizes
            large_effects = df[df['effect_size'].abs() > 0.8]
            if len(large_effects) > 0:
                interpretations.append(f"💪 **Large Effect Sizes**: {', '.join(large_effects['model'].tolist())} show substantial performance differences")
            
            # Best statistical performer
            if len(significant) > 0:
                best_stat = significant.loc[significant['effect_size'].idxmin()]
                interpretations.append(f"🎖️ **Statistically Superior**: {best_stat['model']} shows the strongest positive effect (effect size: {best_stat['effect_size']:.3f})")
                
    except Exception as e:
        interpretations.append(f"⚠️ Could not interpret statistical tests: {e}")
    
    return interpretations

def interpret_eda_results():
    """Interpret exploratory data analysis results."""
    interpretations = []
    
    try:
        # Grade distribution analysis
        grade_sex_path = RESULTS_DIR / "tables" / "grade_by_sex.csv"
        if grade_sex_path.exists():
            df = pd.read_csv(grade_sex_path)
            if len(df) > 1:
                female_grade = df[df['sex'] == 'F']['mean'].iloc[0] if 'F' in df['sex'].values else None
                male_grade = df[df['sex'] == 'M']['mean'].iloc[0] if 'M' in df['sex'].values else None
                
                if female_grade is not None and male_grade is not None:
                    diff = abs(female_grade - male_grade)
                    higher_performer = 'Female' if female_grade > male_grade else 'Male'
                    interpretations.append(f"👥 **Gender Performance**: {higher_performer} students perform {diff:.2f} points higher on average")
                    
                    if diff > 1.0:
                        interpretations.append("⚠️ **Significant Gender Gap**: Performance difference suggests potential bias or systematic factors")
                    else:
                        interpretations.append("✅ **Balanced Performance**: Minimal gender-based performance differences observed")
        
        # Study time analysis
        studytime_path = RESULTS_DIR / "tables" / "grade_by_studytime.csv"
        if studytime_path.exists():
            df = pd.read_csv(studytime_path)
            if len(df) > 1:
                correlation_strength = "strong" if df['mean'].corr(df['studytime']) > 0.7 else "moderate" if df['mean'].corr(df['studytime']) > 0.4 else "weak"
                interpretations.append(f"📚 **Study Time Impact**: {correlation_strength} positive correlation between study time and grades")
                
                highest_studytime = df.loc[df['mean'].idxmax()]
                interpretations.append(f"🎓 **Optimal Study Time**: Students with {highest_studytime['studytime']} hours/week achieve highest grades ({highest_studytime['mean']:.2f})")
                    
    except Exception as e:
        interpretations.append(f"⚠️ Could not interpret EDA results: {e}")
    
    return interpretations

def interpret_rmse_results():
    """Interpret RMSE and regression results."""
    interpretations = []
    
    try:
        rmse_path = RESULTS_DIR / "tables" / "rmse_bootstrap_ci.csv"
        if rmse_path.exists():
            df = pd.read_csv(rmse_path)
            
            # Find best regression model
            best_model = df.loc[df['mean'].idxmin()]
            worst_model = df.loc[df['mean'].idxmax()]
            
            interpretations.append(f"🏆 **Best Regression Model**: {best_model['model']} (RMSE: {best_model['mean']:.3f})")
            interpretations.append(f"📉 **Poorest Performance**: {worst_model['model']} (RMSE: {worst_model['mean']:.3f})")
            
            # Confidence interval analysis
            reliable_models = df[df['ci_upper'] - df['ci_lower'] < 0.5]['model'].tolist()
            if reliable_models:
                interpretations.append(f"🎯 **Most Reliable**: {', '.join(reliable_models[:3])} show narrow confidence intervals")
            
            # Performance categorization
            excellent_models = df[df['mean'] < 1.7]['model'].tolist()
            if excellent_models:
                interpretations.append(f"✅ **Excellent Performance**: {', '.join(excellent_models)} achieve RMSE < 1.7 (excellent for grade prediction)")
                
    except Exception as e:
        interpretations.append(f"⚠️ Could not interpret RMSE results: {e}")
    
    return interpretations

def test_interpretations():
    """Test all interpretation functions."""
    print("🧠 TESTING INTERPRETATION FUNCTIONS")
    print("=" * 50)
    
    print("\n📊 MODEL PERFORMANCE:")
    perf = interpret_model_performance()
    for p in perf:
        print(f"   {p}")
    
    print("\n📈 STATISTICAL ANALYSIS:")
    stats = interpret_statistical_tests()
    for s in stats:
        print(f"   {s}")
    
    print("\n📚 EXPLORATORY DATA ANALYSIS:")
    eda = interpret_eda_results()
    for e in eda:
        print(f"   {e}")
    
    print("\n📉 REGRESSION ANALYSIS:")
    rmse = interpret_rmse_results()
    for r in rmse:
        print(f"   {r}")

if __name__ == "__main__":
    test_interpretations()