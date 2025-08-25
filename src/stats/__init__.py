"""Statistical analysis utilities for model evaluation and comparison."""

from .mcnemar import (
    mcnemar_test,
    paired_t_test,
    bootstrap_difference_test,
    compare_multiple_models,
    save_comparison_results
)

from .anova_tukey import (
    one_way_anova,
    tukey_hsd_test,
    repeated_measures_anova,
    friedman_test,
    nemenyi_post_hoc_test,
    save_anova_results
)

from .calibration import (
    expected_calibration_error,
    maximum_calibration_error,
    brier_score_decomposition,
    reliability_diagram,
    confidence_histogram,
    calibrate_model_predictions,
    threshold_selection_analysis,
    comprehensive_calibration_analysis
)

from .bootstrap import (
    bootstrap_metric,
    bootstrap_confidence_interval,
    bootstrap_hypothesis_test,
    bootstrap_model_comparison,
    permutation_test
)

__all__ = [
    # McNemar and statistical tests
    'mcnemar_test',
    'paired_t_test', 
    'bootstrap_difference_test',
    'compare_multiple_models',
    'save_comparison_results',
    
    # ANOVA and post-hoc tests
    'one_way_anova',
    'tukey_hsd_test',
    'repeated_measures_anova',
    'friedman_test',
    'nemenyi_post_hoc_test',
    'save_anova_results',
    
    # Calibration analysis
    'expected_calibration_error',
    'maximum_calibration_error',
    'brier_score_decomposition',
    'reliability_diagram',
    'confidence_histogram',
    'calibrate_model_predictions',
    'threshold_selection_analysis',
    'comprehensive_calibration_analysis',
    
    # Bootstrap methods
    'bootstrap_metric',
    'bootstrap_confidence_interval',
    'bootstrap_hypothesis_test',
    'bootstrap_model_comparison',
    'permutation_test'
]