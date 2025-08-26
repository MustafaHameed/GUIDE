"""Transfer learning module for domain adaptation."""

# Core transfer learning components
from .feature_bridge import FeatureBridge, create_feature_bridge
from .coral import CORALTransformer, apply_coral_alignment
from .label_shift import LabelShiftCorrector, apply_label_shift_correction
from .weights import ImportanceWeighter, compute_importance_weights
from .mmd import MMDTransformer, compute_mmd, apply_mmd_alignment
from .tent import TENTAdapter, apply_tent_adaptation
from .calibration import (
    CalibratedTransferClassifier, OptimalThresholdFinder,
    expected_calibration_error, evaluate_calibration_transfer
)
from .ablation_runner import TransferLearningAblation

# Legacy modules (with optional imports)
from .diagnostics import generate_shift_report, create_shift_report_summary
from .advanced_transfer import advanced_transfer_learning
from .improved_transfer import improved_feature_alignment, create_ensemble_classifier
from .uci_transfer import run_bidirectional_transfer

# Optional torch-dependent modules
try:
    from .dann import create_dann_classifier
    _has_torch = True
except ImportError:
    _has_torch = False
    def create_dann_classifier(*args, **kwargs):
        raise ImportError("PyTorch not available. Install torch to use DANN.")

__all__ = [
    # Core unified components
    'FeatureBridge', 'create_feature_bridge',
    
    # Domain adaptation
    'CORALTransformer', 'apply_coral_alignment',
    'MMDTransformer', 'compute_mmd', 'apply_mmd_alignment',
    'ImportanceWeighter', 'compute_importance_weights',
    'LabelShiftCorrector', 'apply_label_shift_correction',
    
    # Test-time adaptation
    'TENTAdapter', 'apply_tent_adaptation',
    
    # Calibration and optimization
    'CalibratedTransferClassifier', 'OptimalThresholdFinder',
    'expected_calibration_error', 'evaluate_calibration_transfer',
    
    # Ablation studies
    'TransferLearningAblation',
    
    # Legacy components
    'generate_shift_report', 'create_shift_report_summary',
    'create_dann_classifier', 'advanced_transfer_learning',
    'improved_feature_alignment', 'create_ensemble_classifier',
    'run_bidirectional_transfer'
]
