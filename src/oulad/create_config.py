#!/usr/bin/env python3
"""
OULAD Configuration Management Tool

Creates and manages preprocessing configurations for different OULAD use cases.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any
import sys

# Add the project root to Python path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

try:
    from src.logging_config import setup_logging
except ImportError:
    # Fallback if logging_config not available
    def setup_logging():
        logging.basicConfig(level=logging.INFO)

logger = logging.getLogger(__name__)

TEMPLATE_CONFIGS = {
    "default": {
        "name": "OULAD Default Configuration",
        "description": "Standard preprocessing for general ML tasks",
        "feature_engineering": {
            "vle_features": {"enabled": True},
            "assessment_features": {"enabled": True}
        },
        "sensitive_attributes": {"enabled": True},
        "output": {
            "main_dataset": "data/oulad/processed/oulad_ml.parquet"
        }
    },
    
    "early_prediction": {
        "name": "OULAD Early Prediction Configuration", 
        "description": "Optimized for early risk prediction (first 4 weeks)",
        "temporal_constraints": {
            "enabled": True,
            "cutoff_week": 4,
            "exclude_future_data": True
        },
        "feature_engineering": {
            "vle_features": {
                "enabled": True,
                "temporal_windows": {"early_weeks": 4, "late_weeks": 0}
            }
        },
        "output": {
            "main_dataset": "data/oulad/processed/oulad_early_ml.parquet"
        }
    },
    
    "fairness": {
        "name": "OULAD Fairness Analysis Configuration",
        "description": "Comprehensive fairness and bias analysis",
        "sensitive_attributes": {
            "enabled": True,
            "detailed_statistics": True,
            "intersection_features": ["sex_x_age", "sex_x_education"]
        },
        "fairness_analysis": {
            "enabled": True,
            "compute_parity_metrics": True,
            "analyze_intersectionality": True
        },
        "output": {
            "main_dataset": "data/oulad/processed/oulad_fairness_ml.parquet",
            "bias_report": "data/oulad/processed/bias_report.json"
        }
    },
    
    "minimal": {
        "name": "OULAD Minimal Configuration",
        "description": "Basic preprocessing with minimal features",
        "feature_engineering": {
            "vle_features": {"enabled": False},
            "assessment_features": {"enabled": True}
        },
        "sensitive_attributes": {"enabled": False},
        "output": {
            "main_dataset": "data/oulad/processed/oulad_minimal_ml.parquet"
        }
    }
}

def create_config_from_template(template: str, output_path: Path, 
                              custom_params: Dict[str, Any] = None) -> None:
    """Create a configuration file from a template.
    
    Args:
        template: Template name
        output_path: Output file path
        custom_params: Custom parameters to override defaults
    """
    if template not in TEMPLATE_CONFIGS:
        raise ValueError(f"Unknown template: {template}. Available: {list(TEMPLATE_CONFIGS.keys())}")
    
    config = TEMPLATE_CONFIGS[template].copy()
    
    # Apply custom parameters if provided
    if custom_params:
        config = deep_merge(config, custom_params)
    
    # Ensure output directory exists
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Write configuration
    with open(output_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Created configuration: {output_path}")
    logger.info(f"Template: {template}")
    logger.info(f"Description: {config.get('description', 'No description')}")

def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Deep merge two dictionaries."""
    result = base.copy()
    
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    
    return result

def list_templates() -> None:
    """List available configuration templates."""
    print("Available OULAD Configuration Templates:")
    print("=" * 50)
    
    for name, config in TEMPLATE_CONFIGS.items():
        print(f"\n{name}:")
        print(f"  Description: {config.get('description', 'No description')}")
        
        # Show key features
        features = []
        if config.get('feature_engineering', {}).get('vle_features', {}).get('enabled'):
            features.append("VLE features")
        if config.get('feature_engineering', {}).get('assessment_features', {}).get('enabled'):
            features.append("Assessment features")
        if config.get('sensitive_attributes', {}).get('enabled'):
            features.append("Fairness analysis")
        if config.get('temporal_constraints', {}).get('enabled'):
            features.append("Temporal constraints")
            
        if features:
            print(f"  Features: {', '.join(features)}")

def validate_config(config_path: Path) -> bool:
    """Validate a configuration file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        True if valid, False otherwise
    """
    try:
        with open(config_path) as f:
            config = json.load(f)
        
        # Basic validation
        required_fields = ['name', 'output']
        for field in required_fields:
            if field not in config:
                logger.error(f"Missing required field: {field}")
                return False
        
        # Validate output paths
        output_config = config.get('output', {})
        if 'main_dataset' not in output_config:
            logger.error("Missing main_dataset in output configuration")
            return False
        
        logger.info("Configuration validation passed")
        return True
        
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON: {e}")
        return False
    except Exception as e:
        logger.error(f"Validation error: {e}")
        return False

def main():
    """Main CLI interface."""
    parser = argparse.ArgumentParser(
        description="OULAD Configuration Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List available templates
  python src/oulad/create_config.py --list-templates
  
  # Create default configuration
  python src/oulad/create_config.py --template default --output configs/my_config.json
  
  # Create early prediction configuration  
  python src/oulad/create_config.py --template early_prediction --output configs/early.json
  
  # Validate existing configuration
  python src/oulad/create_config.py --validate configs/my_config.json
        """
    )
    
    parser.add_argument(
        '--template', 
        choices=list(TEMPLATE_CONFIGS.keys()),
        help='Configuration template to use'
    )
    
    parser.add_argument(
        '--output', 
        type=Path,
        help='Output configuration file path'
    )
    
    parser.add_argument(
        '--list-templates',
        action='store_true', 
        help='List available configuration templates'
    )
    
    parser.add_argument(
        '--validate',
        type=Path,
        help='Validate an existing configuration file'
    )
    
    parser.add_argument(
        '--custom-params',
        type=str,
        help='JSON string with custom parameters to override template defaults'
    )
    
    args = parser.parse_args()
    
    if args.list_templates:
        list_templates()
        return
    
    if args.validate:
        if validate_config(args.validate):
            print(f"✓ Configuration {args.validate} is valid")
        else:
            print(f"✗ Configuration {args.validate} has errors")
            sys.exit(1)
        return
    
    if not args.template or not args.output:
        parser.error("Both --template and --output are required for creating configurations")
    
    # Parse custom parameters if provided
    custom_params = {}
    if args.custom_params:
        try:
            custom_params = json.loads(args.custom_params)
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON in custom parameters: {e}")
            sys.exit(1)
    
    try:
        create_config_from_template(args.template, args.output, custom_params)
        print(f"✓ Configuration created: {args.output}")
        
    except Exception as e:
        logger.error(f"Failed to create configuration: {e}")
        sys.exit(1)

if __name__ == "__main__":
    setup_logging()
    main()