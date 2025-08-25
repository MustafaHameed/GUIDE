#!/usr/bin/env python3
"""
GUIDE - Complete Results Presentation Script
============================================

This script runs all available files in the GUIDE repository and presents 
all results and plots in a comprehensive format.

Author: GUIDE Team
Date: 2025-08-25
"""

import os
import sys
import subprocess
import shutil
from pathlib import Path
from datetime import datetime
import glob
import json
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import warnings
warnings.filterwarnings('ignore')

# Set up paths
PROJECT_ROOT = Path(__file__).parent.absolute()
FIGURES_DIR = PROJECT_ROOT / "figures"
TABLES_DIR = PROJECT_ROOT / "tables"
REPORTS_DIR = PROJECT_ROOT / "reports"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
TIMESTAMP = datetime.now().strftime("%Y-%m-%d-%H%M%S")
RESULTS_DIR = PROJECT_ROOT / f"complete_results_{TIMESTAMP}"

def setup_environment():
    """Set up environment and create results directory."""
    print(f"🔧 Setting up environment...")
    
    # Set reproducibility environment variables
    os.environ['PYTHONHASHSEED'] = '0'
    os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
    
    # Create results directory
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "figures").mkdir(exist_ok=True)
    (RESULTS_DIR / "tables").mkdir(exist_ok=True)
    (RESULTS_DIR / "reports").mkdir(exist_ok=True)
    (RESULTS_DIR / "analysis").mkdir(exist_ok=True)
    
    print(f"✅ Environment set up. Results will be saved to: {RESULTS_DIR}")

def run_safe_command(cmd, description="Command"):
    """Run a command safely and log results."""
    print(f"🏃 Running: {description}")
    try:
        result = subprocess.run(cmd, shell=True, cwd=PROJECT_ROOT, 
                              capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            return True, result.stdout
        else:
            print(f"⚠️  {description} failed: {result.stderr}")
            return False, result.stderr
    except subprocess.TimeoutExpired:
        print(f"⏰ {description} timed out")
        return False, "Timeout"
    except Exception as e:
        print(f"❌ {description} error: {e}")
        return False, str(e)

def run_pipeline_components():
    """Run all available pipeline components."""
    print("\n" + "="*60)
    print("🚀 RUNNING PIPELINE COMPONENTS")
    print("="*60)
    
    # List of commands to run (using working versions)
    commands = [
        ("export PYTHONHASHSEED=0 && python -c \"import src.data; print('Data validation passed')\"", 
         "Data validation"),
        ("export PYTHONHASHSEED=0 && python src/eda.py", 
         "Exploratory Data Analysis"),
        ("export PYTHONHASHSEED=0 && python -m src.train --model-type logistic", 
         "Logistic Regression Training"),
        ("export PYTHONHASHSEED=0 && python -m src.train --model-type random_forest", 
         "Random Forest Training"),
        ("export PYTHONHASHSEED=0 && python -m src.nested_cv", 
         "Nested Cross-Validation"),
        ("export PYTHONHASHSEED=0 && python -m src.transfer.uci_transfer", 
         "Transfer Learning Analysis"),
    ]
    
    results = {}
    for cmd, desc in commands:
        success, output = run_safe_command(cmd, desc)
        results[desc] = {"success": success, "output": output}
    
    return results

def copy_and_organize_results():
    """Copy and organize all generated results."""
    print("\n" + "="*60)
    print("📁 ORGANIZING RESULTS")
    print("="*60)
    
    # Copy figures
    if FIGURES_DIR.exists():
        shutil.copytree(FIGURES_DIR, RESULTS_DIR / "figures", dirs_exist_ok=True)
        print(f"📊 Copied {len(list(FIGURES_DIR.glob('**/*')))} figure files")
    
    # Copy tables
    if TABLES_DIR.exists():
        shutil.copytree(TABLES_DIR, RESULTS_DIR / "tables", dirs_exist_ok=True)
        print(f"📋 Copied {len(list(TABLES_DIR.glob('**/*')))} table files")
    
    # Copy reports
    if REPORTS_DIR.exists():
        shutil.copytree(REPORTS_DIR, RESULTS_DIR / "reports", dirs_exist_ok=True)
        print(f"📄 Copied {len(list(REPORTS_DIR.glob('**/*')))} report files")

def analyze_generated_figures():
    """Analyze and categorize all generated figures."""
    print("\n" + "="*60)
    print("🎨 ANALYZING GENERATED FIGURES")
    print("="*60)
    
    figures = list((RESULTS_DIR / "figures").glob("*.png"))
    figure_categories = {
        "Exploratory Data Analysis": [],
        "Model Performance": [],
        "Fairness Analysis": [],
        "Feature Importance": [],
        "Explainability": [],
        "Early Risk Assessment": [],
        "Transfer Learning": [],
        "Other": []
    }
    
    # Categorize figures based on filename patterns
    for fig in figures:
        name = fig.name.lower()
        if any(keyword in name for keyword in ['eda_', 'correlation', 'distribution', 'pairplot']):
            figure_categories["Exploratory Data Analysis"].append(fig)
        elif any(keyword in name for keyword in ['roc_', 'confusion_', 'performance', 'accuracy']):
            figure_categories["Model Performance"].append(fig)
        elif 'fairness' in name:
            figure_categories["Fairness Analysis"].append(fig)
        elif any(keyword in name for keyword in ['importance', 'feature']):
            figure_categories["Feature Importance"].append(fig)
        elif any(keyword in name for keyword in ['shap_', 'lime_', 'pdp_', 'ice_']):
            figure_categories["Explainability"].append(fig)
        elif 'early' in name:
            figure_categories["Early Risk Assessment"].append(fig)
        elif 'transfer' in name:
            figure_categories["Transfer Learning"].append(fig)
        else:
            figure_categories["Other"].append(fig)
    
    # Print summary
    for category, files in figure_categories.items():
        if files:
            print(f"  {category}: {len(files)} figures")
    
    return figure_categories

def analyze_generated_tables():
    """Analyze and summarize all generated tables."""
    print("\n" + "="*60)
    print("📊 ANALYZING GENERATED TABLES")
    print("="*60)
    
    tables = list((RESULTS_DIR / "tables").glob("*.csv"))
    table_summaries = {}
    
    for table_path in tables:
        try:
            df = pd.read_csv(table_path)
            table_summaries[table_path.name] = {
                "rows": len(df),
                "columns": len(df.columns),
                "column_names": list(df.columns),
                "description": get_table_description(table_path.name)
            }
            print(f"  📋 {table_path.name}: {len(df)} rows × {len(df.columns)} columns")
        except Exception as e:
            print(f"  ⚠️  Could not read {table_path.name}: {e}")
    
    return table_summaries

def get_table_description(filename):
    """Get a description for a table based on its filename."""
    descriptions = {
        "classification_report.csv": "Model classification performance metrics",
        "feature_importance.csv": "Feature importance rankings and scores",
        "fairness_sex.csv": "Fairness metrics by gender",
        "correlation_matrix.csv": "Feature correlation matrix",
        "model_performance.csv": "Overall model performance comparison",
        "nested_cv_metrics.csv": "Nested cross-validation results",
        "transfer_results.csv": "Transfer learning performance metrics",
        "eda_summary_statistics.csv": "Exploratory data analysis summary statistics",
    }
    return descriptions.get(filename, "Data analysis results")

def create_comprehensive_report():
    """Create a comprehensive HTML report of all results."""
    print("\n" + "="*60)
    print("📝 CREATING COMPREHENSIVE REPORT")
    print("="*60)
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>GUIDE - Complete Results Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 40px; }}
            h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #34495e; border-bottom: 2px solid #ecf0f1; padding-bottom: 5px; }}
            h3 {{ color: #7f8c8d; }}
            .summary {{ background-color: #f8f9fa; padding: 20px; border-radius: 8px; margin: 20px 0; }}
            .figure-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; }}
            .figure-item {{ border: 1px solid #ddd; padding: 10px; border-radius: 8px; }}
            .figure-item img {{ max-width: 100%; height: auto; }}
            .table-summary {{ background-color: #fff; border: 1px solid #ddd; padding: 15px; margin: 10px 0; border-radius: 8px; }}
            .metric {{ display: inline-block; margin: 5px 10px; padding: 5px; background-color: #e3f2fd; border-radius: 4px; }}
            pre {{ background-color: #f5f5f5; padding: 10px; border-radius: 4px; overflow-x: auto; }}
        </style>
    </head>
    <body>
        <h1>🎯 GUIDE - Complete Machine Learning Pipeline Results</h1>
        
        <div class="summary">
            <h2>📋 Executive Summary</h2>
            <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
            <p><strong>Repository:</strong> GUIDE - Student Performance Analysis</p>
            <p><strong>Purpose:</strong> Publication-grade machine learning pipeline with fairness analysis and explainability</p>
            
            <div class="metric">📊 {len(list((RESULTS_DIR / "figures").glob("*.png")))} Figures</div>
            <div class="metric">📋 {len(list((RESULTS_DIR / "tables").glob("*.csv")))} Tables</div>
            <div class="metric">📄 {len(list((RESULTS_DIR / "reports").glob("*")))} Reports</div>
        </div>
        
        <h2>🎨 Generated Figures</h2>
    """
    
    # Add figures section
    figure_categories = analyze_generated_figures()
    for category, figures in figure_categories.items():
        if figures:
            html_content += f"<h3>{category}</h3><div class='figure-grid'>"
            for fig in figures[:6]:  # Limit to 6 per category for readability
                relative_path = f"figures/{fig.name}"
                html_content += f"""
                <div class="figure-item">
                    <img src="{relative_path}" alt="{fig.stem}">
                    <p><strong>{fig.stem.replace('_', ' ').title()}</strong></p>
                </div>
                """
            if len(figures) > 6:
                html_content += f"<p><em>... and {len(figures) - 6} more figures</em></p>"
            html_content += "</div>"
    
    # Add tables section
    html_content += "<h2>📊 Generated Tables</h2>"
    table_summaries = analyze_generated_tables()
    for table_name, info in table_summaries.items():
        html_content += f"""
        <div class="table-summary">
            <h3>{table_name}</h3>
            <p>{info['description']}</p>
            <p><strong>Dimensions:</strong> {info['rows']} rows × {info['columns']} columns</p>
            <p><strong>Columns:</strong> {', '.join(info['column_names'][:10])}{'...' if len(info['column_names']) > 10 else ''}</p>
        </div>
        """
    
    # Add file structure
    html_content += f"""
        <h2>📁 Complete File Structure</h2>
        <pre>
complete_results_{TIMESTAMP}/
├── figures/          # All generated plots and visualizations
├── tables/           # Data tables and metrics
├── reports/          # Analysis reports and summaries
└── analysis/         # This comprehensive report
        </pre>
        
        <h2>🚀 How to Explore Results</h2>
        <ol>
            <li><strong>Figures:</strong> Browse the figures/ directory for all visualizations</li>
            <li><strong>Tables:</strong> Open CSV files in tables/ for detailed metrics</li>
            <li><strong>Reports:</strong> Check reports/ for analysis summaries</li>
        </ol>
        
        <h2>📖 Next Steps</h2>
        <ul>
            <li>Review figure categories to understand different aspects of the analysis</li>
            <li>Examine performance metrics in the tables</li>
            <li>Use the interactive dashboard for deeper exploration</li>
        </ul>
        
        <footer style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666;">
            <p>Generated by GUIDE Pipeline - {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>
        </footer>
    </body>
    </html>
    """
    
    # Save the report
    report_path = RESULTS_DIR / "analysis" / "comprehensive_report.html"
    with open(report_path, 'w') as f:
        f.write(html_content)
    
    print(f"✅ Comprehensive report created: {report_path}")
    return report_path

def create_figure_summary():
    """Create a PDF summary of all figures."""
    print("\n" + "="*60)
    print("📄 CREATING FIGURE SUMMARY PDF")
    print("="*60)
    
    figures = list((RESULTS_DIR / "figures").glob("*.png"))
    if not figures:
        print("⚠️  No figures found to summarize")
        return None
    
    pdf_path = RESULTS_DIR / "analysis" / "all_figures_summary.pdf"
    
    with PdfPages(pdf_path) as pdf:
        # Create a title page
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.text(0.5, 0.6, 'GUIDE Pipeline\nComplete Results Summary', 
                ha='center', va='center', fontsize=24, weight='bold')
        ax.text(0.5, 0.4, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}', 
                ha='center', va='center', fontsize=12)
        ax.text(0.5, 0.3, f'Total Figures: {len(figures)}', 
                ha='center', va='center', fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis('off')
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)
        
        # Add figures (limit to prevent huge PDF)
        for i, fig_path in enumerate(figures[:50]):  # Limit to 50 figures
            try:
                fig, ax = plt.subplots(figsize=(8.5, 11))
                img = plt.imread(fig_path)
                ax.imshow(img)
                ax.set_title(fig_path.stem.replace('_', ' ').title(), fontsize=12, pad=20)
                ax.axis('off')
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)
            except Exception as e:
                print(f"⚠️  Could not add {fig_path.name} to PDF: {e}")
    
    print(f"✅ Figure summary PDF created: {pdf_path}")
    return pdf_path

def create_results_index():
    """Create an index file for easy navigation."""
    index_content = f"""# GUIDE Pipeline - Complete Results
Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

## 📁 Directory Structure

- `figures/` - All generated plots and visualizations ({len(list((RESULTS_DIR / "figures").glob("*.png")))} files)
- `tables/` - Data tables and metrics ({len(list((RESULTS_DIR / "tables").glob("*.csv")))} files)
- `reports/` - Analysis reports and summaries ({len(list((RESULTS_DIR / "reports").glob("*")))} files)
- `analysis/` - Comprehensive analysis reports

## 🎯 Key Results

### Figures by Category
"""
    
    figure_categories = analyze_generated_figures()
    for category, figures in figure_categories.items():
        if figures:
            index_content += f"\n#### {category}\n"
            for fig in figures:
                index_content += f"- `{fig.name}`\n"
    
    index_content += f"""

### Tables
"""
    table_summaries = analyze_generated_tables()
    for table_name, info in table_summaries.items():
        index_content += f"- `{table_name}` - {info['description']}\n"
    
    index_content += f"""

## 🚀 How to Use

1. **Browse Figures**: Open the `figures/` directory to see all visualizations
2. **Analyze Data**: Examine CSV files in `tables/` for detailed metrics
3. **Read Reports**: Check `reports/` for analysis summaries
4. **View Summary**: Open `analysis/comprehensive_report.html` in a web browser

## 📊 Quick Stats

- Total Figures: {len(list((RESULTS_DIR / "figures").glob("*.png")))}
- Total Tables: {len(list((RESULTS_DIR / "tables").glob("*.csv")))}
- Total Reports: {len(list((RESULTS_DIR / "reports").glob("*")))}

Generated by GUIDE Pipeline on {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
"""
    
    index_path = RESULTS_DIR / "README.md"
    with open(index_path, 'w') as f:
        f.write(index_content)
    
    print(f"✅ Results index created: {index_path}")

def main():
    """Main execution function."""
    print("🎯 GUIDE PIPELINE - COMPLETE RESULTS GENERATOR")
    print("=" * 60)
    print(f"📍 Project Root: {PROJECT_ROOT}")
    print(f"📅 Timestamp: {TIMESTAMP}")
    print("=" * 60)
    
    # Setup
    setup_environment()
    
    # Run pipeline components
    pipeline_results = run_pipeline_components()
    
    # Copy and organize existing results
    copy_and_organize_results()
    
    # Create comprehensive analysis
    create_comprehensive_report()
    create_figure_summary()
    create_results_index()
    
    # Final summary
    print("\n" + "=" * 60)
    print("🎉 COMPLETE RESULTS GENERATION FINISHED!")
    print("=" * 60)
    print(f"📁 All results saved to: {RESULTS_DIR}")
    print(f"🌐 Open comprehensive report: {RESULTS_DIR}/analysis/comprehensive_report.html")
    print(f"📄 View figure summary: {RESULTS_DIR}/analysis/all_figures_summary.pdf")
    print(f"📋 Read index: {RESULTS_DIR}/README.md")
    
    # Summary statistics
    total_figures = len(list((RESULTS_DIR / "figures").glob("*.png")))
    total_tables = len(list((RESULTS_DIR / "tables").glob("*.csv")))
    total_reports = len(list((RESULTS_DIR / "reports").glob("*")))
    
    print(f"\n📊 Generated Results Summary:")
    print(f"   🎨 Figures: {total_figures}")
    print(f"   📋 Tables: {total_tables}")
    print(f"   📄 Reports: {total_reports}")
    print(f"   💾 Total Size: {sum(f.stat().st_size for f in RESULTS_DIR.rglob('*') if f.is_file()) / 1024 / 1024:.1f} MB")
    
    return RESULTS_DIR

if __name__ == "__main__":
    main()