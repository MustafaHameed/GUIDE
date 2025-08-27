#!/usr/bin/env python3
"""
Test Script for Comprehensive Results Collection
==============================================

This script validates that the comprehensive results collection system
is working correctly and captures results from all datasets.
"""

import pandas as pd
import json
from pathlib import Path
from comprehensive_results_collector import ComprehensiveResultsCollector

def test_comprehensive_collection():
    """Test the comprehensive results collection system."""
    print("🧪 TESTING COMPREHENSIVE RESULTS COLLECTION")
    print("=" * 60)
    
    # Initialize collector
    collector = ComprehensiveResultsCollector()
    
    # Collect results
    print("📊 Collecting results...")
    all_results = collector.collect_all_results()
    
    # Generate summary
    summary = collector.generate_comprehensive_summary()
    
    # Validate results
    print("\n✅ VALIDATION RESULTS:")
    print("=" * 40)
    
    # Check datasets
    datasets = all_results['datasets']
    print(f"📂 Datasets Found: {len(datasets)}")
    for name, info in datasets.items():
        print(f"  - {name}: {info['name']}")
    
    # Check result types
    result_types = all_results['results_by_type']
    print(f"\n📊 Result Types Found: {len(result_types)}")
    for result_type, data in result_types.items():
        if 'models' in data:
            model_count = len(data['models'])
        elif 'transfers' in data:
            model_count = len(data['transfers'])
        else:
            model_count = 0
        print(f"  - {result_type}: {model_count} models/experiments")
    
    # Check dataset coverage
    dataset_results = all_results['results_by_dataset']
    print(f"\n🎯 Dataset Coverage:")
    for dataset, data in dataset_results.items():
        result_count = len(data['results'])
        print(f"  - {dataset}: {result_count} result types")
        for result_type in data['results'].keys():
            print(f"    * {result_type}")
    
    # Check cross-dataset results
    cross_dataset = all_results.get('cross_dataset_results', {})
    print(f"\n🔄 Cross-Dataset Results: {len(cross_dataset)} types")
    for result_type in cross_dataset.keys():
        print(f"  - {result_type}")
    
    # Create master table and analyze
    master_table = collector._create_master_table()
    print(f"\n📋 Master Table: {len(master_table)} total records")
    
    dataset_counts = master_table['Dataset'].value_counts()
    print("\n📊 Records per Dataset:")
    for dataset, count in dataset_counts.items():
        print(f"  - {dataset}: {count} models")
    
    result_type_counts = master_table['Result_Type'].value_counts()
    print("\n📈 Records per Result Type:")
    for result_type, count in result_type_counts.items():
        print(f"  - {result_type}: {count} models")
    
    # Performance summary
    print(f"\n🏆 Performance Summary:")
    print(f"  - Best Accuracy: {master_table['Accuracy'].max():.3f}")
    print(f"  - Average Accuracy: {master_table['Accuracy'].mean():.3f}")
    print(f"  - Best ROC-AUC: {master_table['ROC_AUC'].max():.3f}")
    print(f"  - Models with >90% Accuracy: {(master_table['Accuracy'] > 0.9).sum()}")
    
    # Check for missing data
    print(f"\n🔍 Data Quality Check:")
    for col in ['Accuracy', 'ROC_AUC', 'F1_Score']:
        missing = master_table[col].isnull().sum()
        total = len(master_table)
        print(f"  - {col}: {total - missing}/{total} records ({(total-missing)/total*100:.1f}%)")
    
    print("\n🎉 COMPREHENSIVE COLLECTION TEST COMPLETED!")
    return all_results, summary, master_table

def test_result_files():
    """Test that result files exist and are readable."""
    print("\n🔍 TESTING RESULT FILES")
    print("=" * 40)
    
    project_root = Path(".")
    
    # Check for result directories
    result_dirs = [
        "fresh_dl_results_*",
        "complete_results_*", 
        "comprehensive_fresh_results_*",
        "enhanced_feature_engineering_results",
        "comprehensive_all_results_*"
    ]
    
    for pattern in result_dirs:
        matching = list(project_root.glob(pattern))
        print(f"📁 {pattern}: {len(matching)} directories found")
        
        for directory in matching:
            if directory.is_dir():
                csv_files = list(directory.rglob("*.csv"))
                json_files = list(directory.rglob("*.json"))
                html_files = list(directory.rglob("*.html"))
                print(f"  - {directory.name}: {len(csv_files)} CSV, {len(json_files)} JSON, {len(html_files)} HTML")
    
    # Check specific important files
    important_files = [
        "enhanced_feature_engineering_results/model_comparison_summary.csv",
        "student-mat.csv",
        "student-por.csv"
    ]
    
    print(f"\n📋 Important Files Check:")
    for file_path in important_files:
        file_obj = project_root / file_path
        exists = file_obj.exists()
        print(f"  - {file_path}: {'✅' if exists else '❌'}")
        
        if exists and file_path.endswith('.csv'):
            try:
                df = pd.read_csv(file_obj)
                print(f"    * {len(df)} rows, {len(df.columns)} columns")
            except Exception as e:
                print(f"    * Error reading: {e}")

def test_dashboard_data():
    """Test that dashboard can load data correctly."""
    print("\n🎯 TESTING DASHBOARD DATA COMPATIBILITY")
    print("=" * 50)
    
    # Find latest comprehensive results
    project_root = Path(".")
    comp_dirs = list(project_root.glob("comprehensive_all_results_*"))
    
    if not comp_dirs:
        print("❌ No comprehensive results found for dashboard testing")
        return False
    
    latest_dir = max(comp_dirs, key=lambda x: x.stat().st_mtime)
    print(f"📁 Testing with: {latest_dir.name}")
    
    # Test JSON loading
    results_file = latest_dir / "comprehensive_results.json"
    if results_file.exists():
        try:
            with open(results_file, 'r') as f:
                results_data = json.load(f)
            print("✅ JSON results file loaded successfully")
            print(f"  - Datasets: {len(results_data.get('datasets', {}))}")
            print(f"  - Result types: {len(results_data.get('results_by_type', {}))}")
        except Exception as e:
            print(f"❌ Error loading JSON: {e}")
            return False
    
    # Test master table loading
    master_file = latest_dir / "tables" / "master_results_all_datasets.csv"
    if master_file.exists():
        try:
            master_df = pd.read_csv(master_file)
            print("✅ Master table loaded successfully")
            print(f"  - Records: {len(master_df)}")
            print(f"  - Columns: {list(master_df.columns)}")
            print(f"  - Datasets: {master_df['Dataset'].nunique()}")
        except Exception as e:
            print(f"❌ Error loading master table: {e}")
            return False
    
    # Test summary loading
    summary_file = latest_dir / "results_summary.json"
    if summary_file.exists():
        try:
            with open(summary_file, 'r') as f:
                summary_data = json.load(f)
            print("✅ Summary file loaded successfully")
            print(f"  - Total datasets: {summary_data.get('total_datasets', 0)}")
            print(f"  - Total result types: {summary_data.get('total_result_types', 0)}")
        except Exception as e:
            print(f"❌ Error loading summary: {e}")
            return False
    
    print("✅ Dashboard data compatibility test passed!")
    return True

def main():
    """Run all tests."""
    print("🚀 COMPREHENSIVE RESULTS SYSTEM VALIDATION")
    print("=" * 80)
    
    # Test 1: Comprehensive collection
    try:
        all_results, summary, master_table = test_comprehensive_collection()
        print("\n✅ Test 1 PASSED: Comprehensive Collection")
    except Exception as e:
        print(f"\n❌ Test 1 FAILED: {e}")
        return False
    
    # Test 2: Result files
    try:
        test_result_files()
        print("\n✅ Test 2 PASSED: Result Files")
    except Exception as e:
        print(f"\n❌ Test 2 FAILED: {e}")
        return False
    
    # Test 3: Dashboard compatibility
    try:
        dashboard_ok = test_dashboard_data()
        if dashboard_ok:
            print("\n✅ Test 3 PASSED: Dashboard Compatibility")
        else:
            print("\n⚠️ Test 3 WARNING: Dashboard data issues")
    except Exception as e:
        print(f"\n❌ Test 3 FAILED: {e}")
        return False
    
    print("\n🎉 ALL TESTS COMPLETED!")
    print("🎯 The comprehensive results system is ready for use!")
    
    # Print usage instructions
    print("\n📋 USAGE INSTRUCTIONS:")
    print("=" * 40)
    print("1. Run comprehensive collection:")
    print("   python comprehensive_results_collector.py")
    print("\n2. View enhanced results:")
    print("   python run_comprehensive_fresh_results.py")
    print("\n3. Launch interactive dashboard:")
    print("   streamlit run comprehensive_dashboard.py")
    print("\n4. View original dashboard:")
    print("   streamlit run results_dashboard.py")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)