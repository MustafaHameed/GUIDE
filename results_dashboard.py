#!/usr/bin/env python3
"""
GUIDE Results Dashboard
======================

Interactive Streamlit dashboard to explore all generated results and plots.

Run with: streamlit run results_dashboard.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
import base64
from PIL import Image
import json

# Configure page
st.set_page_config(
    page_title="GUIDE - Complete Results Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
PROJECT_ROOT = Path(__file__).parent.absolute()
COMPLETE_RESULTS_DIR = list(PROJECT_ROOT.glob("complete_results_*"))[-1] if list(PROJECT_ROOT.glob("complete_results_*")) else None

def load_css():
    """Load custom CSS."""
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .figure-caption {
        font-size: 0.9rem;
        color: #666;
        text-align: center;
        margin-top: 0.5rem;
    }
    </style>
    """, unsafe_allow_html=True)

def main():
    """Main dashboard function."""
    load_css()
    
    # Header
    st.markdown('<h1 class="main-header">🎯 GUIDE - Complete Results Dashboard</h1>', unsafe_allow_html=True)
    
    if COMPLETE_RESULTS_DIR is None:
        st.error("No complete results directory found. Please run the results generation script first.")
        return
    
    st.info(f"📁 Displaying results from: `{COMPLETE_RESULTS_DIR.name}`")
    
    # Sidebar navigation
    st.sidebar.title("🧭 Navigation")
    
    sections = [
        "📊 Overview",
        "🎨 Figures Gallery",
        "📋 Data Tables",
        "📄 Reports & Analysis",
        "🔍 Detailed Exploration"
    ]
    
    selected_section = st.sidebar.selectbox("Select Section", sections)
    
    # Main content based on selection
    if selected_section == "📊 Overview":
        show_overview()
    elif selected_section == "🎨 Figures Gallery":
        show_figures_gallery()
    elif selected_section == "📋 Data Tables":
        show_data_tables()
    elif selected_section == "📄 Reports & Analysis":
        show_reports()
    elif selected_section == "🔍 Detailed Exploration":
        show_detailed_exploration()

def show_overview():
    """Show overview of all results."""
    st.header("📊 Results Overview")
    
    # Count files
    figures_dir = COMPLETE_RESULTS_DIR / "figures"
    tables_dir = COMPLETE_RESULTS_DIR / "tables"
    reports_dir = COMPLETE_RESULTS_DIR / "reports"
    
    total_figures = len(list(figures_dir.glob("*.png"))) if figures_dir.exists() else 0
    total_tables = len(list(tables_dir.glob("*.csv"))) if tables_dir.exists() else 0
    total_reports = len(list(reports_dir.rglob("*"))) if reports_dir.exists() else 0
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🎨 Total Figures", total_figures)
    with col2:
        st.metric("📋 Total Tables", total_tables)
    with col3:
        st.metric("📄 Total Reports", total_reports)
    with col4:
        total_size = sum(f.stat().st_size for f in COMPLETE_RESULTS_DIR.rglob('*') if f.is_file()) / 1024 / 1024
        st.metric("💾 Total Size", f"{total_size:.1f} MB")
    
    # Figure categories
    st.subheader("🎨 Figure Categories")
    
    if figures_dir.exists():
        figures = list(figures_dir.glob("*.png"))
        categories = categorize_figures(figures)
        
        # Display category counts
        category_data = []
        for category, files in categories.items():
            if files:
                category_data.append({"Category": category, "Count": len(files)})
        
        if category_data:
            df_categories = pd.DataFrame(category_data)
            
            # Bar chart of categories
            fig, ax = plt.subplots(figsize=(10, 6))
            bars = ax.bar(df_categories["Category"], df_categories["Count"], color='skyblue', alpha=0.7)
            ax.set_title("Figures by Category", fontsize=16, fontweight='bold')
            ax.set_xlabel("Category")
            ax.set_ylabel("Number of Figures")
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height,
                       f'{int(height)}', ha='center', va='bottom')
            
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            st.pyplot(fig)
            
            # Show category breakdown
            st.dataframe(df_categories, use_container_width=True)
    
    # Quick preview of key results
    st.subheader("🔍 Quick Preview")
    
    # Show a few sample figures
    if figures_dir.exists():
        key_figures = [
            "model_performance.png",
            "correlation_heatmap.png",
            "fairness_sex.png",
            "eda_grade_distribution_g3.png"
        ]
        
        cols = st.columns(2)
        for i, fig_name in enumerate(key_figures):
            fig_path = figures_dir / fig_name
            if fig_path.exists():
                with cols[i % 2]:
                    st.image(str(fig_path), caption=fig_name.replace('_', ' ').replace('.png', '').title())

def categorize_figures(figures):
    """Categorize figures based on filename patterns."""
    categories = {
        "Exploratory Data Analysis": [],
        "Model Performance": [],
        "Fairness Analysis": [],
        "Feature Importance": [],
        "Explainability": [],
        "Early Risk Assessment": [],
        "Transfer Learning": [],
        "Other": []
    }
    
    for fig in figures:
        name = fig.name.lower()
        if any(keyword in name for keyword in ['eda_', 'correlation', 'distribution', 'pairplot']):
            categories["Exploratory Data Analysis"].append(fig)
        elif any(keyword in name for keyword in ['roc_', 'confusion_', 'performance', 'accuracy']):
            categories["Model Performance"].append(fig)
        elif 'fairness' in name:
            categories["Fairness Analysis"].append(fig)
        elif any(keyword in name for keyword in ['importance', 'feature']):
            categories["Feature Importance"].append(fig)
        elif any(keyword in name for keyword in ['shap_', 'lime_', 'pdp_', 'ice_']):
            categories["Explainability"].append(fig)
        elif 'early' in name:
            categories["Early Risk Assessment"].append(fig)
        elif 'transfer' in name:
            categories["Transfer Learning"].append(fig)
        else:
            categories["Other"].append(fig)
    
    return categories

def show_figures_gallery():
    """Show figures gallery with filtering."""
    st.header("🎨 Figures Gallery")
    
    figures_dir = COMPLETE_RESULTS_DIR / "figures"
    if not figures_dir.exists():
        st.error("Figures directory not found.")
        return
    
    figures = list(figures_dir.glob("*.png"))
    if not figures:
        st.warning("No figures found in the directory.")
        return
    
    # Categorize figures
    categories = categorize_figures(figures)
    
    # Category filter
    selected_category = st.selectbox(
        "Select Category",
        ["All"] + [cat for cat, figs in categories.items() if figs]
    )
    
    # Filter figures based on category
    if selected_category == "All":
        display_figures = figures
    else:
        display_figures = categories[selected_category]
    
    # Search filter
    search_term = st.text_input("🔍 Search figures by name")
    if search_term:
        display_figures = [fig for fig in display_figures if search_term.lower() in fig.name.lower()]
    
    st.write(f"Showing {len(display_figures)} figures")
    
    # Display figures in grid
    cols_per_row = 3
    for i in range(0, len(display_figures), cols_per_row):
        cols = st.columns(cols_per_row)
        for j, fig_path in enumerate(display_figures[i:i+cols_per_row]):
            with cols[j]:
                try:
                    image = Image.open(fig_path)
                    st.image(image, caption=fig_path.stem.replace('_', ' ').title(), use_column_width=True)
                    
                    # Download button
                    with open(fig_path, "rb") as file:
                        btn = st.download_button(
                            label="📥 Download",
                            data=file.read(),
                            file_name=fig_path.name,
                            mime="image/png",
                            key=f"download_{fig_path.name}"
                        )
                except Exception as e:
                    st.error(f"Could not load {fig_path.name}: {e}")

def show_data_tables():
    """Show data tables with filtering and analysis."""
    st.header("📋 Data Tables")
    
    tables_dir = COMPLETE_RESULTS_DIR / "tables"
    if not tables_dir.exists():
        st.error("Tables directory not found.")
        return
    
    csv_files = list(tables_dir.glob("*.csv"))
    if not csv_files:
        st.warning("No CSV files found in the tables directory.")
        return
    
    # Table selector
    selected_table = st.selectbox(
        "Select Table",
        [f.name for f in csv_files]
    )
    
    table_path = tables_dir / selected_table
    
    try:
        df = pd.read_csv(table_path)
        
        # Table info
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Rows", len(df))
        with col2:
            st.metric("Columns", len(df.columns))
        with col3:
            st.metric("Memory Usage", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
        
        # Display table
        st.subheader("📊 Data Preview")
        st.dataframe(df, use_container_width=True)
        
        # Basic statistics for numeric columns
        numeric_cols = df.select_dtypes(include=['number']).columns
        if not numeric_cols.empty:
            st.subheader("📈 Numeric Column Statistics")
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
            
            # Simple visualizations
            if len(numeric_cols) > 0:
                st.subheader("📊 Quick Visualizations")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histogram of first numeric column
                    if len(numeric_cols) > 0:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        df[numeric_cols[0]].hist(bins=20, ax=ax, alpha=0.7)
                        ax.set_title(f"Distribution of {numeric_cols[0]}")
                        ax.set_xlabel(numeric_cols[0])
                        ax.set_ylabel("Frequency")
                        st.pyplot(fig)
                
                with col2:
                    # Correlation heatmap if multiple numeric columns
                    if len(numeric_cols) > 1:
                        fig, ax = plt.subplots(figsize=(8, 6))
                        correlation_matrix = df[numeric_cols].corr()
                        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
                        ax.set_title("Correlation Matrix")
                        st.pyplot(fig)
        
        # Download button
        csv_data = df.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv_data,
            file_name=selected_table,
            mime="text/csv"
        )
        
    except Exception as e:
        st.error(f"Could not load table: {e}")

def show_reports():
    """Show reports and analysis files."""
    st.header("📄 Reports & Analysis")
    
    reports_dir = COMPLETE_RESULTS_DIR / "reports"
    analysis_dir = COMPLETE_RESULTS_DIR / "analysis"
    
    # Show comprehensive report link
    if analysis_dir.exists():
        comprehensive_report = analysis_dir / "comprehensive_report.html"
        if comprehensive_report.exists():
            st.success("🌐 Comprehensive HTML report is available!")
            st.info(f"Open this file in your browser: `{comprehensive_report}`")
    
    # Show reports directory contents
    if reports_dir.exists():
        st.subheader("📁 Available Reports")
        
        report_files = list(reports_dir.rglob("*"))
        for report_file in report_files:
            if report_file.is_file():
                st.write(f"📄 {report_file.relative_to(reports_dir)}")
                
                # Show content for text files
                if report_file.suffix in ['.md', '.txt', '.csv']:
                    with st.expander(f"View {report_file.name}"):
                        try:
                            content = report_file.read_text()
                            if report_file.suffix == '.csv':
                                df = pd.read_csv(report_file)
                                st.dataframe(df)
                            else:
                                st.text(content)
                        except Exception as e:
                            st.error(f"Could not read file: {e}")

def show_detailed_exploration():
    """Show detailed exploration tools."""
    st.header("🔍 Detailed Exploration")
    
    # File browser
    st.subheader("📁 File Browser")
    
    def show_directory_tree(path, prefix=""):
        """Show directory tree structure."""
        if path.is_dir():
            items = sorted(path.iterdir(), key=lambda x: (x.is_file(), x.name))
            for item in items:
                if item.is_dir():
                    st.write(f"{prefix}📁 {item.name}/")
                    if len(prefix) < 4:  # Limit depth
                        show_directory_tree(item, prefix + "  ")
                else:
                    st.write(f"{prefix}📄 {item.name}")
    
    show_directory_tree(COMPLETE_RESULTS_DIR)
    
    # Search functionality
    st.subheader("🔍 Search Files")
    search_term = st.text_input("Search for files by name")
    
    if search_term:
        matching_files = []
        for file_path in COMPLETE_RESULTS_DIR.rglob("*"):
            if search_term.lower() in file_path.name.lower():
                matching_files.append(file_path)
        
        if matching_files:
            st.write(f"Found {len(matching_files)} matching files:")
            for file_path in matching_files[:20]:  # Limit results
                relative_path = file_path.relative_to(COMPLETE_RESULTS_DIR)
                st.write(f"📄 {relative_path}")
        else:
            st.write("No matching files found.")

if __name__ == "__main__":
    main()