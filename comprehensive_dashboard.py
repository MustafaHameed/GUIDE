#!/usr/bin/env python3
"""
Comprehensive Results Dashboard - GUIDE Project
==============================================

Enhanced interactive Streamlit dashboard to explore ALL results from ALL datasets
in the repository with comprehensive filtering and visualization capabilities.

Run with: streamlit run comprehensive_dashboard.py
"""

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import json
import numpy as np
from typing import Dict, List

# Configure page
st.set_page_config(
    page_title="GUIDE - Comprehensive Results Dashboard",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2c3e50;
        text-align: center;
        margin-bottom: 2rem;
        border-bottom: 3px solid #3498db;
        padding-bottom: 1rem;
    }
    .dataset-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        margin: 0.5rem 0;
    }
    .result-type-badge {
        background-color: #e74c3c;
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 15px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
    .performance-excellent {
        background-color: #27ae60;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
    }
    .performance-good {
        background-color: #f39c12;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
    }
    .performance-poor {
        background-color: #e74c3c;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)


class ComprehensiveDashboard:
    """Main dashboard class for comprehensive results visualization."""
    
    def __init__(self):
        self.project_root = Path(__file__).parent.absolute()
        self.results_data = None
        self.master_table = None
        
    def load_latest_results(self):
        """Load the latest comprehensive results."""
        # Find latest comprehensive results directory
        comp_dirs = list(self.project_root.glob("comprehensive_all_results_*"))
        if not comp_dirs:
            st.error("❌ No comprehensive results found! Please run comprehensive_results_collector.py first.")
            return False
            
        latest_dir = max(comp_dirs, key=lambda x: x.stat().st_mtime)
        
        # Load results JSON
        results_file = latest_dir / "comprehensive_results.json"
        if results_file.exists():
            with open(results_file, 'r') as f:
                self.results_data = json.load(f)
        
        # Load master table
        master_file = latest_dir / "tables" / "master_results_all_datasets.csv"
        if master_file.exists():
            self.master_table = pd.read_csv(master_file)
            
        # Load summary
        summary_file = latest_dir / "results_summary.json"
        if summary_file.exists():
            with open(summary_file, 'r') as f:
                self.summary_data = json.load(f)
        
        st.success(f"✅ Loaded results from: {latest_dir.name}")
        return True
    
    def show_overview(self):
        """Show comprehensive overview of all results."""
        st.markdown('<h1 class="main-header">🎯 GUIDE - Comprehensive Results Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        if not self.load_latest_results():
            return
            
        # Key metrics row
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{self.summary_data['total_datasets']}</h3>
                <p>Total Datasets</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{len(self.master_table)}</h3>
                <p>Total Models</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <h3>{self.summary_data['total_result_types']}</h3>
                <p>Result Types</p>
            </div>
            """, unsafe_allow_html=True)
            
        with col4:
            best_accuracy = self.master_table['Accuracy'].max()
            st.markdown(f"""
            <div class="metric-card">
                <h3>{best_accuracy:.3f}</h3>
                <p>Best Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Dataset overview
        st.subheader("📊 Dataset Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Dataset coverage chart
            dataset_counts = self.master_table['Dataset'].value_counts()
            fig = px.bar(
                x=dataset_counts.index, 
                y=dataset_counts.values,
                title="Models per Dataset",
                labels={'x': 'Dataset', 'y': 'Number of Models'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Result type distribution
            result_type_counts = self.master_table['Result_Type'].value_counts()
            fig = px.pie(
                values=result_type_counts.values,
                names=result_type_counts.index,
                title="Distribution of Result Types"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Dataset details
        st.subheader("📂 Dataset Details")
        
        for dataset_name in self.summary_data['dataset_names']:
            dataset_info = self.results_data['datasets'][dataset_name]
            coverage = self.summary_data['dataset_coverage'].get(dataset_name, {'result_types': []})
            
            with st.expander(f"📁 {dataset_name}: {dataset_info['name']}"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Description:** {dataset_info['description']}")
                    st.write(f"**Type:** {dataset_info['type']}")
                    st.write(f"**Result Types:** {coverage['result_type_count']}")
                    
                with col2:
                    st.write("**Available Result Types:**")
                    for result_type in coverage['result_types']:
                        st.markdown(f'<span class="result-type-badge">{result_type}</span>', 
                                  unsafe_allow_html=True)
    
    def show_dataset_comparison(self):
        """Show detailed dataset comparison."""
        st.header("🔍 Dataset Comparison")
        
        if self.master_table is None:
            if not self.load_latest_results():
                return
        
        # Filter controls
        col1, col2, col3 = st.columns(3)
        
        with col1:
            selected_datasets = st.multiselect(
                "Select Datasets",
                options=self.master_table['Dataset'].unique(),
                default=self.master_table['Dataset'].unique()
            )
            
        with col2:
            selected_result_types = st.multiselect(
                "Select Result Types",
                options=self.master_table['Result_Type'].unique(),
                default=self.master_table['Result_Type'].unique()
            )
            
        with col3:
            metric_to_plot = st.selectbox(
                "Primary Metric",
                options=['Accuracy', 'ROC_AUC', 'F1_Score'],
                index=0
            )
        
        # Filter data
        filtered_df = self.master_table[
            (self.master_table['Dataset'].isin(selected_datasets)) &
            (self.master_table['Result_Type'].isin(selected_result_types))
        ].copy()
        
        if filtered_df.empty:
            st.warning("No data matches the selected filters.")
            return
        
        # Performance comparison
        st.subheader(f"📈 {metric_to_plot} Comparison")
        
        fig = px.box(
            filtered_df,
            x='Dataset',
            y=metric_to_plot,
            color='Result_Type',
            title=f"{metric_to_plot} Distribution by Dataset and Result Type"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Top performers
        st.subheader("🏆 Top Performers")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Best models overall
            top_models = filtered_df.nlargest(10, metric_to_plot)[
                ['Dataset', 'Model', metric_to_plot, 'Result_Type']
            ]
            st.write("**Top 10 Models Overall:**")
            st.dataframe(top_models, use_container_width=True)
            
        with col2:
            # Best per dataset
            st.write("**Best Model per Dataset:**")
            best_per_dataset = filtered_df.loc[
                filtered_df.groupby('Dataset')[metric_to_plot].idxmax()
            ][['Dataset', 'Model', metric_to_plot, 'Result_Type']]
            st.dataframe(best_per_dataset, use_container_width=True)
        
        # Detailed results table
        st.subheader("📊 Detailed Results")
        
        # Add performance categorization
        def categorize_performance(value):
            if pd.isna(value):
                return "N/A"
            elif value >= 0.9:
                return "Excellent (≥90%)"
            elif value >= 0.7:
                return "Good (70-90%)"
            else:
                return "Needs Improvement (<70%)"
        
        filtered_df['Performance_Category'] = filtered_df[metric_to_plot].apply(categorize_performance)
        
        st.dataframe(
            filtered_df.sort_values(metric_to_plot, ascending=False),
            use_container_width=True
        )
        
        # Download button
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Results",
            data=csv,
            file_name=f"filtered_results_{metric_to_plot.lower()}.csv",
            mime="text/csv"
        )
    
    def show_transfer_learning_analysis(self):
        """Show detailed transfer learning analysis."""
        st.header("🔄 Transfer Learning Analysis")
        
        if self.master_table is None:
            if not self.load_latest_results():
                return
        
        # Filter for transfer learning results
        transfer_df = self.master_table[
            self.master_table['Result_Type'] == 'transfer_learning'
        ].copy()
        
        if transfer_df.empty:
            st.warning("No transfer learning results found.")
            return
        
        # Extract direction information
        transfer_df['Source_Dataset'] = transfer_df['Dataset'].str.extract(r'Transfer: (.+)_to_(.+)')[0]
        transfer_df['Target_Dataset'] = transfer_df['Dataset'].str.extract(r'Transfer: (.+)_to_(.+)')[1]
        
        # Transfer matrix visualization
        st.subheader("🔄 Transfer Learning Matrix")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Create transfer matrix
            transfer_matrix = transfer_df.groupby(['Source_Dataset', 'Target_Dataset'])['Accuracy'].mean().unstack(fill_value=0)
            
            fig, ax = plt.subplots(figsize=(8, 6))
            sns.heatmap(transfer_matrix, annot=True, cmap='viridis', ax=ax)
            ax.set_title('Transfer Learning Accuracy Matrix')
            st.pyplot(fig)
            
        with col2:
            # Transfer performance by model
            model_perf = transfer_df.groupby('Model')['Accuracy'].agg(['mean', 'std']).round(3)
            st.write("**Transfer Performance by Model:**")
            st.dataframe(model_perf)
            
        # Direction analysis
        st.subheader("📊 Transfer Direction Analysis")
        
        # Group by direction
        direction_stats = transfer_df.groupby('Dataset').agg({
            'Accuracy': ['mean', 'max', 'min'],
            'ROC_AUC': ['mean', 'max', 'min'],
            'F1_Score': ['mean', 'max', 'min']
        }).round(3)
        
        st.dataframe(direction_stats, use_container_width=True)
        
        # Detailed transfer results
        st.subheader("📋 Detailed Transfer Results")
        st.dataframe(transfer_df, use_container_width=True)
    
    def show_model_performance_deep_dive(self):
        """Show detailed model performance analysis."""
        st.header("🔬 Model Performance Deep Dive")
        
        if self.master_table is None:
            if not self.load_latest_results():
                return
        
        # Model type analysis
        st.subheader("📊 Performance by Model Type")
        
        # Calculate statistics by result type
        perf_stats = self.master_table.groupby('Result_Type').agg({
            'Accuracy': ['count', 'mean', 'std', 'min', 'max'],
            'ROC_AUC': ['mean', 'std', 'min', 'max'],
            'F1_Score': ['mean', 'std', 'min', 'max']
        }).round(3)
        
        st.dataframe(perf_stats, use_container_width=True)
        
        # Correlation analysis
        st.subheader("🔗 Metric Correlations")
        
        numeric_cols = ['Accuracy', 'ROC_AUC', 'F1_Score']
        correlation_df = self.master_table[numeric_cols].corr()
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(correlation_df, annot=True, cmap='coolwarm', center=0, ax=ax)
        ax.set_title('Correlation between Performance Metrics')
        st.pyplot(fig)
        
        # Model comparison
        st.subheader("🏁 Model Comparison")
        
        # Interactive scatter plot
        fig = px.scatter(
            self.master_table,
            x='Accuracy',
            y='ROC_AUC',
            color='Result_Type',
            size='F1_Score',
            hover_data=['Model', 'Dataset'],
            title='Model Performance: Accuracy vs ROC-AUC'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance distribution
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.histogram(
                self.master_table,
                x='Accuracy',
                color='Result_Type',
                title='Accuracy Distribution by Result Type'
            )
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            fig = px.violin(
                self.master_table,
                x='Result_Type',
                y='Accuracy',
                title='Accuracy Distribution by Result Type'
            )
            fig.update_xaxis(tickangle=45)
            st.plotly_chart(fig, use_container_width=True)
    
    def show_raw_data_explorer(self):
        """Show raw data exploration interface."""
        st.header("🔍 Raw Data Explorer")
        
        if self.master_table is None:
            if not self.load_latest_results():
                return
        
        # Data overview
        st.subheader("📊 Data Overview")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Records", len(self.master_table))
        with col2:
            st.metric("Unique Models", self.master_table['Model'].nunique())
        with col3:
            st.metric("Missing Values", self.master_table.isnull().sum().sum())
        
        # Data quality summary
        st.subheader("🔍 Data Quality Summary")
        
        quality_df = pd.DataFrame({
            'Column': self.master_table.columns,
            'Non-Null Count': [self.master_table[col].count() for col in self.master_table.columns],
            'Null Count': [self.master_table[col].isnull().sum() for col in self.master_table.columns],
            'Data Type': [str(self.master_table[col].dtype) for col in self.master_table.columns],
            'Unique Values': [self.master_table[col].nunique() for col in self.master_table.columns]
        })
        
        st.dataframe(quality_df, use_container_width=True)
        
        # Interactive filtering
        st.subheader("🎛️ Interactive Data Filtering")
        
        # Filter controls
        col1, col2 = st.columns(2)
        
        with col1:
            # Accuracy range filter
            acc_min, acc_max = st.slider(
                "Accuracy Range",
                min_value=float(self.master_table['Accuracy'].min()),
                max_value=float(self.master_table['Accuracy'].max()),
                value=(float(self.master_table['Accuracy'].min()), 
                       float(self.master_table['Accuracy'].max()))
            )
            
        with col2:
            # Dataset filter
            dataset_filter = st.multiselect(
                "Filter by Dataset",
                options=self.master_table['Dataset'].unique(),
                default=self.master_table['Dataset'].unique()
            )
        
        # Apply filters
        filtered_data = self.master_table[
            (self.master_table['Accuracy'] >= acc_min) &
            (self.master_table['Accuracy'] <= acc_max) &
            (self.master_table['Dataset'].isin(dataset_filter))
        ]
        
        st.write(f"**Filtered Data: {len(filtered_data)} records**")
        st.dataframe(filtered_data, use_container_width=True)
        
        # Export options
        st.subheader("📥 Export Options")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            # Full data export
            full_csv = self.master_table.to_csv(index=False)
            st.download_button(
                "📥 Download Full Dataset",
                data=full_csv,
                file_name="complete_results_all_datasets.csv",
                mime="text/csv"
            )
            
        with col2:
            # Filtered data export
            filtered_csv = filtered_data.to_csv(index=False)
            st.download_button(
                "📥 Download Filtered Data",
                data=filtered_csv,
                file_name="filtered_results.csv",
                mime="text/csv"
            )
            
        with col3:
            # JSON export
            if self.results_data:
                json_str = json.dumps(self.results_data, indent=2)
                st.download_button(
                    "📥 Download Raw JSON",
                    data=json_str,
                    file_name="comprehensive_results.json",
                    mime="application/json"
                )


def main():
    """Main dashboard function."""
    dashboard = ComprehensiveDashboard()
    
    # Sidebar navigation
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.selectbox(
        "Choose a page:",
        ["📊 Overview", "🔍 Dataset Comparison", "🔄 Transfer Learning", 
         "🔬 Model Deep Dive", "🔍 Raw Data Explorer"]
    )
    
    # Add refresh button
    if st.sidebar.button("🔄 Refresh Data"):
        st.experimental_rerun()
    
    # Page routing
    if page == "📊 Overview":
        dashboard.show_overview()
    elif page == "🔍 Dataset Comparison":
        dashboard.show_dataset_comparison()
    elif page == "🔄 Transfer Learning":
        dashboard.show_transfer_learning_analysis()
    elif page == "🔬 Model Deep Dive":
        dashboard.show_model_performance_deep_dive()
    elif page == "🔍 Raw Data Explorer":
        dashboard.show_raw_data_explorer()
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**GUIDE Project**")
    st.sidebar.markdown("Comprehensive Results Dashboard")
    st.sidebar.markdown(f"*Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}*")


if __name__ == "__main__":
    main()