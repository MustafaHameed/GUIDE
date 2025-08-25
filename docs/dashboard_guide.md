# Dashboard Guide

Interactive web-based analysis and visualization tools for the GUIDE student performance pipeline.

## Overview

The GUIDE project includes three specialized Streamlit dashboards:

1. **Student Dashboard** (`dashboard_student.py`) - Individual student analysis and predictions
2. **Teacher Dashboard** (`dashboard_teacher.py`) - Class-level insights and at-risk student identification  
3. **General Dashboard** (`dashboard.py`) - Comprehensive analysis and model exploration

## Getting Started

### Prerequisites

Ensure you have trained models and generated analysis outputs:

```bash
# Run basic training and analysis
make train fairness explain

# Or run the full pipeline
make all
```

### Launching Dashboards

```bash
# Student-focused interface
streamlit run dashboard_student.py

# Teacher/administrator interface
streamlit run dashboard_teacher.py

# General analysis interface
streamlit run dashboard.py
```

Each dashboard will open in your web browser at `http://localhost:8501`.

## Student Dashboard

**Purpose**: Individual student risk assessment and personalized insights.

### Features

#### Student Selection
- **Student ID Input**: Enter any student ID to load their profile
- **Random Student**: Click to analyze a randomly selected student
- **Search by Attributes**: Filter students by characteristics (gender, school, etc.)

#### Risk Assessment
- **Risk Score**: Probability of academic failure (0-100%)
- **Risk Level**: Color-coded classification (Low/Medium/High)
- **Early Warning**: Predictions based on partial grade information
- **Confidence Intervals**: Uncertainty bounds for predictions

#### Academic Progress
- **Grade Trajectory**: Interactive line chart showing G1 → G2 → G3 progression
- **Peer Comparison**: Student's position relative to class averages
- **Subject Performance**: Mathematics vs. Portuguese comparison (if available)

#### Personalized Explanations
- **Top Risk Factors**: Features most influencing this student's risk
- **Protective Factors**: Positive influences on academic success
- **What-If Analysis**: Sliders to explore counterfactual scenarios
- **LIME Explanations**: Local explanations for individual predictions

#### Recommendations
- **Intervention Strategies**: Targeted recommendations based on risk factors
- **Study Suggestions**: Personalized study time and support recommendations
- **Family Engagement**: Suggested family involvement strategies

### Usage Examples

#### Scenario 1: At-Risk Student Analysis
```python
# Example workflow through dashboard:
# 1. Enter student ID: 15
# 2. Review risk score: 75% (High Risk)
# 3. Check top risk factors: low study time, high absences
# 4. Use what-if sliders: increase study time from 1 to 3
# 5. See updated prediction: 45% (Medium Risk)
# 6. Export recommendation report
```

#### Scenario 2: Progress Monitoring
```python
# Track student improvement:
# 1. Select student with partial grades (G1, G2 available)
# 2. Compare early vs. current predictions
# 3. Identify intervention effectiveness
# 4. Plan next steps based on trajectory
```

## Teacher Dashboard

**Purpose**: Class management, at-risk student identification, and resource allocation.

### Features

#### Class Overview
- **Enrollment Statistics**: Total students, pass rate, grade distributions
- **Summary Metrics**: Average performance by demographic groups
- **Trend Analysis**: Performance changes over time
- **Class Composition**: Breakdown by gender, school, background factors

#### At-Risk Student Identification
- **Risk List**: Students ranked by failure probability
- **Intervention Priority**: Urgency scores for targeted support
- **Early Warning System**: Students flagged based on partial grades
- **Risk Factors Summary**: Common issues across at-risk students

#### Fairness Analysis
- **Demographic Parity**: Performance equity across groups
- **Bias Detection**: Algorithmic fairness metrics
- **Group Comparisons**: Side-by-side analysis of subpopulations
- **Intervention Impact**: Effect of support on different groups

#### Resource Planning
- **Support Allocation**: Recommended distribution of tutoring/support
- **Intervention Tracking**: Monitor effectiveness of previous actions
- **Parent Communication**: Priority list for family engagement
- **Professional Development**: Teacher training recommendations

### Dashboard Sections

#### Overview Tab
- Class statistics and key metrics
- Performance distribution visualizations
- Quick access to critical alerts

#### At-Risk Students Tab
- Sortable table of high-risk students
- Individual risk breakdowns
- Bulk intervention planning tools

#### Fairness Analysis Tab
- Bias metrics across demographic groups
- Fairness visualizations and explanations
- Equity monitoring over time

### Usage Examples

#### Scenario 1: Weekly Risk Review
```python
# Teacher workflow:
# 1. Open "At-Risk Students" tab
# 2. Sort by risk score (highest first)
# 3. Review top 10 students
# 4. Check risk factors for each
# 5. Plan interventions for upcoming week
# 6. Export priority contact list
```

#### Scenario 2: Equity Monitoring
```python
# Fairness analysis:
# 1. Navigate to "Fairness Analysis" tab
# 2. Select demographic group: "gender"
# 3. Review performance gaps
# 4. Identify intervention needs
# 5. Track improvement over time
```

## General Dashboard

**Purpose**: Comprehensive model analysis, research insights, and system overview.

### Features

#### Model Performance
- **Performance Metrics**: Accuracy, precision, recall, F1-score
- **ROC Curves**: Interactive receiver operating characteristic plots
- **Confusion Matrices**: Detailed classification breakdowns
- **Calibration Plots**: Prediction reliability analysis

#### Feature Analysis
- **Feature Importance**: Global and local importance rankings
- **Partial Dependence**: Individual feature effect plots
- **SHAP Analysis**: Comprehensive feature interaction analysis
- **Correlation Heatmaps**: Feature relationship exploration

#### Fairness Metrics
- **Demographic Parity**: Equal positive prediction rates
- **Equalized Odds**: Equal TPR and FPR across groups
- **Calibration Fairness**: Prediction reliability by group
- **Individual Fairness**: Similar predictions for similar students

#### Research Tools
- **Hyperparameter Analysis**: Model configuration effects
- **Cross-Validation Results**: Statistical significance testing
- **Transfer Learning**: Cross-domain performance analysis
- **Uncertainty Quantification**: Prediction confidence analysis

### Advanced Features

#### Interactive Exploration
- **Filter Controls**: Subset data by any combination of features
- **Dynamic Updates**: Real-time plot updates based on selections
- **Export Options**: Download plots and data tables
- **Comparison Views**: Side-by-side model/group comparisons

#### Statistical Analysis
- **Significance Testing**: Statistical comparisons between groups
- **Confidence Intervals**: Bootstrap-based uncertainty estimates
- **Effect Sizes**: Practical significance beyond statistical significance
- **Multiple Comparisons**: Correction for multiple testing

## Technical Features

### Performance Optimization

#### Caching
- **Data Caching**: Automatic caching of expensive computations
- **Model Caching**: Reuse of loaded models across sessions
- **Plot Caching**: Cached visualizations for faster rendering

#### Memory Management
- **Lazy Loading**: Load data only when needed
- **Chunked Processing**: Handle large datasets efficiently
- **Resource Monitoring**: Track memory and CPU usage

### Customization Options

#### Appearance
```python
# Custom styling in streamlit config
[theme]
primaryColor = "#FF6B6B"           # Accent color
backgroundColor = "#FFFFFF"        # Background
secondaryBackgroundColor = "#F0F2F6"  # Sidebar
textColor = "#262730"             # Text
```

#### Data Sources
```python
# Configure data paths
DATA_PATH = "student-mat.csv"
MODEL_PATH = "models/model.pkl"
FIGURES_DIR = "figures/"
TABLES_DIR = "tables/"
```

### Snapshot Mode (Publication)

Generate deterministic screenshots for publications:

```bash
# Enable snapshot mode
streamlit run dashboard.py -- --snapshot

# Specify output directory
streamlit run dashboard.py -- --snapshot --output_dir publication_figures/

# Pre-load specific examples
streamlit run dashboard_student.py -- --snapshot --student_ids 1,15,42
```

This creates:
- `dashboard_overview.png` - Main dashboard view
- `dashboard_fairness_sex.png` - Fairness analysis by gender
- `dashboard_student_case.png` - Individual student example

## Troubleshooting

### Common Issues

#### Dashboard Won't Start
```bash
# Check Streamlit installation
streamlit --version

# Reinstall if needed
pip install streamlit --upgrade

# Check port availability
netstat -an | grep 8501
```

#### Missing Data/Models
```bash
# Ensure models are trained
make train

# Check file paths
ls -la models/ figures/ tables/

# Regenerate if needed
make clean && make all
```

#### Performance Issues
```bash
# Clear Streamlit cache
streamlit cache clear

# Reduce data size for testing
python -c "
import pandas as pd
df = pd.read_csv('student-mat.csv', sep=';')
df.sample(100).to_csv('student-mat-small.csv', sep=';', index=False)
"
```

#### Browser Compatibility
- **Recommended**: Chrome 80+, Firefox 75+, Safari 13+
- **Issues**: Disable ad blockers, enable JavaScript
- **Mobile**: Best viewed on tablets/desktop

### Performance Tips

#### Data Loading
- Use cached functions for expensive operations
- Pre-filter large datasets before visualization
- Load models once and reuse across sessions

#### Visualization
- Limit plot complexity for real-time updates
- Use sampling for large scatter plots
- Cache static visualizations

## API Integration

### External Data Sources
```python
# Connect to external APIs
import streamlit as st

@st.cache_data
def load_external_data(api_endpoint):
    """Load data from external source with caching."""
    response = requests.get(api_endpoint)
    return response.json()
```

### Model Serving
```python
# Integrate with model serving platforms
@st.cache_resource
def load_production_model(model_endpoint):
    """Load model from production endpoint."""
    return joblib.load(model_endpoint)
```

## Deployment

### Local Development
```bash
# Development mode with auto-reload
streamlit run dashboard.py --runner.magicEnabled false

# Custom port
streamlit run dashboard.py --server.port 8502
```

### Production Deployment
```bash
# Docker deployment
FROM python:3.9-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "dashboard.py", "--server.address", "0.0.0.0"]
```

### Cloud Platforms
- **Streamlit Cloud**: Direct GitHub integration
- **Heroku**: Use `setup.sh` and `Procfile`
- **AWS/GCP**: Container deployment options

## Best Practices

### User Experience
- Provide clear navigation and instructions
- Include help tooltips for complex metrics
- Offer both summary and detailed views
- Enable data export for further analysis

### Performance
- Cache expensive computations
- Use progress bars for long operations
- Implement responsive design principles
- Test with realistic data sizes

### Security
- Validate all user inputs
- Sanitize data exports
- Implement appropriate access controls
- Log user interactions appropriately

## Advanced Customization

### Custom Components
```python
# Add custom JavaScript components
import streamlit.components.v1 as components

def custom_plot_component(data):
    html_string = f"""
    <div id="custom-plot">
        <!-- Custom visualization code -->
    </div>
    <script>
        // Custom JavaScript
        console.log({data});
    </script>
    """
    components.html(html_string, height=400)
```

### Theme Customization
```python
# Custom CSS injection
def inject_custom_css():
    st.markdown("""
    <style>
    .main > div {
        padding-top: 2rem;
    }
    .stSelectbox > div > div {
        background-color: #f0f0f0;
    }
    </style>
    """, unsafe_allow_html=True)
```

For more information:
- **[Quick Start Guide](quickstart.md)** - Basic usage patterns
- **[CLI Reference](cli_guide.md)** - Command-line tools
- **[Data Card](data_card_student_performance.md)** - Dataset documentation
- **[Streamlit Documentation](https://docs.streamlit.io/)** - Framework reference