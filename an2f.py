#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
import statsmodels.api as sm
from scipy import stats
from scipy.stats import f
# import warnings
# warnings.filterwarnings('ignore')

# Set page config
st.set_page_config(page_title="Two-Way ANOVA Analyzer", layout="wide")

# App title
st.title("Two-Way ANOVA Analysis App")
st.markdown("Analyze your experimental data with two-way ANOVA and post-hoc tests")

# Sidebar for file upload
with st.sidebar:
    st.header("Upload Data")
    uploaded_file = st.file_uploader("Choose an Excel file", type=["xlsx"])
    
    st.header("Settings")
    alpha = st.slider("Significance level (α)", 0.01, 0.10, 0.05, 0.01)
    
    st.markdown("---")
    st.markdown("**Instructions:**")
    st.markdown("1. Upload your Excel file")
    st.markdown("2. Ensure your data has columns for:")
    st.markdown("   - Dependent variable (continuous)")
    st.markdown("   - Two independent variables (categorical)")
    st.markdown("3. The app will automatically analyze your data")

# Main content area
if uploaded_file is not None:
    try:
        # Load data
        data = pd.read_excel(uploaded_file)
        
        # Display raw data
        with st.expander("View Raw Data"):
            st.dataframe(data)
        
        # Get column names
        cols = data.columns.tolist()
        
        # Select variables
        col1, col2, col3 = st.columns(3)
        with col1:
            dependent_var = st.selectbox("Select dependent variable", cols, index=0)
        with col2:
            factor1 = st.selectbox("Select first factor (independent variable)", cols, index=1)
        with col3:
            factor2 = st.selectbox("Select second factor (independent variable)", cols, index=2)
        
        # Perform ANOVA
        st.header("Two-Way ANOVA Results")
        
        # Model formula
        formula = f"{dependent_var} ~ C({factor1}) + C({factor2})"
        model = ols(formula, data=data).fit()
        anova_results = anova_lm(model, typ=2)
        
        # Calculate F-critical values
        df_factor1 = anova_results['df'][f'C({factor1})']
        df_factor2 = anova_results['df'][f'C({factor2})']
        df_resid = anova_results['df']['Residual']
        
        f_crit_factor1 = f.ppf(1 - alpha, df_factor1, df_resid)
        f_crit_factor2 = f.ppf(1 - alpha, df_factor2, df_resid)
        
        # Add F-critical column
        f_crit_column = []
        for index in anova_results.index:
            if index == f'C({factor1})':
                f_crit_column.append(f_crit_factor1)
            elif index == f'C({factor2})':
                f_crit_column.append(f_crit_factor2)
            else:
                f_crit_column.append(None)
        
        anova_results['F crit'] = f_crit_column
        
        # Display ANOVA table
        st.dataframe(anova_results.round(4))
        
        # Interpretation
        st.subheader("Interpretation")
        sig_factor1 = anova_results['PR(>F)'][f'C({factor1})'] < alpha
        sig_factor2 = anova_results['PR(>F)'][f'C({factor2})'] < alpha
        
        if sig_factor1:
            st.success(f"✔ {factor1} has a statistically significant effect (p < {alpha})")
        else:
            st.warning(f"✖ {factor1} does not show a statistically significant effect (p ≥ {alpha})")
            
        if sig_factor2:
            st.success(f"✔ {factor2} has a statistically significant effect (p < {alpha})")
        else:
            st.warning(f"✖ {factor2} does not show a statistically significant effect (p ≥ {alpha})")
        
        # Visualizations
        st.header("Data Visualizations")
        
        fig1, ax1 = plt.subplots(figsize=(10, 6))
        sns.boxplot(x=factor1, y=dependent_var, data=data, ax=ax1)
        ax1.set_title(f"{dependent_var} by {factor1}")
        st.pyplot(fig1)
        
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        sns.boxplot(x=factor2, y=dependent_var, data=data, ax=ax2)
        ax2.set_title(f"{dependent_var} by {factor2}")
        st.pyplot(fig2)
        
        # Interaction plot
        fig3, ax3 = plt.subplots(figsize=(10, 6))
        sns.pointplot(x=factor1, y=dependent_var, hue=factor2, data=data, ax=ax3)
        ax3.set_title(f"Interaction Plot: {factor1} × {factor2}")
        st.pyplot(fig3)
        
        # Post-hoc tests
        st.header("Post-Hoc Tests")
        
        # Factor1 post-hoc if significant
        if sig_factor1:
            st.subheader(f"Post-hoc Analysis for {factor1}")
            
            # Tukey HSD
            st.markdown("**Tukey HSD Test**")
            tukey_factor1 = pairwise_tukeyhsd(data[dependent_var], data[factor1], alpha=alpha)
            st.text(str(tukey_factor1))
            
            # Plot Tukey results
            fig4, ax4 = plt.subplots(figsize=(10, 4))
            tukey_factor1.plot_simultaneous(ax=ax4)
            st.pyplot(fig4)
            
            # Bonferroni
            st.markdown("**Bonferroni Correction**")
            mc_factor1 = MultiComparison(data[dependent_var], data[factor1])
            bonf_results = mc_factor1.allpairtest(stats.ttest_ind, method='bonf')[0]
            st.text(str(bonf_results))
        
        # Factor2 post-hoc if significant
        if sig_factor2:
            st.subheader(f"Post-hoc Analysis for {factor2}")
            
            # Tukey HSD
            st.markdown("**Tukey HSD Test**")
            tukey_factor2 = pairwise_tukeyhsd(data[dependent_var], data[factor2], alpha=alpha)
            st.text(str(tukey_factor2))
            
            # Plot Tukey results
            fig5, ax5 = plt.subplots(figsize=(10, 4))
            tukey_factor2.plot_simultaneous(ax=ax5)
            st.pyplot(fig5)
            
            # Bonferroni
            st.markdown("**Bonferroni Correction**")
            mc_factor2 = MultiComparison(data[dependent_var], data[factor2])
            bonf_results = mc_factor2.allpairtest(stats.ttest_ind, method='bonf')[0]
            st.text(str(bonf_results))
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check your data format and try again.")
else:
    st.info("Please upload an Excel file to begin analysis")
    st.markdown("### Sample Data Format")
    st.markdown("Your Excel file should have three columns:")
    st.markdown("- One continuous dependent variable (e.g., FuelConsumption)")
    st.markdown("- Two categorical independent variables (e.g., Vessel, VoyageBlock)")
    st.markdown("")
    st.markdown("Example:")
    sample_data = pd.DataFrame({
        'FuelConsumption': [45, 50, 55, 40, 42, 48],
        'Vessel': ['A', 'A', 'B', 'B', 'C', 'C'],
        'VoyageBlock': [1, 2, 1, 2, 1, 2]
    })
    st.dataframe(sample_data)

