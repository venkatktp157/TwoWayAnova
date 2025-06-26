#!/usr/bin/env python
# coding: utf-8

# In[1]:


import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from statsmodels.graphics.factorplots import interaction_plot
import itertools

st.title("Two-Way ANOVA with Tukey Post Hoc Test")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Preview of Dataset")
    st.dataframe(df.head())

    factor_a = st.selectbox("Select Factor A", df.columns)
    factor_b = st.selectbox("Select Factor B", df.columns)
    response = st.selectbox("Select Response Variable", df.columns)

    if st.button("Run Analysis"):
        try:
            model = ols(f'{response} ~ C({factor_a}) * C({factor_b})', data=df).fit()
            anova_table = sm.stats.anova_lm(model, typ=2)
            st.write("### ANOVA Table")
            st.dataframe(anova_table)
            
           # Add interpretation
            st.write("#### Interpretation:")
            for index, row in anova_table.iterrows():
                pval = row['PR(>F)']
                if pval < 0.05:
                    st.markdown(f"- **{index}** has a significant effect on the response variable (*p* = {pval:.4f}).")
                else:
                    st.markdown(f"- **{index}** does not have a significant effect (*p* = {pval:.4f}).")


            # Tukey HSD post hoc
            df['Group'] = df[factor_a].astype(str) + "_" + df[factor_b].astype(str)
            tukey = pairwise_tukeyhsd(endog=df[response], groups=df['Group'], alpha=0.05)
            st.write("### Tukey HSD Results")
            st.text(tukey.summary())
            
            # Add interpretation
            st.write("#### Interpretation:")
            summary_df = pd.DataFrame(data=tukey._results_table.data[1:], columns=tukey._results_table.data[0])
            significant = summary_df[summary_df['reject'] == True]
            if not significant.empty:
                for i, row in significant.iterrows():
                    st.markdown(f"- Difference between **{row['group1']}** and **{row['group2']}** is significant (p = {row['p-adj']:.4f}).")
            else:
                st.markdown("- No significant differences found between group pairs.")
                       
            # Boxplot           
            st.write("### Boxplot by Group")
            fig, ax = plt.subplots(figsize=(12, 6))  # Wider figure size
            sns.boxplot(x='Group', y=response, data=df, ax=ax)
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')  # Tilt labels
            st.pyplot(fig)
            st.caption("🔎 Differences in group medians and spread help visually assess variation.")

            
            st.write("### Interaction Plot")

            # Dynamically generate markers and colors
            levels = df[factor_a].unique()
            colors = sns.color_palette("husl", len(levels))
            markers = itertools.cycle(['D', '^', 'o', 's', 'v', '*', 'P', 'X'])

            fig2 = interaction_plot(df[factor_b], df[factor_a], df[response],
                                    colors=colors,
                                    markers=[next(markers) for _ in range(len(levels))],
                                    ms=10)
            st.pyplot(fig2.figure)           

            st.caption("🔎 Non-parallel lines suggest potential interaction effects between the two factors.")


        except Exception as e:
            st.error(f"An error occurred: {e}")
else:
    st.info("Please upload a CSV file to get started.")

