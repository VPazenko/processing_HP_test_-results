# Processing HP test results

## Overview
There are many different methods for detecting Helicobacter Pylori (HP) infection. An overview of these methods can be found, for example, in this article doi:10.3748/wjg.v21.i40.11221 https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4616200/ "Histology is usually considered to be the gold standard in the direct detection of H. pylori infection."
Also, according to the Maastricht III Consensus Report: "A UBT is recommended if available". doi:10.1136/gut.2006.101634

This repository contains an in-depth analysis of HP infected study dataset. The data was collected from a research among patients who came to the hospital (Russia, St. Petersburg)  with the suspicion of HP infection.
According to the study design, each patient was tested by 2 breath tests C13 UBT (Urea Breath Test) , as well as FGDS with biopsy sampling, which were analyzed histologically, by PCR and by RUT (Rapid Urea Test).

The work consists of two large parts:
1. Working with the original dataset "as it is".
2. Trying to recover missing data and working with the full dataset.

## Purpose
The aim of this analysis is 1) To compare UBT test of Russian-made Helicarb with Chinese-made Heliforce test. 2) To explore the correlation between the gold standard and the other tests.

## Methodology
The analysis includes:
- Data cleaning and preprocessing
- Exploratory data analysis (EDA) to visualize and understand data distributions
- Visualization of interactions between various factors that could affecting on test results

- Missing data recovery 
- Statistical analysis to test similarity of data sets before and after recovery

## Files
- `HP_tests.ipynb`: Jupyter Notebook containing the complete analysis workflow
- `config.yaml`: Configuration file with data file path
- `Summary_functions_and_code_clean_v15.py`: Module containing custom statistical functions used in the analysis (Author: M.E.F. Apol)
- 'module.py': : Module containing custom functions used for data processing and visualization 
- `HP_tests_final.xlsx`: Dataset used for analysis (not included in this repository)
- 'recheck_hystology.xlsx': Dataset used for analysis (not included in this repository)
- 'json_dict.json': Auxiliary file. It is used for renaming columns and strict positioning.

## Analysis Details
- Data Inspection: Examining dataset structure, missing values, and data types
- Visualization: Utilizing histograms, scatter plots, and heat maps
- Statistical Tests: Applying 2-sample Mann-Whitney-test for medians(two-sided) and Levene test for 2 variances(two-sided)
- Undersampling: Filling missing data. Checking the influence of these data on the sample (comparison using statistical analysis of the "before" and "after" samples). Compare the results "before" and "after" recovery.

## Conclusion

1. Correlation between two type of load for BT is very strong (if exclude '+/-' results, corr = 90%)

It is notable that the introduction of a cat-value for histology of 10% changed the correlations. 
2. Histology across regions became even more correlated with each other (73% to 89%), which is considered a strong correlation.

3. Both loads for breath tests show strong correlation (70% and 71%) with total gastric infection status.

4. PCR shows good correlation with BT (65% and 58%) and with total Histology status (62%) 

5. RUT shows the good correlation with Heliforce BT (59%)
   
6. The recovery of missing data had some impact on the final Heatmap. For most positions, the correlation changed between -1% and 7%

## How to Use
1. Clone the repository.
2. Ensure required dependencies are installed (see requirements.txt)
3. Execute `HP_tests.ipynb` in a Jupyter Notebook environment.

## Author
V. Pazenko v.pazenko@st.hanze.nl
