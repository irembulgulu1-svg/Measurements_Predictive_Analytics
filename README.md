# B2Spoke: Pre-Purchase Fit Prediction Engine

**Academic Context:** MSc Business Analytics Capstone / Coursework
**Institution:** University College London (UCL)

## Executive Summary
This repository contains the end-to-end machine learning pipeline for **B2Spoke**, a two-sided digital marketplace connecting customers with local bespoke tailors. The objective of this project is to build a "Pre-Purchase Engine"—an AI system that utilizes generative design assistance and customer onboarding measurements to predict garment fit *before* physical manufacturing begins. 

By applying an Expected Value Cost Matrix to the model's predictions, the final deployed pipeline demonstrates a simulated business savings of **£47,010** across 9,935 orders by preemptively catching sizing errors and reducing fabric waste.

## Business Problem & Objective
In bespoke tailoring, the financial cost of predictive errors is highly asymmetric:
* **False Positive (Predicting a fit, but it is too small/large):** Results in a costly physical remake, wasted fabric, and poor customer experience (Estimated Penalty: £50).
* **False Negative (Predicting poor fit, but it would have been fine):** Results in a minor digital pattern verification step by the tailor (Estimated Penalty: £10).
* **True Positive (Correctly catching a bad fit pre-cut):** Allows for a quick, cheap digital pattern adjustment (Estimated Cost: £5).

**Objective:** To train a multiclass classification model (Fit, Small, Large) that aggressively protects the minority classes (Small/Large) to minimize physical remake costs, utilizing *strictly pre-purchase* customer metrics.

## Dataset
This project utilizes a sanitized version of the **ModCloth** e-commerce dataset as a proxy for bespoke measurements. 
* **Target Variable:** `fit` (Multiclass: fit, small, large)
* **Features Used:** `hips`, `height_in`, `size`, `bra size`, `cup size`, `category`.

## 🔬 Methodology & Critical Evaluation

### 1. Leakage-Free Data Engineering
To protect the demographic profiles of petite and plus-size users, missing values for `hips` and `height` were imputed using grouped medians. Crucially, to prevent data leakage, these medians were mapped strictly from the Training set and subsequently applied to the Test set. Dynamic parsing was implemented to convert raw text (e.g., `5ft 6in`) into machine-readable numerics (`66`).

### 2. The Target Leakage Discovery (Model Iteration)
The project represents a critical evolution from a "Naive" baseline to a Production-Ready engine:
* **Iteration 1 (The Illusion):** An initial Random Forest utilized all available features, achieving high simulated savings (£59k). However, a critical business audit revealed **Temporal Target Leakage**. The model relied heavily on post-purchase features (`review_len` and `quality`). 
* **Iteration 2 (Interpretation Leakage):** After removing reviews, the model shifted reliance to the `length` feature. Domain analysis revealed this was not a physical measurement, but rather a post-purchase interpretation of drape by the customer.
* **Iteration 3 (The Production Engine):** All post-purchase and interpretive features were ruthlessly purged. The final model relies strictly on the physical metrics a user provides during the B2Spoke digital onboarding phase.

### 3. Hyperparameter Optimization
The final Pre-Purchase Random Forest was optimized using `RandomizedSearchCV`, targeting the `f1_macro` metric to ensure the algorithm remained balanced across all body types. `class_weight='balanced'` was utilized to penalize majority-class bias.

## Key Findings & Results
The final, strictly pre-purchase model achieved the following on the holdout test set (n=9,935):
* **Cross-Validation Stability:** 5-Fold Stratified CV proved pipeline stability without overfitting.
* **Feature Importance:** Successfully transitioned from target leakage to true physical drivers (`hips`, `size`, `height_in`).
* **Commercial Impact:** Despite a lower overall raw accuracy (47%) due to the strict removal of "cheating" features, the model's cautious recall rates (protecting against False Positives) yielded a **True Business Savings of £47,010** compared to a Status Quo baseline.

## Repository Structure
* `/data/` - Contains raw and processed CSV files.
* `/notebooks/` 
  * `01_EDA_and_Cleaning.ipynb`
  * `02_Feature_Engineering.ipynb`
  * `03_Baseline_Models.ipynb`
  * `04_Target_Leakage_Discovery.ipynb`
  * `05_Final_B2Spoke_Production_Engine.ipynb`
* `requirements.txt` - Required Python dependencies.

## How to Run
1. Clone the repository.
2. Install dependencies via `pip install -r requirements.txt`.
3. Execute the notebooks in sequential order.