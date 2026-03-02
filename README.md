B2Spoke: Pre-Purchase Fit Prediction Engine
Academic Context: MSc Business Analytics Capstone / Coursework
Institution: University College London (UCL)

Executive Summary
This repository contains the end-to-end machine learning pipeline for the B2Spoke startup, a two-sided digital marketplace connecting customers with local bespoke tailors. B2Spoke integrates an AI system that allows customers to easily explain their design ideas, making the bespoke process transparent for both the customer and the tailor. However, a critical scaling challenge is servicing the remote customer segment—users who know their measurements but cannot physically visit a shop for an in-person fitting to verify those AI-generated designs.

The objective of this project is to build a "Pre-Purchase Engine"—a predictive classification model that digitizes the fitting room. It utilizes customer self-reported measurements to verify if a garment will fit before physical manufacturing begins. By applying an Expected Value Cost Matrix, the final deployed Extreme Gradient Boosting (XGBoost) pipeline demonstrates a simulated business savings of £55,290 across 9,935 orders by preemptively catching sizing errors on unseen customers.

The Business Problem: Digitizing the Fitting Room
In traditional bespoke tailoring, an in-person fitting allows the tailor to intuitively assess a customer's proportions. Accepting remote orders based solely on inputted numbers carries a highly asymmetric financial risk:

False Positive (Predicting a fit, but it is too small/large): Results in a costly physical remake, wasted fabric, and poor customer experience (Estimated Penalty: £50).

False Negative (Predicting poor fit, but it would have been fine): Results in a minor digital pattern verification step (a "false alarm" where the tailor double-checks the measurements with the customer) (Estimated Penalty: £10).

True Positive (Correctly catching a bad fit pre-cut): Allows for a quick, cheap digital pattern adjustment before the fabric is cut (Estimated Cost: £5).

Objective: To train an AI model that acts as a digital risk-mitigation tool. The model must trade overall precision for high minority-class recall (Small/Large) to protect tailors from £50 physical remakes, utilizing strictly pre-purchase metrics.

Dataset & Features
This project utilizes a sanitized version of the ModCloth e-commerce dataset as a proxy for bespoke measurements.

Target Variable: fit (Multiclass: fit, small, large)

Pre-Purchase Features Used: hips, height_in, size, bra size, cup size, category.

Methodology & The Ablation Study
The project represents a critical evolution from "Naive" early models to a Production-Ready engine via a strict ablation study.

1. Leakage-Free Data Engineering
To protect demographic profiles, missing values for hips and height were imputed using grouped medians. Crucially, to prevent data leakage, these medians were calculated strictly from the Training set and mapped onto the Test set.

2. Purging Target Leakage (The Journey)
Early exploratory models achieved artificially high accuracies and simulated savings exceeding £59k. However, rigorous data auditing revealed Temporal Target Leakage.

Iteration 1: The model relied on post-purchase reviews (review_len and quality).

Iteration 2: After removing reviews, the model shifted reliance to the length feature. Domain analysis revealed this was not a physical measurement, but a subjective, post-purchase interpretation of drape by the customer.

Iteration 3 (The Production Engine): In our final ablation step, all post-purchase and interpretive features were ruthlessly purged. The models were forced to compete on a purely objective, 100% pre-purchase dataset to simulate a true remote order.

Interpreting the Metrics: The Accuracy Paradox
The final production model achieved an overall accuracy of 43%. In the context of subjective, imbalanced tailoring data and an asymmetric cost matrix, raw accuracy is a vanity metric.

Our objective was Risk Mitigation via Recall. Because a False Positive costs £50 and a False Alarm (False Negative) only costs £10, the model is financially optimized to be highly sensitive. It deliberately sacrifices Precision (~0.22 for minority classes) to maximize Recall on expensive sizing errors. A highly accurate model that misses physical remakes will bankrupt a tailor; a 43% accurate model that catches the critical errors saves the marketplace £55,290.

Model Progression & Results
Three models were evaluated on the strictly pre-purchase test set (n=9,935):

The Linear Baseline (Logistic Regression): * Performance: Acc 0.40 | Savings £54,900

Agent Mistake Correction: The AI initially generated a pipeline without feature scaling, resulting in a mathematical convergence failure. I manually intervened to introduce a StandardScaler, successfully stabilizing the optimization. While it achieved £54,900 in savings, it did so by aggressively predicting garments would not fit (Recall for 'fit' = 0.40). This hyper-conservative filter would reject too many viable remote orders and choke marketplace liquidity.

The Non-Linear Ensemble (Random Forest): * Performance: Acc 0.47 | Savings £47,010

Analysis: Improved raw accuracy and approved more orders (Recall for 'fit' = 0.52). However, it was too lenient and missed critical sizing errors, leaving tailors financially exposed.

The Ultimate Engine (Advanced XGBoost): * Performance: Acc 0.43 | Savings £55,290

Analysis: The optimal balance. By sequentially correcting errors (Boosting) and utilizing sample weights to penalize minority-class mistakes, it aggressively targeted the most expensive errors. It maximized commercial savings while maintaining a sustainable, scalable order approval rate.

Final Recommendation & Model Card
The XGBoost model is formally selected as the B2Spoke Pre-Purchase Engine. By accurately predicting fit outcomes based solely on a remote customer's self-reported measurements, it empowers tailors to confidently verify AI-assisted designs and accept off-site orders, drastically de-risking the bespoke manufacturing process.

What it is for: Pre-purchase digital fit verification for remote customers based strictly on physical onboarding measurements.

What it is not for: Predicting subjective post-purchase style preferences, drape, or fabric quality.

Data Provenance & Constraints: Built on a ready-to-wear proxy dataset. Future B2Spoke iterations must actively harvest genuine bespoke "poor fit" data to further refine failure-state predictions.

Repository Structure
/data/ - Contains processed CSV files. Note: To run Notebook 1 from scratch, download the raw ModCloth dataset from Kaggle and place the .json file in the /data/raw/ directory.

/notebooks/

01_data_loading.ipynb

02_EDA.ipynb

03_data_cleaning.ipynb

04_feature_selection.ipynb

05_model_training.ipynb

06_model_optimization.ipynb

07_production_b2spoke_model.ipynb

requirements.txt - Required Python dependencies for reproducibility.

How to Run
Clone the repository.

Install dependencies via pip install -r requirements.txt.

Execute the notebooks in sequential order, culminating in the Step 7 production engine.