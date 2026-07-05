# AI Fairness in Toxicity Classification

**Skills:** Python, Machine Learning, Natural Language Processing (NLP), Explainable AI (XAI), Fairness Analysis, XGBoost, SHAP, Predictive Modeling

## Overview

This project investigates whether AI-based toxicity detection models exhibit bias toward comments that reference certain identity groups. Using the Jigsaw Unintended Bias in Toxicity Classification dataset, we developed and compared multiple machine learning models to evaluate both predictive performance and fairness in automated content moderation systems.

The project combines machine learning and explainable AI techniques to better understand how model predictions are made and to assess potential ethical concerns in AI-driven decision-making.

## Business Objectives

- Detect and measure potential bias in AI-based toxicity classification models.
- Compare the performance of multiple machine learning algorithms.
- Evaluate the influence of identity-related features on predictions.
- Promote transparency and responsible AI practices through explainable machine learning.

## Dataset

**Dataset:** Jigsaw Unintended Bias in Toxicity Classification Dataset

- Approximately 450,000 online comments
- Toxicity labels and subcategories
- Identity attributes including gender, race, religion, sexual orientation, and disability references
- Engagement and annotator information

## Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SHAP
- Matplotlib
- Jupyter Notebook

## Methodology

### Data Preparation
- Text preprocessing and cleaning
- TF-IDF vectorization
- Binary toxicity classification
- Handling missing identity attributes
- Feature engineering and dataset preparation

### Machine Learning Models
- Logistic Regression
- Random Forest
- XGBoost

### Model Explainability
- SHAP (SHapley Additive exPlanations) analysis to evaluate feature importance and model fairness.

## Key Results

| Model | Accuracy |
|-------|-----------|
| Logistic Regression | 0.5441 |
| Random Forest | 0.5449 |
| XGBoost | 0.5453 |

## Key Findings

- Vocabulary was the primary driver of model predictions.
- Identity-related features had relatively low influence on predictions.
- All models struggled to identify toxic comments due to class imbalance.
- Human oversight remains important for real-world content moderation systems.

## Repository Contents

- `AI_Toxicity_Bias_Analysis_Report.pdf`
- `AI_Toxicity_Bias_Analysis_Presentation.pdf`

## Future Enhancements

Potential improvements include:

- Addressing class imbalance through resampling techniques.
- Evaluating additional fairness metrics.
- Exploring transformer-based NLP models.
- Improving toxic comment detection while maintaining fairness and transparency.
