# DA5401 Assignment 4: GMM-Based Synthetic Sampling for Imbalanced Data

## Overview

This project demonstrates how to address class imbalance in credit card fraud detection using Gaussian Mixture Model (GMM)-based synthetic sampling. The workflow includes data analysis, baseline modeling, synthetic data generation, and performance comparison.

## Dataset

- **File:** `creditcard.csv`
- **Description:** Contains anonymized credit card transactions labeled as fraudulent (`Class = 1`) or non-fraudulent (`Class = 0`).

## Project Structure

- `creditcard.csv`: The dataset used for analysis and modeling.
- `DA5401_Assignment_4_GMM_Sampling.ipynb`: Jupyter notebook with all code, analysis, and results.
- `README.md`: Project documentation.
- `DA5401 A4 GMM.pdf`: Assignment brief/instructions.

## Steps & Methodology

1. **Data Loading & Exploration**
   - Load the dataset and check for missing values.
   - Analyze class distribution and visualize imbalance.

2. **Baseline Model**
   - Train a Logistic Regression model on the original, imbalanced data.
   - Evaluate using accuracy, precision, recall, and F1-score.

3. **GMM-Based Synthetic Sampling**
   - Fit a Gaussian Mixture Model to the minority class (fraud cases).
   - Generate synthetic samples to balance the dataset.

4. **Model Training with Synthetic Data**
   - Combine synthetic and original data.
   - Retrain Logistic Regression on the balanced dataset.
   - Evaluate and compare performance.

5. **Performance Comparison**
   - Summarize and compare metrics for both models.

## How to Run

1. **Requirements**
   - Python 3.7+
   - Jupyter Notebook
   - pandas, numpy, matplotlib, scikit-learn

2. **Instructions**
   - Open `DA5401_Assignment_4_GMM_Sampling.ipynb` in Jupyter Notebook.
   - Run all cells sequentially.
   - Review outputs, visualizations, and performance metrics.

## Results

- The notebook prints and compares the performance of the baseline and GMM-augmented models.
- Metrics include accuracy, precision, recall, and F1-score for both models.

## References

- [Gaussian Mixture Models](https://en.wikipedia.org/wiki/Gaussian_mixture_model)
- [Imbalanced Learning](https://www.oreilly.com/library/view/imbalanced-learning/9781119610080/)
- [scikit-learn Documentation](https://scikit-learn.org/stable/)
