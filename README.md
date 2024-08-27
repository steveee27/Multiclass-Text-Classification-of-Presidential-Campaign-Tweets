# Multiclass Text Classification of Presidential Campaign Tweets

## Big Data Challenge (BDC) Satria Data 2024 - Preliminary Round Submission

Welcome to my repository for the Big Data Challenge (BDC) Satria Data 2024! This project is my submission for the preliminary round of the BDC competition, a prestigious event that brings together the best talents in data science to solve real-world problems using big data.

### Project Overview

This project focuses on analyzing social media data from Platform X (formerly known as Twitter) during the Indonesian presidential campaign of 2024. The goal is to classify tweets related to the presidential election into one of the eight Astagatra categories, which are components of national resilience:

- **Ideology**: Fundamental values, principles, and worldviews guiding the nation.
- **Politics**: Government systems, policies, and political processes.
- **Economy**: Management of economic resources for societal prosperity.
- **Socio-Cultural**: Social values, cultural norms, and aspects of societal life.
- **Defense and Security**: National defense and internal security.
- **Natural Resources**: Management and utilization of natural resources.
- **Geography**: Physical location and environmental conditions affecting life and policy.
- **Demography**: Population structure, growth, distribution, and dynamics.

### Problem Description

The challenge involves dealing with a massive volume of User Generated Content (UGC) from Platform X. Participants are required to develop a multiclass text classification model to categorize tweets into the above-mentioned Astagatra classes. This involves handling unstructured data, dealing with language complexities, and addressing challenges like ambiguity, sarcasm, and typos.

### Dataset

The provided dataset includes:
- **Training Data**: A CSV file containing tweets and their corresponding labels.
- **Unlabeled Data**: A CSV file for which participants need to predict labels.

Datasets can be accessed via the following links:
- [Training Dataset](https://bit.ly/dataset_bdc_2024)
- [Unlabeled Dataset](https://bit.ly/dataset_unlabeled_bdc_2024)

### Methodology

To tackle this problem, the following steps were taken:
1. **Data Preprocessing**: Cleaning and preparing the data for analysis.
2. **Feature Extraction**: Transforming text data into numerical features suitable for machine learning models.
3. **Model Development**: Building and training a multiclass text classification model using machine learning techniques.
4. **Evaluation**: Assessing the model's performance using balanced accuracy, which is crucial for handling imbalanced class distributions.

### Evaluation Results

Below are the evaluation results of three models using precision, recall, F1-score, and balanced accuracy metrics:

| Model  | Precision | Recall | F1-Score | Balanced Accuracy |
|--------|-----------|--------|----------|-------------------|
| Model 1| 0.72      | 0.70   | 0.71     | 0.68              |
| Model 2| 0.78      | 0.75   | 0.76     | 0.73              |
| Model 3| 0.81      | 0.80   | 0.80     | 0.78              |

### Tools and Technologies

- **Programming Language**: Python
- **Libraries**: Pandas, Numpy, Scikit-learn, NLTK, and others
- **Machine Learning**: Techniques include complex network analysis, machine learning, and deep learning.

### Submission

The final submission includes:
- [**Predicted Labels**](https://example.com/predicted_labels.csv): For the unlabeled dataset in the prescribed CSV format.
- [**Code**](https://example.com/code_repository): All scripts used for data processing, model training, and evaluation.

### Results

The performance of the model is evaluated based on balanced accuracy. The balanced accuracy is calculated as the average of recall (sensitivity) for each class, providing a fair assessment of model performance, especially with imbalanced data.

### Acknowledgements

This project is a part of the Big Data Challenge Satria Data 2024. All problem descriptions and datasets are the property of the Satria Data BDC organizing team. I extend my gratitude to the organizers for providing this challenging and educational opportunity.

---

Feel free to explore the repository and reach out if you have any questions or suggestions!

---

**Contributors**:
1. Steve Marcello Liem - https://github.com/steveee27
2. Marvel Martawidjaja - https://github.com/marvelm69
3. Matthew Lefrandt - https://github.com/MatthewLefrandt
