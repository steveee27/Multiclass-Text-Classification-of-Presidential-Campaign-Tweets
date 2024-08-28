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

1. **Text Cleansing**: Preprocessing the text data involved several steps to ensure the data was clean and ready for analysis:
   - **Removing Mentions and Patterns**: Removed Twitter-specific elements like mentions (e.g., `@username`), and patterns such as `re` and `rt` that often appear in retweets.
   - **Hashtag Preprocessing**: Preprocessed hashtags by treating them appropriately in the context of the text.
   - **Lowercasing**: Converted all text to lowercase to maintain consistency.
   - **Removing URLs**: Removed URLs using regular expressions to eliminate links from the text.
   - **Removing Numbers**: Stripped out all numeric characters to focus solely on the textual content.
   - **Removing Non-ASCII Characters**: Removed non-ASCII characters to ensure compatibility across different systems.
   - **Removing Non-Alphanumeric Characters**: Eliminated non-alphanumeric characters (excluding spaces) to clean the text further.
   - **Trimming Excessive Whitespace**: Reduced multiple spaces to a single space for consistency.
   - **Final Cleanup**: Removed any remaining unwanted characters and underscores.
   - **Removing Stopwords**: Using NLP techniques to filter out common stopwords that do not contribute significantly to the model.
2. **Data Splitting**: Dividing the dataset into training and testing sets for model evaluation.
3. **Augmenting Minority Classes**: Applying data augmentation techniques to balance the minority classes in the training data.
4. **TF-IDF Vectorization**: Converting the textual data into numerical features using TF-IDF vectorization.
5. **Model Training**: Training various machine learning models including Logistic Regression, SVM, and others.
6. **Evaluation**: Evaluating the models based on accuracy, balanced accuracy, and other metrics.
7. **Model Selection**: Choosing the best model, which in this case was Logistic Regression, based on the evaluation metrics.
8. **Inference**: Using the selected model to perform inference on the unlabeled data.

### Evaluation Results
The performance of the model is evaluated based on balanced accuracy. The balanced accuracy is calculated as the average of recall (sensitivity) for each class, providing a fair assessment of model performance, especially with imbalanced data.

Below are the evaluation results of three models using accuracy and balanced accuracy metrics:

| Model               | Accuracy | Balanced Accuracy |
|---------------------|----------|-------------------|
| Logistic Regression | 0.75     | 0.52              |
| Random Forest       | 0.73     | 0.51              |
| Gradient Boosting   | 0.66     | 0.54              |
| K-Nearest Neighbors | 0.59     | 0.48              |
| Linear SVM          | 0.72     | 0.48              |
| Poly SVM            | 0.73     | 0.38              |
| RBF SVM             | 0.76     | 0.46              |
| Sigmoid SVM         | 0.72     | 0.48              |
| Multinomial NB      | 0.62     | 0.54              |
| Complement NB       | 0.53     | 0.55              |
| Bernoulli NB        | 0.67     | 0.51              |

### Tools and Technologies

- **Programming Language**: Python
- **Libraries**: 
  - `pandas` for data manipulation
  - `pickle` for model serialization
  - `re` and `string` for text preprocessing
  - `nltk` and `nlp_id.stopword` for natural language processing and stopword removal
  - `nlpaug` for data augmentation
  - `sklearn` (including SVM, KNN, Gradient Boosting, Naive Bayes variants, Logistic Regression, Random Forest, TfidfVectorizer, GridSearchCV, and model evaluation tools)
  - `warnings` for handling warnings

### Submission

The final submission includes:
- [**Predicted Labels**](https://github.com/steveee27/Multiclass-Text-Classification-of-Presidential-Campaign-Tweets/blob/main/jawaban_penyisihan_bdc_2024.csv): For the unlabeled dataset in the prescribed CSV format.
- [**Code**](https://github.com/steveee27/Multiclass-Text-Classification-of-Presidential-Campaign-Tweets/blob/main/code.ipynb): All scripts used for data processing, model training, and evaluation.


### Acknowledgements

This project is a part of the Big Data Challenge Satria Data 2024. All problem descriptions and datasets are the property of the Satria Data BDC organizing team. I extend my gratitude to the organizers for providing this challenging and educational opportunity.

---

Feel free to explore the repository and reach out if you have any questions or suggestions!

---

**Contributors**:
1. [Steve Marcello Liem](https://github.com/steveee27)
2. [Marvel Martawidjaja](https://github.com/marvelm69)
3. [Matthew Lefrandt](https://github.com/MatthewLefrandt)
