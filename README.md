# SMS Spam Classifier (SVM)

A machine learning project that classifies SMS text messages as **spam** or **ham** (not spam). Built with TF-IDF text features and a Support Vector Machine (SVM), benchmarked against a Naive Bayes baseline, and wrapped in an interactive prediction widget.

**Dataset:** [SMS Spam Collection Dataset (Kaggle)](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)
**Notebook:** [Open in Google Colab](https://colab.research.google.com/drive/1RSThX7zJNAAIjHNjg195UG92stJpx8Y6?usp=sharing)

---

## Overview

The project takes raw, labeled SMS messages and builds a full pipeline from cleaning to a usable spam detector:

1. Load and inspect the dataset
2. Explore the data (class balance, message length, word clouds)
3. Clean and normalize the text
4. Convert text to TF-IDF features
5. Train a Linear SVM classifier
6. Tune hyperparameters with GridSearchCV
7. Validate with cross-validation
8. Evaluate with confusion matrix, ROC curve, and AUC
9. Compare against a Multinomial Naive Bayes model
10. Provide an interactive `ipywidgets` prediction tool

---

## Pipeline Details

### 1. Data Loading
The dataset (`spam.csv`) is loaded and trimmed to two columns: `label` (spam/ham) and `message` (the SMS text).

### 2. Exploratory Data Analysis
- Class distribution: dataset is imbalanced — **4,825 ham vs. 747 spam** messages after removing duplicates.
- Message length: spam messages tend to be longer (word/character count) than ham messages.
- Word clouds: spam is dominated by terms like *free, win, prize, urgent, call, txt*; ham reflects everyday conversation (*love, know, get, like, time*).

### 3. Text Preprocessing
Each message is cleaned before modeling:
- Lowercased
- URLs and email addresses removed
- Digits and punctuation stripped
- Stopwords removed
- Words lemmatized to their base form

Labels are mapped to numeric form (`ham` → 0, `spam` → 1).

### 4. Feature Extraction
Cleaned text is vectorized using **TF-IDF** (unigrams + bigrams, top 5,000 features).

### 5. Model Training — SVM
A `LinearSVC` is trained on the TF-IDF features inside a scikit-learn `Pipeline`.

### 6. Hyperparameter Tuning
`GridSearchCV` (5-fold) searches over the SVM's `C` parameter to find the best-performing configuration.

### 7. Cross-Validation
The tuned model is re-validated with 5-fold cross-validation across the full dataset to confirm results are stable, not just a good train/test split.

### 8. Evaluation

**SVM (LinearSVC) — final performance:**
| Metric | Score |
|---|---|
| Accuracy | 0.9845 |
| Precision (Spam) | 0.9832 |
| Recall (Spam) | 0.8931 |
| F1-score (Spam) | 0.9360 |
| AUC | 0.9958 |

Confusion matrix and ROC curve are plotted to visualize true/false positives and negatives across classification thresholds.

### 9. Model Comparison — Naive Bayes
A `MultinomialNB` classifier is trained on the same TF-IDF features for comparison.

**Naive Bayes — final performance:**
| Metric | Score |
|---|---|
| Accuracy | 0.9681 |
| Precision (Spam) | 0.9900 |
| Recall (Spam) | 0.7557 |
| F1-score (Spam) | 0.8571 |
| AUC | 0.9899 |

**Result:** SVM outperforms Naive Bayes overall — better accuracy, recall, F1-score, and AUC. Naive Bayes edges out SVM slightly on precision but misses considerably more spam messages (higher false-negative rate).

### 10. Interactive Prediction Widget
An `ipywidgets`-based UI lets you type any message into a text box, click **Predict**, and instantly see whether it's classified as **Spam** or **Ham** — a live demo of the trained model.

```python
predict_sms("Hi, you've won a lottery. Collect the prize money by paying $100 now.")
# -> "Spam"
```

---

## Key Insights

- **Class imbalance** is real but manageable — the model still achieves strong recall on the minority (spam) class.
- **Message length and vocabulary** are strong signals: spam is longer and uses a distinct promotional vocabulary.
- **SVM is the better choice** for this task, balancing precision and recall better than Naive Bayes — important since missing spam (false negatives) and misflagging real messages (false positives) both have costs.

---

## Tech Stack
- Python, pandas, NumPy
- scikit-learn (TF-IDF, LinearSVC, MultinomialNB, GridSearchCV, metrics)
- NLTK (stopwords, lemmatization)
- matplotlib, seaborn, WordCloud (visualization)
- ipywidgets (interactive prediction tool)

---

## Possible Next Steps
- **Feature engineering:** add signals like special-character count, all-caps word ratio, or message length as explicit features
- **Deep learning:** try LSTM or transformer-based models for potentially higher accuracy
- **Ensembling:** combine SVM, Naive Bayes, and tree-based models
- **Threshold tuning:** adjust the SVM's decision threshold to favor precision or recall depending on tolerance for false positives vs. false negatives
- **User feedback loop:** collect real-world predictions/corrections to retrain and improve the model over time

---

## How to Run
1. Open the [Colab notebook](https://colab.research.google.com/drive/1RSThX7zJNAAIjHNjg195UG92stJpx8Y6?usp=sharing)
2. Download `spam.csv` from [Kaggle](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) and upload it when prompted
3. Run all cells in order — the notebook will train both models, generate visualizations, and launch the interactive predictor at the end
