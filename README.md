# sms_spam_svm
A small project to classify SMS messages as either 'spam' or 'ham' (not spam) using a Support Vector Machine (SVM) model. The project involves loading and preprocessing a dataset, performing exploratory data analysis, training an SVM model with TF-IDF features, tuning hyperparameters using GridSearchCV, evaluating the model's performance with cross-validation and various metrics (accuracy, precision, recall, F1-score), and finally creating an interactive prediction function.

Additional tasks include:

Visualize Confusion Matrix: Generate and display a confusion matrix for the SVM model's predictions on the test set to visualize the true positives, true negatives, false positives, and false negatives.
Plot ROC Curve: Calculate and plot the Receiver Operating Characteristic (ROC) curve and the Area Under the Curve (AUC) for the SVM model to evaluate its performance across various classification thresholds.
Compare with Naive Bayes Classifier: Implement a Naive Bayes classifier (e.g., Multinomial Naive Bayes) with TF-IDF features and compare its performance metrics (accuracy, precision, recall, F1-score, confusion matrix, ROC curve) against the current SVM model.
Interactive Prediction Widget: Develop an interactive widget using ipywidgets that allows a user to input a text message and receive an instant 'Spam' or 'Ham' prediction from the trained model. This will personalize the project by making it directly usable for message classification.
Final Task: Summarize the additional analysis and model comparison results, discuss the insights gained from the visualizations, and highlight the functionality of the interactive prediction tool.
