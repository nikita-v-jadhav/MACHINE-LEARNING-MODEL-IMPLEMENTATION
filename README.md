# MACHINE-LEARNING-MODEL-IMPLEMENTATION
CODTECH INTERNSHIP

**COMPANY**: CODTECH IT SOLUTIONS  
**NAME**: [Nikita Vijay Jadhav]  
**INTERN ID**: [CT04DY2366]  
**DOMAIN**: [Python Programming]  
**DURATION**: 4 Weeks  
**MENTOR**: [Neela Santosh Kumar]  

TASK 4 - MACHINE LEARNING MODEL IMPLEMENTATION

## Project Description

This project demonstrates the implementation of a Machine Learning Model to classify messages as Spam or Ham (Not Spam).
The dataset used is a collection of SMS messages labeled as "spam" (unwanted promotional or fraudulent messages) and "ham" (normal useful messages).

The primary objective of this project is to:

Understand the end-to-end process of building a machine learning model.

Learn how to clean and preprocess textual data.

Apply machine learning algorithms to classify text messages.

Evaluate model performance using metrics such as accuracy, precision, recall, and F1-score.

This project is useful in real-world applications such as spam detection in emails, SMS filtering in telecom companies, and improving user experience by reducing unwanted communication.

 ## Features

Data preprocessing (cleaning, stopword removal, tokenization).

Feature extraction using TF-IDF (Term Frequency-Inverse Document Frequency).

Machine Learning Model training using Naive Bayes.

Evaluation of model performance.

Prediction function for classifying new/unseen messages.

## Technologies Used

Python

Pandas, NumPy

Scikit-learn

NLTK (Natural Language Toolkit)

Matplotlib & Seaborn (for visualization)

## Dataset

The dataset is a CSV file with two columns:

label → spam or ham

message → actual SMS text

Example:

label	message
ham	Hello how are you doing today?
spam	WINNER!! You have won a $1000 prize. Call now!

## How It Works

Import the dataset (spam.csv).

Preprocess the messages (cleaning, tokenization, removing stopwords).

Convert text into numerical features using TF-IDF Vectorizer.

Train the model using Naive Bayes Classifier.

Evaluate model accuracy.

Predict new incoming messages.

## Run the model:

python spam_classifier.py

Example prediction:

model.predict(["Congratulations! You won free entry in a contest."])

# Output: spam
📊 Expected Output

Accuracy: ~95% (depending on dataset size).

Classification report with precision, recall, and F1-score.

Graphs showing distribution of spam vs ham messages.
