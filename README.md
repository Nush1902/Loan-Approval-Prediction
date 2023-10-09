# Loan-Approval-Prediction
A project based on two widely used Data Science algorithms

### Project Description

The project's main objective is to predict whether a loan application will be approved or not based on various factors related to loan applicants. These factors include gender, marital status, education, income, loan amount, credit history, and more. The code reads in a dataset containing this applicant information and performs the following steps:

1. Data Loading and Exploration:
   - The code uses the Pandas library to load the dataset from a CSV file (`loan-train.csv`) located on the local machine.

2. Data Preprocessing:
   - Handling missing values: The code fills missing values in categorical variables with their respective modes and in numerical variables with the mean value.
   - Data transformation: It normalizes the loan amount using the log transformation and computes the total income by summing the applicant's and co-applicant's income. Total income is also normalized using the log transformation.

3. Data Encoding:
   - The code encodes categorical variables using Label Encoding to convert them into numerical form, which is required for machine learning algorithms.

4. Data Splitting:
   - The dataset is split into training and testing sets using the `train_test_split` function from scikit-learn.

5. Model Building and Evaluation:
   - Two machine learning algorithms are trained and evaluated:
     - Decision Tree Classifier (`DecisionTreeClassifier`): It uses the Decision Tree algorithm with the entropy criterion to classify loan approval status.
     - Naive Bayes Classifier (`GaussianNB`): It uses the Naive Bayes algorithm for classification.

6. Model Evaluation:
   - The accuracy of each classifier is calculated and printed as the result. Accuracy is a measure of how well the model predicts loan approval status.

### Algorithms Used

1. **Decision Tree Classifier**:
   - Decision Tree is a supervised machine learning algorithm used for classification tasks.
   - In this project, it is trained using the entropy criterion to build a decision tree that predicts loan approval.

2. **Naive Bayes Classifier (GaussianNB)**:
   - Naive Bayes is a probabilistic classification algorithm based on Bayes' theorem.
   - The Gaussian Naive Bayes variant is used when the features are continuous, as in this case.

### Results

The code prints and displays the accuracy of both the Decision Tree and Naive Bayes classifiers. The accuracy score indicates the percentage of correct predictions made by each classifier on the testing data.

For example:
- "The accuracy of the Decision Tree is: 70.73% " and "The accuracy of NB is: 82.92% " are printed to the console, showing the accuracy of each classifier.

The results provide insights into how well these algorithms perform in predicting loan approval status based on the given dataset. Further evaluation metrics and analysis could be added to enhance the project's insights.
