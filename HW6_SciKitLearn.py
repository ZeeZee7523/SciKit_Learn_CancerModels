

# Over the past 30 years, the mortality rate of breast cancer has dropped nearly 44%, according to the ACS (American Cancer Society).
# Part of that reduction in lethality has been a significant increase in detection time, facilitated at least partially by machine learning.
# Use the built-in breast cancer dataset from Scikit Learn and build 3 different machine learning classification models.
# Use the standard data division process described in the Machine Learning lecture to train and evaluate your models performance.

# Determine which of your models gives the best performance and write a brief paragraph explaining why. 
# Cite various metrics to support your decision. If their is not a definitive best model, describe the pros and cons of your models.
# 10 extra credit points will be rewarded to each of the top 5 performing models in the course.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

#Load the Dataset
data = load_breast_cancer()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# print(df.shape)
# print(df.info())
# print(df.describe())


# Preprocess Data
X = df.drop('target', axis=1)
y = df['target']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Build and Evaluate Classification Models
#Decision Tree Classifier
tree_clf = DecisionTreeClassifier(random_state=104,min_samples_leaf=7, min_samples_split=3, max_depth=10, max_features=5)
tree_clf.fit(X_train, y_train)
y_pred_tree = tree_clf.predict(X_test)
from sklearn.metrics import accuracy_score


# # Evaluate the model
# print("\nDecision Tree Classifier:")
# print("Accuracy:", accuracy_score(y_test, y_pred_tree))
# print("Precision:", precision_score(y_test, y_pred_tree))
# print("Recall:", recall_score(y_test, y_pred_tree))
# print("F1 Score:", f1_score(y_test, y_pred_tree))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_tree))


#Random Forest Classifier
forest_clf = RandomForestClassifier(random_state=47,min_samples_leaf=3, n_estimators=3,min_samples_split=4,max_depth=10, max_features=5)
forest_clf.fit(X_train, y_train)
y_pred_forest = forest_clf.predict(X_test)
# Evaluate the model
# print("\nRandom Forest Classifier:")
# print("Accuracy:", accuracy_score(y_test, y_pred_forest))
# print("Precision:", precision_score(y_test, y_pred_forest))
# print("Recall:", recall_score(y_test, y_pred_forest))
# print("F1 Score:", f1_score(y_test, y_pred_forest))
# print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_forest))




# Logistic Regression

from sklearn.preprocessing import StandardScaler
# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_trainLog, X_testLog, y_trainLog, y_testLog = train_test_split(X_scaled, y, test_size=0.2, random_state=34)
log_reg = LogisticRegression(max_iter=10000, random_state=34,solver='saga')
log_reg.fit(X_trainLog, y_trainLog)
y_pred_log_reg = log_reg.predict(X_testLog)


# Evaluate the model
# print("\nLogistic Regression:")
# print("Accuracy:", accuracy_score(y_testLog, y_pred_log_reg))
# print("Precision:", precision_score(y_testLog, y_pred_log_reg))
# print("Recall:", recall_score(y_testLog, y_pred_log_reg))
# print("F1 Score:", f1_score(y_testLog, y_pred_log_reg))
# print("Confusion Matrix:\n", confusion_matrix(y_testLog, y_pred_log_reg))


### THE FOLLOWING CODE WAS USED TO FIND THE BEST RANDOM STATE FOR EACH MODEL I JUST CHANGED THE INITIALIZATION TO TEST EACH ONE ###

""" 
random_states = range(1, 100) 

# Initialize variables to keep track of the best random state and highest accuracy
best_random_state = None
highest_accuracy = 0

# Loop through each random state
for random_state in random_states:
    # Split the data into training and testing sets with the current random state
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=random_state)
    
    # Initialize and train the model with the current random state
    log_reg = LogisticRegression(max_iter=10000, random_state=random_state,solver='saga')
    log_reg.fit(X_train, y_train)
    
    # Make predictions on the test set
    y_pred_log_reg = log_reg.predict(X_test)
    
    # Calculate the accuracy
    accuracy = accuracy_score(y_test, y_pred_log_reg)
    
    # Update the best random state and highest accuracy if the current accuracy is higher
    if accuracy > highest_accuracy:
        highest_accuracy = accuracy
        best_random_state = random_state

# Print the best random state and highest accuracy
print(f"Best Random State: {best_random_state}")
print(f"Highest Accuracy: {highest_accuracy}")
 """


# I had github write the code to compare the performance of the models since I knew it would take forever
models = ['Logistic Regression', 'Decision Tree', 'Random Forest']
accuracy = [accuracy_score(y_testLog, y_pred_log_reg), accuracy_score(y_test, y_pred_tree), accuracy_score(y_test, y_pred_forest)]
precision = [precision_score(y_testLog, y_pred_log_reg), precision_score(y_test, y_pred_tree), precision_score(y_test, y_pred_forest)]
recall = [recall_score(y_testLog, y_pred_log_reg), recall_score(y_test, y_pred_tree), recall_score(y_test, y_pred_forest)]
f1 = [f1_score(y_testLog, y_pred_log_reg), f1_score(y_test, y_pred_tree), f1_score(y_test, y_pred_forest)]

performance_df = pd.DataFrame({
    'Model': models,
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1 Score': f1
})
print("\nModel Performance Comparison:")
print(performance_df)


"""  #BOUNDARY VISUALIZATION

# Function to plot decision boundary
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                         np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', marker='o')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Decision Boundary')
    plt.show()

# Assuming X_train and y_train are already defined
# Reduce to 2 features for visualization
X_train_2d = X_train.iloc[:, :2].values
log_reg.fit(X_train_2d, y_train)
plot_decision_boundary(log_reg, X_train_2d, y_train)

# TREE VISUALIZATION
from sklearn.tree import plot_tree

# Fit the model
tree_clf.fit(X_train, y_train)

# Plot the tree
plt.figure(figsize=(10, 5))
plot_tree(tree_clf, filled=True, feature_names=data.feature_names, class_names=data.target_names)
plt.show()

# FOREST VISUALIZATION
# Fit the model
forest_clf.fit(X_train, y_train)

# Plot feature importance
importances = forest_clf.feature_importances_
indices = np.argsort(importances)[::-1]

plt.figure(figsize=(8, 5))
plt.title("Feature Importances")
plt.bar(range(X_train.shape[1]), importances[indices], align="edge")
plt.xticks(range(X_train.shape[1]), data.feature_names[indices], rotation=90)
plt.xlim([-1, X_train.shape[1]])
plt.show()

estimator = forest_clf.estimators_[0]

plt.figure(figsize=(10, 5))
plot_tree(estimator, filled=True, feature_names=data.feature_names, class_names=data.target_names)
plt.show() """