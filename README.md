For the most part I was doing trial and error for most of the models' parameters,
and then I used a for loop to test a bunch of different random states and I used the best one. The for loop I used to optimize is commented out below the models for reference.

For both the Random Forest and Logistic Regression I was able to get accuracy, precision, recall, and f1 score to 1.0

For the logistic regression I had to make new variables for the data in which I scaled the X using the StandardScaler() class from SciKitLearn preprocessing.

I wasn't able to get the decision tree to test perfectly but I got the recall to 1.0 and all the others above 0.97.

Im assuming that the perfect scores would no longer be perfect if I tested on more data and they mostly just found a specific random_state that makes it perfect for this specific training and test data.
