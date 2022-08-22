#!/usr/bin/env python
# coding: utf-8

# ## NLP with Hotel Review pt. 2: Modeling
# *by Tyler Jones*

# In[825]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# Next, to import the cleaned test and train datasets for modeling:

# In[826]:


test_df = pd.read_csv('desktop/clean_test_dataframe.csv')
train_df = pd.read_csv('desktop/clean_train_dataframe.csv')


# In[827]:


test_df.head()


# Looking at the dataset, there are over 2700 unique, binarized words from positive and negative hotel reviews that will be used with the other key metrics in the dataset for modeling.

# #### Fitting a logistic regression model
# 
# In order to fit a logistic regression model, I will use the `LogisticRegression` function from `sklearn`.
# I've set the solver to `lbfgs`, of which the 'l' stands for 'limited memory.' Using this solver will save memory because it only stores the most recent few vectors in a logistic regression.
# 
# (Note: another option would be `statsmodels.api` instead of `LogisticRegression`, but I've chosen not to use that here. The former requires the manual addition of a constant when setting up the regression, whereas the latter does it automatically as part of its functionality.)

# In[828]:


from sklearn.linear_model import LogisticRegression
from sklearn import metrics

logreg = LogisticRegression(solver='lbfgs')

y_train = train_df['rating']
X_train = train_df.drop('rating', axis = 1)
y_test = test_df['rating']
X_test = test_df.drop('rating', axis = 1)


print(y_train.shape)
print(X_train.shape)


# The train and test sets are properly separated and defined as dependent and independent variables, so I can now begin the logistic regression.

# In[829]:


logreg_results = logreg.fit(X_train,y_train)


# Now that the logistic regression model has been set up, I will import `classification_report` from `sklearn`. This function is the equivalent of the `.summary` function with `statsmodels.api`, and will display the tested results of the logistic regression.

# In[830]:


predictions = logreg_results.predict(X_test)


# In[831]:


from sklearn.metrics import classification_report

classification_report(y_test, predictions)


# Per the classification report, the accuracy of the model is 0.72. This is confirmed below by running the `score` function:

# In[832]:


score = logreg_results.score(X_test, y_test)
print(f'The accuracy score of the model is {score.round(3)*100}%.')


# The accuracy of 71.8% means that the logistic regression model is moderately accurate at predicting positive and negative reviews. While this number is certainly significant, roughly 28% of reviews cannot be explained by this model.
# 
# ---

# #### Determining the words most predictive of positive and negative reviews

# In order to determine this, I will take the coefficients of regression for each variable and model it in a dataframe. Then, I will look through the dataframe and select the top 20 words from the positive and negative review columns.
# 
# Importantly, any word that is preceded by "p_" appeared in a positive review, and any word preceded by "n_" appeared in a negative review. This is important because the same words appear in both types of review, and the question is asking specifically which words from the positive review column and which words from the negative review column are most predictive of each review type.

# First, I need to set the dataframe parameters to display more than the default 10 rows when calling `.head`.

# In[833]:


pd.set_option('display.max_rows', 100)


# Next, I will create a dataframe using the coefficient values of logistic regression for each column.

# In[834]:


Xcols = X_train.columns
coefs = logreg.coef_
coef_df = pd.DataFrame(data = coefs, columns = Xcols)
coef_df.head()


# That's not easy to read, or to sort, so I will transpose it to a vertical dataframe.

# In[835]:


coef_df1 = coef_df.T
coef_df1.columns = ['Coefficient']


# Now, to get the negative words with the strongest coefficients in the model, I will sort the values in ascending order.

# In[836]:


coef_df1.sort_values(by = 'Coefficient', ascending = True).head(100)


# Finally, I'll perform the same function but sort the dataframe in descending order, so as to get the top positive coefficients of regression:

# In[837]:


coef_df1.sort_values(by = 'Coefficient', ascending = False).head(100)


# The top **negative predictors**:
# 1. Room
# 2. Small
# 3. Bed
# 4. Staff
# 5. Poor
# 6. Noisy
# 7. Shower
# 8. Work
# 9. Tire
# 10. Noise
# 11. Need
# 12. Air
# 13. Tiny
# 14. Date
# 15. Uncomfortable
# 16. Sleep
# 17. Clean
# 18. Water
# 19. Double
# 20. Night
# 
# The top **positive predictors**:
# 1. Staff
# 2. Excellent
# 3. Lovely
# 4. Great
# 5. Friendly
# 6. Helpful
# 7. Room
# 8. Everything
# 9. Hotel
# 10. Amaze
# 11. Comfortable
# 12. Fantastic
# 13. Comfy
# 14. Bed
# 15. Love
# 16. Perfect
# 17. Beautiful
# 18. Service
# 19. Stay
# 20. Really
# 
# Interestingly, there are several words that originated in negative reviews that have high coefficients for modeling positive reviews, and vice versa. This isn't necessariy surprising - after all, any review of a hotel room will likely revolve around key components of a hotel experience, such as "bed", "room", or "service" - but it could pose a problem later on.
# 
# ---

# #### PCA
# 
# The next step is to reduce the dimensionality of the dataset using Principal Component Analysis. There are advantages and disadvantages of doing this, which will be discussed in more detail after I run the PCA.
# 
# The first step of PCA is to scale the data. I will use a standard scaler to do this:

# In[838]:


from sklearn.preprocessing import MinMaxScaler

minmax = MinMaxScaler()

from sklearn.decomposition import PCA


# Next, setting up train and test sets for scaling:

# In[839]:


minmax.fit(X_train)
minmax.fit(X_test)
X_train = minmax.transform(X_train)
X_test = minmax.transform(X_test)


# In[840]:


my_PCA = PCA()
my_PCA.fit(X_train)
my_PCA.fit(X_test)

X_train_PCA = my_PCA.transform(X_train)
X_test_PCA = my_PCA.transform(X_test)


# The PCA has been set up. Because there is a ton of data, I'd like to start by determining just how much variance is actually captured by the top components in this PCA:

# In[841]:


print(f"Proportion of variance captured by PC1: {my_PCA.explained_variance_ratio_[0]: 0.3f}")
print(f"Proportion of variance captured by PC2: {my_PCA.explained_variance_ratio_[1]: 0.3f}")


# Wow, so the top component analyzed by the PCA only capture 4.5% of the variance in the model. This isn't especially high, but not necessarily unexpected for a model with thousands of binary variables, and where many of the variables that explain one outcome (positive reviews) also often appear in the inverse of that outcome (negative reviews).
# 
# I want to take a look at a graph that shows the variables that explain the most variance of the model:

# In[842]:


expl_var = my_PCA.explained_variance_ratio_


# In[843]:


plt.figure()
plt.plot(expl_var,marker='.')
plt.xlabel('Number of PCs')
plt.ylabel('Proportion of Variance Explained')
plt.xticks()
plt.show()


# This graph is a mess, and offers hardly any insight. The vast majority of inputs do little to explain the output, and those that do, explain very little.
# 
# The best way to proceed from here is to use a function that will return those variables which collectively explain 90% of the variance in the model. To do this, I will call a function that returns 0.9 of the components, and then transform the train and test sets to fit the Principal Component Analysis data:

# In[844]:


my_PCA = PCA(n_components = 0.9)
my_PCA.fit(X_train)
my_PCA.fit(X_test)

# Transform train and test
X_train_PCA = my_PCA.transform(X_train)
X_test_PCA = my_PCA.transform(X_test)


# In[845]:


print(f'Original: {X_train.shape}')
print(f'PCA Transformed: {X_train_PCA.shape}')


# In[846]:


1 - 1013/2743


# Shown above, the PCA narrowed the array from 2743 to 1013 variables.
# 
# For perspective, that means roughly 63% of the data only explained 10% of the model's variance. I would say this PCA did a great job of removing variables which seem to be irrelevant to the model.
# 
# Out of curiosity, I want to re-run the model with the updated PCA dataset to see if anything fundamentally changed.

# In[847]:


logreg_results = logreg.fit(X_train_PCA,y_train)


# In[848]:


predictions = logreg_results.predict(X_test_PCA)


# In[849]:


classification_report(y_test, predictions)


# It seems re-running logistic regression using the principal components of the data increase the accuracy of the model, from 72% to 78%.
# 
# ---

# The dimensionality reduction through PCA also reduced the run-time of the model, as shown below:

# In[850]:


## disabling warnings, there are quite a few when timing logistic regressions

import warnings
warnings.filterwarnings("ignore")


# In[851]:


get_ipython().run_cell_magic('timeit', '', '\nlogreg.fit(X_train,y_train)')


# In[852]:


get_ipython().run_cell_magic('timeit', '', 'logreg.fit(X_train_PCA,y_train)')


# In[853]:


print(f'It takes {(1 - 1.47/5.47)*100}% less time to run the reduced dataset.')


# It takes 73% less time to reduce the dimensionality of logistic regression to components that explain 90% of variance. This makes sense, as the number of components analyzed was reduced by over 50% after using PCA to reduce the dimensionality of the dataset.
# 
# ---

# #### KNN Modeling

# In[854]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

KNN_model = KNeighborsClassifier()
KNN_model.fit(X_train, y_train)

# Score the model on the test set
test_predictions = KNN_model.predict(X_test)
test_accuracy = accuracy_score(test_predictions, y_test)
print(f"Test set accuracy: {test_accuracy}")


# The accuracy score of the test set using KNN is 68%, which is lower than logistic regression was.

# #### Finding an optimal K value.
# 
# First, I will split the sampled data into a training and validation set. This is done so that a greater degree of confidence can be had in the model, which can later be measured against the test set for a totally unbiased measure of the model's performance.

# In[859]:


from sklearn.model_selection import train_test_split


# In[860]:


X_train, X_validation, y_train, y_validation =     train_test_split(X_tr_sample, y_tr_sample, test_size = 0.3,
                    random_state=1)


# Next, I'll create a `for` loop, which will run every possible K value through the dataset.
# 
# After all of the possible K values have been run through the loop, I'll call `np.argmax` to select the K value that returned the highest combined train and validation set accuracy.

# In[861]:


neighbors = range(1, len(X_train)+1, 2)

train_acc = []
validation_acc = []

for n in neighbors: 
    print(f"Model is working on {n} neighbors...", end="\r")
    
    #Instantiate and Fit
    KNN = KNeighborsClassifier(n_neighbors=n)
    KNN.fit(X_train, y_train)
    
    
    #Score the model
    train_accuracy = KNN.score(X_train, y_train)
    validation_accuracy = KNN.score(X_validation, y_validation)
    
    
    #Append my accuracy
    train_acc.append(train_accuracy)
    validation_acc.append(validation_accuracy)


# In[862]:


plt.figure()
plt.plot(neighbors, validation_acc, color="red", label="test")
plt.plot(neighbors, train_acc, color="blue", label="train")
plt.ylabel("Accuracy Score")
plt.xlabel("Number of neighbors")
plt.title("KNN Graph")
plt.legend()
plt.show()


# As shown by the graph above, there is massive overfitting that quickly levels out sometime before 100 neighbors are reached. The function below will show the maximum level of test accuracy between the training and validation sets.

# In[863]:


index_of_max = np.argmax(validation_acc)

best_k = neighbors[index_of_max]

best_k


# And the best K value, per the `for` loop, is 41.

# Now to check the accuracy of the model with the optimum 41 neighbors:

# In[866]:


KNN_model = KNeighborsClassifier(n_neighbors=41)
KNN_model.fit(X_train, y_train)

# Score the model on the test set
test_predictions = KNN_model.predict(X_validation)
test_accuracy = accuracy_score(test_predictions, y_validation)
print(f"Test set accuracy: {test_accuracy}")


# ---
# 
# #### Fitting a Decision Tree Model
# 
# To begin the decision tree, I will employ a Decision Tree Classifier to the dataset.

# In[867]:


from sklearn.tree import DecisionTreeClassifier

DT_model = DecisionTreeClassifier()


# In[869]:


X_train = train_df.drop('rating', axis = 1)
y_train = train_df['rating']
X_test = test_df.drop('rating', axis = 1)
y_test = test_df['rating']


# In[870]:


DT_model.fit(X_train, y_train)

print(f"DT training set accuracy: {DT_model.score(X_train, y_train)}")


# As usual, the initial fit decision tree model has perfect accuracy. The reason for that is that the model overfits beyond belief, making decisions which perfectly explain every point in the dataset, but are poorly predictive of new data added to the model.
# 
# The entire tree is shown below, and it is impossible to read for that reason.

# In[871]:


from sklearn.tree import plot_tree

plt.figure()
clf = DT_model.fit(X_train, y_train)
plot_tree(clf, filled=True)
plt.title("Decision tree")
plt.show()


# #### optimal `max_depth` value.
# 
# The `max_depth` of a decision tree is the attribute which determines how far a "leaf" (dependent datapoint) can be from a "root" (the independent variable). The theoretical maximum depth of a decision tree is the length of the dataset, but this of course causes massive overfitting. By choosing a `max_depth` value which is smaller, it will cause the model to have much better predictive capabilities.

# In[897]:


X_tr_sample = X_train.sample(frac = .25, random_state = 1)
y_tr_sample = y_train.sample(frac = .25, random_state = 1)
X_te_sample = X_train.sample(frac = .25, random_state = 1)
y_te_sample = y_train.sample(frac = .25, random_state = 1)


# In[898]:


X_train, X_validation, y_train, y_validation =     train_test_split(X_tr_sample, y_tr_sample, test_size = 0.3,
                    random_state=1)


# In[899]:


train_acc = []
validation_acc = []

depth = range(1, len(X_train)+1)

for i in depth:
    
    print(f"Model is working on {i} depth...", end="\r")
    
    my_dt = DecisionTreeClassifier(max_depth = i)
    my_dt.fit(X_train,y_train)
    
    train_acc.append(my_dt.score(X_train,y_train))
    validation_acc.append(my_dt.score(X_validation,y_validation))


# In[876]:


index_of_max = np.argmax(validation_acc)

best_depth = neighbors[index_of_max]

print(f'The best max depth value for the Decision Tree Model is {best_depth}.')


# In[877]:


DT_model = DecisionTreeClassifier(max_depth = 7)
DT_model.fit(X_train, y_train)

print(f"DT training set accuracy: {DT_model.score(X_train, y_train)}")

