


# Importing necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")



# Reading the dataset
df = pd.read_excel("default of credit card clients.xls", skiprows=1)



# viewing the first five rows of the dataset
df.head()


# viewing the last five rows of the dataset
df.tail()



# Transposing the dataframe to easily view all the rows
df.head().T



# checking the data type of the dataset
df.dtypes



# Checking the Null values in the dataset
df.isnull().sum()



# Turning the columns into lower and replace the space in the 'default payment next month'
df.columns = df.columns.str.lower().str.replace(" ", "_")



df.marriage.unique()



# mapping the categorical data to its respective classes
sex_values = {1: "male", 2: "female"}
df.sex = df.sex = df.sex.map(sex_values)



education_values = {1: "graduate school", 2: "university", 3: "high school", 4: "others", 5: "unk", 6: "unk", 0: "unk"}
df.education = df.education.map(education_values)


marital_status_values = {1: "married", 2: "single", 3: "others", 0: "unk"}
df.marriage = df.marriage.map(marital_status_values)


# Some columns are not neccessarily numerical columns but categorical
categorical_columns_list = ['pay_0', 'pay_2', 'pay_3', 'pay_4', 'pay_5', 'pay_6']
for col in categorical_columns_list:
    df[col] = df[col].astype("O")



categorical_columns = list(df.dtypes[df.dtypes == "object"].index)



# checking for duplicated values
df[df.duplicated()]


# dropping the unnecessary id column
df.drop("id", axis=1, inplace=True)



# removing the "unk" class
df = df[df.education != "unk"].reset_index(drop=True)


df = df[df.marriage != "unk"].reset_index(drop=True)


# ### Exploratory Data Analysis


# Checking all the numerical columns
numerical_cols = df.select_dtypes("int64").columns
numerical_cols


# Age seems to have the highest correlation with default payment
df[numerical_cols].corr()["default_payment_next_month"].sort_values(ascending = False)



# Numerical column without the target varible
target_excluded = numerical_cols[:-1]



# Checking the distribution of the numerical columns
# Some of the columns are skewed
df[target_excluded].hist(figsize=[20,10]);


# From the visualization above looks like there are skewed column in the dataset, this function will check their skewness

def skew_function():
    
    """ I want to pull out all the skewed columns in the dataframe"""
    
    skew_value = []
    
    for col in target_excluded:
        skew_col = df[col].skew()
        skew_value.append((col , skew_col))
    return sorted(skew_value , key = lambda x : x[1] , reverse = True)

skew_function()


skewed_cols = ['pay_amt2','pay_amt3', 'pay_amt1', 'pay_amt4', 'pay_amt5', 'pay_amt6', 'bill_amt3', 'bill_amt5', 'bill_amt6', 'bill_amt4', 'bill_amt2', 'bill_amt1']


df[skewed_cols].isnull().sum()

# Transforming the skewed columns using np.log
df[skewed_cols] = df[skewed_cols].apply(lambda x : np.log(x+1))



# The transformation of the skewed columns gave nan values for zeros in the column


# #### Identifying Outliers in the columns
# Columns with extreme values

def outliers_check():
        cols_store = []
    
        for col in target_excluded: 
            q1 = df[col].quantile(.25)
            q3 = df[col].quantile(.75)
            iqr = q3 - q1
            lower_out= (df[col] < (q1 - (1.5*iqr))).sum()
            upper_out = (df[col]  > (q3 + (1.5*iqr))).sum()

            if lower_out > 0 or upper_out > 0 :
                cols_store.append(col)

        return cols_store



outlier_cols = outliers_check()
outlier_df = df[outlier_cols]


# In[32]:


# there are null values introduced as a result of skewness transformation
df.isnull().sum()


# In[33]:


df.head()


# In[ ]:





# In[34]:


# filling the nan values that was introduced after skewness transformation
# for col in df.columns:
#     df[col] = df[col].fillna(df[col].median())


# In[35]:


df.describe()


# In[36]:


outlier_df.isnull().sum()


# In[37]:


outlier_df = outlier_df.replace([np.inf, -np.inf], np.nan)
outlier_df = outlier_df.fillna(0)
outlier_df.reset_index(drop=True)


# In[38]:


outlier_df.describe()


# In[39]:


# the standard scaler module from the scikit learn library helps to normalize values
from sklearn.preprocessing import StandardScaler


# In[40]:


# normalizing the outlier columns
scaler = StandardScaler()
scaled = scaler.fit_transform(outlier_df)


# In[41]:


outlier_df = pd.DataFrame(scaled, columns=outlier_df.columns)


# In[42]:


outlier_df


# In[43]:


df[outlier_cols] = outlier_df


# In[44]:


# Making sure the data does not contain, missing values or infinity values
df.describe()


# #### Feature Importance
# Mutual Information concept from information theory, it tells us how much we can learn about one varible if we know the value of another.

# In[46]:


# Using mutual information from sklearn library
from sklearn.metrics import mutual_info_score


# In[47]:


# checking the mutual score for all categorical variables
def mutual_info_default_payment_score(series):
    return mutual_info_score(series, df.default_payment_next_month)


# In[48]:


df


# In[49]:


# Applying the mutual info score to all categorical variables
# We can see that History of past payment rank higher than others
df[categorical_columns].apply(mutual_info_default_payment_score).sort_values(ascending=False)


# ### Modelling Training

# In[50]:


# splitting the dataset into train/validation/ test data
from sklearn.model_selection import train_test_split

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)


# In[51]:


# reseting the index of the dataframes
df_train = df_train.reset_index(drop=True)
df_val = df_val.reset_index(drop=True)
df_test = df_test.reset_index(drop=True)


# In[52]:


# Assigning the outputs
y_train = df_train.default_payment_next_month
y_val = df_val.default_payment_next_month
y_test = df_test.default_payment_next_month


# In[53]:


del df_train["default_payment_next_month"]
del df_val["default_payment_next_month"]
del df_test["default_payment_next_month"]


# #### Decision Tree

# In[57]:


from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import roc_auc_score


# In[58]:


# Turning the training data into dictionary
train_dicts = df_train.to_dict(orient="records")


# In[59]:


# Vectorizing the dictionary
dv = DictVectorizer(sparse=False)
X_train = dv.fit_transform(train_dicts)


# In[60]:


# Training the model
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)


# In[61]:


# Testing the model
val_dicts = df_val.to_dict(orient="records")
X_val = dv.transform(val_dicts)


# In[66]:


# Prediction
y_pred = dt.predict_proba(X_val)[:, 1]
y_pred


# In[67]:


# computing the auc score for testing dataset
roc_auc_score(y_val, y_pred)


# In[59]:


# limiting the dept of the tree to 3
dt =  DecisionTreeClassifier(max_depth=3)
dt.fit(X_train, y_train)


# In[60]:


y_pred = dt.predict_proba(X_train)[:,1]
auc = roc_auc_score(y_train, y_pred)
print("train:", auc)

y_pred = dt.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred)
print("val:", auc)


# In[61]:


# Visualing the tree
from sklearn.tree import export_text


# In[64]:


export_text(dt, feature_names=dv.get_feature_names_out())


# ### Decision trees parameter tuning
# - selecting max_depth
# - selecting min_samples_leaf

# In[62]:


# Getting the appropriate max_depth
for d in [1, 2, 3, 4, 5, 6, 7, 8, 10, 15, 20, None]:
    dt = DecisionTreeClassifier(max_depth=d)
    dt.fit(X_train, y_train)

    y_pred = dt.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_pred)

    print("%4s  -> %.3f" % (d, auc))



# In[63]:


# Getting the appropriate min sample leaf
scores = []

for d in [4, 5, 6, 7, 10, 15, 20, None]:
    for s in [1, 2, 5, 10, 15, 20, 100, 200, 500]:
        dt = DecisionTreeClassifier(max_depth=d, min_samples_leaf=s)
        dt.fit(X_train, y_train)

        y_pred = dt.predict_proba(X_val)[:, 1]
        auc = roc_auc_score(y_val, y_pred)

        scores.append((d, s,  auc))


# In[64]:


# Creating a DataFrame of the scores
columns = ["max_depth", "min_samples_leaf", "auc"]
df_scores = pd.DataFrame(scores, columns=columns)
df_scores.tail(20)


# In[65]:


df_scores.sort_values(by="auc", ascending=False).head()


# In[66]:


# The min_samplesLeaf of 200 seems to be the best, I will pick 15 max depth
dt = DecisionTreeClassifier(max_depth= 15, min_samples_leaf= 200)
dt.fit(X_train, y_train)


# ### Random forest
# 
# - Random forest - ensembling decision trees
# - turning random forest

# In[67]:


from sklearn.ensemble import RandomForestClassifier


# In[68]:


rf = RandomForestClassifier(n_estimators= 100)
rf.fit(X_train, y_train)


# In[69]:


y_pred = rf.predict_proba(X_val)[:, 1]
roc_auc_score(y_val, y_pred)


# In[70]:


# iterating over different values of estimators
scores = []
for n in range(10, 201, 10):
    rf = RandomForestClassifier(n_estimators=n, random_state=1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict_proba(X_val)[:,1]
    auc = roc_auc_score(y_val, y_pred)
    scores.append((n, auc))


# In[71]:


df_score = pd.DataFrame(scores, columns= ["n_estimators", "auc"])
df_score


# In[73]:


# 200 estimators seems to be good

# iterating over different values of depths
scores = []

for d in [5, 10, 15]:
    for n in range(10, 201, 10):
        rf = RandomForestClassifier(n_estimators=n,
                                    max_depth = d,
                                    random_state=1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict_proba(X_val)[:,1]
        auc = roc_auc_score(y_val, y_pred)
        scores.append((d, n, auc))


# In[74]:


column= ["max_depth", "n_estimators", "auc"]
df_score = pd.DataFrame(scores, columns= column)
df_score


# In[ ]:





# In[75]:


# getting the best min leave parameter 

scores = []

for s in [1, 3, 5, 10, 50]:
    for n in range(10, 201, 10):
        rf = RandomForestClassifier(n_estimators=n,
                                    max_depth = 10,
                                    min_samples_leaf= s,
                                    random_state=1)
        rf.fit(X_train, y_train)
        y_pred = rf.predict_proba(X_val)[:,1]
        auc = roc_auc_score(y_val, y_pred)
        scores.append((s, n, auc))


# In[76]:


# the min_samples_leaf does not affect the performance of the model in this
# instance, I will go with 1
column= ["min_samples_leaf", "n_estimators", "auc"]
df_score = pd.DataFrame(scores, columns= column)
df_score.tail(30)


# In[77]:


# 10 max_depth, 50 n_estimators and min_sample_leaf 1 gives the best result

rf = RandomForestClassifier(n_estimators= 50,
                            max_depth = 10,
                            min_samples_leaf= 1,
                            random_state=1,
                           n_jobs=-1)
rf.fit(X_train, y_train)


# ### Gradient boosting and XGBoost
# 
# - Performance monitoring
# - Parsing xgboost's monitoring output

# In[78]:


import xgboost as xgb


# In[79]:


# wrap the data into special data structure called DMatrix(optimized to train xgboost faster)
features = dv.feature_names_
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names= features)
dval = xgb.DMatrix(X_val, label=y_val, feature_names= features)


# In[80]:


# setting xgboost parameters, training the model
xgb_params ={
    "eta": 0.3,
    "max_depth": 6,
    "min_child_weight": 1,
    "objective": "binary:logistic",
    "nthread": 8,
    "seed": 1,
    "verbosity": 2,
}

model = xgb.train(xgb_params, dtrain, num_boost_round=10)


# In[81]:


y_pred = model.predict(dval)


# In[82]:


roc_auc_score(y_val, y_pred)


# In[83]:


# We can evaluate the data on validation dataset after each tree is trained
watchlist = [(dtrain, "train"), (dval, "val")]


# In[84]:


get_ipython().run_cell_magic('capture', 'output', '# setting xgboost parameters\nxgb_params ={\n    "eta": 0.3,\n    "max_depth": 6,\n    "min_child_weight": 1,\n    "objective": "binary:logistic",\n    "eval_metric": "auc",\n    "nthread": 8,\n    "seed": 1,\n    "verbosity": 2,\n}\n\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=200,\n                  verbose_eval=5,\n                  evals=watchlist)\n')


# In[85]:


print(output.stdout)


# In[86]:


def parse_xgb_output(output):
    results = []

    for line in output.stdout.strip().split("\n"):
        it_line, train_line, val_line = line.split("\t")

        it = int(it_line.strip("[]"))
        train = float(train_line.split(":")[1])
        val = float(val_line.split(":")[1])

        results.append((it, train, val))

    columns = ["num_iter", "train_auc", "val_auc"]
    df_results = pd.DataFrame(results, columns=columns)
    return df_results


# In[87]:


df_score = parse_xgb_output(output)


# In[88]:


# plotting training against validation
plt.plot(df_score.num_iter, df_score.train_auc, label="train")
plt.plot(df_score.num_iter, df_score.val_auc, label="val")

plt.legend()


# In[89]:


plt.plot(df_score.num_iter, df_score.val_auc, label="val")


# ### XGBOOST parameter turning
# 
# Tuning the following parameters:
# - `eta`
# - `max`
# - `min_child_weight`

# In[93]:


scores = {}


# In[100]:


get_ipython().run_cell_magic('capture', 'output', '# setting xgboost parameters\nxgb_params ={\n    "eta": 0.1,\n    "max_depth": 6,\n    "min_child_weight": 1,\n    "objective": "binary:logistic",\n    "eval_metric": "auc",\n    "nthread": 8,\n    "seed": 1,\n    "verbosity": 2,\n}\n\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=200,\n                  verbose_eval=5,\n                  evals=watchlist)\n')


# In[ ]:


etas = ["eta=1.0", "eta=0.3", "eta=0.1"]


# In[101]:


key = "eta=%s" % (xgb_params["eta"])
scores[key] = parse_xgb_output(output)


# In[102]:


scores.keys()


# In[103]:


# Plotting the eta's more accurately
etas = ["eta=1.0", "eta=0.3", "eta=0.1"]

for eta in etas:
    df_score = scores[eta]
    plt.plot(df_score.num_iter, df_score.val_auc, label=eta)
plt.legend()


# In[107]:


# Tuning max_depth
scores = {}
max_depth = [3, 4, 5, 10]


# In[117]:


get_ipython().run_cell_magic('capture', 'output', '# setting xgboost parameters\nxgb_params ={\n    "eta": 0.3,\n    "max_depth": 10,\n    "min_child_weight": 1,\n    "objective": "binary:logistic",\n    "eval_metric": "auc",\n    "nthread": 8,\n    "seed": 1,\n    "verbosity": 2,\n}\n\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=200,\n                  verbose_eval=5,\n                  evals=watchlist)\n')


# In[118]:


key = "max_depth=%s" % (xgb_params["max_depth"])
scores[key] = parse_xgb_output(output)


# In[119]:


scores.keys()


# In[121]:


# Plotting the max_depth's more accurately
etas = ["eta=1.0", "eta=0.3", "eta=0.1"]

for max_depth, df_score in scores.items():
    plt.plot(df_score.num_iter, df_score.val_auc, label=max_depth)
#plt.ylim(0.76, 0.80)
plt.legend()


# In[126]:


# Tuning the min_child_weight
scores = {}
min_child_weight = [1, 5, 10, 20, 30]


# In[139]:


get_ipython().run_cell_magic('capture', 'output', '# setting xgboost parameters\nxgb_params ={\n    "eta": 0.1,\n    "max_depth": 4,\n    "min_child_weight": 30,\n    "objective": "binary:logistic",\n    "eval_metric": "auc",\n    "nthread": 8,\n    "seed": 1,\n    "verbosity": 2,\n}\n\nmodel = xgb.train(xgb_params, dtrain, num_boost_round=200,\n                  verbose_eval=5,\n                  evals=watchlist)\n')


# In[140]:


key = "min_child_weight=%s" % (xgb_params["min_child_weight"])
scores[key] = parse_xgb_output(output)


# In[141]:


scores.keys()


# In[143]:


# Plotting the max_depth's more accurately
etas = ["eta=1.0", "eta=0.3", "eta=0.1"]

for min_child_weight, df_score in scores.items():
    plt.plot(df_score.num_iter, df_score.val_auc, label=min_child_weight)
plt.ylim(0.70, 0.77)
plt.legend()


# In[ ]:





# In[144]:


# Final model
xgb_params ={
    "eta": 0.3,
    "max_depth": 4,
    "min_child_weight": 30,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "nthread": 8,
    "seed": 1,
    "verbosity": 2,
}

model = xgb.train(xgb_params, dtrain, num_boost_round=10)


# ### Selecting the final model
# - Choosing between xgboost, random forest and decision tree
# - Training the final model
# - saving the model

# In[145]:


# The min_samplesLeaf of 200 seems to be the best, I will pick 15 max depth
dt = DecisionTreeClassifier(max_depth= 15, min_samples_leaf= 200)
dt.fit(X_train, y_train)


# In[146]:


# Performance of Decision tree
y_pred = dt.predict_proba(X_val)[:, 1]
roc_auc_score(y_val, y_pred)


# In[147]:


# 10 max_depth, 50 n_estimators and min_sample_leaf 1 gives the best result

rf = RandomForestClassifier(n_estimators= 50,
                            max_depth = 10,
                            min_samples_leaf= 1,
                            random_state=1,
                           n_jobs=-1)
rf.fit(X_train, y_train)


# In[148]:


# Performance of Random Forest
y_pred = rf.predict_proba(X_val)[:, 1]
roc_auc_score(y_val, y_pred)


# In[149]:


# Xgboost
xgb_params ={
    "eta": 0.3,
    "max_depth": 4,
    "min_child_weight": 30,
    "objective": "binary:logistic",
    "eval_metric": "auc",
    "nthread": 8,
    "seed": 1,
    "verbosity": 2,
}

model = xgb.train(xgb_params, dtrain, num_boost_round=10)


# In[150]:


# Performance of Xgboost
y_pred = model.predict(dval)
roc_auc_score(y_val, y_pred)


# In[68]:


import pickle


# In[156]:


# Picking Random Forest as the final model
with open("model.bin", "wb") as f_out:
    pickle.dump((dv,  rf), f_out)


# ### Load model

# In[69]:


with open("model.bin", "rb") as f_in:
    dv, rf = pickle.load(f_in)


# In[70]:


dv, rf


# In[81]:


customer = df_train.iloc[20]


# In[82]:


customer = customer.to_dict()


# In[83]:


X = dv.transform([customer])


# In[84]:


y_pred = rf.predict_proba(X)[0, 1]


# In[85]:


# This research employed a binary variable, default payment (Yes = 1, No = 0), as the response variable.
if y_pred < 0.5:
    print("This customer is not likely to default Payment")
else:
    print("This customer is likely to default Payment")


# In[86]:


def predict(data):
    customer = data.to_dict()
    X = dv.transform([customer])
    y_pred = rf.predict_proba(X)[0, 1]
    
    if y_pred < 0.5:
        return "This customer is not likely to default Payment"
    else:
        return "This customer is likely to default Payment"


# In[87]:


predict(df_train.iloc[20])


# In[88]:


df_train.iloc[20]


# In[ ]:





# In[ ]:





# In[ ]:




