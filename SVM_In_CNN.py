#!/usr/bin/env python
# coding: utf-8

# ## Problem Statement:
# ### Pedicting effective treatments  for diabetes in turn reducing the readmission into the hospital

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


# In[2]:


data = pd.read_csv('diabetic_data.csv')
data.shape


# # Data Preparation

# In[3]:


data.columns


# In[4]:


data.info()


# In[5]:


data.isnull().values.any()


# In[6]:


data.race.value_counts().plot(kind = 'bar' )


# In[7]:


data.payer_code.value_counts().plot(kind = 'bar' )


# In[8]:


data.medical_specialty.value_counts()


# In[9]:


data.max_glu_serum.value_counts().plot(kind = 'bar' )


# In[10]:


data.A1Cresult.value_counts().plot(kind = 'bar' )


# In[11]:


data.change.value_counts().plot(kind = 'bar' )


# In[12]:


data.diabetesMed.value_counts().plot(kind = 'bar' )


# In[13]:


data.readmitted.value_counts().plot(kind = 'bar' )


# In[14]:


data.age.value_counts().plot(kind = 'bar')


# ## Filtering patients with Diabetes
# ### diabetesMed = Yes

# In[15]:


data=data[data.diabetesMed=='Yes']
data.shape


# ## Filtering patients who didn't readmit
# ### readmission = NO

# In[16]:


data=data[data.readmitted=='NO']
data.shape


# ## Excluding patients who are Dead and are in hospise

# In[17]:


data=data[~data.discharge_disposition_id.isin([11,13,14,19,20])]
data.shape


# # Handling Missing Values

# ### We can observe that, Payer code, medical speciality & weight have more than 50% of the missing data, and prefer to drop those features.

# In[18]:


data = data.drop(['medical_specialty','payer_code','weight'],axis=1)


# **We can observe that the "Race" Feature has some missing values**

# **Missing value Imputation using MODE for Race Feature as most of the people in the Dataset are Caucasian**

# ##### 1. Replacing the ? with NaN's

# In[19]:


data['race']=data.race.replace('?',np.nan)


# ##### 2. Filling the NaN's with the mode

# In[20]:


data['race'].fillna(data['race'].mode()[0], inplace=True)


# In[21]:


data.race.isnull().sum()


# In[22]:


data.shape


# In[23]:


data.columns


# In[24]:


treatments = data[['encounter_id','metformin', 'repaglinide', 'nateglinide',
       'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide',
       'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
       'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton',
       'insulin', 'glyburide-metformin', 'glipizide-metformin',
       'glimepiride-pioglitazone', 'metformin-rosiglitazone',
       'metformin-pioglitazone']].copy()


# In[25]:


treatments.head()


# # Feature Engineering

#  ### Custom encoding for the 23 Drug Features
# 

# In[26]:


treatments=treatments.replace(['No','Steady','Up','Down'],[0,1,1,1])
treatments.set_index('encounter_id',inplace=True)


# In[27]:


treatments.head()


# In[28]:


treatments.sum(axis=1).value_counts()


# # Patients are Given at max a combination of 6 drugs for treating diabetes

# ### Feature Engineering - Creating a new feature "Treatments"

# **1. When the value of Insuin is '1' , creating the classes "insulin" & "io" (insulin + others )********

# In[29]:


i1 = treatments[treatments['insulin']==1].sum(axis = 1).replace([1,2,3,4,5,6],['insulin','io','io','io','io','io'])


# In[30]:


i1.value_counts()


# **2. When the value of Insuin is '0' , creating the classes "others" & "no med"**

# In[31]:


i0=treatments[treatments['insulin']==0].sum(axis=1).replace([0,1,2,3,4,5,6],['no med','other','other','other','other','other','other'])


# In[32]:


i0.value_counts()


# In[33]:


treatments=pd.concat([i1,i0])
treatments = pd.DataFrame({'treatments':treatments})


# In[34]:


treatments.head()


# **Adding the new feature to the Actual Dataframe**

# In[35]:


data=data.join(treatments,on='encounter_id') #setting index as encounter_id


# In[36]:


data.head()


# ## Since the treatments column was created from the 23 Drugs, We will be removing them

# In[37]:


data = data.drop(['metformin', 'repaglinide', 'nateglinide',
       'chlorpropamide', 'glimepiride', 'acetohexamide', 'glipizide',
       'glyburide', 'tolbutamide', 'pioglitazone', 'rosiglitazone', 'acarbose',
       'miglitol', 'troglitazone', 'tolazamide', 'examide', 'citoglipton',
       'insulin', 'glyburide-metformin', 'glipizide-metformin',
       'glimepiride-pioglitazone', 'metformin-rosiglitazone',
       'metformin-pioglitazone'],axis=1)


# ## Choosing the records with treatments Insulin and Insulin + other ( w.r.t Problem Statement)

# In[38]:


data=data[data.treatments!='other']
data.shape


# In[39]:


data.columns


# # Here the features which contains numeric values are of type Discrete Quantitative and has a finite set of values. Discrete data can be both Quantitative and Qualitative. So treating outliers in this dataset is not possible

# **One hot encoding the nominal categorical values**

# In[40]:


data = pd.get_dummies(data, columns=['race', 'gender','max_glu_serum', 'A1Cresult', 'change',
       'diabetesMed', 'readmitted'])


# In[41]:


data.head()


# ** Encoding the AGE(ordinal) categorical column**

# In[42]:


data.age.value_counts()


# In[43]:


labels = data['age'].astype('category').cat.categories.tolist()
replace_age = {'age' : {k: v for k,v in zip(labels,list(range(1,len(labels)+1)))}}

print(replace_age)


# In[44]:


data.replace(replace_age, inplace=True)


# In[45]:


data.age.value_counts()


# # Exploratory Data Analysis

# ### UNI VARIATE ANALYSIS

# In[46]:


data.num_lab_procedures.plot(kind='hist')


# In[47]:


import seaborn as sns
sns.distplot(data.time_in_hospital)


# In[48]:


import matplotlib.pyplot as plt
age_count = data['age'].value_counts()
sns.set(style="darkgrid")
sns.barplot(age_count.index, age_count.values, alpha=0.9)
plt.title('Frequency Distribution of age')
plt.ylabel('Number of Occurrences', fontsize=12)
plt.xlabel('Age', fontsize=12)
plt.show()


# In[49]:


labels = data['age'].astype('category').cat.categories.tolist()
counts = data['age'].value_counts()
sizes = [counts[var_cat] for var_cat in labels]
fig1, ax1 = plt.subplots()
ax1.pie(sizes, labels=labels, autopct='%1.1f%%', shadow=True) #autopct is show the % on plot
ax1.axis('equal')
plt.show()


# # Feature Identification

# In[50]:


data.columns


# ### Considering the Domain knowledge, we would like to drop the Columns "diag_1" , "diag_2" ,"diag_3"
# 
# ##### Since they contain the information about the codes of different types of treatments given to the patient. They don't contribute to the effectiveness of the treat (i.e, our problemm statement)

# In[51]:


data = data.drop(['diag_1','diag_2','diag_3'],axis = 1)


# ## With respect to the problem statement given, the output variable is observed to be the “treatments” feature
# ## The input variables are both Discrete Quantitative and Categorical and our output variable is Categorical
# 

# ## Since we have a combination of Discrete Quantitative Variables and Categorical Variables, we cannot perform general Correlation tests

# In[52]:


#from IPython.display import Image
#Image("../input/correlation/Picture1.png")


# ### We will be performing Chi-Square Test of Independence for finding the Correlation btw the variables

# # Chi-Square Test of Independence

# In[53]:


import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import chi2_contingency

class ChiSquare:
    def __init__(self, dataframe):
        self.df = dataframe
        self.p = None #P-Value
        self.chi2 = None #Chi Test Statistic
        self.dof = None
        
        self.dfObserved = None
        self.dfExpected = None
        
    def _print_chisquare_result(self, colX, alpha):
        result = ""
        if self.p<alpha:
            result="{0} is IMPORTANT for Prediction".format(colX)
        else:
            result="{0} is NOT an important predictor. (Discard {0} from model)".format(colX)

        print(result)
        
    def TestIndependence(self,colX,colY, alpha=0.05):
        X = self.df[colX].astype(str)
        Y = self.df[colY].astype(str)
        
        self.dfObserved = pd.crosstab(Y,X) 
        chi2, p, dof, expected = stats.chi2_contingency(self.dfObserved.values)
        self.p = p
        self.chi2 = chi2
        self.dof = dof 
        
        self.dfExpected = pd.DataFrame(expected, columns=self.dfObserved.columns, index = self.dfObserved.index)
        
        self._print_chisquare_result(colX,alpha)


# In[54]:


data['dummyCat'] = np.random.choice([0, 1], size=(len(data),), p=[0.5, 0.5])

data.dummyCat.value_counts()


# In[55]:


#Initialize ChiSquare Class
cT = ChiSquare(data)

#Feature Selection
testColumns = ['encounter_id', 'patient_nbr', 'age', 'admission_type_id',
       'discharge_disposition_id', 'admission_source_id', 'time_in_hospital',
       'num_lab_procedures', 'num_procedures', 'num_medications',
       'number_outpatient', 'number_emergency', 'number_inpatient','number_diagnoses',
       'race_AfricanAmerican', 'race_Asian', 'race_Caucasian', 'race_Hispanic',
       'race_Other', 'gender_Female', 'gender_Male',
       'max_glu_serum_>200', 'max_glu_serum_>300', 'max_glu_serum_None',
       'max_glu_serum_Norm', 'A1Cresult_>7', 'A1Cresult_>8', 'A1Cresult_None',
       'A1Cresult_Norm', 'change_Ch', 'change_No', 'diabetesMed_Yes',
       'readmitted_NO', 'dummyCat']
for var in testColumns:
    cT.TestIndependence(colX=var,colY="treatments" ) 


# # Model Building
# ## Train Test Split

# Since our target variable is Categorical , We would be importing the required Classification model packages

# In[56]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB,GaussianNB,BernoulliNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report


# In[57]:


X = data.drop(['encounter_id','patient_nbr','num_lab_procedures','number_outpatient','number_emergency',
                      'race_Asian','race_Other','diabetesMed_Yes','max_glu_serum_>200','A1Cresult_>8','A1Cresult_Norm',
                      'readmitted_NO','dummyCat','treatments'],axis=1)
Y = data['treatments']
print(X.shape)
print(Y.shape)


# In[58]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=2)
X_train.shape,X_test.shape,y_train.shape,y_test.shape


# # Base Model

# In[59]:


y_p=[]
for i in range(y_test.shape[0]):
    y_p.append(y_test.mode()[0])#Highest class is assigned to a list which is compared with ytest
len(y_p) 


# In[60]:


y_pred=pd.Series(y_p)


# In[61]:


print("Accuracy : ",accuracy_score(y_test,y_pred))


# ## Our Baseline accuracy is 54% 
# #### We can set the accuracy as 54% and the models we build should be giving us accuracies greater than 54%

# # Predictive Model Development - Iteration 1 

# ## Baseline Models - Logistic Regression 

# In[62]:


#Logistic Regression
m1=LogisticRegression()
m1.fit(X_train,y_train)
y_pred_lr=m1.predict(X_test)
Train_Score_lr = m1.score(X_train,y_train)
Test_Score_lr = accuracy_score(y_test,y_pred_lr)


print('Training Accuracy is:',Train_Score_lr)
print('Testing Accuracy is:',Test_Score_lr)
print(classification_report(y_test,y_pred_lr))


# ## KNN

# In[63]:


m2 = KNeighborsClassifier()
m2.fit(X_train,y_train)
y_pred_knn = m2.predict(X_test)
Train_Score_knn = m2.score(X_train,y_train)
Test_Score_knn = accuracy_score(y_test,y_pred_knn)

print('Training Accuracy is :',Train_Score_knn)
print('Testing Accuracy is:',Test_Score_knn)
print(classification_report(y_test,y_pred_knn))


# ## Decision Trees

# In[64]:


m4 = DecisionTreeClassifier()
m4.fit(X_train,y_train)
y_pred_dt=m4.predict(X_test)
Train_Score_dt = m4.score(X_train,y_train)
Test_Score_dt = accuracy_score(y_test,y_pred_dt)

print('Training Accuracy :',Train_Score_dt)
print('Testing Accuracy :',Test_Score_dt)
print(classification_report(y_test,y_pred_dt))


# ## Random Forest

# In[65]:


m5 = RandomForestClassifier()
m5.fit(X_train,y_train)
y_pred_rf=m5.predict(X_test)
Train_Score_rf = m5.score(X_train,y_train)
Test_Score_rf = accuracy_score(y_test,y_pred_rf)

print('Training Accuracy :',Train_Score_rf)
print('Testing Accuracy :',Test_Score_rf)
print(classification_report(y_test,y_pred_rf))


# ## SVM

# In[66]:


from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
#X_train, X_test, y_train, y_test = train_test_split(X, y,random_state=0)
pipe = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=2)
X_train.shape,X_test.shape,y_train.shape,y_test.shape


# In[67]:


from sklearn.svm import SVC
poly_kernel_svm_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm_clf", SVC(kernel="poly", degree=3, coef0=1, C=5))
])
poly_kernel_svm_clf.fit(X,Y)


# In[68]:


poly_kernel_svm_clf.score(X_test, y_test)


# # Predictive Model Development - Iteration 2

# ## Hyperparameter Tuning

# ### For Decision Tree

# In[69]:


# GridSearchCV to find optimal max_depth
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV


# specify number of folds for k-fold CV
n_folds = 3

# parameters to build the model on
parameters = {'max_depth': range(5, 15, 5),
    'min_samples_leaf': range(50, 150, 50),
    'min_samples_split': range(50, 150, 50),
    'criterion': ["entropy", "gini"]}

# instantiate the model
dtree = DecisionTreeClassifier(random_state = 100)

# fit tree on training data
tree = GridSearchCV(dtree, parameters, 
                    cv=n_folds, 
                   scoring="accuracy")
tree.fit(X_train, y_train)


# In[70]:


tree.best_params_


# In[71]:


m6 = DecisionTreeClassifier(criterion='gini',max_depth=5,min_samples_leaf=50,min_samples_split=50)
m6.fit(X_train,y_train)
y_pred_tdt=m6.predict(X_test)
Train_Score_tdt = m6.score(X_train,y_train)
Test_Score_tdt = accuracy_score(y_test,y_pred_tdt)

print('Training Accuracy :',Train_Score_tdt)
print('Testing Accuracy  :',Test_Score_tdt)
print(classification_report(y_test,y_pred_tdt))


# ### For KNN

# In[72]:


#Gridsearch CV to find Optimal K value for KNN model
grid = {'n_neighbors':np.arange(1,50)}
knn=KNeighborsClassifier()
knn_cv=GridSearchCV(knn,grid,cv=3)
knn_cv.fit(X_train,y_train)


print("Tuned Hyperparameter k: {}".format(knn_cv.best_params_))


# In[73]:


m7 = KNeighborsClassifier(n_neighbors=19)
m7.fit(X_train,y_train)
y_pred_tknn=m7.predict(X_test)
Train_Score_tknn = m7.score(X_train,y_train)
Test_Score_tknn = accuracy_score(y_test,y_pred_tknn)


print('Training Accuracy :',Train_Score_tknn)
print('Testing Accuracy  :',Test_Score_tknn)
print(classification_report(y_test,y_pred_tknn))


# ### For Random Forest 

# In[74]:


parameter={'n_estimators':np.arange(1,101)}
gs = GridSearchCV(m5,parameter,cv=3)
gs.fit(X_train,y_train)
gs.best_params_


# In[75]:


m8 = RandomForestClassifier(n_estimators=73)
m8.fit(X_train,y_train) 
y_pred_trf=m8.predict(X_test)
Train_Score_trf = m8.score(X_train,y_train)
Test_Score_trf = accuracy_score(y_test,y_pred_trf)


print('Training Accuracy :',Train_Score_trf)
print('Testing Accuracy  :',Test_Score_trf)
print(classification_report(y_test,y_pred_trf))


# In[76]:


Model_Scores=pd.DataFrame({'Models':['Logistic Regression','KNN','Decision Tree','Random Forest','Tuned Decison Tree','Tuned KNN','Tuned Random Forest'],
             'Training Accuracy':[Train_Score_lr,Train_Score_knn,Train_Score_dt,Train_Score_rf,Train_Score_tdt,Train_Score_tknn,Train_Score_trf],
             'Testing Accuracy':[Test_Score_lr,Test_Score_knn,Test_Score_dt,Test_Score_rf,Test_Score_tdt,Test_Score_tknn,Test_Score_trf],
                })

Model_Scores.sort_values(by=('Testing Accuracy'),ascending=False)


# In[77]:


## from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=2)
X_train.shape,X_test.shape,y_train.shape,y_test.shape


# In[78]:


X_train.shape


# In[79]:


X_test.shape


# In[80]:


X = data.drop(['encounter_id','patient_nbr','num_lab_procedures','number_outpatient','number_emergency',
                      'race_Asian','race_Other','diabetesMed_Yes','max_glu_serum_>200','A1Cresult_>8','A1Cresult_Norm',
                      'readmitted_NO','dummyCat','treatments'],axis=1)
Y = data['treatments']
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X= sc.fit_transform(X)


# In[81]:


from sklearn import preprocessing
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from pandas.plotting import scatter_matrix
from sklearn.metrics import f1_score
from sklearn.linear_model import LogisticRegression
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import GradientBoostingClassifier


# In[82]:


models = []
models.append(('KNN', KNeighborsClassifier(n_neighbors = 5)))
models.append(('SVM', SVC(gamma='auto')))
models.append(('GNB', GaussianNB()))
models.append(('LR', LogisticRegression(solver='liblinear',random_state=0)))
models.append(('decisiontree', tree.DecisionTreeClassifier()))
models.append(('randomforest', RandomForestClassifier(max_depth=2, random_state=10, n_estimators=10)))
models.append(('GB', GradientBoostingClassifier()))

scores = []
names = []
        
for name, model in models:
    
    score = cross_val_score(model, X, Y, scoring='accuracy', cv=5).mean()
    
    names.append(name)
    scores.append(score)
kf_cross_val = pd.DataFrame({'Name': names, 'Score': scores})
print(kf_cross_val)

axis = sns.barplot(x = 'Name', y = 'Score', data = kf_cross_val)
axis.set(xlabel='Classifier', ylabel='Accuracy')
for p in axis.patches:
    height = p.get_height()
    axis.text(p.get_x() + p.get_width()/2, height + 0.005, '{:1.4f}'.format(height), ha="center") 
    
plt.show()


# In[83]:


from sklearn.model_selection import GridSearchCV, KFold
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam


# In[84]:


def create_model():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim = 21, kernel_initializer='normal', activation='relu'))
    model.add(Dense(4, input_dim = 21, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    # compile the model
    adam = Adam(lr = 0.01)
    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
    return model

model = create_model()
print(model.summary())


# In[85]:


model = KerasClassifier(build_fn = create_model, verbose = 1)
model.fit(X,Y)


# In[86]:


from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam

# Define a random seed
seed = 6
np.random.seed(seed)

# Start defining the model
def create_model():
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim = 21, kernel_initializer='normal', activation='relu'))
    model.add(Dense(4, input_dim = 21, kernel_initializer='normal', activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    # compile the model
    adam = Adam(lr = 0.01)
    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
    return model

# create the model
model = KerasClassifier(build_fn = create_model, verbose = 1)

# define the grid search parameters
batch_size = [10,15, 20]
epochs = [10, 50, 100]

# make a dictionary of the grid search parameters
param_grid = dict(batch_size=batch_size, epochs=epochs)

# build and fit the GridSearchCV
grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = KFold(random_state=None), verbose = 10)
grid_results = grid.fit(X, Y)

# summarize the results
print("Best: {0}, using {1}".format(grid_results.best_score_, grid_results.best_params_))
means = grid_results.cv_results_['mean_test_score']
stds = grid_results.cv_results_['std_test_score']
params = grid_results.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print('{0} ({1}) with: {2}'.format(mean, stdev, param))


# In[87]:


from keras.layers import Dropout

# Define a random seed
seed = 6
np.random.seed(seed)

# Start defining the model
def create_model(learn_rate, dropout_rate):
    # create model
    model = Sequential()
    model.add(Dense(8, input_dim = 21, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(4, input_dim = 21, kernel_initializer='normal', activation='relu'))
    model.add(Dropout(dropout_rate))
    model.add(Dense(1, activation='sigmoid'))
    
    # compile the model
    adam = Adam(lr = learn_rate)
    model.compile(loss = 'binary_crossentropy', optimizer = adam, metrics = ['accuracy'])
    return model

# create the model
model = KerasClassifier(build_fn = create_model, epochs = 10, batch_size = 10, verbose = 0)

# define the grid search parameters
learn_rate = [0.001, 0.01, 0.1]
dropout_rate = [0.0, 0.1, 0.2]

# make a dictionary of the grid search parameters
param_grid = dict(learn_rate=learn_rate, dropout_rate=dropout_rate)

# build and fit the GridSearchCV
grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = KFold(random_state=None), verbose = 10)
grid_results = grid.fit(X, Y)

# summarize the results
print("Best: {0}, using {1}".format(grid_results.best_score_, grid_results.best_params_))
means = grid_results.cv_results_['mean_test_score']
stds = grid_results.cv_results_['std_test_score']
params = grid_results.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print('{0} ({1}) with: {2}'.format(mean, stdev, param))


# In[88]:


seed = 6
np.random.seed(seed)

# Start defining the model
def create_model(activation, init):
    # create model
    model = Sequential()
    model.add(Dense(4, input_dim = 21, kernel_initializer= init, activation= activation))
    model.add(Dense(4, input_dim = 21, kernel_initializer= init, activation= activation))
    model.add(Dense(1, activation='sigmoid'))
    
    # compile the model
    adam = Adam(lr = 0.001)
    model.compile(loss = 'hinge', optimizer = adam, metrics = ['accuracy'])
    return model

# create the model
model = KerasClassifier(build_fn = create_model, epochs = 100, batch_size = 10, verbose = 0)

# define the grid search parameters
activation = ['softmax', 'relu', 'tanh', 'linear']
init = ['uniform', 'normal', 'zero']

# make a dictionary of the grid search parameters
param_grid = dict(activation = activation, init = init)

# build and fit the GridSearchCV
grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = KFold(random_state=None), verbose = 10)
grid_results = grid.fit(X, Y)

# summarize the results
print("Best: {0}, using {1}".format(grid_results.best_score_, grid_results.best_params_))
means = grid_results.cv_results_['mean_test_score']
stds = grid_results.cv_results_['std_test_score']
params = grid_results.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print('{0} ({1}) with: {2}'.format(mean, stdev, param))


# In[89]:


seed = 6
np.random.seed(seed)
from keras.layers import Dropout
# Start defining the model
def create_model(neuron1, neuron2):
    # create model
    model = Sequential()
    model.add(Dense(neuron1, input_dim = 21, kernel_initializer= 'normal', activation= 'relu'))
    model.add(Dense(neuron2, input_dim = neuron1, kernel_initializer= 'normal', activation= 'linear'))
    model.add(Dense(1, activation='sigmoid'))
    
    # compile the model
    adam = Adam(lr = 0.001)
    model.compile(loss = 'hinge', optimizer = adam, metrics = ['accuracy'])
    return model

# create the model
model = KerasClassifier(build_fn = create_model, epochs = 10, batch_size = 10, verbose = 0)

# define the grid search parameters
neuron1 = [4, 8, 16]
neuron2 = [2, 4, 8]

# make a dictionary of the grid search parameters
param_grid = dict(neuron1 = neuron1, neuron2 = neuron2)

# build and fit the GridSearchCV
grid = GridSearchCV(estimator = model, param_grid = param_grid, cv = KFold(random_state=None), refit = True, verbose = 10)
grid_results = grid.fit(X, Y)

# summarize the results
print("Best: {0}, using {1}".format(grid_results.best_score_, grid_results.best_params_))
means = grid_results.cv_results_['mean_test_score']
stds = grid_results.cv_results_['std_test_score']
params = grid_results.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print('{0} ({1}) with: {2}'.format(mean, stdev, param))


# In[90]:


y_pred = grid.predict(X)
from sklearn.metrics import classification_report, accuracy_score

print(accuracy_score(Y, y_pred))
print(classification_report(Y, y_pred))


# In[171]:


# confusion matrix
import seaborn as sns
from sklearn.metrics import confusion_matrix
# compute the confusion matrix
confusion_mtx = confusion_matrix(Y, y_pred) 
# plot the confusion matrix
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()


# In[174]:


from sklearn.metrics import classification_report

# Generate a classification report
report = classification_report(Y, y_pred, target_names=['0','1'])

print(report)

