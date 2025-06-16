import numpy as np
import pandas as pd
import matplotlib 
import matplotlib.pyplot as plt
import seaborn as sns
import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("rakeshkapilavai/extrovert-vs-introvert-behavior-data")

# print("Path to dataset files:", path)

Data = pd.read_csv("C:\\Users\\USER\\.cache\\kagglehub\\datasets\\rakeshkapilavai\\extrovert-vs-introvert-behavior-data\\versions\\1\\personality_dataset.csv")
# printing the first 5
# Data.head()

# Dataset description 
# Data.describe()

# info about the dataset
# Data.info()

# checking for null values
# print(Data[Data.isnull()].any())
Data.dropna(inplace=True)



# shape of the dataset
# Data.shape

# checking for dupolicated values
# Data.duplicated().value_counts()

# dropping dupicated values
Data = Data.drop_duplicates(keep='first')
# Data.shape

# Data.head()

# splitting the dataset int features and target
X = Data.drop(columns='Personality') #features
y = Data['Personality'] #Target
# print(X.shape)
# print(y.shape)

# splitting the dataset int train and test set
from sklearn.model_selection import train_test_split
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=0.8, random_state=42)
# print(Xtrain.shape)
# print(Xtest.shape)

from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
# from sklearn.model_selection import GridSearchCV 
#preprocessing the target


#preprocessing the features
num_features = X.select_dtypes(include=['int64','float64']).columns
cat_features = X.select_dtypes(include=['object']).columns
# print(num_features)
# print(cat_features)

# instantiating the preprocessing model
num_transformer = StandardScaler()
cat_transformer = OneHotEncoder()

# ColumnTransformer for synching the model and the features
preprocessor = ColumnTransformer(transformers=[('num', num_transformer, num_features),('cat', cat_transformer, cat_features)])
# preprocessor

# creating pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', RandomForestClassifier())])
# pipeline

# training 
pipeline.fit(Xtrain,ytrain)

# prediction
y_pred = pipeline.predict(Xtest)
# print('prediction:', y_pred)
# print('actual:', ytest)

score = pipeline.score(Xtrain, ytrain)
print('score:', score)

report = classification_report(ytest, y_pred)
print(report)

# from sklearn.metrics import confusion_matrixs
# import seaborn as sns
# mat = confusion_matrix(ytest, y_pred)
# sns.heatmap(mat, square=True, annot=True, cbar=False)
# plt.xlabel('predicted value')
# plt.ylabel('true value');


# creating a pickle file
import pickle
with open("model.pkl", 'wb') as f:
    pickle.dump(pipeline, f)
print('model saved as model.pkl')
