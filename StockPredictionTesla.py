import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier
from sklearn import metrics
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

import warnings
warnings.filterwarnings('ignore')

# Read the TESLA.csv file and take the content into a dataframe 
df = pd.read_csv('./TESLA.csv')
df.head()

# Print the share of the dataframe 
print("Dataframe shape: ", df.shape);
df.describe()
df.info()

# Now start EDA: Exploratory Data Analysis 
# Plot the closing price of this stock 
plt.figure(figsize=(15,5))
plt.plot(df['Close'])
plt.title('Tesla Close price.', fontsize=15)
plt.ylabel('Price in dollars.')
plt.show()

df.head()

# There is a possibility of few redundant columns and it seems like "Close" and "Adj Close" could be same
# Check that by comparing two columns
df[df['Close'] == df['Adj Close']].shape

# if the shape of this is same as shape of the original dataframe, which it is, drop the "Adj Close" column
df = df.drop(['Adj Close'], axis=1)

# Check for the null value and do the necessary cleanup 
df.isnull().sum()

# Plot few distribution graphs for 'Open', 'High', 'Low', 'Close' and 'Volume' columns
features = ['Open', 'High', 'Low', 'Close', 'Volume']
plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
	plt.subplot(2,3,i+1)
	sb.distplot(df[col])
plt.show()

# Now do boxplots for those same columns 
plt.subplots(figsize=(20,10))
for i, col in enumerate(features):
	plt.subplot(2,3,i+1)
	sb.boxplot(df[col])
plt.show()

# Feature Engineering 
splitted = df['Date'].str.split('-', expand=True)
print(splitted)
print("Dataframe shape: ", splitted.shape);
df['day'] = splitted[2].astype('int')
df['month'] = splitted[1].astype('int')
df['year'] = splitted[0].astype('int')
# drop Date column
df = df.drop(['Date'], axis=1)
df.head()

# Now we have three more columns namely ‘day’, ‘month’ and ‘year’ derived from Date
df['is_quarter_end'] = np.where(df['month']%3==0,1,0)
df.head()

# Lets group data by Year and plot those 
print("df year: ", df['year'])
print("Groupby output: ", df.groupby('year'))
tmp = df.groupby('year')
#print("tmp groupby output: ", tmp)
#print("tmp groupby output BEFORE reset_index type: ", type(tmp))
#tmp
#tmp = tmp.reset_index()
#print("tmp groupby output after reset_index: ", tmp.head())
#print("tmp groupby output after reset_index data type: ", type(tmp))
data_grouped = tmp.mean()
plt.subplots(figsize=(20,10))
for i, col in enumerate(['Open', 'High', 'Low', 'Close']):
	plt.subplot(2,2,i+1)
	data_grouped[col].plot.bar()
plt.show()

df.groupby('is_quarter_end').mean()
df['open-close'] = df['Open'] - df['Close']
df['low-high'] = df['Low'] - df['High']
df['target'] = np.where(df['Close'].shift(-1) > df['Close'], 1, 0)

# Check the target using pie chart
plt.pie(df['target'].value_counts().values, 
		labels=[0, 1], autopct='%1.1f%%')
plt.show()

# Heatmap
plt.figure(figsize=(10, 10))
# As our concern is with the highly
# correlated features only so, we will visualize
# our heatmap as per that criteria only.
sb.heatmap(df.corr() > 0.9, annot=True, cbar=False)
plt.show()

# Data Splitting and Normalization
features = df[['open-close', 'low-high', 'is_quarter_end']]
target = df['target']
scaler = StandardScaler()
features = scaler.fit_transform(features)
X_train, X_valid, Y_train, Y_valid = train_test_split(
	features, target, test_size=0.1, random_state=2022)
print(X_train.shape, X_valid.shape)

# Model development and Evaluation
clf = SVC(kernel='poly', probability=True)
models = [LogisticRegression(), clf, XGBClassifier()]
for i in range(3):
	models[i].fit(X_train, Y_train)
	print(f'{models[i]} : ')
	print('Training Accuracy : ', metrics.roc_auc_score(
	Y_train, models[i].predict_proba(X_train)[:,1]))
	print('Validation Accuracy : ', metrics.roc_auc_score(
	Y_valid, models[i].predict_proba(X_valid)[:,1]))
	Y_predicted = np.where(models[i].predict_proba(X_valid)[:,1] > 0.5, 1, 0)
	cm = confusion_matrix(Y_valid, Y_predicted)
	ConfusionMatrixDisplay(confusion_matrix=cm)
	plt.show()
	print("CONF MATRIX: \n", cm)
	print(classification_report(Y_valid, Y_predicted))
	print()
