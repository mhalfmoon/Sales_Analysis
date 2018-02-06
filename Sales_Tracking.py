import pandas as pd
import numpy as np
from scipy.stats import mode
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,r2_score
from sklearn.ensemble import GradientBoostingRegressor
#from sklearn import cross_validation, metrics
pd.options.mode.chained_assignment = None
from sklearn.externals import joblib

test = pd.read_csv('Test.csv')
train = pd.read_csv('Train.csv')
train['source']='train'
test['source']='test'
data = pd.concat([train, test],ignore_index=True)

categorical_columns = [x for x in data.dtypes.index if data.dtypes[x]=='object']
categorical_columns = [x for x in categorical_columns if x not in ['Item_Identifier','Outlet_Identifier','source']]

item_avg_weight = data.groupby(by='Item_Identifier').Item_Weight.mean()
miss_bool = data['Item_Weight'].isnull()
data.loc[miss_bool,'Item_Weight']=data.loc[miss_bool,'Item_Identifier'].apply(lambda x: item_avg_weight[x])
visibility_avg = data.groupby(by ='Item_Identifier').Item_Visibility.mean()
miss_bool2 = (data['Item_Visibility'] == 0)
data.loc[miss_bool2,'Item_Visibility'] = data.loc[miss_bool2,'Item_Identifier'].apply(lambda x: visibility_avg[x])
df = pd.DataFrame({'Outlet_Type':['Grocery Store','Supermarket Type1','Supermarket Type2','Supermarket Type3'],'Outlet_Size':['Small','Small','Medium','Medium']})
data.Outlet_Size = data.Outlet_Size.fillna(data.Outlet_Type.map(df.set_index('Outlet_Type').Outlet_Size))

data['Outlet_Age'] = 2018 - data['Outlet_Establishment_Year']
data['Item_Type_Combined'] = data['Item_Identifier'].apply(lambda x: x[0:2])
data['Item_Type_Combined'] = data['Item_Type_Combined'].map({'FD':'Food','NC':'Non-Consumable','DR':'Drinks'})


#Make the fat content consistent, and assign Non-edible items.
data['Item_Fat_Content'] = data['Item_Fat_Content'].replace({'LF':'Low Fat','reg':'Regular','low fat':'Low Fat'})
data.loc[data['Item_Type_Combined']=="Non-Consumable",'Item_Fat_Content'] = "Non-Edible"

le = LabelEncoder()
#New variable for outlet
data['Outlet'] = le.fit_transform(data['Outlet_Identifier'])
var_mod = ['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Item_Type_Combined','Outlet_Type','Outlet']

le = LabelEncoder()
for i in var_mod:
    data[i] = le.fit_transform(data[i])
#print(data.dtypes)
data = pd.get_dummies(data, columns=['Item_Fat_Content','Outlet_Location_Type','Outlet_Size','Outlet_Type','Item_Type_Combined','Outlet'])

IDcol = ['Item_Identifier','Outlet_Identifier']
#Drop the columns which have been converted to different types:
data.drop(['Item_Type','Outlet_Establishment_Year'],axis=1,inplace=True)

#Divide into test and train:
train = data.loc[data['source']=="train"]
test = data.loc[data['source']=="test"]

#Drop unnecessary columns:
test.drop(['source'],axis=1,inplace=True)
train.drop(['source'],axis=1,inplace=True)
#print(data.dtypes)
train['Items_Sold']=round(train.Item_Outlet_Sales/train.Item_MRP)
train.drop(['Item_Outlet_Sales'],axis=1,inplace=True)
target = 'Items_Sold'
predictors = [x for x in train.columns if x not in [target]+IDcol]
#print(train)
rf = RandomForestRegressor(n_estimators = 5000,max_depth=6, min_samples_leaf=50,n_jobs=4)
rf.fit(train[predictors],train[target])
test[target] = rf.predict(test[predictors])
#params = {'n_estimators': 500, 'max_depth': 6,'learning_rate': 0.1, 'loss': 'huber','alpha':0.95}
#clf = GradientBoostingRegressor(**params).fit(train[predictors], train[target])
test[target]=rf.predict(test[predictors])
test['Item_Outlet_Sales']=test.Items_Sold*test.Item_MRP
target2 = 'Item_Outlet_Sales'
IDcol.append(target2)

submission = pd.DataFrame({ x: test[x] for x in IDcol})
submission.to_csv('Submission.csv', index=False)
