#1.套件引用
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from sklearn.model_selection import train_test_split #2、資料切割
from sklearn.neighbors import KNeighborsClassifier #最近鄰

#2.資料準備 本機端讀取資料與整理 最後儲存至df
titanic=pd.read_csv('titanic_train.csv')
titanic2=titanic.drop_duplicates(subset='PassengerId',keep='first')
titanic2.drop(['PassengerId'],inplace=True,axis=1)
titanic2.drop(['Ticket'],inplace=True,axis=1)
titanic2.drop(['Cabin'],inplace=True,axis=1)
titanic2.drop(['Pclass'],inplace=True,axis=1)

titanic2['Age'].fillna(titanic2['Age'].mean(),inplace=True)
titanic2['Embarked'].fillna('S',inplace=True)
titanic2['FamilySize'] = titanic2['SibSp'] + titanic2['Parch'] + 1
bins=[0,1,2,3,12]
titanic2['FamilySizeGroup']=pd.cut(titanic2['FamilySize'],bins,labels=['Single','Small','Medium','Large'])
titanic2['FamilySizeGroup'] =titanic2['FamilySizeGroup'].astype('object')
titanic2['Title'] = titanic2['Name'].str.extract('([A-Za-z]+)\.', expand=False)
keep_titles = ['Mr', 'Miss', 'Mrs', 'Master']
titanic2['Title']=titanic2['Title'].apply(lambda x: x if x in keep_titles else 'Other')
titanic2['FareBin'] = pd.qcut(titanic2['Fare'], 3, labels=['Low', 'Medium', 'High'])
titanic2['FareBin'] =titanic2['FareBin'].astype('object')
titanic2.drop('SibSp',axis=1,inplace=True)
titanic2.drop('Parch',axis=1,inplace=True)
titanic2.drop('Fare',axis=1,inplace=True)
titanic2.drop('Name',axis=1,inplace=True)
titanic2.drop('FamilySize',axis=1,inplace=True)

list1=[] #數值
list2=[] #字串
for i in titanic2.columns:
  if titanic2[i].dtype==object:
    try:
      titanic2[i]=titanic2[i].astype(float)
      list1.append(i)
    except:
      list2.append(i)
  else:
    list1.append(i)
titanic3=titanic2[list2]
titanic3=pd.get_dummies(titanic3)
titanic3=titanic3.astype(float)
titanic3['Age']=titanic2['Age']
titanic3['Survived']=titanic2['Survived']

#3.模型訓練與之前的各項流程
X=titanic3.iloc[:,:-1]
y=titanic3.iloc[:,-1]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3)
model=KNeighborsClassifier(n_neighbors=8)
model.fit(X_train,y_train)

#4.表格方式資料呈現
st.title("titanic3")
st.write("### 資料集前幾行")
st.dataframe(titanic3.head())

#5A.以滑桿呈現特徵值的範圍 放在網頁左側
Embarked_C,Embarked_Q,Embarked_S=0,0,0
choices1=['Embarked_C','Embarked_Q','Embarked_S']
select1=st.sidebar.radio('Embarked:',choices1)
if select1=='Embarked_C':
  Embarked_C=1
  Embarked_Q=0
  Embarked_S=0
elif select1=='Embarked_Q':
  Embarked_C=0
  Embarked_Q=1
  Embarked_S=0
else:
  Embarked_C=0
  Embarked_Q=0
  Embarked_S=1

FamilySizeGroup_Large,FamilySizeGroup_Medium,FamilySizeGroup_Single,FamilySizeGroup_Small=0,0,0,0
choices2=['FamilySizeGroup_Large','FamilySizeGroup_Medium','FamilySizeGroup_Single','FamilySizeGroup_Small']
select2=st.sidebar.radio('FamilySizeGroup:',choices2)
if select2=='FamilySizeGroup_Large':
  FamilySizeGroup_Large=1
  FamilySizeGroup_Medium=0
  FamilySizeGroup_Single=0
  FamilySizeGroup_Small=0
if select2=='FamilySizeGroup_Medium':
  FamilySizeGroup_Large=0
  FamilySizeGroup_Medium=1
  FamilySizeGroup_Single=0
  FamilySizeGroup_Small=0
if select2=='FamilySizeGroup_Single':
  FamilySizeGroup_Large=0
  FamilySizeGroup_Medium=0
  FamilySizeGroup_Single=1
  FamilySizeGroup_Small=0
if select2=='FamilySizeGroup_Small':
  FamilySizeGroup_Large=0
  FamilySizeGroup_Medium=0
  FamilySizeGroup_Single=0
  FamilySizeGroup_Small=1

Title_Master,Title_Miss,Title_Mr,Title_Mrs,Title_Other=0,0,0,0,0
choices3=['Title_Master','Title_Miss','Title_Mr','Title_Mrs','Title_Other']
select3=st.sidebar.radio('Title:',choices3)
if select3=='Title_Master':
   Title_Master=1
   Title_Miss=0
   Title_Mr=0
   Title_Mrs=0
   Title_Other=0
if select3=='Title_Miss':
   Title_Master=0
   Title_Miss=1
   Title_Mr=0
   Title_Mrs=0
   Title_Other=0
if select3=='Title_Mr':
   Title_Master=0
   Title_Miss=0
   Title_Mr=1
   Title_Mrs=0
   Title_Other=0      
if select3=='Title_Mrs':
   Title_Master=0
   Title_Miss=0
   Title_Mr=0
   Title_Mrs=1
   Title_Other=0 
if select3=='Title_Other':
   Title_Master=0
   Title_Miss=0
   Title_Mr=0
   Title_Mrs=0
   Title_Other=1 

FareBin_High,FareBin_Low,FareBin_Medium=0,0,0
choices4=['FareBin_High','FareBin_Low','FareBin_Medium']
select4=st.sidebar.radio('FareBin:',choices4)
if select4=='FareBin_High':
   FareBin_High=1
   FareBin_Low=0
   FareBin_Medium=0
if select4=='FareBin_Low':
   FareBin_High=0
   FareBin_Low=1
   FareBin_Medium=0
if select4=='FareBin_Medium':
   FareBin_High=0
   FareBin_Low=0
   FareBin_Medium=1  

Sex_female,Sex_male=0,0
choices5=['Sex_female','Sex_male']
select5=st.sidebar.radio('Sex:',choices5)
if select5=='Sex_female':
   Sex_female=1
   Sex_male=0
if select5=='Sex_male':
   Sex_male=1
   Sex_female=0
   
   
#5B.以滑桿呈現特徵值的範圍 放在網頁左側
st.sidebar.header("輸入特徵值")
df=titanic3.copy()
Age = st.sidebar.slider("Age", float(df.iloc[:, 17].min()), float(df.iloc[:, 17].max()), float(df.iloc[:,17].mean()))

#6.以上面的特徵值進行預測
input_data = [[Sex_female,Sex_male,Embarked_C,Embarked_Q,Embarked_S,FamilySizeGroup_Large,FamilySizeGroup_Medium,FamilySizeGroup_Single,
FamilySizeGroup_Small,Title_Master,Title_Miss,Title_Mr,Title_Mrs,Title_Other,FareBin_High,
FareBin_Low,FareBin_Medium,Age]]
prediction = model.predict(input_data)
prediction_proba = model.predict_proba(input_data)

#7.文字呈現預測結果
st.write("### 預測結果")
st.write(f"分類結果：{prediction[0]}")

#8.圖表呈現預測的機率
st.write("### 預測機率")
st.bar_chart(prediction_proba[0])

#9.文字呈現預測的準確率
accuracy = model.score(X_test, y_test)
st.write(f"### 模型準確率：{accuracy:.2f}")

import matplotlib.pyplot as plt
import seaborn as sns
df["Prediction"] = model.predict(X)

st.write("### 全部資料年齡與女性預測結果散點圖:女性為1非女性為0")
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(
    x="Age", y="Sex_female", hue="Prediction",data=df, ax=ax, palette="coolwarm"
)
ax.set_title("Age vs Sex_female: Prediction ")
st.pyplot(fig)

st.write("### 全部資料年齡與女性實際結果散點圖:女性為1非女性為0")
fig, ax = plt.subplots(figsize=(8, 5))
sns.scatterplot(
    x="Age", y="Sex_female", hue="Survived",data=df, ax=ax, palette="coolwarm"
)
ax.set_title("Age vs Sex_female:Actual")
st.pyplot(fig)