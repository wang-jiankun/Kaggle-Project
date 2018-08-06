"""
Kaggle 练习 -- Titanic
author: 王建坤
date: 2018-7-28
"""
import pandas as pd

# 导入数据集
train = pd.read_csv('Titanic_dataset/train.csv')
test = pd.read_csv('Titanic_dataset/test.csv')

# 查看数据集的信息
# print(train.info())
# print(test.info())

# 选择特征
selected_features = ['Pclass', 'Sex', 'Age', 'Embarked', 'SibSp', 'Parch', 'Fare']
X_train = train[selected_features]
X_test = test[selected_features]
y_train = train['Survived']

# 查看具体特征的所有值
# print(X_train['Embarked'].value_counts())
# print(X_test['Embarked'].value_counts())

# 填充缺失值，Embarked 用出现频率最高的特征值
X_train['Embarked'].fillna('S', inplace=True)
X_test['Embarked'].fillna('S', inplace=True)

# 填充缺失值，Age 和 Fare 用特征值的平均值
X_train['Age'].fillna(X_train['Age'].mean(), inplace=True)
X_test['Age'].fillna(X_test['Age'].mean(), inplace=True)
X_test['Fare'].fillna(X_test['Fare'].mean(), inplace=True)

# 查看数据集的信息
# X_train.info()

# 特征向量化
from sklearn.feature_extraction import DictVectorizer
dict_vec = DictVectorizer(sparse=False)
X_train = dict_vec.fit_transform(X_train.to_dict(orient='record'))
# 查看特征向量的特征名称
# dict_vec.feature_names_
X_test = dict_vec.transform(X_test.to_dict(orient='record'))

# 创建随机森林模型分类器
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()

# 交叉验证的精度
from sklearn.model_selection import cross_val_score
print(cross_val_score(rfc, X_train, y_train, cv=5))

# 分类器拟合数据集
rfc.fit(X_train, y_train)
rfc_y_predict = rfc.predict(X_test)

# 保存对测试集的预测结果，以便提交
rfc_submission = pd.DataFrame({'PassengerId': test['PassengerId'], 'Survived': rfc_y_predict})
rfc_submission.to_csv('titanic_submission.csv', index=False)
