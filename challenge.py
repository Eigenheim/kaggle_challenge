import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error

# parameters
train_flag = True
k = 5

def rmse(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

def importData():
    train = pd.read_csv('data/train.csv')
    test = pd.read_csv('data/test.csv')

    return (train, test)

def processData(data):
    # TODO: Implement data processing
    return data

def buildModel(data, labels):
    regression = xgb.XGBRegressor(
        colsample_bytree=0.2,
        gamma=0.0,
        learning_rate=0.01,
        max_depth=4,
        min_child_weight=1.5,
        n_estimators=7200,
        reg_alpha=0.9,
        reg_lambda=0.6,
        subsample=0.2,
        seed=42,
        silent=1)

    regression.fit(data, labels)

    return regression

# Predict housing prices
(kaggle_train, kaggle_test) = importData()
train_processed = processData(kaggle_train)
train_processed = train_processed.drop("SalePrice", axis = 1)
test_processed = processData(kaggle_test)

total_processed = pd.concat([train_processed, test_processed], keys = ["train", "test"])
total_processed = pd.get_dummies(total_processed)

total_processed.to_csv("total_processed.csv", header = True)

train_processed = total_processed.ix["train"]
test_processed = total_processed.ix["test"]

train_processed.to_csv("train_processed.csv", header = True)

train_labels = pd.DataFrame(index = kaggle_train.index, columns=["SalePrice"])
train_labels["SalePrice"] = np.log(kaggle_train["SalePrice"])

if train_flag:
    errors = list()
    k_fold = KFold(k)

    for k, (train, test) in enumerate(k_fold.split(train_processed)):
        model = buildModel(train_processed.values[train], train_labels.values[train])
        prediction = model.predict(train_processed.values[test])
        error = rmse(prediction, train_labels.values[test])

        errors.append(error)

    print np.mean(errors)

else:
    model = buildModel(train_processed, train_labels)
    prediction = model.predict(test_processed)

    # reverse previous log transformation
    prediction = np.exp(prediction)

    result = pd.DataFrame(prediction, index = test_processed["Id"], columns = ["SalePrice"])
    result.to_csv('result.csv', header = True, index_label = 'Id')
