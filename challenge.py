import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso

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
    # TODO: Implement correct model building
    regression = Lasso(alpha = 0.00099 , max_iter=50000)
    regression.fit(data, labels)

    return regression


# Predict housing prices
(kaggle_train, kaggle_test) = importData()
train_processed = processData(kaggle_train)
test_processed = processData(kaggle_test)

train_labels = pd.DataFrame(index = kaggle_train.index, columns=["SalePrice"])
train_labels["SalePrice"] = np.log(kaggle_train["SalePrice"])

if train_flag:
    errors = list()
    k_fold = KFold(k)

    for k, (train, test) in enumerate(k_fold.split(train_processed)):
        model = buildModel(train_processed.index[train], train_labels.index[train])
        prediction = model.predict(train_processed.index[test])
        error = rmse(prediction, train_labels.index[test])

        errors.append(error)

    print np.mean(errors)

else:
    model = buildModel(train_processed, train_labels)
    prediction = model.predict(test_processed)

    # reverse previous log transformation
    prediction = np.exp(prediction)

    result = pd.DataFrame(prediction, index = test_processed["Id"], columns = ["SalePrice"])
    result.to_csv('result.csv', header = True, index_label = 'Id')
