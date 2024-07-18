import pandas as pd
import numpy as np
np.set_printoptions(precision=3)
from datetime import time, timedelta
import time
from sklearn.model_selection import train_test_split, cross_val_predict, cross_val_score, KFold, RandomizedSearchCV
from sklearn.metrics import accuracy_score, f1_score
from sklearn.ensemble import GradientBoostingClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from xgboost.sklearn import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
import warnings
warnings.filterwarnings("ignore")


class ModelResult:
    def __init__(self, model_name, train_time, train_acc, train_score, train_F1, test_time, test_acc, test_score, test_F1):
        self.model_name = model_name
        self.train_time = train_time
        self.train_acc = train_acc
        self.train_score = train_score
        self.train_F1 = train_F1
        self.test_time = test_time
        self.test_acc = test_acc
        self.test_score = test_score
        self.test_F1 = test_F1
        self.columns = ["model_name", "train_time", "train_acc", "train_score", "train_F1", "test_time", "test_acc",
                        "test_score", "test_F1"]

    def to_list(self):
        return [self.model_name, self.train_time, self.train_acc, self.train_score, self.train_F1, self.test_time,
                self.test_acc, self.test_score, self.test_F1]


class Result:
    def __init__(self):
        self.model_list = []

    def save(self, file_name):
        model_list = [line.to_list() for line in self.model_list]
        output = pd.DataFrame(model_list, columns=self.model_list[0].columns)
        output.to_csv(file_name, encoding="utf-8-sig", index=0)


class BoostMethod:
    def __init__(self, datapath, labelpath, k=5, cv=4, search=False):
        """
        :param datapath: 数据路径
        :param labelpath: 标签路径
        :param k: k折训练
        :param cv: 交叉验证次数
        :param search: 是否需要网格调参
        """
        self.data_path = datapath
        self.labelpath = labelpath
        self.dataset = self.loading_data()  # [train_x, test_x, train_y, test_y]
        self.cv = cv
        self.k = k
        self.search = search
        self.model = {
            "AdaBoost": AdaBoostClassifier(n_estimators=100),
            "GTBoost": GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=0),
            "HBGBoost": HistGradientBoostingClassifier(max_iter=100),
            "xgboost": XGBClassifier(eval_metric=['logloss', 'auc', 'error']),
            "CatBoost": CatBoostClassifier(learning_rate=0.1, depth=6, iterations=100, verbose=False),
            "LightGBM": LGBMClassifier(learning_rate=0.1, max_depth=3, num_leaves=16),
        }

    def loading_data(self):#读取数据，并划分实验集和验证集
        data = pd.read_csv(self.data_path, encoding="utf-8-sig", header=0)
        label = pd.read_csv(self.labelpath, encoding="utf-8-sig", header=0)
        train_x, test_x, train_y, test_y = train_test_split(data, label, test_size=0.3, random_state=1129)
        return {"train_x": train_x, "test_x": test_x, "train_y": train_y, "test_y": test_y}

    def fitting(self):
        result = Result()
        for item in self.model.items():
            model_name = item[0]
            model = item[1]
            print(model_name)
            model, train_time, (train_acc, train_score, train_F1) = self.train(model, self.dataset["train_x"],
                                                                               self.dataset["train_y"])
            (test_time, test_acc, test_score, test_F1) = self.test(model, self.dataset["test_x"],
                                                                   self.dataset["test_y"])
            model_result = ModelResult(model_name, train_time, train_acc, train_score, train_F1, test_time, test_acc,
                                       test_score, test_F1)
            result.model_list.append(model_result)
        return result

    def evaluate(self, model, data, label, test=False):
        start_time = time.time()
        predict = cross_val_predict(model, data, label, cv=self.cv)
        time_ret = self.get_time_dif(start_time)
        acc = accuracy_score(predict, label)
        score = cross_val_score(model, data, label, cv=self.cv).mean()
        F1 = f1_score(label, predict)
        if test:
            return str(time_ret), acc, score, F1
        else:
            return acc, score, F1

    def train(self, model, data, label):
        start_time = time.time()
        kf = KFold(n_splits=self.k, random_state=1129, shuffle=True)
        for train, evaluate in kf.split(data):
            model.fit(data.iloc[train], label.iloc[train])
        time_ret = self.get_time_dif(start_time)
        return model, str(time_ret), self.evaluate(model, data, label)

    def test(self, model, data, label):
        return self.evaluate(model, data, label, test=True)

    def get_time_dif(self, start_time):
        end_time = time.time()
        time_dif = end_time - start_time
        # print("Time usage:", timedelta(seconds=int(round(time_dif))))
        return timedelta(seconds=int(round(time_dif)))


if __name__ == '__main__':
    method = BoostMethod("dataset.csv", "label.csv")
    result = method.fitting()
    result.save("boosting{}.csv".format(time.strftime('_%Y%m%d_%H%M', time.localtime())))

