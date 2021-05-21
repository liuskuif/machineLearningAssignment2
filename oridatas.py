import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_iris
import adaboost as adb

class Ori_datas:
    def __init__(self):
        self.datas = [] # 存放数据集
        self.labels = [] # 存放数据标签
        self.range_min = [] # 存放数据集特征的最小值
        self.range_max = [] # 存放数据集特征的最大值

    # 导入数据集,并更新数据集相关信息
    def create_data(self):
        iris = load_iris()
        df = pd.DataFrame(iris.data, columns=iris.feature_names)
        df['label'] = iris.target
        df.columns = ['sepal length', 'sepal width', 'petal length', 'petal width', 'label']
        data = np.array(df.iloc[:100, [0, 1, -1]])
        for i in range(len(data)):
            if data[i, -1] == 0:
                data[i, -1] = -1
        self.datas = data[:, :2]
        self.labels = data[:, -1]
        data_matrix = np.mat(self.datas)
        m, n = np.shape(data_matrix)
        for i in range(n):
            self.range_min.append(data_matrix[:, i].min())
            self.range_max.append(data_matrix[:, i].max())

        print(self.range_min)
        print(self.range_max)

    # 可视化
    def show_figure(self, classifier_arr):
        # 绘制数据点
        x_cord_1 = []
        y_cord_1 = []
        x_cord_01 = []
        y_cord_01 = []
        for i, data_item in enumerate(self.datas):
            xPt = float(data_item[0])
            yPt = float(data_item[1])
            label = int(self.labels[i])
            if label == 1:
                x_cord_1.append(xPt)
                y_cord_1.append(yPt)
            else:
                x_cord_01.append(xPt)
                y_cord_01.append(yPt)

        fig = plt.figure()
        ax = fig.add_subplot(111)
        plt.title('Support Vectors')

        # 绘制各个弱分类器
        for best_stump in classifier_arr:
            if best_stump['dim'] == 1:
                plt.axhline(y=best_stump['thresh'], c="blue")  # 添加水平直线
            else:
                plt.axvline(x=best_stump['thresh'], c="red")  # 添加垂直直线

        # 绘制预测结果
        x = np.linspace(self.range_min[0] - 1, self.range_max[0] + 1, 200)
        y = np.linspace(self.range_min[1] - 1, self.range_max[1] + 1, 200)
        pred_arr_x_1 = []
        pred_arr_y_1 = []
        pred_arr_x_01 = []
        pred_arr_y_01 = []
        for x_i in x:
            for y_j in y:
                pred_result = adb.ada_classify([x_i, y_j], classifier_arr)
                if pred_result == 1:
                    pred_arr_x_1.append(x_i)
                    pred_arr_y_1.append(y_j)
                else:
                    pred_arr_x_01.append(x_i)
                    pred_arr_y_01.append(y_j)
        # plt.fill_between(0, 0, 3, facecolor='blue', where=y1 > y2, alpha=0.5, interpolate=True)
        # ax.axis([3, 7.5, 0, 6])
        ax.scatter(pred_arr_x_1, pred_arr_y_1, marker=',', s=30, c="yellow", alpha=0.1)
        ax.scatter(pred_arr_x_01, pred_arr_y_01, marker=',', s=30, c="green", alpha=0.1)

        # 绘制数据点
        ax.scatter(x_cord_1, y_cord_1, marker='.', s=80, c="blue")
        ax.scatter(x_cord_01, y_cord_01, marker='*', s=80, c='red')

        plt.show()
