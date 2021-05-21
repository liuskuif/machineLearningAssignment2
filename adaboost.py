import numpy as np

# Adaboost类
class Adaboost(object):
    def __init__(self, ori_datas, items):
        self.ori_datas = ori_datas
        # 初始化数据集和数据标签
        self.data_matrix = np.mat(self.ori_datas.datas)
        self.label_mat = np.mat(self.ori_datas.labels).T
        self.items = items

    # 初始化参数
    def __init_argus(self):
        # 初始化列表，用来存放单层决策树(弱分类器)的信息
        self.best_stump_arr = []
        # 获取数据集行数和列数
        self.m, self.n = np.shape(self.data_matrix)
        # 初始化向量D每个值均为1/m，D包含每个数据点的权重
        self.D = np.mat(np.ones((self.m, 1)) / self.m)
        # 初始化列向量，记录每个数据点的类别估计累计值
        self.est_agg = np.mat(np.zeros((self.m, 1)))


    # 通过阈值比较对数据进行分类
    def classify(self, data_matrix, dimen, thresh_val, in_equal):
        """
        输入：dimen：数据集列数， data_matrix :要分类的数据
        thresh_val：阈值， in_equal：比较方式：low(小于)， greate(大于)

        输出： result_arr：分类结果
        """
        # 新建一个数组用于存放分类结果，初始化都为1
        result_arr = np.ones((np.shape(data_matrix)[0], 1))
        # lt：小于，gt；大于；根据阈值进行分类，并将分类结果存储到ret_array
        if in_equal == 'low':
            result_arr[data_matrix[:, dimen] <= thresh_val] = -1.0
        else:
            result_arr[data_matrix[:, dimen] > thresh_val] = -1.0
        # 返回分类结果
        return result_arr


    # 找到错误率最低的单层决策树，即一个弱分类器
    def build_stump(self, num_steps = 10):
        """
        输入： num_steps，步数，用于在特征的所有可能值上进行遍历，默认为10
        输出： best_stump：分类结果
                    min_error：最小错误率
                    best_est：最佳单层决策树
        """

        # 初始化类别估计值
        best_est = np.mat(np.zeros((self.m, 1)))

        # 初始化字典，用于存储给定权重向量D时所得到的最佳单层决策树的相关信息
        best_stump = {}

        # 将最小错误率设无穷大，之后用于寻找可能的最小错误率
        min_error = np.inf

        # 遍历数据集中每一个特征
        for i in range(self.n):
            # 根据步数求得步长
            step_size = (self.ori_datas.range_max[i] - self.ori_datas.range_min[i]) / num_steps
            # 遍历每个步长
            for j in range(-1, int(num_steps) + 1):
                # 遍历每个不等号
                for in_equal in ['low', 'greate']:
                    # 设定阈值
                    thresh_val = (self.ori_datas.range_min[i] + float(j) * step_size)
                    # 通过阈值比较对数据进行分类
                    predicted_vals = self.classify(self.data_matrix, i, thresh_val, in_equal)
                    # 初始化错误计数向量
                    err_arr = np.mat(np.ones((self.m, 1)))
                    # 如果预测结果和标签相同，则相应位置0
                    err_arr[predicted_vals == self.label_mat] = 0
                    # 计算权值误差，这就是AdaBoost和分类器交互的地方
                    weighted_error = self.D.T * err_arr
                    # 如果错误率低于min_error，则将当前单层决策树设为最佳单层决策树，更新各项值
                    if weighted_error < min_error:
                        min_error = weighted_error
                        best_est = predicted_vals.copy()
                        best_stump['dim'] = i
                        best_stump['thresh_val'] = thresh_val
                        best_stump['in_equal'] = in_equal
        # 返回最佳单层决策树，最小错误率，类别估计值
        return best_stump, min_error, best_est


    # 为下一次迭代计算D
    def updata_D(self, alpha, best_est):

        expon = np.multiply(-1 * alpha * self.label_mat, best_est)
        self.D = np.multiply(self.D, np.exp(expon))
        self.D = self.D / self.D.sum()


    # 通过迭代来找出最佳单层决策树集
    # 更新weak_class_arr
    def find_wkc_arr(self):
        # 初始化参数
        self.__init_argus()

        # 开始迭代
        for i in range(self.items):
            # 利用build_stump()函数找到最佳的单层决策树
            best_stump, error, best_est = self.build_stump()
            # 根据公式计算alpha的值，max(error, 1e-16)用来确保在没有错误时不会发生除零溢出
            alpha = float(0.5 * np.log((1.0 - error) / max(error, 1e-16)))
            # 保存alpha的值
            best_stump['alpha'] = alpha
            # 填入数据到列表
            self.best_stump_arr.append(best_stump)

            # 更新D值
            self.updata_D(alpha, best_est)

            # 累加类别估计值
            self.est_agg += alpha * best_est

            # 计算错误率，agg_class_est本身是浮点数，需要通过sign来得到二分类结果
            agg_errors = np.multiply(np.sign(self.est_agg) != np.mat(self.ori_datas.labels).T, np.ones((self.m, 1)))
            error_rate = agg_errors.sum() / self.m
            print("error_rate: ", error_rate)

            # 如果总错误率为0则跳出循环
            if error_rate == 0.0:
                break



    # 分类预测函数
    def ada_classify(self, dat_to_class):
        """
        输入：dat_to_class：待分类样例

        输出： sign(est_agg)：分类结果
        """
        # 初始化数据集
        data_matrix = np.mat(dat_to_class)
        # 获得待分类样例个数
        m = np.shape(data_matrix)[0]
        # 构建一个初始化为0的列向量，记录每个数据点的类别估计累计值
        agg_class_est = np.mat(np.zeros((m, 1)))
        # 遍历每个弱分类器
        for i in range(len(self.best_stump_arr)):
            # 基于classify得到类别估计值
            classEst = self.classify(data_matrix, self.best_stump_arr[i]['dim'], self.best_stump_arr[i]['thresh_val'],
                                      self.best_stump_arr[i]['in_equal'])
            # 累加类别估计值
            agg_class_est += self.best_stump_arr[i]['alpha'] * classEst

        # 返回分类结果，aggClassEst大于0则返回+1，否则返回-1
        return np.sign(agg_class_est)



