from numpy import *


# 通过阈值比较对数据进行分类
def stump_classify(data_matrix, dimen, thresh_val, thresh_I_neq):
    """
    输入：data_matrix：数据集， thresh_val：数据集列数，
    threshVal：阈值， threshIneq：比较方式：lt（小于），gt（大于）

    输出： res_array：分类结果
    """
    # 新建一个数组用于存放分类结果，初始化都为1
    ret_array = ones((shape(data_matrix)[0], 1))
    # lt：小于，gt；大于；根据阈值进行分类，并将分类结果存储到ret_array
    if thresh_I_neq == 'lt':
        ret_array[data_matrix[:, dimen] <= thresh_val] = -1.0
    else:
        ret_array[data_matrix[:, dimen] > thresh_val] = -1.0
    # 返回分类结果
    return ret_array


# 找到错误率最低的单层决策树，即一个弱分类器
def build_stump(ori_data, D):
    """
    输入：ori_data 数据集类 D：权重向量

    输出： best_stump：分类结果
                min_error：最小错误率
                best_class_est：最佳单层决策树
    """
    # 初始化数据集和数据标签
    data_matrix = mat(ori_data.datas)
    label_mat = mat(ori_data.labels).T
    # 获取行列值
    m, n = shape(data_matrix)
    # 初始化步数，用于在特征的所有可能值上进行遍历
    num_steps = 10.0
    # 初始化字典，用于存储给定权重向量D时所得到的最佳单层决策树的相关信息
    best_stump = {}
    # 初始化类别估计值
    best_class_est = mat(zeros((m, 1)))
    # 将最小错误率设无穷大，之后用于寻找可能的最小错误率
    min_error = inf
    # 遍历数据集中每一个特征
    for i in range(n):
        # 获取数据集的最大最小值
        range_min = data_matrix[:, i].min()
        range_max = data_matrix[:, i].max()
        # 根据步数求得步长
        step_size = (range_max - range_min) / num_steps
        # 遍历每个步长
        for j in range(-1, int(num_steps) + 1):
            # 遍历每个不等号
            for in_equal in ['lt', 'gt']:
                # 设定阈值
                thresh_val = (range_min + float(j) * step_size)
                # 通过阈值比较对数据进行分类
                predicted_vals = stump_classify(data_matrix, i, thresh_val, in_equal)
                # 初始化错误计数向量
                err_arr = mat(ones((m, 1)))
                # 如果预测结果和标签相同，则相应位置0
                err_arr[predicted_vals == label_mat] = 0
                # 计算权值误差，这就是AdaBoost和分类器交互的地方
                weighted_error = D.T * err_arr
                # 如果错误率低于min_error，则将当前单层决策树设为最佳单层决策树，更新各项值
                if weighted_error < min_error:
                    min_error = weighted_error
                    best_class_est = predicted_vals.copy()
                    best_stump['dim'] = i
                    best_stump['thresh'] = thresh_val
                    best_stump['ineq'] = in_equal
    # 返回最佳单层决策树，最小错误率，类别估计值
    return best_stump, min_error, best_class_est


# 找到最低错误率的单层决策树
def adaboost_train_ds(ori_data, items=40):
    """
    输入：ori_data：原始数据集类,  items：迭代次数

    输出： weak_class_arr：单层决策树列表
    """
    # 初始化列表，用来存放单层决策树的信息
    weak_class_arr = []
    # 获取数据集行数
    m = shape(ori_data.datas)[0]
    # 初始化向量D每个值均为1/m，D包含每个数据点的权重
    D = mat(ones((m, 1)) / m)
    # 初始化列向量，记录每个数据点的类别估计累计值
    agg_class_est = mat(zeros((m, 1)))
    # 开始迭代
    for i in range(items):
        # 利用buildStump()函数找到最佳的单层决策树
        best_stump, error, class_est = build_stump(ori_data, D)
        # print("D: ", D.T)
        # 根据公式计算alpha的值，max(error, 1e-16)用来确保在没有错误时不会发生除零溢出
        alpha = float(0.5 * log((1.0 - error) / max(error, 1e-16)))
        # 保存alpha的值
        best_stump['alpha'] = alpha
        # 填入数据到列表
        weak_class_arr.append(best_stump)
        # print("class_est: ", class_est.T)
        # 为下一次迭代计算D
        expon = multiply(-1 * alpha * mat(ori_data.labels).T, class_est)
        D = multiply(D, exp(expon))
        D = D / D.sum()
        # 累加类别估计值
        agg_class_est += alpha * class_est
        # 计算错误率，agg_class_est本身是浮点数，需要通过sign来得到二分类结果
        agg_errors = multiply(sign(agg_class_est) != mat(ori_data.labels).T, ones((m, 1)))
        error_rate = agg_errors.sum() / m
        print("total error: ", error_rate)
        # 如果总错误率为0则跳出循环
        if error_rate == 0.0:
            break
    # 返回单层决策树列表
    return weak_class_arr


# 分类预测函数
def ada_classify(dat_to_class, classifier_arr):
    """
    输入：dat_to_class：待分类样例， classifier_arr：多个弱分类器组成的数组

    输出： sign(agg_class_est)：分类结果
    """
    # 初始化数据集
    data_matrix = mat(dat_to_class)
    # 获得待分类样例个数
    m = shape(data_matrix)[0]
    # 构建一个初始化为0的列向量，记录每个数据点的类别估计累计值
    agg_class_est = mat(zeros((m, 1)))
    # 遍历每个弱分类器
    for i in range(len(classifier_arr)):
        # 基于stump_classify得到类别估计值
        classEst = stump_classify(data_matrix, classifier_arr[i]['dim'], classifier_arr[i]['thresh'],
                                  classifier_arr[i]['ineq'])
        # 累加类别估计值
        agg_class_est += classifier_arr[i]['alpha'] * classEst

    # 返回分类结果，aggClassEst大于0则返回+1，否则返回-1
    return sign(agg_class_est)



