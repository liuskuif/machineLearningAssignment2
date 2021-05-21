from oridatas import Ori_datas
from adaboost import Adaboost


if __name__ == "__main__":
    # 初始化数据
    ori_datas = Ori_datas()
    # 读入数据
    ori_datas.create_data()
    # 定义adaboost对象
    adaboost = Adaboost(ori_datas, 35)
    # 找最佳单层决策树列表
    adaboost.find_wkc_arr()
    # 打印单层最佳决策树列表
    for i, best_stump in enumerate(adaboost.best_stump_arr):
        print("单层最佳决策树{}：{}".format(i + 1, best_stump))
    # 可视化
    ori_datas.show_figure(adaboost)
