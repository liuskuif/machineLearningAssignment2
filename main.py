from oridatas import Ori_datas
import adaboost as adb


if __name__ == "__main__":
    ori_data = Ori_datas()
    ori_data.create_data()
    classifier_arr = adb.adaboost_train_ds(ori_data, 40)
    print(classifier_arr)
    ori_data.show_figure(classifier_arr)
