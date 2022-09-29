import numpy as np


def extract_params():
    # 内参
    mtx = [[1057.35,       0,    936.730],
           [0,       1056.54,    588.998],
           [0,             0,          1]]
    # 畸变参数
    dist = [[-0.0437674, 0.0129022, -0.00589165, 0.000164766, 0.000125511]]
    np.savez(r".\results", mtx=mtx, dist=dist)
    print("内参和畸变参数保存成功！")


if __name__ == "__main__":
    extract_params()
