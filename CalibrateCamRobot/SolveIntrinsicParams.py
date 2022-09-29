import numpy as np
import cv2
import glob


def calibration_chessboard(images_path):
    # 设置要标定的内角点个数
    global gray
    nums_width = 11
    nums_height = 8
    length_side = 30
    # 设置标定板在世界坐标中的坐标
    world_point = np.zeros((nums_width * nums_height, 3), np.float32)  # xyz，其中z先置为0，对象点
    world_point[:, :2] = length_side * np.mgrid[:nums_width, : nums_height].T.reshape(-1, 2)
    # print(world_point)
    # 保存角点坐标
    world_position = []  # 存放世界坐标，对象点
    image_position = []  # 存放棋盘角点对应的图片像素坐标，图像点
    # 设置终止条件，迭代30次或移动0.001
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    # 获取所有标定图
    images = glob.glob(images_path + r"\*.png")
    for image_path in images:
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 查找角点
        ret, corners = cv2.findChessboardCorners(gray, (nums_width, nums_height), None)
        if ret:
            # 把每一幅图像的世界坐标放到world_position中
            world_position.append(world_point)
            # 亚像素级角点检测，在角点检测中精确化角点位置
            exact_corners = cv2.cornerSubPix(gray, corners, (11,11), (-1,-1), criteria)
            # 把获取的角点坐标放到image_position中
            image_position.append(exact_corners)
            # 可视化角点
            image = cv2.drawChessboardCorners(image, (nums_width, nums_height), exact_corners, ret)
            cv2.imshow("image_corner", image)
            cv2.waitKey()
    """
    计算内参、畸变矩阵、外参
    """
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(world_position, image_position, gray.shape[::-1], None, None)
    # 内参是mtx, 3x3矩阵
    # 畸变矩阵是dist，1x5矩阵
    # 旋转向量（要得到矩阵还要进行罗德里格斯变换）rvecs
    # 外参：平移矩阵tvecs
    # 将内参保存起来
    np.savez(r".\results", mtx=mtx, dist=dist)
    # print(mtx.shape,dist.shape)
    print('内参是：\n', mtx, '\n畸变参数是：\n', dist)
    # 计算偏差
    mean_error = 0
    for i in range(len(world_position)):
        image_position2, _ = cv2.projectPoints(world_position[i], rvecs[i], tvecs[i], mtx, dist)  # shape is 88，1，2
        error = cv2.norm(image_position[i], image_position2, cv2.NORM_L2) / len(image_position2)
        mean_error += error
    print("Re-projection error: ", mean_error / len(image_position))


def main():
    # path
    images_path = r".\Images"
    calibration_chessboard(images_path)


if __name__ == "__main__":
    main()
