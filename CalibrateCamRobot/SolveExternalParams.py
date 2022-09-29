import cv2
import numpy as np
import glob


def draw(img, corners, imgpts):
    corner = corners[0].ravel()
    corner = tuple([int(corner[0]), int(corner[1])])
    img = cv2.line(img, corner, tuple([int(imgpts[0].ravel()[0]), int(imgpts[0].ravel()[1])]), (255, 0, 0), 5)
    img = cv2.line(img, corner, tuple([int(imgpts[1].ravel()[0]), int(imgpts[1].ravel()[1])]), (0, 255, 0), 5)
    img = cv2.line(img, corner, tuple([int(imgpts[2].ravel()[0]), int(imgpts[2].ravel()[1])]), (0, 0, 255), 5)
    return img


def quat2rotation(xyzwrxryrz):
    """
    将aubo机械臂给的位姿转换为齐次矩阵
    """
    x, y, z, w, rx, ry, rz = xyzwrxryrz
    # 转
    converted_r = np.array([[-1 + 2*w*w + 2*rx*rx, 2*rx*ry -2*w*rz, 2*rx*rz + 2*w*ry],
                                   [2*rx*ry + 2*w*rz, -1 + 2*w*w + 2*ry*ry, 2*ry*rz - 2*w*rx],
                                   [2*rx*rz - 2*w*ry, 2*ry*rz + 2*w*rx, -1 + 2*w*w + 2*rz*rz]])
    converted_t = np.array([x, y, z])

    return converted_r, converted_t


# 标定图像
def calibration_images(images_path):
    # 设置要标定的角点个数
    nums_width = 11
    nums_height = 8
    length_side = 30
    # 设置标定图在世界坐标中的坐标
    world_point = np.zeros((nums_width * nums_height, 3), np.float32)
    world_point[:, :2] = length_side * np.mgrid[:nums_width, :nums_height].T.reshape(-1, 2)
    print('world pointL: ', world_point)
    # 设置世界坐标的坐标
    axis = 30 * np.float32([[3, 0, 0], [0, 3, 0], [0, 0, -3]]).reshape(-1, 3)
    # 设置终止条件，迭代30次或移动0.001
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    images = glob.glob(images_path + r"/*.png")
    external_rotation_r = []
    external_rotation_t = []
    for i, image_path in enumerate(images):
        print(image_path)
        image = cv2.imread(image_path)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 查找角点
        ret, corners = cv2.findChessboardCorners(gray, (nums_width, nums_height), None)
        if ret is True:
            # 亚像素级角点检测，在角点检测中精确化角点位置
            exact_corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            # 获取外参
            # 使用Ransac方法从2D点到3D点查找对应关系 ，PnP (Perspective-n-Point)
            # cv2.solvePnP() 估计标定板的姿势
            _, rvec, tvec, inliers = cv2.solvePnPRansac(world_point, exact_corners, mtx, dist)
            # 获得的旋转矩阵是向量，是3×1的矩阵，想要还原回3×3的矩阵，需要罗德里格斯变换Rodrigues，
            rotation_m, _ = cv2.Rodrigues(rvec)  # 罗德里格斯变换，变成旋转矩阵
            external_rotation_r.append(rotation_m)
            external_rotation_t.append(tvec)
            # 画图
            imgpts, jac = cv2.projectPoints(axis, rvec, tvec, mtx, dist)
            img = draw(image, corners, imgpts)
            cv2.namedWindow("img{}".format(i), cv2.WINDOW_NORMAL)
            cv2.imshow("img{}".format(i), img)
            cv2.waitKey(100)
    return external_rotation_r, external_rotation_t


if __name__ == "__main__":
    # 读取相机内参，这里建议读厂家的试一试
    """
    use SN23884712.conf
    """
    with np.load(r'results.npz') as params:
        mtx, dist = [params[i] for i in ('mtx', 'dist')]
        print(mtx, '\n', dist)

    images_path = r"Images"

    external_rotation_r, external_rotation_t = calibration_images(images_path)  # 这是外参矩阵
    external_rotation_r = np.array(external_rotation_r)
    external_rotation_t = np.array(external_rotation_t) / 1000.0

    pos = []  # 机械臂对应的姿态
    # 读入机械臂位姿数据
    with open(r"robot_info.txt") as f:
        lines = f.readlines()
        for i, line in enumerate(lines):
            each_pos = []
            # 当前行为空，则读取信息结束
            if not line:
                break
            line = line.strip().split(" ")
            for j in line:
                each_pos.append(float(j))
            pos.append(each_pos)

    # print(external_rotation_r, external_rotation_t)
    # 机械臂的齐次矩阵
    robot_rotation_r = []
    robot_rotation_t = []
    for i, pt in enumerate(pos):
        rr, rt = quat2rotation(pt)
        robot_rotation_r.append(rr)
        robot_rotation_t.append(rt)

    r_eye2hand, t_eye2hand = cv2.calibrateHandEye(robot_rotation_r, robot_rotation_t,
                                                  external_rotation_r, external_rotation_t,
                                                  cv2.CALIB_HAND_EYE_TSAI)  # CALIB_HAND_EYE_HORAUD CALIB_HAND_EYE_PARK
    print("手眼旋转矩阵: {}\n手眼平移矩阵: {}".format(r_eye2hand, t_eye2hand))
