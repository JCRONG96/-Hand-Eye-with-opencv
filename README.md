# -Hand-Eye-with-opencv
Use opencv-python for eye-in-hand calibration

1.尽量将相机在不同的位置拍摄棋盘格，前提是使棋盘格完整出现在视场中。

2.left文件夹中存放对应采集的图片

3.坐标6.txt文件夹里存放对应的机械臂位置和姿态，每一行7个元素，分别是位置和四元数的值（通过机械臂sdk反馈）

4.运行SolveIntrinsicParams.py可以求解相机内参，并保存至results.npz

5.运行SolveExternalParams.py可以求解手眼转换矩阵

6.对于一些商业相机，存在一些厂家标定的内参，运行convert2npz.py直接写入results.npz可以用厂家标定内参进行手眼矩阵计算，如果用厂家内参，跳过执行4，执行步骤6
