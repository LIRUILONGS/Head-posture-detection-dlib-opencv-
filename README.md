# Head-posture-detection-dlib-opencv-

头部姿态检测（dlib+opencv）Demo

---
title: 基于OpenCV 和 Dlib 进行头部姿态估计
tags:
  - 头部姿态评估
categories:
  - 头部姿态评估
toc: true
recommend: 1
keywords: 头部姿态评估
uniqueId: '2023-06-02 10:57:09/基于OpenCV 和 Dlib 进行头部姿态估计.html'
mathJax: false
date: 2023-06-02 18:57:09
thumbnail:
---

**<font color="009688"> 庐山烟雨浙江潮，未到千般恨不消。到得还来别无事，庐山烟雨浙江潮。----《庐山烟雨浙江潮》苏轼**</font>

<!-- more -->
## 写在前面

***
+ 工作中遇到，简单整理
+ 理解不足小伙伴帮忙指正


**<font color="009688"> 庐山烟雨浙江潮，未到千般恨不消。到得还来别无事，庐山烟雨浙江潮。 ----《庐山烟雨浙江潮》苏轼**</font>

***


实验项目以上传，只需 git 克隆，就可以开始使用了，但是需要说明的是 Dlib 的基于 `HOG特征和SVM分类器`的人脸检测器很一般，很多脸都检测不到，实际情况中可以考虑使用深度学习模型来做关键点检测，然后评估姿态。可以查看文章末尾大佬的开源项目


### 实现效果

|Demo|
|--|
|原图|
|![原图](https://img-blog.csdnimg.cn/0c9a7eb4b5b54dbe8b3bd710f8eb1d37.jpeg)
|特征点标记后|
|![特征点标记后](https://img-blog.csdnimg.cn/d94c728ecd63423d881510f3ba5f0fcd.jpeg)|
|姿态标记|
|![姿态标记](https://img-blog.csdnimg.cn/61b0456e4ea24bb78b554179c011d419.jpeg)|
|姿态对应的Yaw，Pitch，Roll 度数|
|![姿态对应的Yaw，Pitch，Roll 度数](https://img-blog.csdnimg.cn/73660aecceb14300ba9c702229308083.jpeg)|
### 步骤

三个主要步骤：

#### 人脸检测

`人脸检测`：引入人脸检测器 `dlib.get_frontal_face_detector()` 以检测包含人脸的图片，多个人脸会选择面积最大的人脸。

`dlib.get_frontal_face_detector()` 是 `dlib` 库中的一个函数，用于获取一个基于`HOG特征和SVM分类器`的人脸检测器。该函数返回一个可以用于检测图像中人脸的对象。

具体来说，HOG（Histogram of Oriented Gradients，梯度方向直方图）是一种常用于图像识别中的特征描述子，SVM（Support Vector Machine，支持向量机）是一种常用的分类器。将HOG特征与SVM分类器结合起来，可以得到一个有效的人脸检测器。

在使用 `dlib.get_frontal_face_detector()`函数时，只需将待检测的图像作为参数传入，即可得到一个用于检测人脸的对象。一个Demo
```py
import dlib
import cv2

# 读取图像
img = cv2.imread('image.jpg')

# 获取人脸检测器
detector = dlib.get_frontal_face_detector()

# 在图像中检测人脸
faces = detector(img)

# 输出检测到的人脸数
print("检测到的人脸数为：", len(faces))

```

#### 面部特征点检测

`面部特征点检测`，利用预训练模型 `shape_predictor_68_face_landmarks.dat` 以人脸图像为输入，输出`68个人脸特征点`。

`shape_predictor_68_face_landmarks.dat` 是基于 dlib 库中的人脸特征点检测模型，该模型使用了基于 HOG 特征和 SVM 分类器的人脸检测器来检测图像中的人脸，并使用回归算法来预测人脸的 68 个关键点位置。这些关键点包括眼睛、鼻子、嘴巴等部位，可以用于进行人脸识别、表情识别、姿态估计等应用。

这个模型文件可以在dlib的官方网站上下载。在使用它之前，需要安装dlib库并将模型文件加载到程序中。

```py
predictor = dlib.shape_predictor(r".\shape_predictor_68_face_landmarks.dat")
```
#### 姿势估计

姿势估计。在获得 68 个面部特征点后，选择部分特征点，通过 `PnP`算法计算姿势 `Yaw、Pitch、Roll` 度数

```py
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
```


`Yaw、Pitch、Roll` 是用于描述物体或相机在三维空间中的旋转角度的术语，常用于`姿态估计和姿态控制`中。

+ `Yaw(左右)`：绕垂直于物体或相机的轴旋转的角度，也称为偏航角。通常以 z 轴为轴进行旋转，正值表示逆时针旋转，负值表示顺时针旋转。
+ `Pitch(上下)`：绕物体或相机的横轴旋转的角度，也称为俯仰角。通常以 x 轴为轴进行旋转，正值表示向上旋转，负值表示向下旋转。
+ `Roll(弯曲)`：绕物体或相机的纵轴旋转的角度，也称为翻滚角。通常以 y 轴为轴进行旋转，正值表示向右旋转，负值表示向左旋转。

这三个角度通常以欧拉角的形式表示，可以用于描述物体或相机的姿态信息。在计算机视觉中，常用于人脸识别、动作捕捉、机器人控制等应用场景。



### 完整 Demo 代码
```py
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
"""
@File    :   face_ypr_demo.py
@Time    :   2023/06/05 21:32:45
@Author  :   Li Ruilong
@Version :   1.0
@Contact :   liruilonger@gmail.com
@Desc    :   根据68个人脸关键点，获取人头部姿态评估
"""

# here put the import lib

import cv2
import numpy as np
import dlib
import math
import uuid

# 头部姿态检测（dlib+opencv）

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(r".\shape_predictor_68_face_landmarks.dat")
POINTS_NUM_LANDMARK = 68


# shape_predictor_68_face_landmarks.dat 是一个预训练的人脸关键点检测模型，可以用于识别人脸的68个关键点，如眼睛、鼻子、嘴巴等。这个模型可以被用于人脸识别、人脸表情分析、面部姿势估计等领域。
# 它是由dlib库提供的，可以在Python中使用。如果你想使用它，可以在dlib的官方网站上下载。

# 获取最大的人脸
def _largest_face(dets):
    """
    @Time    :   2023/06/05 21:30:37
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   从一个由 dlib 库检测到的人脸框列表中，找到最大的人脸框，并返回该框在列表中的索
                如果只有一个人脸，直接返回
                 Args:
                   dets： 一个由 `dlib.rectangle` 类型的对象组成的列表，每个对象表示一个人脸框
                 Returns:
                   人脸索引
    """
    # 如果列表长度为1，则直接返回
    if len(dets) == 1:
        return 0
    # 计算每个人脸框的面积
    face_areas = [(det.right() - det.left()) * (det.bottom() - det.top()) for det in dets]
    import heapq
    # 找到面积最大的人脸框的索引
    largest_area = face_areas[0]
    largest_index = 0
    for index in range(1, len(dets)):
        if face_areas[index] > largest_area:
            largest_index = index
            largest_area = face_areas[index]
    # 打印最大人脸框的索引和总人脸数
    print("largest_face index is {} in {} faces".format(largest_index, len(dets)))

    return largest_index


def get_image_points_from_landmark_shape(landmark_shape):
    """
    @Time    :   2023/06/05 22:30:02
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   从dlib的检测结果抽取姿态估计需要的点坐标
                 Args:
                   landmark_shape:  所有的位置点
                 Returns:
                   void
    """

    if landmark_shape.num_parts != POINTS_NUM_LANDMARK:
        print("ERROR:landmark_shape.num_parts-{}".format(landmark_shape.num_parts))
        return -1, None

    # 2D image points. If you change the image, you need to change vector

    image_points = np.array([
        (landmark_shape.part(17).x, landmark_shape.part(17).y),  # 17 left brow left corner
        (landmark_shape.part(21).x, landmark_shape.part(21).y),  # 21 left brow right corner
        (landmark_shape.part(22).x, landmark_shape.part(22).y),  # 22 right brow left corner
        (landmark_shape.part(26).x, landmark_shape.part(26).y),  # 26 right brow right corner
        (landmark_shape.part(36).x, landmark_shape.part(36).y),  # 36 left eye left corner
        (landmark_shape.part(39).x, landmark_shape.part(39).y),  # 39 left eye right corner
        (landmark_shape.part(42).x, landmark_shape.part(42).y),  # 42 right eye left corner
        (landmark_shape.part(45).x, landmark_shape.part(45).y),  # 45 right eye right corner
        (landmark_shape.part(31).x, landmark_shape.part(31).y),  # 31 nose left corner
        (landmark_shape.part(35).x, landmark_shape.part(35).y),  # 35 nose right corner
        (landmark_shape.part(48).x, landmark_shape.part(48).y),  # 48 mouth left corner
        (landmark_shape.part(54).x, landmark_shape.part(54).y),  # 54 mouth right corner
        (landmark_shape.part(57).x, landmark_shape.part(57).y),  # 57 mouth central bottom corner
        (landmark_shape.part(8).x, landmark_shape.part(8).y),  # 8 chin corner
    ], dtype="double")
    return 0, image_points


def get_image_points(img):
    """
    @Time    :   2023/06/05 22:30:43
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   用dlib检测关键点，返回姿态估计需要的几个点坐标
                 Args:

                 Returns:
                   void
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # 图片调整为灰色

    dets = detector(img, 0)

    if 0 == len(dets):
        print("ERROR: found no face")
        return -1, None
    largest_index = _largest_face(dets)
    face_rectangle = dets[largest_index]

    landmark_shape = predictor(img, face_rectangle)
    draw = im.copy()
    cv2.circle(draw, (landmark_shape.part(0).x, landmark_shape.part(0).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(1).x, landmark_shape.part(1).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(2).x, landmark_shape.part(2).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(3).x, landmark_shape.part(3).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(4).x, landmark_shape.part(4).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(5).x, landmark_shape.part(5).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(6).x, landmark_shape.part(6).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(7).x, landmark_shape.part(7).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(8).x, landmark_shape.part(8).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(9).x, landmark_shape.part(9).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(10).x, landmark_shape.part(10).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(11).x, landmark_shape.part(11).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(12).x, landmark_shape.part(12).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(13).x, landmark_shape.part(13).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(14).x, landmark_shape.part(14).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(15).x, landmark_shape.part(15).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(16).x, landmark_shape.part(16).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(17).x, landmark_shape.part(17).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(18).x, landmark_shape.part(18).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(19).x, landmark_shape.part(19).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(20).x, landmark_shape.part(20).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(21).x, landmark_shape.part(21).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(22).x, landmark_shape.part(22).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(23).x, landmark_shape.part(23).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(24).x, landmark_shape.part(24).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(25).x, landmark_shape.part(25).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(26).x, landmark_shape.part(26).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(27).x, landmark_shape.part(27).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(28).x, landmark_shape.part(28).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(29).x, landmark_shape.part(29).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(30).x, landmark_shape.part(30).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(31).x, landmark_shape.part(31).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(32).x, landmark_shape.part(32).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(33).x, landmark_shape.part(33).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(34).x, landmark_shape.part(34).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(35).x, landmark_shape.part(35).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(36).x, landmark_shape.part(36).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(37).x, landmark_shape.part(37).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(38).x, landmark_shape.part(38).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(39).x, landmark_shape.part(39).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(40).x, landmark_shape.part(40).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(41).x, landmark_shape.part(41).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(42).x, landmark_shape.part(42).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(43).x, landmark_shape.part(43).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(44).x, landmark_shape.part(44).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(45).x, landmark_shape.part(45).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(46).x, landmark_shape.part(46).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(47).x, landmark_shape.part(47).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(48).x, landmark_shape.part(48).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(49).x, landmark_shape.part(49).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(50).x, landmark_shape.part(50).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(51).x, landmark_shape.part(51).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(52).x, landmark_shape.part(52).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(53).x, landmark_shape.part(53).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(54).x, landmark_shape.part(54).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(55).x, landmark_shape.part(55).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(56).x, landmark_shape.part(56).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(57).x, landmark_shape.part(57).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(58).x, landmark_shape.part(58).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(59).x, landmark_shape.part(59).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(60).x, landmark_shape.part(60).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(61).x, landmark_shape.part(61).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(62).x, landmark_shape.part(62).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(63).x, landmark_shape.part(63).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(64).x, landmark_shape.part(64).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(65).x, landmark_shape.part(65).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(66).x, landmark_shape.part(66).y), 2, (0, 255, 0), -1)
    cv2.circle(draw, (landmark_shape.part(67).x, landmark_shape.part(67).y), 2, (0, 255, 0), -1)

    # 部分关键点特殊标记
    cv2.circle(draw, (landmark_shape.part(17).x, landmark_shape.part(17).y), 2, (0, 165, 255),
               -1)  # 17 left brow left corner
    cv2.circle(draw, (landmark_shape.part(21).x, landmark_shape.part(21).y), 2, (0, 165, 255),
               -1)  # 21 left brow right corner
    cv2.circle(draw, (landmark_shape.part(22).x, landmark_shape.part(22).y), 2, (0, 165, 255),
               -1)  # 22 right brow left corner
    cv2.circle(draw, (landmark_shape.part(26).x, landmark_shape.part(26).y), 2, (0, 165, 255),
               -1)  # 26 right brow right corner
    cv2.circle(draw, (landmark_shape.part(36).x, landmark_shape.part(36).y), 2, (0, 165, 255),
               -1)  # 36 left eye left corner
    cv2.circle(draw, (landmark_shape.part(39).x, landmark_shape.part(39).y), 2, (0, 165, 255),
               -1)  # 39 left eye right corner
    cv2.circle(draw, (landmark_shape.part(42).x, landmark_shape.part(42).y), 2, (0, 165, 255),
               -1)  # 42 right eye left corner
    cv2.circle(draw, (landmark_shape.part(45).x, landmark_shape.part(45).y), 2, (0, 165, 255),
               -1)  # 45 right eye right corner
    cv2.circle(draw, (landmark_shape.part(31).x, landmark_shape.part(31).y), 2, (0, 165, 255),
               -1)  # 31 nose left corner
    cv2.circle(draw, (landmark_shape.part(35).x, landmark_shape.part(35).y), 2, (0, 165, 255),
               -1)  # 35 nose right corner
    cv2.circle(draw, (landmark_shape.part(48).x, landmark_shape.part(48).y), 2, (0, 165, 255),
               -1)  # 48 mouth left corner
    cv2.circle(draw, (landmark_shape.part(54).x, landmark_shape.part(54).y), 2, (0, 165, 255),
               -1)  # 54 mouth right corner
    cv2.circle(draw, (landmark_shape.part(57).x, landmark_shape.part(57).y), 2, (0, 165, 255),
               -1)  # 57 mouth central bottom corner
    cv2.circle(draw, (landmark_shape.part(8).x, landmark_shape.part(8).y), 2, (0, 165, 255), -1)

    # 保存关键点标记后的图片
    cv2.imwrite('new_' + "KeyPointDetection.jpg", draw)

    return get_image_points_from_landmark_shape(landmark_shape)


def get_pose_estimation(img_size, image_points):
    """
    @Time    :   2023/06/05 22:31:31
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   获取旋转向量和平移向量
                 Args:

                 Returns:
                   void
    """

    # 3D model points.
    model_points = np.array([
        (6.825897, 6.760612, 4.402142),  # 33 left brow left corner
        (1.330353, 7.122144, 6.903745),  # 29 left brow right corner
        (-1.330353, 7.122144, 6.903745),  # 34 right brow left corner
        (-6.825897, 6.760612, 4.402142),  # 38 right brow right corner
        (5.311432, 5.485328, 3.987654),  # 13 left eye left corner
        (1.789930, 5.393625, 4.413414),  # 17 left eye right corner
        (-1.789930, 5.393625, 4.413414),  # 25 right eye left corner
        (-5.311432, 5.485328, 3.987654),  # 21 right eye right corner
        (2.005628, 1.409845, 6.165652),  # 55 nose left corner
        (-2.005628, 1.409845, 6.165652),  # 49 nose right corner
        (2.774015, -2.080775, 5.048531),  # 43 mouth left corner
        (-2.774015, -2.080775, 5.048531),  # 39 mouth right corner
        (0.000000, -3.116408, 6.097667),  # 45 mouth central bottom corner
        (0.000000, -7.415691, 4.070434)  # 6 chin corner
    ])
    # Camera internals

    focal_length = img_size[1]
    center = (img_size[1] / 2, img_size[0] / 2)
    camera_matrix = np.array(
        [[focal_length, 0, center[0]],
         [0, focal_length, center[1]],
         [0, 0, 1]], dtype="double"
    )

    dist_coeffs = np.array([7.0834633684407095e-002, 6.9140193737175351e-002, 0.0, 0.0, -1.3073460323689292e+000],
                           dtype="double")  # Assuming no lens distortion

    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix,
                                                                  dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    # print("Rotation Vector:\n {}".format(rotation_vector))
    # print("Translation Vector:\n {}".format(translation_vector))
    return success, rotation_vector, translation_vector, camera_matrix, dist_coeffs


def draw_annotation_box(image, rotation_vector, translation_vector, camera_matrix, dist_coeefs, color=(0, 255, 0),
                        line_width=2):
    """
    @Time    :   2023/06/05 22:09:14
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   标记一个人脸朝向的3D框
                 Args:

                 Returns:
                   void
    """

    """Draw a 3D box as annotation of pose"""
    point_3d = []
    rear_size = 10
    rear_depth = 0
    point_3d.append((-rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, rear_size, rear_depth))
    point_3d.append((rear_size, -rear_size, rear_depth))
    point_3d.append((-rear_size, -rear_size, rear_depth))

    front_size = 10
    # 高度
    front_depth = 10
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d.append((-front_size, front_size, front_depth))
    point_3d.append((front_size, front_size, front_depth))
    point_3d.append((front_size, -front_size, front_depth))
    point_3d.append((-front_size, -front_size, front_depth))
    point_3d = np.array(point_3d, dtype=np.float32).reshape(-1, 3)

    # Map to 2d image points
    (point_2d, _) = cv2.projectPoints(point_3d,
                                      rotation_vector,
                                      translation_vector,
                                      camera_matrix,
                                      dist_coeefs)
    point_2d = np.int32(point_2d.reshape(-1, 2))

    # Draw all the lines
    cv2.polylines(image, [point_2d], True, color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[1]), tuple(
        point_2d[6]), color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[2]), tuple(
        point_2d[7]), color, line_width, cv2.LINE_AA)
    cv2.line(image, tuple(point_2d[3]), tuple(
        point_2d[8]), color, line_width, cv2.LINE_AA)


# 从旋转向量转换为欧拉角
def get_euler_angle(rotation_vector):
    """
    @Time    :   2023/06/05 22:31:52
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   从旋转向量转换为欧拉角
                 Args:

                 Returns:
                   void
    """

    # calculate rotation angles
    theta = cv2.norm(rotation_vector, cv2.NORM_L2)

    # transformed to quaterniond
    w = math.cos(theta / 2)
    x = math.sin(theta / 2) * rotation_vector[0][0] / theta
    y = math.sin(theta / 2) * rotation_vector[1][0] / theta
    z = math.sin(theta / 2) * rotation_vector[2][0] / theta

    ysqr = y * y
    # pitch (x-axis rotation)
    t0 = 2.0 * (w * x + y * z)
    t1 = 1.0 - 2.0 * (x * x + ysqr)

    # print('t0:{}, t1:{}'.format(t0, t1))
    pitch = math.atan2(t0, t1)

    # yaw (y-axis rotation)
    t2 = 2.0 * (w * y - z * x)
    if t2 > 1.0:
        t2 = 1.0
    if t2 < -1.0:
        t2 = -1.0
    yaw = math.asin(t2)

    # roll (z-axis rotation)
    t3 = 2.0 * (w * z + x * y)
    t4 = 1.0 - 2.0 * (ysqr + z * z)
    roll = math.atan2(t3, t4)

    print('pitch:{}, yaw:{}, roll:{}'.format(pitch, yaw, roll))

    # 单位转换：将弧度转换为度
    pitch_degree = int((pitch / math.pi) * 180)
    yaw_degree = int((yaw / math.pi) * 180)
    roll_degree = int((roll / math.pi) * 180)

    return 0, pitch, yaw, roll, pitch_degree, yaw_degree, roll_degree


def get_pose_estimation_in_euler_angle(landmark_shape, im_szie):
    try:
        ret, image_points = get_image_points_from_landmark_shape(landmark_shape)
        if ret != 0:
            print('get_image_points failed')
            return -1, None, None, None

        ret, rotation_vector, translation_vector, camera_matrix, dist_coeffs = get_pose_estimation(im_szie,
                                                                                                   image_points)
        if ret != True:
            print('get_pose_estimation failed')
            return -1, None, None, None

        ret, pitch, yaw, roll = get_euler_angle(rotation_vector)
        if ret != 0:
            print('get_euler_angle failed')
            return -1, None, None, None

        euler_angle_str = 'Pitch:{}, Yaw:{}, Roll:{}'.format(pitch, yaw, roll)
        print(euler_angle_str)
        return 0, pitch, yaw, roll

    except Exception as e:
        print('get_pose_estimation_in_euler_angle exception:{}'.format(e))
        return -1, None, None, None


def build_img_text_marge(img_, text, height):
    """
    @Time    :   2023/06/01 05:29:09
    @Author  :   liruilonger@gmail.com
    @Version :   1.0
    @Desc    :   生成文字图片拼接到 img 对象
                 Args:

                 Returns:
                   void
    """
    import cv2
    from PIL import Image, ImageDraw, ImageFont

    # 定义图片大小和背景颜色
    width = img_.shape[1]
    background_color = (255, 255, 255)

    # 定义字体、字号和颜色
    font_path = 'arial.ttf'
    font_size = 26
    font_color = (0, 0, 0)

    # 创建空白图片
    image = Image.new('RGB', (width, height), background_color)

    # 创建画笔
    draw = ImageDraw.Draw(image)

    # 加载字体
    font = ImageFont.truetype(font_path, font_size)

    # 写入文字
    text_width, text_height = draw.textsize(text, font)
    text_x = (width - text_width) // 2
    text_y = (height - text_height) // 2
    draw.text((text_x, text_y), text, font=font, fill=font_color)

    # 将Pillow图片转换为OpenCV图片
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    montage_size = (width, img_.shape[0])
    import imutils
    montages = imutils.build_montages([img_, image_cv], montage_size, (1, 2))

    # 保存图片
    return montages[0]


if __name__ == '__main__':
    from imutils import paths

    # for imagePath in paths.list_images("W:\\python_code\\deepface\\huge_1.jpg"):
    for imagePath in range(1):
        print(f"处理的图片路径为： {imagePath}")
        # Read Image
        im = cv2.imread("image.jpg")
        size = im.shape
        # 对图像进行缩放的操作
        if size[0] > 700:
            h = size[0] / 3
            w = size[1] / 3
            # 如果图像的高度大于700，就将其高度和宽度分别缩小为原来的1/3，然后使用双三次插值的方法进行缩放。最后返回缩放后的图像的大小。
            im = cv2.resize(im, (int(w), int(h)), interpolation=cv2.INTER_CUBIC)
            size = im.shape
        # 获取坐标点
        ret, image_points = get_image_points(im)
        if ret != 0:
            print('get_image_points failed')
            continue

        ret, rotation_vector, translation_vector, camera_matrix, dist_coeffs = get_pose_estimation(size, image_points)

        if ret != True:
            print('get_pose_estimation failed')
            continue
        draw_annotation_box(im, rotation_vector, translation_vector, camera_matrix, dist_coeffs)
        cv2.imwrite('new_' + "draw_annotation_box.jpg", im)

        ret, pitch, yaw, roll, pitch_degree, yaw_degree, roll_degree = get_euler_angle(rotation_vector)

        draw = im.copy()
        # Yaw:

        if yaw_degree < 0:
            output_yaw = "left : " + str(abs(yaw_degree)) + " degrees"
        elif yaw_degree > 0:
            output_yaw = "right :" + str(abs(yaw_degree)) + " degrees"
        else:
            output_yaw = "No left or right"
        print(output_yaw)

        # Pitch:
        if pitch_degree > 0:
            output_pitch = "dow :" + str(abs(pitch_degree)) + " degrees"
        elif pitch_degree < 0:
            output_pitch = "up :" + str(abs(pitch_degree)) + " degrees"
        else:
            output_pitch = "No downwards or upwards"
        print(output_pitch)

        # Roll:
        if roll_degree < 0:
            output_roll = "bends to the right: " + str(abs(roll_degree)) + " degrees"
        elif roll_degree > 0:
            output_roll = "bends to the left: " + str(abs(roll_degree)) + " degrees"
        else:
            output_roll = "No bend  right or left."
        print(output_roll)

        # Initial status:
        if abs(yaw) < 0.00001 and abs(pitch) < 0.00001 and abs(roll) < 0.00001:
            cv2.putText(draw, "Initial ststus", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, .5, (0, 255, 0))
            print("Initial ststus")

        # 姿态检测完的数据写在对应的照片
        imgss = build_img_text_marge(im, output_yaw + "\n" + output_pitch + "\n" + output_roll, 200)
        cv2.imwrite('new_' + str(uuid.uuid4()).replace('-', '') + ".jpg", imgss)

```








## 博文部分内容参考

© 文中涉及参考链接内容版权归原作者所有，如有侵权请告知，这是一个开源项目，如果你认可它，不要吝啬星星哦 :)


***
https://blog.csdn.net/zhang2gongzi/article/details/124520896

https://github.com/JuneoXIE/

https://github.com/yinguobing/head-pose-estimation

***

© 2018-2023 liruilonger@gmail.com, All rights reserved. 保持署名-非商用-相同方式共享(CC BY-NC-SA 4.0)
