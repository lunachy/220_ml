import cv2
import numpy as np
cap = cv2.VideoCapture('第2节：百度订单系统度小店课程.mp4')
isOpened = cap.isOpened   # 判断视频是否可读
print(isOpened)
fps = cap.get(cv2.CAP_PROP_FPS)  # 获取图像的帧，即该视频每秒有多少张图片
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取图像的宽度和高度
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(fps, width, height)
videoname = "2.mp4"   # 要创建的视频文件名称
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码格式
video = cv2.VideoWriter(videoname, fourcc, fps, (width, height))
while isOpened:
    # 读取每一帧，flag表示是否读取成功，frame为图片的内容
    flag, frame = cap.read()
    if not flag: break
    img_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)  # 转为opencv的BGR格式
    video.write(img_bgr)
video.release()
cv2.destroyAllWindows()

# videoname = "2.mp4"   # 要创建的视频文件名称
# fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')  # 编码器
#
#
# # 1.要创建的视频文件名称 2.编码器 3.帧率 4.size
# videoWrite = cv2.VideoWriter(videoname, fourcc, fps, (width, height))
# for i in range(10):
#     filename = 'img' + str(i) + '.jpg'
#     img = cv2.imread(filename)
#     videoWrite.write(img) # 写入