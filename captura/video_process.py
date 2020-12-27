import subprocess

import cv2
import numpy as np
from pathlib import Path
import time
from captura.courses import COURSES

course_names = [i["name"] for i in COURSES[38].values()]


def cvt_color(path):
    for file in Path(path).rglob("*.mp4"):
        s_time = time.time()
        # file = Path(r"D:\Projects\220_ml\captura\订单系统\第2节：百度订单系统度小店课程_tmp.mp4")
        # if file.stem not in course_names: continue
        cap = cv2.VideoCapture(str(file.absolute()))
        isOpened = cap.isOpened  # 判断视频是否可读
        fps = cap.get(cv2.CAP_PROP_FPS)  # 获取图像的帧，即该视频每秒有多少张图片
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))  # 获取图像的宽度和高度
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('\n', file.name, fps, width, height, frames)
        assert width == 1920 and height == 1080
        continue
        video_tmp_name = f"./我是钱冷门竞价38期/{file.parent.name}/{file.name}"  # 要创建的视频文件名称
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码格式
        video = cv2.VideoWriter(video_tmp_name, fourcc, fps, (width, height))
        while isOpened:
            # 读取每一帧，flag表示是否读取成功，frame为图片的内容
            flag, frame = cap.read()
            if not flag: break
            img_bgr = cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR)  # 转为opencv的BGR格式
            video.write(img_bgr)
        video.release()
        # cv2.destroyAllWindows()
        print(time.time() - s_time)
        audio_tmp_name = file.with_suffix(".wav")  # 要创建的视频文件名称
        video_out_name = f"./我是钱冷门竞价38期/{file.parent.name}/{file.stem[:-4] + '.mp4'}"  # 要创建的视频文件名称
        subprocess.call(f"ffmpeg -i {video_tmp_name} -i {audio_tmp_name} {video_out_name}", shell=True)


def video2img():
    for file in Path(r"D:\Projects\220_ml\captura\new1").rglob("*.mp4"):
        s_time = time.time()
        prefix = int(s_time) % 100000
        subprocess.call(f"ffmpeg -i {file} -r 0.3 -q:v 2 -f image2 images_new/{prefix}_%05d.jpg", shell=True)
        print(time.time() - s_time)


def cvr_fmt():
    for file in Path(r"D:\Projects\220_ml\captura\images").glob("*.jpeg"):
        file.rename(file.with_suffix(".jpg"))


def generate_xml():
    pass


if __name__ == "__main__":
    cvt_color(r"D:\BaiduNetdiskDownload\我是钱冷门竞价38期\微信加粉")
