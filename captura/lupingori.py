import wave
from pathlib import Path

import pyaudio
from PIL import ImageGrab
import mss
import numpy as np
import cv2
from moviepy.editor import *
from moviepy.audio.fx import all
import time
import subprocess

time_s = time.strftime('%H_%M_%S', time.localtime(time.time()))
course = {
    38: {
        "1_1": {"name": "第1节：百度竞价冷门产品课程安排", "time": 887},
        "1_2": {"name": "第2节：百度订单系统度小店课程", "time": 46},
        "1_3": {"name": "第3节：冷门竞价选产品的标准", "time": 916},
        "1_4": {"name": "第4节：常用找产品的方法", "time": 1097},
        "1_5": {"name": "第5节：教程类找产品方法演示", "time": 152},
        "1_6": {"name": "第6节：二类找产品方法演示", "time": 1357},
        "2_7": {"name": "第7节：竞价单页基础知识", "time": 1143},
        "2_7_": {"name": "第7节课前补充：竞价单页差异化和竞价账户效果差解决方案", "time": 3019},
        "2_8_1": {"name": "第8节1：服务器购买", "time": 584},
        "2_8_2": {"name": "第8节2：服务器安装", "time": 600},
        "2_8_3": {"name": "第8节3：订单系统上传、修改、配置", "time": 600},
        "2_8_4": {"name": "第8节4：单页复制、修改、上传（上）", "time": 600},
        "2_8_5": {"name": "第8节5：单页复制、修改、上传（下）", "time": 569},
        "2_9_1": {"name": "第9节1：利用模板自己做单页", "time": 1260},
        "2_9_2": {"name": "第9节2：度小店操作方法", "time": 1890},
        "2_10": {"name": "第10节：原有竞价单页优化", "time": 827},
        "2_11": {"name": "第11节：竞价单页制作注意事项", "time": 607},
        "3_12": {"name": "第12节：竞价账户操作基础", "time": 1889},
        "3_13": {"name": "第13节：百度竞价账户搭建", "time": 2214},
        "3_14_1": {"name": "第14节1：百度竞价账户优化（上）", "time": 1436},
        "3_14_2": {"name": "第14节2：百度竞价账户优化（下）", "time": 2628},
        "3_15": {"name": "第15节：360竞价账户搭建", "time": 622},
        "4_16": {"name": "第16节：竞价实战客服话术", "time": 850},
        "4_17": {"name": "第17节：核对订单话术和在线支付", "time": 519},
        "4_18": {"name": "第18节：竞价发货流程", "time": 1253},
        "4_19": {"name": "第19节：冷门产品总结和答疑", "time": 811},
        "5_1": {"name": "波总分享竞价日利润3000+实战经验", "time": 3010},
        "5_2": {"name": "王总分享竞价从0到2000利润的实战经验", "time": 3948},
        "6_1": {"name": "第1节：微信加粉的思维框架", "time": 1394},
        "6_2": {"name": "第2节：微信加粉选产品的标准", "time": 670},
        "6_3": {"name": "第3节：常用找产品的方法", "time": 728},
        "6_4": {"name": "第4节：食品类找产品方法演示", "time": 767},
        "6_5": {"name": "第5节：文玩类找产品方法演示", "time": 462},
        "6_6": {"name": "第6节：其他找产品的方法", "time": 938},
        "6_7": {"name": "第7节：如何保证产品能够赚钱", "time": 1817},
        "7_8": {"name": "第8节：竞价单页基础知识", "time": 379},
        "7_9": {"name": "第9节：高转化竞价单页设计", "time": 1337},
        "8_10": {"name": "第10节：竞价账户操作基础", "time": 1845},
        "8_11": {"name": "第11节：百度竞价账户搭建", "time": 2446},
        "8_12": {"name": "第12节：微信加粉产品测试方法", "time": 1348},
        "8_13": {"name": "第13节：百度竞价账户优化", "time": 2591},
        "9_14": {"name": "第14节：微信加分客服话术", "time": 1119},
        "9_15": {"name": "第15节：实物如何跟卖家谈代发", "time": 317},
        "9_16": {"name": "第16节：微信加粉课程总结", "time": 560},
    }
}
course = course[38]["1_1"]
file_name, record_time = course["name"], course["time"]
audio_tmp_name = f"{file_name}_tmp.wav"
video_tmp_name = f"{file_name}_tmp.mp4"
video_out_name = f"{file_name}.mp4"
if Path(audio_tmp_name).exists(): Path(audio_tmp_name).unlink()
if Path(audio_tmp_name).exists(): Path(video_tmp_name).unlink()
if Path(audio_tmp_name).exists(): Path(video_out_name).unlink()
fps = 14.5
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 48000

p = pyaudio.PyAudio()
wf = wave.open(audio_tmp_name, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
audio_record_flag = True


def callback(in_data, frame_count, time_info, status):
    wf.writeframes(in_data)
    if audio_record_flag:
        return in_data, pyaudio.paContinue
    else:
        return in_data, pyaudio.paComplete


stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
                channels=wf.getnchannels(),
                rate=wf.getframerate(),
                input=True,
                stream_callback=callback)
image = ImageGrab.grab()  # 获得当前屏幕
width = image.size[0]
height = image.size[1]
# print("width:", width, "height:", height)
# print("image mode:", image.mode)
k = np.zeros((width, height), np.uint8)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码格式
video = cv2.VideoWriter(video_tmp_name, fourcc, fps, (width, height))
# 经实际测试，单线程下最高帧率为10帧/秒，且会变动，因此选择9.5帧/秒
# 若设置帧率与实际帧率不一致，会导致视频时间与音频时间不一致

print("video recording!!!!!")
time.sleep(1.5)
stream.start_stream()
print("audio recording!!!!!")
s_time = time.time()
with mss.mss() as sct:
    sct.compression_level = 0
    # Part of the screen to capture
    monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080, "mon": 1}
    sct.grab(monitor)
    while True:
        last_time = time.time()
        # Get raw pixels from the screen, save it to a Numpy array
        img = sct.grab(monitor)

        print("fps: {}".format(1.0 / (time.time() - last_time)))
        img_bgra = np.array(img)
        img_bgr = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)  # 转为opencv的BGR格式
        video.write(img_bgr)

        if time.time() - s_time > record_time:
            break

audio_record_flag = False
while stream.is_active():
    time.sleep(1)

stream.stop_stream()
stream.close()
wf.close()
p.terminate()
print("audio recording done!!!!!")

video.release()
cv2.destroyAllWindows()
print("video recording done!!!!!")

print("video audio merge!!!!!")
# audioclip = AudioFileClip(audio_name)
# videoclip = VideoFileClip(video_name)
# videoclip2 = videoclip.set_audio(audioclip)
# video = CompositeVideoClip([videoclip2])
# video.write_videofile(video_out_name,
#                       codec='mpeg4',
#                       # bitrate="3000k",
#                       threads=100)
subprocess.call(f"ffmpeg -i {video_tmp_name} -i {audio_tmp_name} {video_out_name}", shell=True)
