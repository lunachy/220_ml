import wave
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
audio_name = f"{time_s}.wav"
video_name = f"{time_s}.mp4"
video_out_name = f"{time_s}_out.mp4"
video_sub_name = f"{time_s}_sub.mp4"
video_out_avi = f"{time_s}_out.avi"
record_time = 15
fps = 14.6
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100

p = pyaudio.PyAudio()
wf = wave.open(audio_name, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
audio_record_flag = True


def callback(in_data, frame_count, time_info, status):
    wf.writeframes(in_data)
    if audio_record_flag:
        return (in_data, pyaudio.paContinue)
    else:
        return (in_data, pyaudio.paComplete)


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

fourcc = cv2.VideoWriter_fourcc(*'XVID')  # 编码格式
video = cv2.VideoWriter(f'{time_s}.mp4', fourcc, fps, (width, height))
# 经实际测试，单线程下最高帧率为10帧/秒，且会变动，因此选择9.5帧/秒
# 若设置帧率与实际帧率不一致，会导致视频时间与音频时间不一致

print("video recording!!!!!")
time.sleep(1)
stream.start_stream()
print("audio recording!!!!!")
s_time = time.perf_counter()
with mss.mss() as sct:
    sct.compression_level = 0
    # Part of the screen to capture
    monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080, "mon": 1}
    sct.grab(monitor)
    while True:
        last_time = time.perf_counter()
        # Get raw pixels from the screen, save it to a Numpy array
        img = sct.grab(monitor)

        print("fps: {}".format(1.0 / (time.perf_counter() - last_time)))
        img_rgb = np.array(img)
        img_bgr = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)  # 转为opencv的BGR格式
        video.write(img_bgr)

        if time.perf_counter() - s_time > record_time:
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
subprocess.call(f"ffmpeg -i {video_name} -i {audio_name} {video_sub_name}", shell=True)
