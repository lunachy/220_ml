import subprocess
import time
import wave
from pathlib import Path

import cv2
import mss
import numpy as np
import pyaudio
import yaml

CUR_PATH = Path(__file__).parent
OUT_PATH = CUR_PATH / "output"
OUT_PATH.mkdir(exist_ok=True)
with open(CUR_PATH / "config.yaml", encoding="utf-8") as f:
    cfg = yaml.load(f, Loader=yaml.Loader)

time_s = time.strftime('%Y%m%d_%H%M%S', time.localtime(time.time()))

audio_name = OUT_PATH / f"{time_s}.wav"
video_tmp1_name = OUT_PATH / f"{time_s}_tmp1.mp4"
video_tmp2_name = OUT_PATH / f"{time_s}_tmp2.mp4"
video_name = OUT_PATH / f"{time_s}.mp4"

p = pyaudio.PyAudio()
CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
SAMPLE_WIDTH = p.get_sample_size(FORMAT)

wf = wave.open(str(audio_name), 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(SAMPLE_WIDTH)
wf.setframerate(RATE)
audio_record_flag = True


def callback(in_data, frame_count, time_info, status):
    wf.writeframes(in_data)
    if audio_record_flag:
        return in_data, pyaudio.paContinue
    else:
        return in_data, pyaudio.paComplete


stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, stream_callback=callback)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码格式

i = 1
with mss.mss() as sct:
    video_tmp1 = cv2.VideoWriter(str(video_tmp1_name), fourcc, cfg["fps"], (cfg["width"], cfg["height"]))
    sct.compression_level = 0
    # Part of the screen to capture
    monitor = {"top": cfg["top"], "left": cfg["left"],
               "width": cfg["width"], "height": cfg["height"], "mon": cfg["mon"]}
    sct.grab(monitor)
    print(f"start recording, wait {cfg['countdown']} seconds!")
    time.sleep(cfg["countdown"])
    stream.start_stream()
    s_time = time.time()

    while True:
        last_time = time.time()
        # Get raw pixels from the screen, save it to a Numpy array
        img = sct.grab(monitor)

        # print("fps: {}".format(1.0 / (time.time() - last_time)))
        img_bgra = np.array(img)
        img_bgr = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)  # 转为opencv的BGR格式
        video_tmp1.write(img_bgr)

        if time.time() - s_time > cfg["record_time"]:
            break
        i += 1
    video_tmp1.release()
    print(f"grab {i} images in {cfg['record_time']}s, fps: {i / cfg['record_time']}")

real_fps = i / cfg['record_time']

audio_record_flag = False
while stream.is_active():
    time.sleep(0.1)

stream.stop_stream()
stream.close()
wf.close()
p.terminate()
print("finish recording.")

cap = cv2.VideoCapture(str(video_tmp1_name))
video_tmp2 = cv2.VideoWriter(str(video_tmp2_name), fourcc, real_fps, (cfg["width"], cfg["height"]))
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break
    video_tmp2.write(frame)
video_tmp2.release()
cap.release()

subprocess.call(f"ffmpeg -i {video_tmp2_name} -i {audio_name} {video_name}", shell=True)

if not cfg["debug"]:
    audio_name.unlink()
    video_tmp1_name.unlink()
    video_tmp2_name.unlink()
