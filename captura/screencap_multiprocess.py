# coding=utf-8
import subprocess
import time
import wave
from multiprocessing import Process, Queue
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
audio_name = OUT_PATH / f"tmp.wav"
video_tmp_name = OUT_PATH / f"tmp.mp4"
video_name = OUT_PATH / f"{time_s}.mp4"

p = pyaudio.PyAudio()
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 44100
SAMPLE_WIDTH = p.get_sample_size(FORMAT)

audio_frames = []
audio_record_flag = True


def gen_audio():
    with wave.open(str(audio_name), 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(RATE)
        for _frame in audio_frames:
            wf.writeframes(_frame)


def process_image(queue_image):
    """do some operations and save to image"""
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码格式
    video = cv2.VideoWriter(str(video_tmp_name), fourcc, cfg["fps"], (cfg["width"], cfg["height"]))
    i = 0
    while True:
        img = queue_image.get()
        if img is None:
            video.release()
            break
        img_bgra = np.array(img)
        img_bgr = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)  # 转为opencv的BGR格式
        video.write(img_bgr)
        # cv2.imwrite(str(OUT_PATH / f"{i:08d}.jpg"), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])
        i += 1


def clean():
    for img in OUT_PATH.glob("*.jpg"):
        img.unlink()
    # for audio in OUT_PATH.glob("*.wav"):
    #     audio.unlink()


def callback(in_data, frame_count, time_info, status):
    audio_frames.append(in_data)
    if audio_record_flag:
        return in_data, pyaudio.paContinue
    else:
        return in_data, pyaudio.paComplete


def record_image(queue_image):
    with mss.mss() as sct:
        sct.compression_level = 0
        # Part of the screen to capture
        monitor = {"top": cfg["top"], "left": cfg["left"],
                   "width": cfg["width"], "height": cfg["height"], "mon": cfg["mon"]}
        sct.grab(monitor)  # first grab screen
        s_time = time.time()
        while True:
            img = sct.grab(monitor)
            queue_image.put(img)
            if time.time() - s_time > cfg["record_time"]:
                break
        queue_image.put(None)


def record(queue_image):
    print("开始录屏，倒计时2秒。")
    time.sleep(2)
    global audio_record_flag
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, stream_callback=callback)
    stream.start_stream()
    record_image(queue_image)
    audio_record_flag = False
    while stream.is_active():
        time.sleep(0.1)
    stream.stop_stream()
    stream.close()
    p.terminate()
    print("录屏结束。")
    gen_audio()


if __name__ == "__main__":
    clean()
    queue_image = Queue()
    p1 = Process(target=process_image, args=(queue_image,))
    p1.start()
    record(queue_image)
    # p2 = Process(target=record, args=(queue_image,))
    # p2.start()
    while True:
        if p1.is_alive():
            time.sleep(5)
            print(queue_image.qsize())
        else:
            subprocess.call(
                # f"ffmpeg -f image2 -i {OUT_PATH}/%08d.jpg -i {audio_name} -t {cfg['record_time']} {video_name}",
                f"ffmpeg -f image2 -i {video_tmp_name} -i {audio_name} -t {cfg['record_time']} {video_name}",
                shell=True
            )
            break
    clean()
