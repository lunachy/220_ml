import os
import subprocess
import sys
import time
import wave
from pathlib import Path

import cv2
import mss
import numpy as np
import pyaudio

from captura.courses import COURSES

time_s = time.strftime('%H_%M_%S', time.localtime(time.time()))

course = COURSES[38]["4_19"]
file_name, record_time = course["name"], course["time"]
audio_tmp_name = f"{file_name}_tmp.wav"
video_tmp_name = f"{file_name}_tmp.mp4"
video_out_name = f"{file_name}.mp4"
if Path(audio_tmp_name).exists(): Path(audio_tmp_name).unlink()
if Path(audio_tmp_name).exists(): Path(video_tmp_name).unlink()
if Path(audio_tmp_name).exists(): Path(video_out_name).unlink()
fps = 14.5

width, height = 1920, 1080

# print("video recording!!!!!")
# time.sleep(2)
print("audio recording!!!!!")

p = pyaudio.PyAudio()
FORMAT = pyaudio.paInt16
CHANNELS = 2
RATE = 48000
SAMPLE_WIDTH = p.get_sample_size(FORMAT)


def gen_audio(_frames, audio_name):
    with wave.open(audio_name, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(SAMPLE_WIDTH)
        wf.setframerate(RATE)
        for _frame in _frames:
            wf.writeframes(_frame)


def gen_image_video(_frames, image_dir="", video_name="", gen_image=True, gen_video=True):
    if gen_video:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # 编码格式
        video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    for i, _frame in enumerate(_frames):
        img_bgra = np.array(_frame)
        img_bgr = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)  # 转为opencv的BGR格式
        if gen_image:
            cv2.imwrite(os.path.join(image_dir, f"{i:08d}.jpg"), img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 100])
            # cv2.imwrite(os.path.join(image_dir, f"{i:08d}.png"), img_bgr, [cv2.IMWRITE_PNG_COMPRESSION, 0])

        if gen_video:
            video.write(img_bgr)

        # if i == 3: break
    subprocess.call(f"ffmpeg -f image2 -i {image_dir}/%08d.jpg {image_dir}/jpg.mp4", shell=True)
    # subprocess.call(f"ffmpeg -f image2 -i {image_dir}/%08d.png -t 10 {image_dir}/png.mp4", shell=True)


# audio_record_flag = True

audio_frames = []
image_frames = []
audio_record_flag = True


def callback(in_data, frame_count, time_info, status):
    # wf.writeframes(in_data)
    audio_frames.append(in_data)
    if audio_record_flag:
        return in_data, pyaudio.paContinue
    else:
        return in_data, pyaudio.paComplete


def record_image(rec_time):
    with mss.mss() as sct:
        sct.compression_level = 0
        # Part of the screen to capture
        monitor = {"top": 0, "left": 0, "width": 1920, "height": 1080, "mon": 1}
        sct.grab(monitor)  # first grab screen
        s_time = time.time()
        while True:
            last_time = time.time()
            # Get raw pixels from the screen, save it to a Numpy array
            img = sct.grab(monitor)
            image_frames.append(img)
            # # print("fps: {}".format(1.0 / (time.time() - last_time)))
            # img_bgra = np.array(img)
            # img_bgr = cv2.cvtColor(img_bgra, cv2.COLOR_BGRA2BGR)  # 转为opencv的BGR格式
            # video.write(img_bgr)

            if time.time() - s_time > rec_time:
                break


# exit()
# stream.stop_stream()
# stream.close()
# wf.close()
# p.terminate()
# print("audio recording done!!!!!")
#
# video.release()
# cv2.destroyAllWindows()
# print("video recording done!!!!!")
#
# print("video audio merge!!!!!")
# # subprocess.call(f"ffmpeg -i {video_tmp_name} -i {audio_tmp_name} {video_out_name}", shell=True)
def get_size(obj, seen=None):
    # From
    # Recursively finds size of objects
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size


if __name__ == "__main__":
    stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, stream_callback=callback)
    stream.start_stream()

    record_image(10)
    audio_record_flag = False
    while stream.is_active():
        time.sleep(1)
    stream.stop_stream()
    stream.close()
    p.terminate()

    gen_audio(audio_frames, "D:/test/test.wav")
    gen_image_video(image_frames, "D:/test", "D:/test/test.mp4")

    print(get_size(audio_frames))
    print(get_size(audio_frames[0]))
    print(get_size(image_frames))
    print(get_size(image_frames[0]))
    s = 'abc'
    print(sys.getsizeof(s))
    print(get_size(s))
    pass
