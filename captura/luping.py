import wave
import pyaudio
import mss
import numpy as np
import cv2
from moviepy.editor import *
import time
from multiprocessing import Process, Queue

record_time = 10
height, width, fps = 1080, 1920, 20
fourcc = cv2.VideoWriter_fourcc('M', 'P', '4', 'V')  # 编码格式
video = cv2.VideoWriter('test.mp4', fourcc, fps, (width, height))


def record_audio(p, wf):
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
    return stream


def record_image(queue):
    s_time = time.perf_counter()
    with mss.mss() as sct:
        sct.compression_level = 0
        # Part of the screen to capture
        monitor = {"top": 0, "left": 0, "width": width, "height": height, "mon": 1}
        sct.grab(monitor)
        while True:
            last_time = time.perf_counter()
            # Get raw pixels from the screen, save it to a Numpy array
            queue.put(sct.grab(monitor))
            print("fps: {}".format(1.0 / (time.perf_counter() - last_time)))
            if time.perf_counter() - s_time > record_time:
                break


def write_image(queue):
    while True:
        img = queue.get()
        if img is None:
            break
        img = queue.get()
        img_rgb = np.array(img)
        img_bgr = cv2.cvtColor(np.array(img_rgb), cv2.COLOR_RGB2BGR)  # 转为opencv的BGR格式
        video.write(img_bgr)


if __name__ == "__main__":
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 2
    RATE = 44100
    WAVE_OUTPUT_FILENAME = "output.wav"

    p = pyaudio.PyAudio()
    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    audio_record_flag = True

    stream = record_audio(p, wf)
    stream.start_stream()
    print("audio recording!!!!!")

    queue = Queue()
    print("video recording!!!!!")
    p1=Process(target=record_image, args=(queue,))
    p2=Process(target=write_image, args=(queue,))
    p1.start(), p2.start()
    p1.join(), p2.join()
    # 经实际测试，单线程下最高帧率为10帧/秒，且会变动，因此选择9.5帧/秒
    # 若设置帧率与实际帧率不一致，会导致视频时间与音频时间不一致
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
    audioclip = AudioFileClip("output.wav")
    videoclip = VideoFileClip("test.mp4")
    videoclip2 = videoclip.set_audio(audioclip)
    video = CompositeVideoClip([videoclip2])
    video.write_videofile("test2default.mp4", codec='mpeg4', threads=200)
