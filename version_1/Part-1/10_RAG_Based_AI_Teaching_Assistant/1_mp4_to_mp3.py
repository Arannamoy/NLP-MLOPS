import os
import whisper
import subprocess
import tensorflow as tf

files=os.listdir("music")
print(files)

for file in files:
    file_no=file.split("-")[1].split(".")[0]
    print(file_no,file)
    # subprocess.run(["python","./9_Tensorflow/1_Intro.py"]) # usage of subprocess
    subprocess.run(["ffmpeg","-i",f"music/{file}",f"audios/{file_no}_music.mp3"]) # this line convert mp4 to mp3
