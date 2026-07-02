import whisper
import pandas as pd
import json
import torch

gpu_count=torch.cuda.device_count()

print("GPU Count:",gpu_count)

model=whisper.load_model("tiny").to("cuda")

result=model.transcribe(audio="./audios/1.mp3",language="bn",task="Translate")


chunks=[]


for segment in result["segments"]:
    chunks.append({"start":segment["start"],"end":segment["end"],"text":segment["text"]})


print(result["text"])

#with open("./10_RAG_Based_AI_Teaching_Assistant/Data/2_mp3_to_text.json","w") as fl:
#     json.dump(chunks,fl)
