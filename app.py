# See the demo with voices:
# https://huggingface.co/spaces/Matthijs/speecht5-tts-demo

from datasets import load_dataset
import soundfile as sf
import torch
from transformers import SpeechT5Processor, SpeechT5ForTextToSpeech, SpeechT5HifiGan
from dotenv import load_dotenv
import os


# Models: https://huggingface.co/tasks


load_dotenv()
HUGGINGFACEHUB_API_TOKEN = os.getenv('HUGGINGFACEHUB_API_TOKEN')


def text2speech(n: int, text: str):
    text2speech_with_transformers_microsoft(n, text)


processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

# load xvector containing speaker's voice characteristics from a dataset
embeddings_dataset = load_dataset(
    "Matthijs/cmu-arctic-xvectors",
    split="validation")
speaker_embeddings = torch.tensor(
    embeddings_dataset[7306]["xvector"]).unsqueeze(0)


# Use dataset and transformers directly.
def text2speech_with_transformers_microsoft(file: str, text: str):
    inputs = processor(text=text, return_tensors="pt")
    speech = model.generate_speech(
        inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

    sf.write(file, speech.numpy(),
             samplerate=16000, format="OGG")


def speech_file(n: int, k: int):
    return f"{n}_{k}_speech.ogg"


# Returns a list with splitted paragraph.
def splitted(paragraph: str, limit: int = 300):
    r = []

    sentences = f"{paragraph}".split(". ")
    acc = ""
    for sentence in sentences:
        if len(acc) + len(sentence) > limit and len(sentence) > 2:
            r.append(acc)
            acc = ""
        acc += f"{sentence}. "

    if acc:
        r.append(acc[:-2])

    return r


def main():
    with open(f"text.txt", "r") as file:
        n = 0
        lines = file.readlines()
        for line in lines:
            text = line.strip()
            if text:
                n += 1
                k = 0
                splitted_text = splitted(text)
                for st in splitted_text:
                    k += 1
                    print(f"\n{n}_{k}\t{st}")
                    file = speech_file(n, k)
                    if not os.path.exists(file):
                        text2speech(file, st)


if __name__ == "__main__":
    main()
