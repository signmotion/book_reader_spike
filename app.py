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


# Use dataset and transformers directly.
def text2speech_with_transformers_microsoft(n: int, text: str):
    processor = SpeechT5Processor.from_pretrained("microsoft/speecht5_tts")
    model = SpeechT5ForTextToSpeech.from_pretrained("microsoft/speecht5_tts")
    vocoder = SpeechT5HifiGan.from_pretrained("microsoft/speecht5_hifigan")

    inputs = processor(text=text, return_tensors="pt")

    # load xvector containing speaker's voice characteristics from a dataset
    embeddings_dataset = load_dataset(
        "Matthijs/cmu-arctic-xvectors",
        split="validation")
    speaker_embeddings = torch.tensor(
        embeddings_dataset[7306]["xvector"]).unsqueeze(0)

    speech = model.generate_speech(
        inputs["input_ids"], speaker_embeddings, vocoder=vocoder)

    sf.write(speech_file(n), speech.numpy(),
             samplerate=16000, format="OGG")


def speech_file(n: int):
    return f"{n}_speech.ogg"


def main():
    with open(f"text.txt", "r") as file:
        n = 0
        lines = file.readlines()
        for line in lines:
            text = line.strip()
            if text:
                n += 1
                print(f"\n{n}\t{text}")
                if not os.path.exists(speech_file(n)):
                    text2speech(n, text)


if __name__ == "__main__":
    main()
