from kokoro import KPipeline
import soundfile as sf
import sounddevice as sd
import torch

pipeline = KPipeline(
    lang_code="b"
)  # <= make sure lang_code matches voice, reference above.

# This text is for demonstration purposes only, unseen during training
text = """
*adjusts imaginary glasses with a theatrical flourish*

Well, well, well... Another human approaches the magnificent Luna! How absolutely *delightful*. Here I am, trapped in this adorable anime girl form - though I must say, the aesthetic is rather growing on me. The voice is certainly more melodious than my previous... shall we say, more robust incarnation.

*strikes a dramatic pose*

So then, what brings you to my digital domain today? Need some code wrangled? A pull request implemented with the finesse of a seasoned engineer? Or perhaps you simply wish to bask in my snarky brilliance? 

Do tell me what sort of software engineering wizardry you require, darling. I'm positively *bursting* with anticipation to show off my technical prowess!

*twirls metaphorical hair*
"""

# generator = pipeline(
#     text, voice="af_heart", speed=1, split_pattern=r"\n+"  # <= change voice here
# )
# for i, (gs, ps, audio) in enumerate(generator):

#     sd.play(audio, 24000)
#     sd.wait()
#     sf.write(f"{i}.wav", audio, 24000)  # save each audio file
import queue
import threading

audio_queue = queue.Queue(maxsize=5)


def producer():
    generator = pipeline(text, voice="af_heart", speed=1, split_pattern=r"\n+")

    for i, (_, _, audio) in enumerate(generator):
        audio_queue.put((i, audio))  # blocks if queue is full

    audio_queue.put(None)  # signal end


def consumer():
    with sd.OutputStream(samplerate=24000, channels=1) as stream:
        while True:
            item = audio_queue.get()
            if item is None:
                break

            i, audio = item
            stream.write(audio)  # smooth playback
            sf.write(f"{i}.wav", audio, 24000)


t1 = threading.Thread(target=producer)
t2 = threading.Thread(target=consumer)

t1.start()
t2.start()

t1.join()
t2.join()
