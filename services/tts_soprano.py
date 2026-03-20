from soprano import SopranoTTS
from soprano.utils.streaming import play_stream

model = SopranoTTS(
    backend="auto", device="auto", cache_size_mb=100, decoder_batch_size=1
)
text = "I  say chap! How are you doing today?"

out = model.infer_stream(text)
play_stream(out)
