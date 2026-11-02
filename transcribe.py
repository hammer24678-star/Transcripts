"""Transcribe one audio file with a Quran-tuned Whisper model -> raw JSON of timed words.
usage: python transcribe.py AUDIO OUT.json MODEL [BEAMS]"""
import sys, json, re
from transformers import pipeline, GenerationConfig
audio, out, model = sys.argv[1], sys.argv[2], sys.argv[3]
beams = int(sys.argv[4]) if len(sys.argv) > 4 else 1
pipe = pipeline("automatic-speech-recognition", model=model, device=-1, chunk_length_s=30, stride_length_s=(6, 6))
gen = {"language": "arabic", "task": "transcribe", "num_beams": beams}
def run(ts):
    return pipe(audio, return_timestamps=ts, generate_kwargs=gen)
mode = "word"
try:
    if not getattr(pipe.model.generation_config, "alignment_heads", None):
        # older fine-tunes lack alignment heads; borrow the ones from the base checkpoint of the same size
        size = "tiny" if "tiny" in model else "small" if "small" in model else "medium" if "medium" in model else "base"
        pipe.model.generation_config.alignment_heads = GenerationConfig.from_pretrained(f"openai/whisper-{size}").alignment_heads
    res = run("word")
    words = [{"w": c["text"].strip(), "t0": c["timestamp"][0], "t1": c["timestamp"][1]} for c in res["chunks"] if c["text"].strip()]
except Exception as e:
    print("word timestamps failed, falling back to segment timestamps:", repr(e)[:300])
    mode = "segment"
    res = run(True); words = []
    for c in res["chunks"]:
        ws = c["text"].split(); t0, t1 = c["timestamp"]
        t1 = t1 if t1 is not None else t0 + 0.4 * len(ws)
        for i, w in enumerate(ws):   # evenly interpolated inside the segment (approximate)
            words.append({"w": w, "t0": t0 + (t1 - t0) * i / len(ws), "t1": t0 + (t1 - t0) * (i + 1) / len(ws)})
json.dump({"model": model, "audio": audio, "timestamp_mode": mode, "beams": beams, "text": res["text"], "words": words},
          open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
print("words:", len(words), "mode:", mode)
