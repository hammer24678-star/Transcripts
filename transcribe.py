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
def base_name(cfg):
    d = cfg.d_model
    return {384: "tiny", 512: "base", 768: "small", 1024: "medium"}.get(d, "large-v3" if getattr(cfg, "num_mel_bins", 80) == 128 else "large-v2")
gc = pipe.model.generation_config
if not getattr(gc, "no_timestamps_token_id", None) or not getattr(gc, "alignment_heads", None):
    # old fine-tunes ship an incomplete generation config; the base checkpoint's one is token-compatible
    ref = GenerationConfig.from_pretrained("openai/whisper-" + base_name(pipe.model.config))
    pipe.model.generation_config = ref
    print("using generation config from openai/whisper-" + base_name(pipe.model.config))
try:
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
