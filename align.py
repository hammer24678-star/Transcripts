"""Match ASR words against the full Quran text, keeping every repetition.
usage: python align.py RAW.json quran.json OUT.json
Output entries are runs of consecutive Quran words; a run whose start lies at/before the end of an earlier
run is marked repeat=true. Nothing is merged or skipped: unmatched ASR words are listed explicitly."""
import sys, json, re, difflib, collections
DIAC = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
MAP = str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ٱ": "ا", "ى": "ي", "ة": "ه", "ؤ": "و", "ئ": "ي", "ء": ""})
def norm(w): return DIAC.sub("", w).translate(MAP)
def sim(a, b): return 1.0 if a == b else difflib.SequenceMatcher(None, a, b).ratio()

def load_quran(path):
    W = []                                  # (surah, ayah, word_index_in_ayah, display, norm)
    for o in json.load(open(path, encoding="utf-8")):
        for k, w in enumerate(o["text"].split(), 1):
            if norm(w): W.append((o["s"], o["a"], k, w, norm(w)))
    return W

def align(asr, W, accept=0.72, anchor_n=3):
    N = [norm(a["w"]) for a in asr]
    idx = collections.defaultdict(list)
    for j in range(len(W) - anchor_n + 1): idx[tuple(x[4] for x in W[j:j + anchor_n])].append(j)
    # fuzzy anchor: also index by first 3 letters per word to survive 1-letter ASR errors
    pidx = collections.defaultdict(list)
    for j in range(len(W) - anchor_n + 1): pidx[tuple(x[4][:3] for x in W[j:j + anchor_n])].append(j)
    pos = [None] * len(asr); score = [0.0] * len(asr); p = None; i = 0
    while i < len(asr):
        if not N[i]: i += 1; continue
        if p is not None and p + 1 < len(W):
            s0 = sim(N[i], W[p + 1][4])
            if s0 >= accept: pos[i], score[i], p = p + 1, s0, p + 1; i += 1; continue
            # ASR dropped one Quran word
            if p + 2 < len(W):
                s1 = sim(N[i], W[p + 2][4])
                if s1 >= accept + 0.08 and (i + 1 >= len(asr) or sim(N[i + 1], W[p + 3][4] if p + 3 < len(W) else "") >= accept):
                    pos[i], score[i], p = p + 2, s1, p + 2; i += 1; continue
            # ASR inserted a spurious word
            if i + 1 < len(asr) and sim(N[i + 1], W[p + 1][4]) >= accept + 0.1:
                i += 1; continue
        # (re)locate: anchor on an n-gram of the next ASR words
        cand = []
        if i + anchor_n <= len(asr):
            key = tuple(N[i:i + anchor_n])
            cand = idx.get(key, []) or pidx.get(tuple(x[:3] for x in key), [])
            cand = [c for c in cand if sum(sim(N[i + t], W[c + t][4]) for t in range(anchor_n)) / anchor_n >= accept]
        if cand:
            if p is None: c = cand[0] if len(cand) == 1 else cand[0]
            else: c = min(cand, key=lambda c: (c <= p - 40 and c > p + 400, abs(c - (p + 1))))
            pos[i], score[i], p = c, sim(N[i], W[c][4]), c; i += 1; continue
        i += 1
    return pos, score

def runs(asr, pos, score, W):
    out, cur = [], None
    for i, q in enumerate(pos):
        if q is None:
            if cur: cur["unm"] += 1
            continue
        # continue a run across up to 3 words the ASR failed to hear (forward only; never across a repeat jump)
        if cur and 0 < q - cur["q1"] <= 4 and cur["unm"] <= 3:
            cur["q1"], cur["i1"], cur["unm"] = q, i, 0; cur["obs"][q] = i; cur["sc"].append(score[i])
        else:
            if cur: out.append(cur)
            cur = {"q0": q, "q1": q, "i0": i, "i1": i, "unm": 0, "sc": [score[i]], "obs": {q: i}}
    if cur: out.append(cur)
    res, max_end = [], -1
    for r in out:
        a0, a1 = W[r["q0"]], W[r["q1"]]
        rep = r["q0"] <= max_end; max_end = max(max_end, r["q1"])
        # canonical words with times; unheard words get interpolated times (observed=false)
        words = []
        qs = list(range(r["q0"], r["q1"] + 1)); known = sorted(r["obs"])
        for q in qs:
            if q in r["obs"]:
                t0, t1, ob = asr[r["obs"][q]]["t0"], asr[r["obs"][q]]["t1"], True
            else:
                lo = max(k for k in known if k < q); hi = min(k for k in known if k > q)
                a_, b_ = asr[r["obs"][lo]]["t1"], asr[r["obs"][hi]]["t0"]
                t0 = a_ + (b_ - a_) * (q - lo - 1) / (hi - lo - 1); t1 = a_ + (b_ - a_) * (q - lo) / (hi - lo - 1); ob = False
            words.append({"surah": W[q][0], "ayah": W[q][1], "k": W[q][2], "w": W[q][3], "t0": round(t0, 3), "t1": round(t1, 3), "observed": ob})
        per = []
        for q in qs:
            s_, a_, k_ = W[q][0], W[q][1], W[q][2]
            if not per or per[-1]["ayah"] != (s_, a_): per.append({"ayah": (s_, a_), "w0": k_, "w1": k_})
            else: per[-1]["w1"] = k_
        unheard = sum(not w["observed"] for w in words); ms = sum(r["sc"]) / len(r["sc"])
        starts_mid = a0[2] != 1
        ends_mid = not (r["q1"] + 1 >= len(W) or (W[r["q1"] + 1][0], W[r["q1"] + 1][1]) != (a1[0], a1[1]))
        res.append({
            "start": asr[r["i0"]]["t0"], "end": asr[r["i1"]]["t1"],
            "from": f"{a0[0]}:{a0[1]}:{a0[2]}", "to": f"{a1[0]}:{a1[1]}:{a1[2]}",
            "repeat": rep, "mean_match": round(ms, 3), "unheard_words": unheard,
            "ayat": [{"surah": p["ayah"][0], "ayah": p["ayah"][1], "word_from": p["w0"], "word_to": p["w1"]} for p in per],
            "text": " ".join(w["w"] for w in words), "asr_text": " ".join(a["w"] for a in asr[r["i0"]:r["i1"] + 1]),
            "starts_mid_ayah": starts_mid, "ends_mid_ayah": ends_mid,
            "words": words, "needs_review": ms < 0.88 or unheard > 0 or len(qs) < 3 or ends_mid,
        })
    return res

if __name__ == "__main__":
    raw = json.load(open(sys.argv[1], encoding="utf-8")); W = load_quran(sys.argv[2])
    pos, score = align(raw["words"], W); segs = runs(raw["words"], pos, score, W)
    un = [{"t": raw["words"][i]["t0"], "w": raw["words"][i]["w"]} for i, q in enumerate(pos) if q is None]
    json.dump({"source": raw["audio"], "model": raw["model"], "timestamp_mode": raw["timestamp_mode"],
               "asr_words": len(raw["words"]), "unmatched_asr_words": un, "segments": segs},
              open(sys.argv[3], "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(len(segs), "segments,", sum(s["repeat"] for s in segs), "repeats,", len(un), "unmatched,",
          sum(s["needs_review"] for s in segs), "to review")
