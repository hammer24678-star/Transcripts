"""Parse ARQURAN.TXT (page-based; ayah numbers as {n}; '@' marks page-first lines;
'*2* N. surah' marks a surah start) into quran.json [{"s","a","text"}].
The standalone basmalah that opens each surah (except Al-Fatiha, where it is ayah 1) is dropped."""
import re, json, sys
src, dst = sys.argv[1], sys.argv[2]
lines = open(src, encoding="utf-8", errors="replace").read().replace("\r", "").split("\n")
DIAC = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭـ]")
def plain(s): return re.sub(r"\s+", " ", DIAC.sub("", s)).strip()
BASM = "بسم الله الرحمن الرحيم"
surah, buf, fresh = None, {}, False
for l in lines:
    if l.startswith("*EOF*"): break
    m = re.match(r"\*2\*\s*(\d+)\s*\.?\s*سورة", l)
    if m:
        n = int(m.group(1))
        if n != surah: surah, fresh = n, True; buf.setdefault(n, [])
        continue
    if l.startswith("*") or surah is None: continue
    l = l.strip().lstrip("@").strip()
    if not l: continue
    if plain(l) == BASM and surah != 1:       # standalone basmalah = a surah opening
        if fresh: fresh = False
        else: surah += 1; buf.setdefault(surah, [])
        continue
    fresh = False
    buf[surah].append(l)
COUNTS = [7,286,200,176,120,165,206,75,129,109,123,111,43,52,99,128,111,110,98,135,112,78,118,64,77,227,93,88,69,60,34,30,73,54,45,83,182,88,75,85,54,53,89,59,37,35,38,29,18,45,60,49,62,55,78,96,29,22,24,13,14,11,11,18,12,12,30,52,52,44,28,28,20,56,40,31,50,40,46,42,29,19,36,25,22,17,19,26,30,20,15,21,11,8,8,19,5,8,8,11,11,8,3,9,5,4,7,3,6,3,5,4,5,6]
out, problems = [], []
for s_ in sorted(buf):
    # source typos: some markers are written (n) instead of {n}
    parts = re.split(r"[{(](\d+)[})]", " ".join(buf[s_]))
    texts = [re.sub(r"\s+", " ", parts[i]).strip() for i in range(0, len(parts) - 1, 2)]
    texts = [t for t in texts if t]
    if s_ == 38 and len(texts) == 87:           # marker for 38:40 missing in the source file
        for i, t in enumerate(texts):
            w = t.split(); pw = [re.sub("[أإآ]", "ا", plain(x)) for x in w]
            for k in range(len(w) - 2):
                if pw[k:k+3] == ["واذكر", "عبدنا", "ايوب"]:
                    texts[i:i+1] = [" ".join(w[:k]), " ".join(w[k:])]; break
            else: continue
            break
    if len(texts) != COUNTS[s_ - 1]: problems.append((s_, len(texts), COUNTS[s_ - 1]))
    for n, t in enumerate(texts, 1): out.append({"s": s_, "a": n, "text": t})
json.dump(out, open(dst, "w", encoding="utf-8"), ensure_ascii=False)
print(len(out), "ayat,", len(buf), "surahs; count mismatches vs canonical:", problems)
