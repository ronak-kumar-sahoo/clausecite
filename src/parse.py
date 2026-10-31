import re, json, bisect, pymupdf
from pathlib import Path

doc = pymupdf.open("data/raw/dpdp_act.pdf")
full, offsets, note = "", [], ""
for i in range(4, 24):  # PDF pages 5-24: Act + Schedule
    t = doc[i].get_text()
    m = re.search(r"\*\s*1\..*?s\. 3\(i\)\.", t, re.S)
    if m:
        note = re.sub(r"\s+", " ", m.group(0))
        t = t.replace(m.group(0), "")
    t = re.sub(r"IndiaCode[ \t]*\d{0,2}", "", t)
    t = re.sub(r"such date\s*1\s*as", "such date as", t)
    t = re.sub(r"\*\s*different", "different", t)
    offsets.append(len(full))
    full += t + "\n"

start = full.index("as follows:") + len("as follows:")
rest = full[start:]
parts = re.split(r"\n\s*THE SCHEDULE\s*\n", rest, maxsplit=1)
body = parts[0]
schedule = parts[1].strip() if len(parts) > 1 else ""

chap = [(m.start(), m.end(), m.group(1), m.group(2).strip())
        for m in re.finditer(r"(?m)^CHAPTER ([IVX]+)\s*\n([A-Z ,]+)\n", body)]
sec = [(m.start(), m.end(), int(m.group(1)), re.sub(r"\s+", " ", m.group(2)).strip())
       for m in re.finditer(
           r"(?m)^[ \t]*(\d{1,2})\.[ \t]+([A-Z][^\n.\u2014\u2013]*(?:\n[^\n.\u2014\u2013]*)?)\.\s*[\u2014\u2013-]+", body)]

nums = [s[2] for s in sec]
print("sections found:", len(nums), "| sequence OK:", nums == list(range(1, 45)))

events = sorted([(a, b, "c", r, t) for a, b, r, t in chap] +
                [(a, b, "s", n, t) for a, b, n, t in sec])
cur, sections = ("", ""), []
for k, (a, b, kind, x, y) in enumerate(events):
    if kind == "c":
        cur = (x, y)
        continue
    end = events[k + 1][0] if k + 1 < len(events) else len(body)
    text = re.sub(r"[ \t]+", " ", body[b:end]).strip()
    text = re.sub(r"\s*\n\s*", "\n", text)
    page = 5 + bisect.bisect_right(offsets, start + a) - 1
    sections.append({"id": f"s{x}", "act": "DPDP Act 2023", "section": x, "title": y,
                     "chapter": cur[0], "chapter_title": cur[1], "page": page, "text": text})

out = {"act": "The Digital Personal Data Protection Act, 2023 (Act 22 of 2023)",
       "as_on": "2025-11-19", "source": "India Code", "commencement_note": note,
       "sections": sections, "schedule": schedule}
Path("data/processed").mkdir(parents=True, exist_ok=True)
Path("data/processed/sections.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")

for s in sections[:3] + sections[-2:]:
    print(s["section"], s["title"], "| ch", s["chapter"], "| p", s["page"], "|", len(s["text"]), "chars")
print("note:", note[:70], "| schedule chars:", len(schedule))
