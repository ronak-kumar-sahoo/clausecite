import sys
from graph import app

q = " ".join(sys.argv[1:])
for step in app.stream({"question": q}):
    for node, upd in step.items():
        short = {k: (v[:80] if isinstance(v, str) else ([d.metadata["section"] for d in v] if k == "docs" else v))
                 for k, v in upd.items()}
        print(node, short)