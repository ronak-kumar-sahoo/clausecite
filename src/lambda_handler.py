import os, shutil
HERE = os.path.dirname(os.path.abspath(__file__))
if not os.path.isdir("/tmp/chroma_db"):
    shutil.copytree(os.path.join(HERE, "chroma_db"), "/tmp/chroma_db")
os.environ["CHROMA_DIR"] = "/tmp/chroma_db"
from mangum import Mangum
from api import api
handler = Mangum(api, lifespan="off")
