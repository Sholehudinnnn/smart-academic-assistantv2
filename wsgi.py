import os
import sys

ROOT_DIR = os.path.dirname(__file__)
APP_DIR = os.path.join(ROOT_DIR, "")

if APP_DIR not in sys.path:
    sys.path.insert(0, APP_DIR)

from app import app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "5000")))
