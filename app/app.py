import json
import os
import subprocess
import sys
import threading
import time

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.append(os.path.join(ROOT, "src"))

from flask import Flask, jsonify, render_template, request

import predict

app = Flask(__name__)
TRAIN_SCRIPT = os.path.join(ROOT, "src", "train_all.py")
STATE = {"running": False, "log": [], "returncode": None, "started": None}


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def do_predict():
    try:
        return jsonify(predict.run(request.get_json()))
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/metrics")
def metrics():
    path = os.path.join(ROOT, "models", "metrics.json")
    if not os.path.exists(path):
        return jsonify({})
    with open(path) as f:
        return jsonify(json.load(f))


def _run_training():
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    p = subprocess.Popen([sys.executable, "-u", TRAIN_SCRIPT], cwd=ROOT, env=env,
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                         text=True, encoding="utf-8", errors="replace", bufsize=1)
    for line in p.stdout:
        STATE["log"].append(line.rstrip())
    p.wait()
    code = p.returncode

    finished_ok = any("DONE" in line for line in STATE["log"])
    if code == 0 and not finished_ok:
        STATE["log"].append("ERROR: the training script ended without doing any work. "
                            "Open src\\train_all.py, make sure it contains the full code, and save it.")
        code = 1

    if code == 0:
        try:
            predict.load()
        except Exception as e:
            STATE["log"].append(f"ERROR: could not load the new models: {e}")
            code = 1

    STATE["returncode"] = code
    STATE["running"] = False


@app.route("/train", methods=["POST"])
def train():
    if STATE["running"]:
        return jsonify({"started": False, "reason": "Training is already running."})
    STATE.update(running=True, log=[], returncode=None, started=time.time())
    threading.Thread(target=_run_training, daemon=True).start()
    return jsonify({"started": True})


@app.route("/train_status")
def train_status():
    elapsed = int(time.time() - STATE["started"]) if STATE["started"] else 0
    return jsonify({"running": STATE["running"], "log": STATE["log"],
                    "returncode": STATE["returncode"], "elapsed": elapsed})


if __name__ == "__main__":
    app.run(debug=False, threaded=True)