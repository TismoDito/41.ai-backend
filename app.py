from waitress import serve
from flask import Flask, request, jsonify, send_file, render_template
import threading
import queue
import uuid
import os
from utils import T2S


app = Flask(__name__)
# Job queue
job_queue = queue.Queue()
job_results = {}  # job_id -> {"status": ..., "audio_path": str or None, "error": str}

# --- Replace this with your actual model call ---
def tts_infer(arch, vocoder_arch, voice_name, text, speaker, torchmoji_overwrite, superress, maxdecode, gate_tresh, arpaconv, skip_sr) -> str:
    print(arch,vocoder_arch,voice_name, text, speaker, torchmoji_overwrite, superress, maxdecode, gate_tresh, arpaconv, skip_sr)
    if os.path.exists(os.path.join("Models",voice_name,"taco.pt")):
        tacopth = os.path.join("Models",voice_name,"taco.pt")
    elif os.path.exists(os.path.join("Models",voice_name,"taco.onnx")):
        tacopth = os.path.join("Models",voice_name,"taco.onnx")
    handler = T2S(arch=arch, vocoder_arch=vocoder_arch, taco_path=tacopth, vocoder_path=os.path.join("Models",voice_name,"vocoder"))
    audio_pth = handler.synthesize(
            text=text,
            speaker_id=speaker,
            torchmoji_text=torchmoji_overwrite or text,
            superress=superress,
            maxdecode=maxdecode,
            gate_tresh=gate_tresh,
            arpaconv=arpaconv,
            skip_sr=skip_sr
        )
    audio_pth = os.path.abspath(audio_pth)
    return audio_pth

# Worker thread
def worker():
    while True:
        job_id, arch, vocoder_arch, voice_name, text, speaker, torchmoji_overwrite, superress, maxdecode, gate_tresh, arpaconv, skip_sr = job_queue.get()
        try:
            audio_path = tts_infer(arch, vocoder_arch, voice_name, text, speaker, torchmoji_overwrite, superress, maxdecode, gate_tresh, arpaconv, skip_sr)
            job_results[job_id]["status"] = "done"
            job_results[job_id]["audio_path"] = audio_path
        except Exception as e:
            job_results[job_id]["status"] = "error"
            job_results[job_id]["error"] = str(e)
            print(e)
        job_queue.task_done()

# Start worker
for i in range(2):
    threading.Thread(target=worker, daemon=True).start()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/tts", methods=["POST"])
def tts():
    data = request.json
    if not data or "text" not in data or "voice" not in data:
        return jsonify({"error": "Missing 'text' or 'voice'"}), 400

    text = data["text"]
    voice = data["voice"]
    speaker = data.get("speaker", 0)
    superress = data.get("superress", 10)
    maxdecode = 3000
    gate_tresh = .25
    arpaconv = data.get("arpaconv", True)
    torchmoji_overwrite = data.get("torchmoji_overwrite", "")
    skip_sr = data.get("skip_sr", False)

    job_id = str(uuid.uuid4())
    job_results[job_id] = {"status": "pending", "audio_path": None, "error": None}
    job_queue.put((
        job_id,
        data.get("arch", "tacotron2"),
        data.get("vocoder_arch", "hifi-gan"),
        voice,
        text,
        speaker,
        torchmoji_overwrite,
        superress,
        maxdecode,
        gate_tresh,
        arpaconv,
        skip_sr
    ))

    return jsonify({"job_id": job_id, "status": "queued"})

from flask import jsonify

@app.route("/GetSpeakerIds/<source>", methods=["POST"])
def GetVoices(source):
    speakerids = os.path.join("Models", source, "speaker_id.txt")
    if not os.path.exists(speakerids):
        return jsonify({"error": "Path not found"}), 404

    voices = []
    with open(speakerids, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("|", 1)
            if len(parts) == 2:
                voices.append({"id": parts[0], "name": parts[1]})

    return jsonify({"voices": voices})

@app.route("/raw_metadata/<voice>")
def raw_metadata(voice):
    yaml_path = os.path.join("Models", voice, "config.yml")
    app.logger.debug("Attempting to send YAML file: %s", yaml_path)

    if not os.path.exists(yaml_path):
        app.logger.error("Metadata not found at: %s", yaml_path)
        return jsonify({"error": "Metadata not found"}), 404

    try:
        return send_file(yaml_path, mimetype="text/yaml")
    except Exception as e:
        app.logger.exception("Error sending metadata file:")
        return jsonify({"error": str(e)}), 500



@app.route("/status/<job_id>")
def status(job_id):
    if job_id not in job_results:
        return jsonify({"error": "Job not found"}), 404
    return jsonify({"status": job_results[job_id]["status"]})

@app.route("/download/<job_id>")
def download(job_id):
    if job_id not in job_results:
        return jsonify({"error": "Job not found"}), 404

    job = job_results[job_id]
    if job["status"] != "done":
        return jsonify({"error": "Audio not ready"}), 400

    audio_path = job["audio_path"]
    if not os.path.isfile(audio_path):
        return jsonify({"error": f"File not found: {audio_path}"}), 500

    return send_file(
        audio_path,
        mimetype="audio/wav",
        as_attachment=True,
        download_name=f"{job_id}.wav"
    )

@app.route("/voices")
def get_voices():
    """Return a list of immediate subdirectories in the Models directory."""
    return [name for name in os.listdir("Models") 
            if os.path.isdir(os.path.join("Models", name))]

@app.route("/queue/raw")
def raw_queue():
    return jsonify([str(x[0]) for x in list(job_queue.queue)])
    

# Cleanup route (optional)
@app.route("/cleanup/<job_id>", methods=["POST"])
def cleanup(job_id):
    """Delete old audio files after download"""
    if job_id in job_results and job_results[job_id]["audio_path"]:
        try:
            os.remove(job_results[job_id]["audio_path"])
        except FileNotFoundError:
            pass
        del job_results[job_id]
        return jsonify({"status": "deleted"})
    return jsonify({"error": "Job not found"}), 404


@app.route("/Credits")
def credits():
    return render_template("credits.html")

if __name__ == "__main__":
    serve(app, host="127.0.0.1", port=5000)