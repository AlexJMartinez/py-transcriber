#!/usr/bin/env python3
import argparse
import os
import time
import tempfile
import requests
import sounddevice as sd
import numpy as np
import soundfile as sf
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas
from datetime import timedelta

AAI_UPLOAD_URL = "https://api.assemblyai.com/v2/upload"
AAI_TRANSCRIBE_URL = "https://api.assemblyai.com/v2/transcript"


def read_audio_bytes(path):
    """Read audio file and return bytes."""
    with open(path, "rb") as f:
        return f.read()


def upload_to_assemblyai(path):
    """Upload audio to AssemblyAI and return the URL."""
    token = os.getenv("AAI_TOKEN")
    headers = {"authorization": token}
    data = read_audio_bytes(path)
    resp = requests.post(AAI_UPLOAD_URL, headers=headers, data=data)
    resp.raise_for_status()
    return resp.json()["upload_url"]


def request_transcription(audio_url):
    """Request transcription with speaker diarization and return the transcription ID."""
    token = os.getenv("AAI_TOKEN")
    headers = {
        "authorization": token,
        "content-type": "application/json"
    }
    payload = {
        "audio_url": audio_url,
        "speaker_labels": True,
        "format_text": True
    }
    resp = requests.post(AAI_TRANSCRIBE_URL, json=payload, headers=headers)
    resp.raise_for_status()
    return resp.json()["id"]


def poll_transcription(transcript_id, interval=5):
    """Poll AssemblyAI for completion, then return full response JSON."""
    token = os.getenv("AAI_TOKEN")
    headers = {"authorization": token}
    url = f"{AAI_TRANSCRIBE_URL}/{transcript_id}"
    while True:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        j = resp.json()
        if j.get("status") in ("completed", "error"):
            return j
        time.sleep(interval)


def format_segments(assembly_json):
    """Convert AssemblyAI JSON into a list of speaker-tagged segments."""
    labels_data = assembly_json.get("speaker_labels")
    # If diarization not provided, fallback to full transcript
    if not labels_data or not isinstance(labels_data, dict):
        duration = assembly_json.get("audio_duration", 0)
        return [{
            "speaker": "Speaker_0",
            "start": 0,
            "end": duration,
            "text": assembly_json.get("text", "")
        }]

    labels = labels_data.get("segments", [])
    words = assembly_json.get("words", [])
    segments = []
    for seg in labels:
        spk = seg.get("speaker_label")
        start = seg.get("start", 0) / 1000.0
        end = seg.get("end", 0) / 1000.0
        # collect words in this segment
        text = " ".join(
            w["text"] for w in words
            if seg.get("start", 0) <= w.get("start", -1) < seg.get("end", 0)
        )
        segments.append({"speaker": spk, "start": start, "end": end, "text": text})
    return sorted(segments, key=lambda x: x["start"])


def main():
    if not os.getenv("AAI_TOKEN"):
        print("❌ Please set AAI_TOKEN in your shell, e.g.: export AAI_TOKEN=\"your_key\"")
        return

    parser = argparse.ArgumentParser(
        description="AssemblyAI-powered speaker-diarizing transcriber"
    )
    parser.add_argument(
        "-o", "--output", help="Output file (.pdf or .txt)", required=False
    )
    args = parser.parse_args()

    # Input selection
    choice = input("Select input [(1) file, (2) microphone]: ").strip()
    if choice == "1":
        audio_path = input("Path to audio file: ").strip()
    elif choice == "2":
        samplerate = 16000
        print("Recording… press Enter to stop.")
        chunks = []
        def cb(indata, frames, time, status):
            chunks.append(indata.copy())
        with sd.InputStream(samplerate=samplerate, channels=1, callback=cb):
            input()
        data = np.concatenate(chunks, axis=0).squeeze()
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, data, samplerate)
        audio_path = tmp.name
    else:
        print("Invalid choice.")
        return

    print("Uploading audio…")
    audio_url = upload_to_assemblyai(audio_path)

    print("Requesting transcription…")
    tid = request_transcription(audio_url)

    print("Waiting for completion…")
    result = poll_transcription(tid)
    if result.get("error"):
        print("❌ Error:", result["error"])
        return

    segments = format_segments(result)
    out = args.output

    # PDF output
    if out and out.lower().endswith(".pdf"):
        c = canvas.Canvas(out, pagesize=letter)
        width, height = letter
        margin = 72
        y = height - margin
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, y, "Speaker-Tagged Transcript")
        y -= 24
        c.setFont("Helvetica", 10)
        for seg in segments:
            start_ts = str(timedelta(seconds=int(seg["start"])))
            end_ts = str(timedelta(seconds=int(seg["end"])))
            line = f"{seg['speaker']} [{start_ts} → {end_ts}]: {seg['text']}"
            if y < margin:
                c.showPage()
                y = height - margin
                c.setFont("Helvetica", 10)
            c.drawString(margin, y, line)
            y -= 14
        c.save()
        print(f"✅ PDF saved to {out}")
    else:
        # TXT fallback
        lines = [
            f"{seg['speaker']} [{seg['start']:.1f}s→{seg['end']:.1f}s]: {seg['text']}"
            for seg in segments
        ]
        text = "\n".join(lines)
        if out:
            if not out.lower().endswith(".txt"):
                out += ".txt"
            with open(out, "w", encoding="utf-8") as f:
                f.write(text)
            print(f"✅ Text saved to {out}")
        else:
            print(text)

if __name__ == "__main__":
    main()