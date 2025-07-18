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
from reportlab.pdfbase.pdfmetrics import stringWidth
from datetime import timedelta

# AssemblyAI endpoints
AAI_UPLOAD_URL     = "https://api.assemblyai.com/v2/upload"
AAI_TRANSCRIBE_URL = "https://api.assemblyai.com/v2/transcript"

def read_audio_bytes(path):
    """Read an audio file and return raw bytes."""
    with open(path, "rb") as f:
        return f.read()

def upload_to_assemblyai(path):
    """Upload audio to AssemblyAI and return the upload URL."""
    token = os.getenv("AAI_TOKEN")
    if not token:
        raise RuntimeError('Please set AAI_TOKEN: export AAI_TOKEN="your_key_here"')
    headers = {"authorization": token}
    data = read_audio_bytes(path)
    resp = requests.post(AAI_UPLOAD_URL, headers=headers, data=data)
    resp.raise_for_status()
    return resp.json()["upload_url"]

def request_transcription(audio_url):
    """Request a diarized transcription and return the transcript ID."""
    token = os.getenv("AAI_TOKEN")
    headers = {"authorization": token, "content-type": "application/json"}
    payload = {
        "audio_url":      audio_url,
        "speaker_labels": True,
        "format_text":    True
    }
    resp = requests.post(AAI_TRANSCRIBE_URL, headers=headers, json=payload)
    resp.raise_for_status()
    return resp.json()["id"]

def poll_transcription(transcript_id, interval=5):
    """Poll the transcription endpoint until completion or error."""
    token = os.getenv("AAI_TOKEN")
    headers = {"authorization": token}
    url = f"{AAI_TRANSCRIBE_URL}/{transcript_id}"
    while True:
        resp = requests.get(url, headers=headers)
        resp.raise_for_status()
        j = resp.json()
        status = j.get("status")
        if status in ("completed", "error"):
            return j
        print(f"... waiting (status: {status})")
        time.sleep(interval)

def format_segments(assembly_json):
    """Convert AssemblyAI JSON into speaker-tagged segments."""
    labels = assembly_json.get("speaker_labels")
    if not labels or not isinstance(labels, dict):
        duration = assembly_json.get("audio_duration", 0)
        return [{
            "speaker": "Speaker_0",
            "start":   0.0,
            "end":     duration,
            "text":    assembly_json.get("text", "")
        }]
    segments = []
    words = assembly_json.get("words", [])
    for seg in labels.get("segments", []):
        spk   = seg.get("speaker_label")
        start = seg.get("start", 0) / 1000.0
        end   = seg.get("end",   0) / 1000.0
        text  = " ".join(
            w["text"] for w in words
            if seg["start"] <= w.get("start", -1) < seg["end"]
        )
        segments.append({
            "speaker": spk,
            "start":   start,
            "end":     end,
            "text":    text
        })
    return sorted(segments, key=lambda x: x["start"])

def auto_device(s):
    """Convert digit strings to int, else return the string."""
    return int(s) if s.isdigit() else s

def wrap_text(text, font, size, max_width):
    """Wrap a single string to fit within max_width."""
    words = text.split()
    lines = []
    current = ""
    for w in words:
        test = w if not current else f"{current} {w}"
        if stringWidth(test, font, size) <= max_width:
            current = test
        else:
            if current:
                lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines

def main():
    parser = argparse.ArgumentParser(
        description="AssemblyAI-powered speaker-diarizing transcriber"
    )
    parser.add_argument("-o", "--output",
        help="Output file (.pdf or .txt)", required=False
    )
    parser.add_argument("--list-devices", action="store_true",
        help="List all audio I/O devices and exit"
    )
    parser.add_argument("--device", type=auto_device, default=None,
        help="Input device (index or name substring)"
    )
    args = parser.parse_args()

    if args.list_devices:
        print(sd.query_devices())
        return

    choice = input("Select input [(1) file, (2) microphone/interface]: ").strip()
    if choice == "1":
        audio_path = input("Path to audio file: ").strip()
    elif choice == "2":
        info = sd.query_devices(args.device, kind="input")
        samplerate = int(info["default_samplerate"])
        channels   = info["max_input_channels"]
        name       = info["name"]
        print(f"Recording from {name!r} (index={args.device})")
        print(f"  samplerate={samplerate} Hz, channels={channels}")
        print("  Press Enter to stop.")
        rec_chunks = []
        def callback(indata, frames, t, status):
            rec_chunks.append(indata.copy())
        with sd.InputStream(samplerate=samplerate,
                            channels=channels,
                            device=args.device,
                            callback=callback):
            input()
        audio_arr = np.concatenate(rec_chunks, axis=0)
        if channels > 1:
            audio_arr = audio_arr.mean(axis=1)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, audio_arr, samplerate)
        audio_path = tmp.name
    else:
        print("Invalid choice.")
        return

    print("Uploading audio…")
    url = upload_to_assemblyai(audio_path)
    print("Requesting transcription…")
    tid = request_transcription(url)
    print("Waiting for completion…")
    result = poll_transcription(tid)
    if result.get("error"):
        print("❌ Error:", result["error"])
        return

    segments = format_segments(result)
    out      = args.output

    # PDF output with wrapping
    if out and out.lower().endswith(".pdf"):
        c = canvas.Canvas(out, pagesize=letter)
        width, height = letter
        margin = 72
        max_width = width - 2 * margin
        y = height - margin
        font_name = "Helvetica"
        font_size = 10
        c.setFont("Helvetica-Bold", 14)
        c.drawString(margin, y, "Speaker-Tagged Transcript")
        y -= 24
        c.setFont(font_name, font_size)
        for seg in segments:
            st = str(timedelta(seconds=int(seg["start"])))
            et = str(timedelta(seconds=int(seg["end"])))
            header = f"{seg['speaker']} [{st} → {et}]:"
            wrapped = wrap_text(header + " " + seg["text"],
                                font_name, font_size, max_width)
            for line in wrapped:
                if y < margin:
                    c.showPage()
                    y = height - margin
                    c.setFont(font_name, font_size)
                c.drawString(margin, y, line)
                y -= font_size + 2
        c.save()
        print(f"✅ PDF saved to {out}")
    else:
        lines = [f"{seg['speaker']} [{seg['start']:.1f}s→{seg['end']:.1f}s]: {seg['text']}"
                 for seg in segments]
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