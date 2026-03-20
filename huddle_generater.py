import os
import re
import csv
import json
import uuid
import subprocess
import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import uvicorn

app = FastAPI(title="Transcription Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

SUPPORTED_AUDIO = {".mp3", ".wav", ".ogg", ".flac", ".m4a", ".aac", ".opus", ".wma"}
SUPPORTED_VIDEO = {".mp4", ".mkv", ".avi", ".mov", ".webm", ".flv", ".wmv", ".3gp"}
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# CSV output paths
CSV_DAILY_LOG        = Path("huddle_daily_log.csv")
CSV_STAFF_PERF       = Path("huddle_staff_performance.csv")
CSV_BLOCKERS         = Path("huddle_blockers.csv")

# ─────────────────────────────────────────────────────────────
# CSV INITIALISATION — write headers if files don't exist yet
# ─────────────────────────────────────────────────────────────
def _init_csv(path: Path, headers: list[str]):
    if not path.exists():
        with open(path, "w", newline="", encoding="utf-8") as f:
            csv.writer(f).writerow(headers)

_init_csv(CSV_DAILY_LOG, [
    "DATE","BRANCH_ID","BRANCH_NAME","STATE","BRANCH_TYPE","CLUSTER",
    "HUDDLE_HELD","DURATION_MINS","SPEAKER_COUNT","SENTIMENT_SCORE",
    "MANAGER_TALK_PCT","STRESS_MARKER","COMPLIANCE_SCORE","SHAME_FLAG",
    "CASA_TARGET","CASA_ACTUAL","ESCALATION_RAISED"
])
_init_csv(CSV_STAFF_PERF, [
    "DATE","BRANCH_ID","BRANCH_NAME","BRANCH_TYPE",
    "STAFF_NAME","STAFF_ROLE","CASA_OPENED","CROSS_SELL","FD_OPENED","IS_STAR_DAY"
])
_init_csv(CSV_BLOCKERS, [
    "DATE","BRANCH_ID","BRANCH_NAME",
    "BLOCKER_TYPE","BLOCKER_DESC","RAISED_IN_HUDDLE","RESOLVED_SAME_DAY","ESCALATED_TO_RM"
])


# ─────────────────────────────────────────────────────────────
# AUDIO UTILITIES
# ─────────────────────────────────────────────────────────────
def extract_audio_from_video(video_path: str) -> str:
    audio_path = video_path.replace(Path(video_path).suffix, "_audio.wav")
    cmd = ["ffmpeg", "-y", "-i", video_path, "-vn", "-acodec", "pcm_s16le",
           "-ar", "16000", "-ac", "1", audio_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"ffmpeg error: {result.stderr}")
    return audio_path


def convert_to_wav(input_path: str) -> str:
    output_path = input_path.rsplit(".", 1)[0] + "_16k.wav"
    cmd = ["ffmpeg", "-y", "-i", input_path, "-ar", "16000", "-ac", "1",
           "-acodec", "pcm_s16le", output_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"Audio conversion error: {result.stderr}")
    return output_path


def get_audio_duration(audio_path: str) -> float:
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", audio_path],
        capture_output=True, text=True
    )
    try:
        return round(float(result.stdout.strip()) / 60, 1)   # minutes
    except Exception:
        return 0.0


# ─────────────────────────────────────────────────────────────
# TRANSCRIPTION ENGINES
# ─────────────────────────────────────────────────────────────
def transcribe_whisper(audio_path: str, model_size: str = "base",
                       language: Optional[str] = None) -> dict:
    try:
        import whisper
        model = whisper.load_model(model_size)
        options = {}
        if language:
            options["language"] = language
        result = model.transcribe(audio_path, **options)
        raw_segments: list[dict] = result.get("segments") or []
        return {
            "text": str(result["text"]).strip(),
            "language": result.get("language", "unknown"),
            "segments": [
                {
                    "start": round(float(s["start"]), 2),
                    "end":   round(float(s["end"]),   2),
                    "text":  str(s["text"]).strip(),
                }
                for s in raw_segments
            ]
        }
    except ImportError:
        raise HTTPException(status_code=500,
            detail="Whisper not installed. Run: pip install openai-whisper")


def transcribe_faster_whisper(audio_path: str, model_size: str = "base",
                               language: Optional[str] = None) -> dict:
    try:
        from faster_whisper import WhisperModel
        model = WhisperModel(model_size, device="cpu", compute_type="int8")
        segments, info = model.transcribe(audio_path, language=language or None)
        seg_list, full_text = [], []
        for seg in segments:
            seg_list.append({"start": round(seg.start, 2), "end": round(seg.end, 2),
                              "text": seg.text.strip()})
            full_text.append(seg.text.strip())
        return {"text": " ".join(full_text), "language": info.language,
                "segments": seg_list}
    except ImportError:
        raise HTTPException(status_code=500,
            detail="faster-whisper not installed. Run: pip install faster-whisper")


def transcribe_sarvam(audio_path: str, language_code: str = "hi-IN",
                      api_key: str = "") -> dict:
    try:
        import httpx
        if not api_key:
            raise HTTPException(status_code=400,
                detail="Sarvam.ai API key is required.")
        wav_path = convert_to_wav(audio_path) if not audio_path.endswith("_16k.wav") else audio_path
        with open(wav_path, "rb") as f:
            audio_bytes = f.read()
        if len(audio_bytes) > 4 * 1024 * 1024:
            return _transcribe_sarvam_chunked(wav_path, language_code, api_key)
        with httpx.Client(timeout=120) as client:
            response = client.post(
                "https://api.sarvam.ai/speech-to-text",
                headers={"api-subscription-key": api_key},
                files={"file": ("audio.wav", audio_bytes, "audio/wav")},
                data={"language_code": language_code, "model": "saaras:v1",
                      "with_timestamps": "true"}
            )
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code,
                detail=f"Sarvam API error: {response.text}")
        data = response.json()
        return {"text": data.get("transcript", ""), "language": language_code,
                "segments": data.get("subtitles", [])}
    except ImportError:
        raise HTTPException(status_code=500,
            detail="httpx not installed. Run: pip install httpx")


def _transcribe_sarvam_chunked(wav_path: str, language_code: str,
                                api_key: str) -> dict:
    import httpx
    chunk_dir = Path(wav_path).parent / "chunks"
    chunk_dir.mkdir(exist_ok=True)
    chunk_pattern = str(chunk_dir / "chunk_%03d.wav")
    subprocess.run(["ffmpeg", "-y", "-i", wav_path, "-f", "segment",
                    "-segment_time", "30", "-c", "copy", chunk_pattern],
                   capture_output=True)
    chunks = sorted(chunk_dir.glob("chunk_*.wav"))
    all_text, all_segments, time_offset = [], [], 0.0
    with httpx.Client(timeout=120) as client:
        for chunk_path in chunks:
            with open(chunk_path, "rb") as f:
                chunk_bytes = f.read()
            response = client.post(
                "https://api.sarvam.ai/speech-to-text",
                headers={"api-subscription-key": api_key},
                files={"file": ("chunk.wav", chunk_bytes, "audio/wav")},
                data={"language_code": language_code, "model": "saaras:v1"}
            )
            if response.status_code == 200:
                data = response.json()
                all_text.append(data.get("transcript", ""))
                for seg in data.get("subtitles", []):
                    seg["start"] = seg.get("start", 0) + time_offset
                    seg["end"]   = seg.get("end",   0) + time_offset
                    all_segments.append(seg)
            result = subprocess.run(
                ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
                 "-of", "default=noprint_wrappers=1:nokey=1", str(chunk_path)],
                capture_output=True, text=True
            )
            try:
                time_offset += float(result.stdout.strip())
            except Exception:
                time_offset += 30.0
    return {"text": " ".join(all_text), "language": language_code,
            "segments": all_segments}


# ─────────────────────────────────────────────────────────────
# HYBRID ANALYSIS PIPELINE
# ─────────────────────────────────────────────────────────────

# ── Rule-based helpers ────────────────────────────────────────

STRESS_KEYWORDS = [
    "problem", "issue", "delay", "stuck", "pending", "blocked", "complaint",
    "error", "failed", "missing", "not working", "escalate", "urgent",
    "critical", "overdue", "backlog", "offline", "slow", "lag", "queue",
    "behind", "short", "absent", "leave", "unavailable"
]

SHAME_PATTERNS = [
    r"\bwhy (didn't|did not|haven't|have not) you\b",
    r"\bfailure\b", r"\bincompetent\b", r"\buseless\b",
    r"\balways (late|wrong|missing)\b", r"\bnever (on time|complete)\b",
    r"\bwhat were you (doing|thinking)\b",
    r"\byou should (be ashamed|know better)\b"
]

ESCALATION_PATTERNS = [
    r"\bescalat(e|ed|ing)\b", r"\breport(ed|ing) to (RM|regional|head office)\b",
    r"\bbrought to (senior|management|higher)\b",
    r"\btaking this (up|forward) to\b",
    r"\binform(ed|ing) the (cluster|regional|zonal)\b"
]

COMPLIANCE_TOPICS = [
    "kyc", "aadhaar", "pan", "nomination", "fatca", "ckyc",
    "aml", "compliance", "audit", "regulatory", "seeding",
    "document", "upload", "pending account"
]

BLOCKER_TYPES = {
    "Staffing":   ["leave", "absent", "rm on leave", "short staff", "reduced capacity",
                   "understaffed", "vacancy"],
    "System":     ["offline", "slow", "biometric", "cbs", "upi", "settlement lag",
                   "system down", "login slow", "device offline", "network"],
    "Compliance": ["kyc", "aadhaar", "seeding", "backlog", "document", "ckyc",
                   "fatca", "pan", "pending upload"],
    "Process":    ["queue", "fd renewal", "backed up", "process", "manual",
                   "workflow", "delay", "waiting"]
}

ROLES = ["RM", "Sales Officer", "Branch Head", "Loan Officer", "BCA Officer",
         "Operations Officer", "Teller", "CSO", "Manager"]

def _rule_sentiment(text: str) -> float:
    """Score 1-5: positive words push up, negative push down."""
    pos = len(re.findall(
        r"\b(good|great|excellent|achieved|target|met|success|well done|congrats|"
        r"star|top|best|improve|growth|positive|on track)\b", text, re.I))
    neg = len(re.findall(
        r"\b(problem|issue|delay|fail|blocked|complaint|error|behind|"
        r"short|missed|overdue|stuck|urgent|escalat)\b", text, re.I))
    raw = 3.0 + (pos * 0.3) - (neg * 0.25)
    return round(max(1.0, min(5.0, raw)), 1)

def _rule_stress_markers(text: str) -> int:
    count = sum(1 for kw in STRESS_KEYWORDS if kw.lower() in text.lower())
    return count

def _rule_shame_flag(text: str) -> int:
    for p in SHAME_PATTERNS:
        if re.search(p, text, re.I):
            return 1
    return 0

def _rule_escalation(text: str) -> int:
    for p in ESCALATION_PATTERNS:
        if re.search(p, text, re.I):
            return 1
    return 0

def _rule_compliance_score(text: str) -> float:
    found = sum(1 for t in COMPLIANCE_TOPICS if t.lower() in text.lower())
    return round(min(100.0, found * 12.5), 1)

def _rule_manager_talk_pct(segments: list) -> float:
    """Estimate manager talk % — first speaker heuristic (branch head talks most)."""
    if not segments:
        return 0.0
    # Count words per segment index (proxy for speaker)
    # Without diarization, estimate: first 30% of segments = manager
    total_words  = sum(len(s.get("text","").split()) for s in segments)
    if total_words == 0:
        return 0.0
    cutoff = max(1, int(len(segments) * 0.35))
    mgr_words = sum(len(s.get("text","").split()) for s in segments[:cutoff])
    return round((mgr_words / total_words) * 100, 1)

def _rule_speaker_count(segments: list) -> int:
    """Estimate speakers from transcript segment gaps — fallback only."""
    if not segments:
        return 0
    gaps = [segments[i]["start"] - segments[i-1]["end"]
            for i in range(1, len(segments))]
    large_gaps = sum(1 for g in gaps if g > 2.5)
    return min(max(2, large_gaps + 1), 8)


def _audio_speaker_count(audio_path: str) -> int:
    """
    Estimate speaker count from raw audio using ZCR (Zero Crossing Rate) quartile bands.
    Works on continuous speech with NO long silences — the correct approach for
    video huddles where speakers talk back-to-back without long pauses.

    Method:
      1. Compute ZCR over 1-second speech-only windows.
         ZCR tracks voice timbre/pitch — different speakers occupy different ZCR ranges.
      2. Compute quartile thresholds (Q1, Q2, Q3) across all speech windows.
      3. Count how many of the 4 quartile bands are meaningfully occupied
         (>= 5% of total windows). Each occupied band ≈ one distinct voice type.
    Returns int in [2, 8].
    """
    import wave, struct, math
    try:
        with wave.open(audio_path) as wf:
            rate = wf.getframerate()
            n_frames = wf.getnframes()
            chunk = int(rate * 1.0)   # 1-second windows
            zcr_values = []

            for i in range(0, n_frames, chunk):
                wf.setpos(i)
                raw = wf.readframes(chunk)
                if len(raw) < 4:
                    continue
                samples = struct.unpack("<" + "h" * (len(raw) // 2), raw)
                rms = math.sqrt(sum(s * s for s in samples) / len(samples))
                if rms < 800:           # skip near-silence frames
                    continue
                zcr = sum(
                    1 for j in range(1, len(samples))
                    if (samples[j] >= 0) != (samples[j - 1] >= 0)
                ) / len(samples)
                zcr_values.append(zcr)

        if len(zcr_values) < 10:
            return 2

        N = len(zcr_values)
        zcr_sorted = sorted(zcr_values)
        q1 = zcr_sorted[N // 4]
        q2 = zcr_sorted[N // 2]
        q3 = zcr_sorted[3 * N // 4]

        # Count windows in each quartile band
        band_counts = [0, 0, 0, 0]
        for z in zcr_values:
            if   z < q1: band_counts[0] += 1
            elif z < q2: band_counts[1] += 1
            elif z < q3: band_counts[2] += 1
            else:         band_counts[3] += 1

        # A band is "occupied" if it holds at least 5% of all speech windows
        min_occupancy = max(3, N * 0.05)
        occupied = sum(1 for c in band_counts if c >= min_occupancy)

        return min(8, max(2, occupied))

    except Exception as e:
        print(f"[audio_speaker_count] Error: {e}")
        return 2

def _rule_extract_numbers(text: str, keywords: list[str]) -> Optional[int]:
    """Extract a number that follows one of the keywords."""
    for kw in keywords:
        pattern = rf"\b{re.escape(kw)}\b[^0-9]{{0,20}}?(\d+)"
        m = re.search(pattern, text, re.I)
        if m:
            return int(m.group(1))
    return None

def _rule_blockers(text: str) -> list[dict]:
    """Detect blockers from transcript using keyword matching."""
    blockers = []
    lower = text.lower()
    for btype, keywords in BLOCKER_TYPES.items():
        for kw in keywords:
            if kw in lower:
                # Extract surrounding sentence as description
                sentences = re.split(r'[.!?\n]', text)
                desc = next(
                    (s.strip() for s in sentences if kw.lower() in s.lower()),
                    f"{btype} issue: {kw}"
                )
                # Truncate long descriptions
                desc = desc[:120].strip()
                # Avoid duplicate blocker types
                if not any(b["BLOCKER_TYPE"] == btype for b in blockers):
                    blockers.append({
                        "BLOCKER_TYPE": btype,
                        "BLOCKER_DESC": desc,
                        "RAISED_IN_HUDDLE": 1,
                        "RESOLVED_SAME_DAY": 0,   # LLM will refine this
                        "ESCALATED_TO_RM": _rule_escalation(desc)
                    })
                break
    return blockers

def _rule_staff_performance(text: str, branch_type: str) -> list[dict]:
    """Extract staff names and performance numbers using regex patterns."""
    staff_entries = []
    # Pattern: "Name (Role): X CASA, Y cross-sell, Z FD"
    # Also catches "Arjun opened 9 accounts"
    name_pattern  = r"\b([A-Z][a-z]+ (?:[A-Z][a-z]+ )?[A-Z][a-z]+)\b"
    number_context = r"(\d+)"

    # Try to find any mention of names with numbers nearby
    sentences = re.split(r'[.!?\n]', text)
    seen_names = set()
    for sentence in sentences:
        names = re.findall(name_pattern, sentence)
        numbers = re.findall(number_context, sentence)
        role = next((r for r in ROLES if r.lower() in sentence.lower()), "Staff")
        for name in names:
            if name in seen_names or len(name.split()) < 2:
                continue
            seen_names.add(name)
            vals = [int(n) for n in numbers[:3]]
            while len(vals) < 3:
                vals.append(0)
            star = 1 if re.search(
                rf"\b(star|top|best|well done|congrats|excellent)\b.*{re.escape(name)}|"
                rf"{re.escape(name)}.*\b(star|top|best|well done|congrats|excellent)\b",
                text, re.I) else 0
            staff_entries.append({
                "STAFF_NAME":  name,
                "STAFF_ROLE":  role,
                "CASA_OPENED": vals[0],
                "CROSS_SELL":  vals[1],
                "FD_OPENED":   vals[2],
                "IS_STAR_DAY": star,
                "BRANCH_TYPE": branch_type
            })
    return staff_entries


# ── LLM-based extraction (Claude API via httpx) ───────────────

LLM_SYSTEM_PROMPT = """You are a banking operations analyst. 
You receive a transcript of a daily branch huddle meeting and extract structured data.
Return ONLY valid JSON. No markdown fences, no preamble.

Extract and return this exact JSON structure:
{
  "sentiment_score": <float 1.0-5.0>,
  "manager_talk_pct": <float 0-100>,
  "stress_markers": <int>,
  "compliance_score": <float 0-100>,
  "shame_flag": <0 or 1>,
  "escalation_raised": <0 or 1>,
  "speaker_count": <int>,
  "casa_target": <int or null>,
  "casa_actual": <int or null>,
  "staff_performance": [
    {
      "staff_name": "<full name>",
      "staff_role": "<RM|Sales Officer|Branch Head|Loan Officer|BCA Officer|Operations Officer|Teller|CSO|Manager>",
      "casa_opened": <int>,
      "cross_sell": <int>,
      "fd_opened": <int>,
      "is_star_day": <0 or 1>
    }
  ],
  "blockers": [
    {
      "blocker_type": "<Staffing|System|Compliance|Process>",
      "blocker_desc": "<short description under 120 chars>",
      "resolved_same_day": <0 or 1>,
      "escalated_to_rm": <0 or 1>
    }
  ]
}

Rules:
- sentiment_score: 1=very negative, 5=very positive, based on overall team mood
- manager_talk_pct: estimated % of conversation dominated by the manager/branch head
- stress_markers: count of distinct stress/problem indicators mentioned
- compliance_score: 0-100 % of standard compliance topics covered
- shame_flag: 1 if any staff member was shamed, belittled or publicly humiliated
- escalation_raised: 1 if any issue was formally escalated to RM or above
- For missing numbers, use 0. For unknown names, skip that entry.
- If no staff or blockers are mentioned, return empty arrays.
"""

def _llm_analyze(transcript_text: str, anthropic_api_key: str) -> Optional[dict]:
    """Call Claude Sonnet via Anthropic API for deep transcript analysis."""
    if not anthropic_api_key:
        return None
    try:
        import httpx
        payload = {
            "model": "claude-sonnet-4-20250514",
            "max_tokens": 1500,
            "system": LLM_SYSTEM_PROMPT,
            "messages": [
                {"role": "user", "content": f"Analyze this huddle transcript:\n\n{transcript_text[:6000]}"}
            ]
        }
        with httpx.Client(timeout=60) as client:
            resp = client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": anthropic_api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                },
                json=payload
            )
        if resp.status_code != 200:
            print(f"[LLM] API error {resp.status_code}: {resp.text[:200]}")
            return None
        raw = resp.json()["content"][0]["text"].strip()
        # Strip any accidental markdown fences
        raw = re.sub(r"^```(?:json)?|```$", "", raw, flags=re.M).strip()
        return json.loads(raw)
    except Exception as e:
        print(f"[LLM] Analysis failed: {e}")
        return None


# ── Merge rule-based + LLM results ───────────────────────────

def _merge(rule_val, llm_val, prefer_llm: bool = True):
    """Return LLM value when available and prefer_llm=True, else rule value."""
    if prefer_llm and llm_val is not None:
        return llm_val
    return rule_val


def analyze_transcript(
    transcript: dict,
    audio_path: str,
    branch_meta: dict,
    anthropic_api_key: str = ""
) -> dict:
    """
    Hybrid analysis pipeline:
      1. Audio-level analysis (always runs — works even with empty transcript).
      2. Rule-based NLP on transcript text (runs when text is available).
      3. LLM enrichment via Claude API (runs when anthropic_api_key is provided).
    Later layers override earlier layers when they return non-null values.
    """
    text     = transcript.get("text", "") or ""
    segments = transcript.get("segments", []) or []
    has_text = len(text.strip()) > 20   # meaningful transcript available

    # ── Layer 1: Audio-level analysis (no transcript needed) ──────────────────
    duration_mins = get_audio_duration(audio_path)

    # Speaker count: always use the ZCR-band method on raw audio.
    # The old segment-gap method only works when Whisper produces timestamps;
    # ZCR works on the waveform directly regardless of transcription engine.
    wav_path = audio_path if audio_path.endswith(".wav") else None
    if not wav_path:
        # Convert to wav for analysis
        try:
            wav_path = convert_to_wav(audio_path)
        except Exception:
            wav_path = None

    a_speakers = _audio_speaker_count(wav_path) if wav_path else 2

    # Audio energy ratio → rough sentiment proxy when no text
    a_sentiment = 3.0   # neutral default
    if wav_path:
        try:
            import wave as _wave, struct as _struct, math as _math
            with _wave.open(wav_path) as wf:
                rate = wf.getframerate()
                n = wf.getnframes()
                chunk = int(rate * 5)
                rms_vals = []
                for i in range(0, n, chunk):
                    wf.setpos(i)
                    raw = wf.readframes(chunk)
                    if len(raw) < 4: continue
                    samps = _struct.unpack("<" + "h" * (len(raw) // 2), raw)
                    rms_vals.append(_math.sqrt(sum(s*s for s in samps) / len(samps)))
            if rms_vals:
                avg_rms = sum(rms_vals) / len(rms_vals)
                # High energy + consistent = engaged/positive huddle
                cv = (max(rms_vals) - min(rms_vals)) / avg_rms if avg_rms else 1
                a_sentiment = round(min(5.0, max(1.0, 3.5 - cv * 0.5)), 1)
        except Exception:
            pass

    # ── Layer 2: Rule-based NLP (only when transcript text exists) ────────────
    r_sentiment   = _rule_sentiment(text)   if has_text else a_sentiment
    r_stress      = _rule_stress_markers(text)  if has_text else 0
    r_shame       = _rule_shame_flag(text)      if has_text else 0
    r_escalation  = _rule_escalation(text)      if has_text else 0
    r_compliance  = _rule_compliance_score(text) if has_text else 0.0
    r_mgr_pct     = _rule_manager_talk_pct(segments) if has_text else 0.0
    # For speaker count: use audio method as primary; segment gaps only as fallback
    r_speakers    = a_speakers if a_speakers > 0 else _rule_speaker_count(segments)
    r_casa_target = _rule_extract_numbers(text, ["target", "casa target", "goal"]) if has_text else None
    r_casa_actual = _rule_extract_numbers(text, ["opened", "actual", "achieved", "done"]) if has_text else None
    r_blockers    = _rule_blockers(text) if has_text else []
    r_staff       = _rule_staff_performance(text, branch_meta.get("branch_type", "")) if has_text else []

    # ── Layer 3: LLM enrichment (optional, needs API key + non-empty text) ────
    # If transcript is empty, send a note to LLM so it can still fill audio-derivable fields
    if anthropic_api_key:
        llm_input = text if has_text else (
            f"[No transcript available. Audio duration: {duration_mins:.1f} min, "
            f"estimated {a_speakers} speakers. Please return default/neutral JSON values "
            f"and empty arrays for staff_performance and blockers.]"
        )
        llm = _llm_analyze(llm_input, anthropic_api_key)
    else:
        llm = None

    # ── Merge layers: LLM > rule-based > audio-level ──────────────────────────
    def _l(rule_val, llm_key):
        return _merge(rule_val, llm.get(llm_key) if llm else None)

    sentiment   = _l(r_sentiment,  "sentiment_score")
    stress      = _l(r_stress,     "stress_markers")
    shame       = _l(r_shame,      "shame_flag")
    escalation  = _l(r_escalation, "escalation_raised")
    compliance  = _l(r_compliance, "compliance_score")
    mgr_pct     = _l(r_mgr_pct,    "manager_talk_pct")
    # Speaker count: LLM may override audio estimate if text is available
    speakers    = _l(r_speakers,   "speaker_count")
    casa_target = (_l(r_casa_target, "casa_target") or 0)
    casa_actual = (_l(r_casa_actual, "casa_actual") or 0)

    # Staff list
    if llm and llm.get("staff_performance"):
        staff_list = llm["staff_performance"]
    elif r_staff:
        staff_list = [
            {"staff_name": s["STAFF_NAME"], "staff_role": s["STAFF_ROLE"],
             "casa_opened": s["CASA_OPENED"], "cross_sell": s["CROSS_SELL"],
             "fd_opened": s["FD_OPENED"], "is_star_day": s["IS_STAR_DAY"]}
            for s in r_staff
        ]
    else:
        staff_list = []

    # Blockers list
    if llm and llm.get("blockers"):
        blocker_list = llm["blockers"]
    elif r_blockers:
        blocker_list = [
            {"blocker_type": b["BLOCKER_TYPE"], "blocker_desc": b["BLOCKER_DESC"],
             "resolved_same_day": b["RESOLVED_SAME_DAY"], "escalated_to_rm": b["ESCALATED_TO_RM"]}
            for b in r_blockers
        ]
    else:
        blocker_list = []

    return {
        "duration_mins":   duration_mins,
        "speaker_count":   speakers,
        "sentiment_score": sentiment,
        "manager_talk_pct": mgr_pct,
        "stress_marker":   stress,
        "compliance_score": compliance,
        "shame_flag":      shame,
        "escalation_raised": escalation,
        "casa_target":     casa_target,
        "casa_actual":     casa_actual,
        "staff_list":      staff_list,
        "blocker_list":    blocker_list,
    }


# ─────────────────────────────────────────────────────────────
# CSV WRITERS
# ─────────────────────────────────────────────────────────────

def write_daily_log(date: str, branch_meta: dict, analysis: dict):
    row = [
        date,
        branch_meta["branch_id"],
        branch_meta["branch_name"],
        branch_meta.get("state", ""),
        branch_meta.get("branch_type", ""),
        branch_meta.get("cluster", ""),
        1,                                      # HUDDLE_HELD
        analysis["duration_mins"],
        analysis["speaker_count"],
        analysis["sentiment_score"],
        analysis["manager_talk_pct"],
        analysis["stress_marker"],
        analysis["compliance_score"],
        analysis["shame_flag"],
        analysis["casa_target"],
        analysis["casa_actual"],
        analysis["escalation_raised"],
    ]
    with open(CSV_DAILY_LOG, "a", newline="", encoding="utf-8") as f:
        csv.writer(f).writerow(row)


def write_staff_performance(date: str, branch_meta: dict, staff_list: list):
    with open(CSV_STAFF_PERF, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for s in staff_list:
            w.writerow([
                date,
                branch_meta["branch_id"],
                branch_meta["branch_name"],
                branch_meta.get("branch_type", ""),
                s.get("staff_name", ""),
                s.get("staff_role", ""),
                s.get("casa_opened", 0),
                s.get("cross_sell", 0),
                s.get("fd_opened", 0),
                s.get("is_star_day", 0),
            ])


def write_blockers(date: str, branch_meta: dict, blocker_list: list):
    with open(CSV_BLOCKERS, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for b in blocker_list:
            w.writerow([
                date,
                branch_meta["branch_id"],
                branch_meta["branch_name"],
                b.get("blocker_type", ""),
                b.get("blocker_desc", ""),
                1,                              # RAISED_IN_HUDDLE always 1
                b.get("resolved_same_day", 0),
                b.get("escalated_to_rm", 0),
            ])


# ─────────────────────────────────────────────────────────────
# FASTAPI ROUTES
# ─────────────────────────────────────────────────────────────

@app.get("/")
async def serve_frontend():
    return FileResponse("huddles.html")


@app.post("/transcribe")
async def transcribe(
    file:             UploadFile = File(...),
    engine:           str = Form("whisper"),
    whisper_model:    str = Form("base"),
    language:         str = Form(""),
    sarvam_api_key:   str = Form(""),
    sarvam_language:  str = Form("hi-IN"),
    # Branch metadata — provided by front-end form fields
    huddle_date:      str = Form(""),          # YYYY-MM-DD
    branch_id:        str = Form("YBL00000"),
    branch_name:      str = Form(""),
    branch_state:     str = Form(""),
    branch_type:      str = Form(""),          # Metro / Urban / Rural
    cluster:          str = Form(""),
    # Optional: Anthropic API key for LLM enrichment
    anthropic_api_key: str = Form(""),
):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_AUDIO and suffix not in SUPPORTED_VIDEO:
        raise HTTPException(status_code=400,
            detail=f"Unsupported file type: {suffix}.")

    file_id   = str(uuid.uuid4())
    save_path = UPLOAD_DIR / f"{file_id}{suffix}"
    with open(save_path, "wb") as f:
        f.write(await file.read())

    # Keep wav path alive for analysis even if transcription partially fails
    audio_path = str(save_path)
    wav_for_analysis = None

    try:
        # ── Step 1: Extract audio ──────────────────────────────────────────────
        if suffix in SUPPORTED_VIDEO:
            audio_path = extract_audio_from_video(audio_path)

        # Always produce a WAV for audio-level analysis (speaker count, energy)
        try:
            wav_for_analysis = convert_to_wav(audio_path)
        except Exception as wav_err:
            print(f"[warn] WAV conversion failed: {wav_err}")
            wav_for_analysis = audio_path  # use whatever we have

        # ── Step 2: Transcribe ─────────────────────────────────────────────────
        # If no engine is installed, fall back to audio-only analysis gracefully
        transcript_error = None
        lang = language.strip() or None
        try:
            if engine == "whisper":
                result = transcribe_whisper(audio_path, model_size=whisper_model, language=lang)
            elif engine == "faster-whisper":
                result = transcribe_faster_whisper(audio_path, model_size=whisper_model, language=lang)
            elif engine == "sarvam":
                result = transcribe_sarvam(audio_path, language_code=sarvam_language,
                                           api_key=sarvam_api_key)
            else:
                raise HTTPException(status_code=400, detail=f"Unknown engine: {engine}")
        except HTTPException as he:
            # Engine not installed → proceed with empty transcript so audio analysis still runs
            if he.status_code == 500 and "not installed" in str(he.detail):
                transcript_error = he.detail
                result = {"text": "", "language": "unknown", "segments": []}
            else:
                raise

        result["engine"]   = engine
        result["filename"] = file.filename
        if transcript_error:
            result["warning"] = f"Transcription skipped: {transcript_error}. Audio-level analysis still ran."

        # ── Step 3: Analyze → write all 3 CSVs ────────────────────────────────
        today = huddle_date or datetime.date.today().isoformat()
        branch_meta = {
            "branch_id":   branch_id   or "YBL00000",
            "branch_name": branch_name or file.filename.rsplit(".", 1)[0],
            "state":       branch_state,
            "branch_type": branch_type or "Metro",
            "cluster":     cluster     or "Central",
        }

        analysis = analyze_transcript(
            transcript=result,
            audio_path=wav_for_analysis,   # always the WAV for audio methods
            branch_meta=branch_meta,
            anthropic_api_key=anthropic_api_key,
        )

        write_daily_log(today, branch_meta, analysis)
        write_staff_performance(today, branch_meta, analysis["staff_list"])
        write_blockers(today, branch_meta, analysis["blocker_list"])

        result["analysis"] = {
            "daily_log_written":    True,
            "staff_rows_written":   len(analysis["staff_list"]),
            "blocker_rows_written": len(analysis["blocker_list"]),
            "sentiment_score":      analysis["sentiment_score"],
            "compliance_score":     analysis["compliance_score"],
            "stress_marker":        analysis["stress_marker"],
            "shame_flag":           bool(analysis["shame_flag"]),
            "escalation_raised":    bool(analysis["escalation_raised"]),
            "speaker_count":        analysis["speaker_count"],
            "llm_used":             bool(anthropic_api_key),
            "transcript_available": bool(result.get("text", "").strip()),
        }
        return result

    finally:
        # Cleanup temp files
        try:
            save_path.unlink(missing_ok=True)
            for pattern in ["_audio.wav", "_audio_16k.wav"]:
                (UPLOAD_DIR / f"{file_id}{pattern}").unlink(missing_ok=True)
        except Exception:
            pass


@app.get("/engines")
async def list_engines():
    engines = []
    try:
        import whisper
        engines.append({"id": "whisper", "name": "OpenAI Whisper", "type": "local",
            "description": "Open-source speech recognition. Runs fully offline.",
            "available": True, "models": ["tiny","base","small","medium","large"]})
    except ImportError:
        engines.append({"id": "whisper", "name": "OpenAI Whisper", "type": "local",
            "description": "Not installed. Run: pip install openai-whisper",
            "available": False})
    try:
        from faster_whisper import WhisperModel
        engines.append({"id": "faster-whisper", "name": "Faster Whisper", "type": "local",
            "description": "4x faster Whisper variant. Optimised for CPU.",
            "available": True, "models": ["tiny","base","small","medium","large-v2","large-v3"]})
    except ImportError:
        engines.append({"id": "faster-whisper", "name": "Faster Whisper", "type": "local",
            "description": "Not installed. Run: pip install faster-whisper",
            "available": False})
    engines.append({"id": "sarvam", "name": "Sarvam.ai (Saaras)", "type": "api",
        "description": "Optimised for Indian languages. Requires API key from sarvam.ai",
        "available": True,
        "languages": [
            {"code": "hi-IN", "name": "Hindi"},
            {"code": "en-IN", "name": "English (India)"},
            {"code": "te-IN", "name": "Telugu"},
            {"code": "ta-IN", "name": "Tamil"},
            {"code": "kn-IN", "name": "Kannada"},
            {"code": "ml-IN", "name": "Malayalam"},
            {"code": "bn-IN", "name": "Bengali"},
            {"code": "mr-IN", "name": "Marathi"},
            {"code": "gu-IN", "name": "Gujarati"},
            {"code": "pa-IN", "name": "Punjabi"},
            {"code": "od-IN", "name": "Odia"},
        ]})
    return {"engines": engines}


@app.get("/export/{csv_name}")
async def export_csv(csv_name: str):
    """Download any of the 3 generated CSVs."""
    allowed = {
        "daily_log":       CSV_DAILY_LOG,
        "staff_performance": CSV_STAFF_PERF,
        "blockers":        CSV_BLOCKERS,
    }
    if csv_name not in allowed:
        raise HTTPException(status_code=404, detail="Unknown CSV. Use: daily_log, staff_performance, blockers")
    path = allowed[csv_name]
    if not path.exists():
        raise HTTPException(status_code=404, detail="CSV not generated yet.")
    return FileResponse(path, media_type="text/csv", filename=path.name)


if __name__ == "__main__":
    uvicorn.run("huddle_generater:app", host="0.0.0.0", port=8000, reload=True)