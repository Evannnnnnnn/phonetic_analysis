import os
import re
import tempfile
import nltk
import torch
import torchaudio
import librosa
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import whisperx
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
)

# -----------------------------
# Setup + Model Initialization
# -----------------------------
app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

DEVICE = "cpu"
WHISPER_MODEL = whisperx.load_model("base", DEVICE, compute_type="int8")

PHONEME_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(PHONEME_MODEL)
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(PHONEME_MODEL)
PHONEME_PROCESSOR = Wav2Vec2Processor(feature_extractor=feature_extractor, tokenizer=tokenizer)
PHONEME_MODEL_LOADED = Wav2Vec2ForCTC.from_pretrained(PHONEME_MODEL)

# CMU dictionary
try:
    nltk.data.find("corpora/cmudict")
except LookupError:
    nltk.download("cmudict")
CMU_DICT = nltk.corpus.cmudict.dict()

# -----------------------------
# Utility functions
# -----------------------------
def clean_phoneme_sequence(phoneme_string: str):
    phonemes = phoneme_string.replace("|", " ").split()
    return [p.strip() for p in phonemes if p.strip()]


ARPABET_TO_IPA = {
    "AA": "ɑ", "AE": "æ", "AH": "ʌ", "AO": "ɔ", "AW": "aʊ", "AY": "aɪ",
    "B": "b", "CH": "tʃ", "D": "d", "DH": "ð", "EH": "ɛ", "ER": "ɝ",
    "EY": "eɪ", "F": "f", "G": "ɡ", "HH": "h", "IH": "ɪ", "IY": "i",
    "JH": "dʒ", "K": "k", "L": "l", "M": "m", "N": "n", "NG": "ŋ",
    "OW": "oʊ", "OY": "ɔɪ", "P": "p", "R": "ɹ", "S": "s", "SH": "ʃ",
    "T": "t", "TH": "θ", "UH": "ʊ", "UW": "u", "V": "v", "W": "w",
    "Y": "j", "Z": "z", "ZH": "ʒ"
}

def arpabet_to_ipa(arpabet_seq):
    ipa_seq = []
    for sym in arpabet_seq:
        sym = sym.rstrip("012")  # remove stress markers
        ipa = ARPABET_TO_IPA.get(sym, sym)
        ipa_seq.append(ipa)
    return ipa_seq


def normalize_ipa(seq):
    EQUIV = {"ɝ": "ɚ", "ɜ": "ɚ"}
    normalized = []
    for p in seq:
        p = EQUIV.get(p, p)
        p = p.replace("ː", "")  # remove length markers
        normalized.append(p)
    return normalized


def normalize_word_for_cmu(word: str) -> str:
    word = re.sub(r"[^\w\s]", "", word.lower())
    contractions = {
        "im": "i'm", "ive": "i've", "dont": "don't", "doesnt": "doesn't",
        "cant": "can't", "isnt": "isn't", "hows": "how's", "wont": "won't",
        "werent": "weren't", "didnt": "didn't",
    }
    return contractions.get(word, word)


def compare_phoneme_sequences(std, det):
    errors = []
    for i, (s, d) in enumerate(zip(std, det)):
        if s != d:
            errors.append({"index": i, "expected": s, "actual": d})
    return errors

# -----------------------------
# Routes
# -----------------------------
@app.get("/")
async def root():
    return FileResponse("static/index.html")


@app.get("/health")
async def health():
    return {"status": "ok", "message": "IPA alignment service running"}


@app.post("/align")
async def align_audio(file: UploadFile = File(...)):
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "audio.wav")
        audio_data = await file.read()
        with open(path, "wb") as f:
            f.write(audio_data)

        print(f"📁 Received {len(audio_data)} bytes audio")

        # 1️⃣ Transcribe with WhisperX (with word timestamps)
        print("🎤 Starting WhisperX transcription...")
        result = WHISPER_MODEL.transcribe(path, language="en")
        print(f"🔍 Raw WhisperX segments: {len(result.get('segments', []))}")

        # 2️⃣ Load and run alignment model manually for word-level timings
        try:
            align_model, metadata = whisperx.load_align_model(language_code="en", device=DEVICE)
            aligned = whisperx.align(
                result["segments"], align_model, metadata, path, DEVICE
            )
            result = aligned
            print("✅ WhisperX alignment complete (word-level timings enabled).")
        except Exception as e:
            print(f"⚠️ WhisperX alignment failed: {e}")

        transcript = " ".join([s["text"] for s in result["segments"]]).strip()
        print(f"✅ WHISPERX: '{transcript}'")

        if not transcript:
            return JSONResponse({
                "transcribed_text": "",
                "results": [],
                "accuracy": 0,
                "note": "No speech detected. Try speaking louder or closer to the mic."
            })

        # 2️⃣ Convert to proper audio format if needed
        try:
            from pydub import AudioSegment
            audio = AudioSegment.from_file(path)
            audio = audio.set_channels(1).set_frame_rate(16000).set_sample_width(2)
            converted_path = os.path.join(tmp, "converted.wav")
            audio.export(converted_path, format="wav")
            path = converted_path
        except Exception as e:
            print(f"⚠️  Audio conversion failed: {e}")

        # 3️⃣ Phoneme recognition (wav2vec2-espeak)
        try:
            waveform, sr = torchaudio.load(path)
        except Exception:
            waveform, sr = librosa.load(path, sr=16000)
            waveform = torch.tensor(waveform).unsqueeze(0)

        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)

        inputs = PHONEME_PROCESSOR(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            logits = PHONEME_MODEL_LOADED(**inputs).logits[0]

        pred_ids = torch.argmax(logits, dim=-1).cpu().numpy()
        tokens = PHONEME_PROCESSOR.tokenizer.convert_ids_to_tokens(pred_ids)

        time_per_frame = PHONEME_MODEL_LOADED.config.inputs_to_logits_ratio / 16000.0
        phoneme_segments = []
        current = None
        for i, token in enumerate(tokens):
            if token == PHONEME_PROCESSOR.tokenizer.pad_token:
                continue
            if current and token != current["phoneme"]:
                current["end"] = i * time_per_frame
                phoneme_segments.append(current)
                current = {"phoneme": token, "start": i * time_per_frame}
            elif not current:
                current = {"phoneme": token, "start": i * time_per_frame}
        if current:
            current["end"] = len(tokens) * time_per_frame
            phoneme_segments.append(current)

        phoneme_segments = [p for p in phoneme_segments if p["phoneme"] != "<pad>"]
        print("🔍 PHONEMES (IPA):", " ".join(p["phoneme"] for p in phoneme_segments))

        # 4️⃣ Group phonemes using WhisperX word timings
        word_groups = []
        segments = result.get("segments", [])
        if segments and "words" in segments[0]:
            for seg in segments:
                for w in seg["words"]:
                    start, end = w["start"], w["end"]
                    text = w["word"].lower()
                    word_groups.append({
                        "word": text,
                        "start": start,
                        "end": end,
                        "phonemes": []
                    })

            # assign phonemes to matching or nearest word interval
            for p in phoneme_segments:
                ph_mid = 0.5 * (p["start"] + p["end"])
                matches = [w for w in word_groups if w["start"] <= ph_mid <= w["end"]]
                if matches:
                    matches[0]["phonemes"].append(p["phoneme"])
                elif word_groups:
                    nearest = min(word_groups, key=lambda w: abs(ph_mid - 0.5 * (w["start"] + w["end"])))
                    nearest["phonemes"].append(p["phoneme"])
        else:
            print("⚠️ WhisperX returned no word segments — using fallback.")
            word_groups.append({
                "word": transcript.lower(),
                "start": 0.0,
                "end": phoneme_segments[-1]["end"] if phoneme_segments else 0.0,
                "phonemes": [p["phoneme"] for p in phoneme_segments],
            })

        print(f"📊 Grouped {len(word_groups)} words via WhisperX timings (snapped).")

        # 5️⃣ Compare with CMUdict (IPA)
        results = []
        for w in word_groups:
            word = normalize_word_for_cmu(w["word"])
            detected_ipa = normalize_ipa(w["phonemes"])
            std_arpabet = CMU_DICT.get(word, [[""]])[0]
            std_ipa = normalize_ipa(arpabet_to_ipa(std_arpabet))
            errs = compare_phoneme_sequences(std_ipa, detected_ipa)
            results.append({
                "word": word,
                "standard": std_ipa,
                "detected": detected_ipa,
                "errors": errs,
                "match": len(errs) == 0,
                "timing": {"start": w["start"], "end": w["end"]}
            })

        accuracy = sum(1 for r in results if r["match"]) / len(results) * 100 if results else 0
        return JSONResponse({
            "transcribed_text": transcript,
            "results": results,
            "accuracy": accuracy,
            "note": "Grouped via WhisperX word-level timings; pronunciations compared in IPA."
        })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
