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
from gtts import gTTS
import pyttsx3
import io
import base64
from google import genai

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()

# -----------------------------
# Setup + Model Initialization
# -----------------------------
app = FastAPI()

# Conversation History Storage
conversation_history = {}

# Pronunciation Tracking Storage
pronunciation_tracking = {}

# Text-to-Speech Engine
tts_engine = None
try:
    tts_engine = pyttsx3.init()
    tts_engine.setProperty('rate', 150)  # Speed of speech
    tts_engine.setProperty('volume', 0.8)  # Volume level (0.0 to 1.0)
except Exception as e:
    print(f"⚠️  TTS engine initialization failed: {e}")

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

def generate_gemini_response(transcript, session_id="default"):
    """Generate a conversational response using GenAI with conversation history."""
    # Get or create conversation history for this session
    if session_id not in conversation_history:
        conversation_history[session_id] = []
    
    # Add user message to history
    conversation_history[session_id].append({"role": "user", "content": transcript})
    
    # Build context from conversation history (last 10 messages to avoid token limits)
    recent_history = conversation_history[session_id][-10:]
    context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in recent_history])
    
    # Generate response with context
    prompt = f"""You are having a continuous conversation with a user. Here's the conversation history:

{context}

Respond naturally and engagingly to the user's latest message. Keep responses conversational and concise (1-2 sentences)."""
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash", 
            contents=prompt
        )
        ai_response = response.text
        
        # Add AI response to history
        conversation_history[session_id].append({"role": "assistant", "content": ai_response})
        
        return ai_response
    except Exception as e:
        print(f"GenAI API error: {e}")
        return "I'm having trouble generating a response right now. Please try again!"

def text_to_speech(text):
    """Convert text to speech and return audio data."""
    try:
        # Try Google TTS first (better quality)
        tts = gTTS(text=text, lang='en', slow=False)
        audio_buffer = io.BytesIO()
        tts.write_to_fp(audio_buffer)
        audio_buffer.seek(0)
        return audio_buffer.getvalue()
    except Exception as e:
        print(f"Google TTS failed: {e}")
        
        # Fallback to pyttsx3
        if tts_engine:
            try:
                audio_buffer = io.BytesIO()
                tts_engine.save_to_buffer(audio_buffer, text)
                audio_buffer.seek(0)
                return audio_buffer.getvalue()
            except Exception as e2:
                print(f"pyttsx3 TTS failed: {e2}")
        
        return None

def track_pronunciation_errors(session_id, word, detected_phonemes, standard_phonemes, errors):
    """Track pronunciation errors for analysis."""
    if session_id not in pronunciation_tracking:
        pronunciation_tracking[session_id] = {
            "total_words": 0,
            "error_count": 0,
            "word_errors": {},
            "phoneme_errors": {},
            "error_patterns": []
        }
    
    tracking = pronunciation_tracking[session_id]
    tracking["total_words"] += 1
    
    if errors:
        tracking["error_count"] += len(errors)
        
        # Track word-level errors
        if word not in tracking["word_errors"]:
            tracking["word_errors"][word] = {"count": 0, "errors": []}
        tracking["word_errors"][word]["count"] += 1
        tracking["word_errors"][word]["errors"].extend(errors)
        
        # Track phoneme-level errors
        for error in errors:
            expected_ph = error["expected"]
            actual_ph = error["actual"]
            if expected_ph not in tracking["phoneme_errors"]:
                tracking["phoneme_errors"][expected_ph] = {"count": 0, "substitutions": {}}
            tracking["phoneme_errors"][expected_ph]["count"] += 1
            if actual_ph not in tracking["phoneme_errors"][expected_ph]["substitutions"]:
                tracking["phoneme_errors"][expected_ph]["substitutions"][actual_ph] = 0
            tracking["phoneme_errors"][expected_ph]["substitutions"][actual_ph] += 1
        
        # Track error patterns
        tracking["error_patterns"].append({
            "word": word,
            "detected": detected_phonemes,
            "standard": standard_phonemes,
            "errors": errors
        })

def generate_pronunciation_feedback(session_id):
    """Generate structural pronunciation feedback using GenAI."""
    if session_id not in pronunciation_tracking:
        return "No pronunciation data available for analysis."
    
    tracking = pronunciation_tracking[session_id]
    
    if tracking["total_words"] == 0:
        return "No pronunciation data available for analysis."
    
    # Calculate overall accuracy
    accuracy = ((tracking["total_words"] - tracking["error_count"]) / tracking["total_words"]) * 100
    
    # Find most problematic words
    problematic_words = sorted(tracking["word_errors"].items(), 
                             key=lambda x: x[1]["count"], reverse=True)[:5]
    
    # Find most common phoneme substitutions
    common_substitutions = {}
    for expected, data in tracking["phoneme_errors"].items():
        for actual, count in data["substitutions"].items():
            if actual not in common_substitutions:
                common_substitutions[actual] = []
            common_substitutions[actual].append((expected, count))
    
    # Build analysis prompt
    analysis_data = {
        "total_words": tracking["total_words"],
        "error_count": tracking["error_count"],
        "accuracy": accuracy,
        "problematic_words": problematic_words,
        "common_substitutions": common_substitutions,
        "error_patterns": tracking["error_patterns"][-10:]  # Last 10 errors
    }
    
    prompt = f"""As a pronunciation coach, analyze the following pronunciation data and provide structured feedback:

OVERALL PERFORMANCE:
- Total words spoken: {analysis_data['total_words']}
- Pronunciation errors: {analysis_data['error_count']}
- Overall accuracy: {analysis_data['accuracy']:.1f}%

MOST PROBLEMATIC WORDS:
{chr(10).join([f"- '{word}': {data['count']} errors" for word, data in analysis_data['problematic_words']])}

COMMON PHONEME SUBSTITUTIONS:
{chr(10).join([f"- '{actual}' instead of '{expected}' ({count} times)" for actual, subs in analysis_data['common_substitutions'].items() for expected, count in subs[:3]])}

RECENT ERROR PATTERNS:
{chr(10).join([f"- '{pattern['word']}': Expected {pattern['standard']}, said {pattern['detected']}" for pattern in analysis_data['error_patterns'][-5:]])}

Please provide:
1. A brief overall assessment
2. The top 3 pronunciation challenges
3. Specific improvement recommendations
4. Practice suggestions for the most common errors

Keep the feedback encouraging and actionable."""
    
    try:
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt
        )
        return response.text
    except Exception as e:
        print(f"GenAI feedback generation error: {e}")
        return f"Pronunciation Analysis:\n- Overall accuracy: {accuracy:.1f}%\n- Most problematic words: {[word for word, _ in problematic_words[:3]]}\n- Total errors tracked: {tracking['error_count']}"

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
async def align_audio(file: UploadFile = File(...), session_id: str = "default"):
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
            
            # Track pronunciation errors for analysis
            track_pronunciation_errors(session_id, word, detected_ipa, std_ipa, errs)
            
            results.append({
                "word": word,
                "standard": std_ipa,
                "detected": detected_ipa,
                "errors": errs,
                "match": len(errs) == 0,
                "timing": {"start": w["start"], "end": w["end"]}
            })

        accuracy = sum(1 for r in results if r["match"]) / len(results) * 100 if results else 0
        
        # Generate GenAI response with conversation history
        print("🤖 Generating AI response...")
        ai_response = generate_gemini_response(transcript, session_id)
        print(f"🤖 AI Response: {ai_response}")
        
        # Generate audio for the AI response
        print("🔊 Converting to speech...")
        audio_data = text_to_speech(ai_response)
        audio_base64 = None
        
        if audio_data:
            audio_base64 = base64.b64encode(audio_data).decode('utf-8')
            print("✅ Audio generated successfully")
        else:
            print("⚠️  Audio generation failed")
        
        # Get conversation history for this session
        session_history = conversation_history.get(session_id, [])
        
        return JSONResponse({
            "transcribed_text": transcript,
            "results": results,
            "accuracy": accuracy,
            "ai_response": ai_response,
            "ai_audio": audio_base64,
            "conversation_history": session_history,
            "session_id": session_id,
            "note": "Grouped via WhisperX word-level timings; pronunciations compared in IPA."
        })


@app.post("/pronunciation-feedback")
async def get_pronunciation_feedback(session_id: str):
    """Get pronunciation analysis and feedback for a session."""
    try:
        feedback = generate_pronunciation_feedback(session_id)
        
        # Get tracking data for additional context
        tracking_data = pronunciation_tracking.get(session_id, {})
        
        return JSONResponse({
            "feedback": feedback,
            "tracking_data": {
                "total_words": tracking_data.get("total_words", 0),
                "error_count": tracking_data.get("error_count", 0),
                "accuracy": ((tracking_data.get("total_words", 0) - tracking_data.get("error_count", 0)) / max(tracking_data.get("total_words", 1), 1)) * 100,
                "problematic_words": list(tracking_data.get("word_errors", {}).keys())[:5],
                "session_id": session_id
            }
        })
    except Exception as e:
        return JSONResponse({
            "error": f"Failed to generate pronunciation feedback: {str(e)}",
            "feedback": "Unable to analyze pronunciation data at this time."
        })


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
