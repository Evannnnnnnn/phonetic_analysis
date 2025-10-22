import os
import re
import tempfile
import torch
import torchaudio
import torchaudio.functional as F
import librosa
import nltk
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import (
    Wav2Vec2FeatureExtractor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2Processor,
    Wav2Vec2ForCTC,
)

from g2p_en import G2p
from gtts import gTTS
import pyttsx3
import io
import base64
from google import genai

import shutil

# Download NLTK data if not already present
try:
    nltk.data.find('taggers/averaged_perceptron_tagger_eng')
except LookupError:
    nltk.download('averaged_perceptron_tagger_eng', quiet=True)
try:
    nltk.data.find('corpora/cmudict')
except LookupError:
    nltk.download('cmudict', quiet=True)
try:
    nltk.data.find('tagsets/universal_tagset')
except LookupError:
    nltk.download('universal_tagset', quiet=True)

# The client gets the API key from the environment variable `GEMINI_API_KEY`.
client = genai.Client()

# -----------------------------
# Setup + Model Initialization
# -----------------------------
app = FastAPI()

# Conversation History Storage
conversation_history = {}

# Pronunciation Tracking Storage

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

WHISPER_MODEL_NAME = "openai/whisper-base.en"
PHONEME_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"

print("🎤 Loading Whisper model...")
whisper_processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_NAME)
whisper_model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_NAME).to(DEVICE)

print("\n🧠 Loading Wav2Vec2 phoneme model...")
feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(PHONEME_MODEL)
tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(PHONEME_MODEL)
processor = Wav2Vec2Processor(
    feature_extractor=feature_extractor, 
    tokenizer=tokenizer
)
model = Wav2Vec2ForCTC.from_pretrained(PHONEME_MODEL).to(DEVICE)
vocab = processor.tokenizer.get_vocab()
print(f"✅ Model vocabulary size: {len(vocab)}")

def text_to_phonemes_g2p(text, vocab):
    """
    Convert text to phonemes using g2p-en
    Maps to eSpeak-compatible IPA that matches the model vocabulary
    """
    g2p = G2p()
    
    # Enhanced ARPAbet to eSpeak IPA mapping
    # Based on the actual vocab from facebook/wav2vec2-lv-60-espeak-cv-ft
    ARPABET_TO_ESPEAK = {
        # Vowels - monophthongs
        "AA": "ɑː",  # father
        "AE": "æ",   # cat
        "AH": "ə",   # about (schwa) - changed from ʌ
        "AO": "ɔː",  # thought
        "EH": "ɛ",   # bed
        "ER": "ɜː",  # bird
        "IH": "ɪ",   # bit
        "IY": "iː",  # beat
        "UH": "ʊ",   # book
        "UW": "uː",  # boot
        
        # Diphthongs
        "AW": "aʊ",  # now
        "AY": "aɪ",  # bite
        "EY": "eɪ",  # bait
        "OW": "oʊ",  # boat
        "OY": "ɔɪ",  # boy
        
        # Consonants
        "B": "b",
        "CH": "tʃ",
        "D": "d",
        "DH": "ð",   # this
        "F": "f",
        "G": "ɡ",
        "HH": "h",
        "JH": "dʒ",  # judge
        "K": "k",
        "L": "l",
        "M": "m",
        "N": "n",
        "NG": "ŋ",   # sing
        "P": "p",
        "R": "ɹ",    # red
        "S": "s",
        "SH": "ʃ",   # ship
        "T": "t",
        "TH": "θ",   # think
        "V": "v",
        "W": "w",
        "Y": "j",
        "Z": "z",
        "ZH": "ʒ",   # measure
    }
    
    phonemes = []
    words = text.lower().split()
    
    for word in words:
        # Get ARPAbet from g2p
        arpabet_list = g2p(word)
        
        for arpa in arpabet_list:
            # Remove stress markers (0, 1, 2)
            arpa_clean = ''.join(c for c in arpa if c.isalpha())
            
            if arpa_clean in ARPABET_TO_ESPEAK:
                espeak_phone = ARPABET_TO_ESPEAK[arpa_clean]
                
                # Only add if it exists in the model's vocabulary
                if espeak_phone in vocab:
                    phonemes.append(espeak_phone)
                else:
                    # Try fallback alternatives
                    fallbacks = {
                        "ɑː": ["ɑ", "a"],
                        "iː": ["i"],
                        "uː": ["u"],
                        "ɜː": ["ɜ"],
                        "ɔː": ["ɔ"],
                    }
                    if espeak_phone in fallbacks:
                        for alt in fallbacks[espeak_phone]:
                            if alt in vocab:
                                phonemes.append(alt)
                                break
    
    return phonemes


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

        local_path = "saved_audio.wav"  # or any path you want
        shutil.copy(path, local_path)
        print(f"💾 Saved to {local_path}")

        print(f"📂 Loading audio from: {path}")
        try:
            waveform, sr = torchaudio.load(path)
        except Exception as e:
            print(f"⚠️ Torchaudio failed, using librosa fallback: {e}")
            import librosa
            waveform, sr = librosa.load(path, sr=16000)
            waveform = torch.from_numpy(waveform).unsqueeze(0)
        
        if sr != 16000:
            waveform = torchaudio.functional.resample(waveform, sr, 16000)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        inputs = whisper_processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            predicted_ids = whisper_model.generate(**inputs)
        transcript = whisper_processor.batch_decode(predicted_ids, skip_special_tokens=True)[0].strip()

        print(f"✅ Transcript: \"{transcript}\"")

        if not transcript:
            return JSONResponse({
                "transcribed_text": "",
                "results": [],
                "accuracy": 0,
                "note": "No speech detected. Try speaking louder or closer to the mic."
            })


        print("\n🔤 Converting transcript to phonemes (using g2p-en)...")
        phonemes = text_to_phonemes_g2p(transcript, vocab)

        if not phonemes:
            print("❌ ERROR: No phonemes generated!")
            print("Make sure g2p-en is installed: pip install g2p-en")
            exit(1)

        print(f"📜 Target phonemes: {phonemes}")
        print(f"   Total: {len(phonemes)} phonemes")

        # Verify all phonemes are in vocab
        missing = [p for p in phonemes if p not in vocab]
        if missing:
            print(f"⚠️ WARNING: Some phonemes not in vocab: {set(missing)}")
            phonemes = [p for p in phonemes if p in vocab]
            print(f"   Filtered to {len(phonemes)} valid phonemes")

        print("\n🎯 Generating emissions...")
        inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

        with torch.no_grad():
            logits = model(**inputs).logits

        log_probs = torch.log_softmax(logits, dim=-1)
        print(f"   Emission shape: {log_probs.shape}")

        print("\n🎯 Running forced alignment...")

        target_ids = processor.tokenizer.convert_tokens_to_ids(phonemes)
        targets = torch.tensor([target_ids], dtype=torch.int32, device=DEVICE)
        blank = processor.tokenizer.pad_token_id

        alignment, scores = F.forced_align(
            log_probs,
            targets,
            blank=blank
        )

        alignment = alignment[0]
        scores = scores[0]

        print("✅ Alignment complete!")

        print("\n📊 Merging repeated tokens...")
        scores = scores.exp()
        token_spans = F.merge_tokens(alignment, scores)
        print(f"✅ Merged into {len(token_spans)} token spans")

        # ==========================================
        # GENERATE RESULTS
        # ==========================================

        time_per_frame = model.config.inputs_to_logits_ratio / 16000.0
        phoneme_results = []

        for span in token_spans:
            token_id = span.token
            phoneme = processor.tokenizer.convert_ids_to_tokens([token_id])[0]
            
            start_time = span.start * time_per_frame
            end_time = span.end * time_per_frame
            
            phoneme_results.append({
                "phoneme": phoneme,
                "start": round(start_time, 3),
                "end": round(end_time, 3),
                "duration": round(end_time - start_time, 3),
                "posterior": round(float(span.score), 4),
            })

        # ==========================================
        # PRINT RESULTS
        # ==========================================

        print("\n" + "="*70)
        print("🎯 Phoneme-level alignment and posterior scores:")
        print("="*70)
        print(f"{'Phoneme':<10} {'Start':>8} {'End':>8} {'Duration':>8} {'Score':>8}")
        print("-"*70)

        for p in phoneme_results:
            print(f"{p['phoneme']:<10} {p['start']:>7.3f}s {p['end']:>7.3f}s {p['duration']:>7.3f}s {p['posterior']:>8.4f}")

        print("="*70)

        if phoneme_results:
            avg_score = sum(p['posterior'] for p in phoneme_results) / len(phoneme_results)
            print(f"\n📊 Statistics:")
            print(f"   Average confidence: {avg_score:.4f}")
            print(f"   Phonemes aligned: {len(phoneme_results)}")
            print(f"   Expected phonemes: {len(phonemes)}")
            
            low_conf = [p for p in phoneme_results if p['posterior'] < 0.2]
            if low_conf:
                print(f"\n⚠️  {len(low_conf)} phonemes with low confidence (<0.2)")

        print("\n✅ Done.")
        print("\nNote: This uses g2p-en for phoneme conversion instead of espeak.")
        print("Results may be less accurate than espeak for some words.")

        # Generate phoneme-level analysis with confidence scores
        low_confidence_count = len(low_conf)
        total_phonemes = len(phoneme_results)
        
        # Calculate accuracy score (percentage of high-confidence phonemes)
        accuracy_score = ((total_phonemes - low_confidence_count) / max(total_phonemes, 1)) * 100
        
        print(f"📊 Phoneme Analysis:")
        print(f"   Total phonemes: {total_phonemes}")
        print(f"   Low confidence (<0.1): {low_confidence_count}")
        print(f"   Accuracy score: {accuracy_score:.1f}%")
        
        # Generate AI response
        print("\n🤖 Generating AI response...")
        ai_response = generate_gemini_response(transcript, session_id)
        print(f"✅ AI response: {ai_response[:100]}...")
        
        # Generate AI audio
        print("\n🔊 Generating AI audio...")
        try:
            ai_audio_bytes = text_to_speech(ai_response)
            audio_base64 = base64.b64encode(ai_audio_bytes).decode('utf-8')
            print("✅ AI audio generated")
        except Exception as e:
            print(f"⚠️ TTS failed: {e}")
            audio_base64 = None
        
        # Get conversation history
        session_history = conversation_history.get(session_id, [])
        
        return JSONResponse({
            "transcribed_text": transcript,
            "phoneme_analysis": phoneme_results,
            "accuracy_score": accuracy_score,
            "total_phonemes": total_phonemes,
            "low_confidence_count": low_confidence_count,
            "ai_response": ai_response,
            "ai_audio": audio_base64,
            "conversation_history": session_history,
            "session_id": session_id,
            "note": "Phoneme-level analysis with confidence scores. Red highlighting for confidence < 0.1"
        })




if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
