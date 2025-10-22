import os
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2CTCTokenizer,
    Wav2Vec2ForCTC
)
import torchaudio.functional as F
from g2p_en import G2p

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# ==========================================
# PHONEME CONVERSION (NO ESPEAK NEEDED)
# ==========================================

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

# ==========================================
# CONFIGURATION
# ==========================================

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
AUDIO_PATH = "audio2.wav"
WHISPER_MODEL_NAME = "openai/whisper-base.en"
PHONEME_MODEL = "facebook/wav2vec2-lv-60-espeak-cv-ft"

# ==========================================
# TRANSCRIBE
# ==========================================

print("🎤 Loading Whisper model...")
whisper_processor = WhisperProcessor.from_pretrained(WHISPER_MODEL_NAME)
whisper_model = WhisperForConditionalGeneration.from_pretrained(WHISPER_MODEL_NAME).to(DEVICE)

print(f"📂 Loading audio from: {AUDIO_PATH}")
import librosa
waveform, sr = librosa.load(AUDIO_PATH, sr=16000)
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

# ==========================================
# LOAD PHONEME MODEL
# ==========================================

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

# ==========================================
# CONVERT TO PHONEMES (using g2p-en)
# ==========================================

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

# ==========================================
# GENERATE EMISSIONS
# ==========================================

print("\n🎯 Generating emissions...")
inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")
inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

with torch.no_grad():
    logits = model(**inputs).logits

log_probs = torch.log_softmax(logits, dim=-1)
print(f"   Emission shape: {log_probs.shape}")

# ==========================================
# FORCED ALIGNMENT
# ==========================================

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

# ==========================================
# MERGE TOKENS
# ==========================================

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
    
    low_conf = [p for p in phoneme_results if p['posterior'] < 0.5]
    if low_conf:
        print(f"\n⚠️  {len(low_conf)} phonemes with low confidence (<0.5)")

print("\n✅ Done.")
print("\nNote: This uses g2p-en for phoneme conversion instead of espeak.")
print("Results may be less accurate than espeak for some words.")