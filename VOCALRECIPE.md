# VocalRecipe — GPU Worker Integration Spec

Add a `/analyze` endpoint to the existing VocalEnhancer GPU worker.
No GPU required — Librosa runs on CPU only.

## Endpoint

```
POST /analyze
Authorization: Bearer {WORKER_SECRET}
Content-Type: application/json
```

## Input
```json
{
  "audio_url": "https://ddgwskkdjdelhxhmiasv.supabase.co/storage/v1/...",
  "daw": "fl-studio",
  "genre": "hip-hop"
}
```

Supported DAW values: `fl-studio`, `ableton-live`, `logic-pro`, `pro-tools`, `garageband`, `studio-one`
Supported genre values: `hip-hop`, `trap`, `r-and-b`, `pop`, `melodic-rap`, `other`

## Dependencies to add to requirements.txt
```
librosa>=0.10.0
soundfile>=0.12.1
numpy>=1.24.0
```

## Analysis Logic

```python
import librosa
import numpy as np
import soundfile as sf
import tempfile
import httpx

def analyze_vocal(audio_url: str, daw: str, genre: str) -> dict:
    # 1. Download audio
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        response = httpx.get(audio_url)
        tmp.write(response.content)
        tmp_path = tmp.name

    # 2. Load with librosa (max 60 seconds)
    y, sr = librosa.load(tmp_path, sr=22050, duration=60.0, mono=True)

    # 3. Extract features
    spectral_centroid = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    spectral_rolloff = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=sr)))
    rms = float(np.mean(librosa.feature.rms(y=y)))
    rms_db = float(librosa.amplitude_to_db(np.array([rms]))[0])
    zcr = float(np.mean(librosa.feature.zero_crossing_rate(y)))
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_means = np.mean(mfccs, axis=1).tolist()
    spectral_contrast = float(np.mean(librosa.feature.spectral_contrast(y=y, sr=sr)))
    
    # Fundamental frequency
    f0, voiced_flag, _ = librosa.pyin(y, fmin=80, fmax=400)
    f0_mean = float(np.nanmean(f0)) if np.any(voiced_flag) else 200.0

    # 4. Classify voice characteristics
    brightness = "bright" if spectral_centroid > 2500 else "dark" if spectral_centroid < 1500 else "balanced"
    voice_type = "female" if f0_mean > 200 else "male"
    dynamics = "compressed" if (rms_db > -20) else "dynamic"
    muddiness = "muddy" if spectral_contrast < 20 else "clear"
    breathiness = "breathy" if zcr > 0.1 else "solid"

    # 5. Generate chain
    chain = generate_chain(
        spectral_centroid=spectral_centroid,
        rms_db=rms_db,
        brightness=brightness,
        voice_type=voice_type,
        dynamics=dynamics,
        muddiness=muddiness,
        breathiness=breathiness,
        genre=genre
    )

    # 6. Format for DAW
    instructions = format_for_daw(chain, daw)

    return {
        "features": {
            "spectral_centroid": round(spectral_centroid),
            "rms_db": round(rms_db, 1),
            "brightness": brightness,
            "voice_type": voice_type,
            "dynamics": dynamics,
            "muddiness": muddiness,
            "breathiness": breathiness,
            "f0_mean": round(f0_mean)
        },
        "chain": chain,
        "daw_instructions": instructions,
        "genre_notes": get_genre_notes(genre)
    }


def generate_chain(spectral_centroid, rms_db, brightness, voice_type,
                   dynamics, muddiness, breathiness, genre):
    # High pass filter
    hpf_freq = 120 if voice_type == "male" else 100

    # Low-mid cut (mud removal)
    low_mid_cut = {"freq": 320, "gain": -3, "q": 1.2} if muddiness == "muddy" else {"freq": 300, "gain": -1.5, "q": 1.0}

    # Presence boost
    presence_boost = {"freq": 3000, "gain": 3.0, "q": 0.8} if brightness == "dark" else {"freq": 4000, "gain": 1.5, "q": 0.8}

    # Air
    air = {"freq": 10000, "gain": 2.5} if brightness != "bright" else {"freq": 12000, "gain": 1.0}

    # Compression
    threshold = max(-24, min(-12, rms_db - 4))
    ratio = "6:1" if dynamics == "compressed" else "4:1"
    attack = 5 if genre in ["trap", "hip-hop"] else 15
    release = 60 if genre in ["trap", "hip-hop"] else 100

    # Saturation
    sat_type = "tape" if genre in ["r-and-b", "pop"] else "tube"
    sat_drive = 20 if brightness == "dark" else 12

    # Reverb
    reverb_types = {
        "hip-hop": "room",
        "trap": "room",
        "r-and-b": "plate",
        "pop": "hall",
        "melodic-rap": "plate",
        "other": "room"
    }
    reverb_wet = 10 if genre in ["trap", "hip-hop"] else 14

    # Delay
    delay_types = {
        "hip-hop": {"type": "slapback", "time_ms": 80},
        "trap": {"type": "slapback", "time_ms": 60},
        "r-and-b": {"type": "ping-pong", "time_ms": 120},
        "pop": {"type": "ping-pong", "time_ms": 100},
        "melodic-rap": {"type": "ping-pong", "time_ms": 110},
        "other": {"type": "slapback", "time_ms": 80}
    }

    return {
        "eq": {
            "high_pass": {"freq": hpf_freq, "slope": 12},
            "low_mid_cut": low_mid_cut,
            "presence": presence_boost,
            "air": air
        },
        "compression": {
            "threshold": round(threshold),
            "ratio": ratio,
            "attack": attack,
            "release": release,
            "makeup_gain": 3
        },
        "saturation": {
            "type": sat_type,
            "drive": sat_drive
        },
        "reverb": {
            "type": reverb_types.get(genre, "room"),
            "wet": reverb_wet
        },
        "delay": {
            **delay_types.get(genre, {"type": "slapback", "time_ms": 80}),
            "feedback": 20,
            "wet": 8
        }
    }


# DAW-specific formatting
DAW_PLUGINS = {
    "fl-studio": {
        "eq": "Parametric EQ 2",
        "compressor": "Fruity Peak Controller + Compressor",
        "saturation": "Fruity Blood Overdrive (low drive)",
        "reverb": "Fruity Reeverb 2",
        "delay": "Fruity Delay 3"
    },
    "ableton-live": {
        "eq": "EQ Eight",
        "compressor": "Compressor",
        "saturation": "Saturator",
        "reverb": "Reverb",
        "delay": "Simple Delay"
    },
    "logic-pro": {
        "eq": "Channel EQ",
        "compressor": "Compressor",
        "saturation": "Exciter",
        "reverb": "ChromaVerb",
        "delay": "Tape Delay"
    },
    "pro-tools": {
        "eq": "EQ III",
        "compressor": "Dyn3 Compressor",
        "saturation": "AIR Distortion",
        "reverb": "AIR Reverb",
        "delay": "AIR Dynamic Delay"
    },
    "garageband": {
        "eq": "Channel EQ",
        "compressor": "Compressor",
        "saturation": "Amp Designer (clean setting)",
        "reverb": "Space Designer",
        "delay": "Tape Delay"
    },
    "studio-one": {
        "eq": "Pro EQ3",
        "compressor": "Compressor",
        "saturation": "Ampire (clean)",
        "reverb": "Room Reverb",
        "delay": "Analog Delay"
    }
}


def format_for_daw(chain: dict, daw: str) -> list:
    plugins = DAW_PLUGINS.get(daw, DAW_PLUGINS["fl-studio"])
    eq = chain["eq"]
    comp = chain["compression"]
    sat = chain["saturation"]
    rev = chain["reverb"]
    dly = chain["delay"]

    instructions = [
        f"1. Open {plugins['eq']}",
        f"   → High Pass Filter: {eq['high_pass']['freq']}Hz, {eq['high_pass']['slope']}dB/oct",
        f"   → Bell Cut: {eq['low_mid_cut']['freq']}Hz, {eq['low_mid_cut']['gain']}dB, Q {eq['low_mid_cut']['q']}",
        f"   → Presence Boost: {eq['presence']['freq']}Hz, +{eq['presence']['gain']}dB, Q {eq['presence']['q']}",
        f"   → Air Boost: {eq['air']['freq']}Hz, +{eq['air']['gain']}dB (shelf)",
        f"",
        f"2. Open {plugins['compressor']}",
        f"   → Threshold: {comp['threshold']}dB",
        f"   → Ratio: {comp['ratio']}",
        f"   → Attack: {comp['attack']}ms",
        f"   → Release: {comp['release']}ms",
        f"   → Makeup Gain: +{comp['makeup_gain']}dB",
        f"",
        f"3. Open {plugins['saturation']}",
        f"   → Type: {sat['type'].title()}",
        f"   → Drive: {sat['drive']}%",
        f"",
        f"4. Open {plugins['reverb']}",
        f"   → Type: {rev['type'].title()}",
        f"   → Wet: {rev['wet']}%",
        f"",
        f"5. Open {plugins['delay']}",
        f"   → Type: {dly['type'].title()}",
        f"   → Time: {dly.get('time_ms', 80)}ms",
        f"   → Feedback: {dly['feedback']}%",
        f"   → Wet: {dly['wet']}%",
    ]
    return [line for line in instructions]


def get_genre_notes(genre: str) -> str:
    notes = {
        "hip-hop": "Hip-Hop vocals thrive on clarity and punch. Keep the high end crisp, use a fast compressor to control dynamics, and keep reverb short and tight so the vocal cuts through the beat.",
        "trap": "Trap vocals need presence and edge. The slapback delay adds depth without washing out the mix. Keep saturation subtle — too much and it competes with 808s.",
        "r-and-b": "R&B vocals need warmth and space. The plate reverb adds smoothness, and tape saturation brings out the harmonic richness in the voice. Compress gently to preserve dynamics.",
        "pop": "Pop vocals need to sit on top of everything. Boost the presence and air, keep reverb medium-length, and use ping-pong delay to add width without muddiness.",
        "melodic-rap": "Melodic rap sits between Hip-Hop and R&B. The chain balances clarity with warmth — enough reverb to feel musical but tight enough to stay in the pocket.",
        "other": "This chain is a solid starting point. Adjust the EQ to taste based on your specific vocal and mix."
    }
    return notes.get(genre, notes["other"])
```

## Adding to app.py

Add the route to the existing FastAPI app in `app.py`:

```python
from analyze import analyze_vocal  # new module

@app.post("/analyze")
async def analyze_route(
    request: Request,
    _: None = Depends(verify_bearer)
):
    body = await request.json()
    audio_url = body.get("audio_url")
    daw = body.get("daw", "fl-studio")
    genre = body.get("genre", "hip-hop")

    if not audio_url:
        raise HTTPException(status_code=400, detail="audio_url required")

    try:
        result = analyze_vocal(audio_url, daw, genre)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
```

## Testing

```bash
curl -X POST https://your-runpod-endpoint/analyze \
  -H "Authorization: Bearer your_worker_secret" \
  -H "Content-Type: application/json" \
  -d '{
    "audio_url": "https://example.com/test-vocal.mp3",
    "daw": "fl-studio",
    "genre": "hip-hop"
  }'
```
