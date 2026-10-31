# JingleGen AI – Smart Brand Jingle Creator
> **Final Year / College Capstone Generative AI Project**
> Built with Python, Streamlit, Digital Signal Processing (DSP), NumPy, SciPy, Matplotlib, and Google Gemini AI.

---

## 📌 Project Overview
**JingleGen AI** is a complete, full-stack Generative AI application designed to create original, memorable advertising jingles for fictional brands. The system generates:
- **Original advertising lyrics** (Verses, Earworm Hooks, Brand Slogans).
- **Multilingual jingles** in English, Hindi (with Romanized transliteration), and Telugu.
- **Polyphonic musical arrangements** across 8 genres (Pop, Acoustic, Electronic, Jazz, Hip-Hop, Classical, Corporate, Folk) and 6 moods.
- **High-fidelity audio synthesis (44.1 kHz WAV)** via physical modeling (Karplus-Strong guitar synthesis, harmonic piano, analog synth, percussion, and bass).
- **Heuristic evaluation**: Transparent memorability score, brand-fit rating, and interactive radar comparison charts.
- **Originality & Rights Verification Checklist**: Ensuring compliance with intellectual property standards.

---

## 🛠️ Technology Stack
- **Language**: Python 3.10+
- **Frontend / UI**: Streamlit with custom SaaS-grade CSS
- **Audio DSP & Synthesis**: NumPy, SciPy (`scipy.io.wavfile`, `scipy.signal`)
- **Visualizations**: Matplotlib (Oscilloscope waveforms, Spectrograms, Radar charts)
- **Generative AI (Optional / Supported)**: Google GenAI SDK (`gemini-3.8-flash`)
- **Offline Fallback**: Algorithmic Advertising Grammar & Harmony Engine (100% offline)
- **Persistence**: Lightweight JSON database (`data/history.json`) and Pandas export

---

## 📂 Project Structure
```text
jingle_generator/
│
├── app.py                      # Main Streamlit application and dashboard
├── requirements.txt            # Python package dependencies
├── .env.example                # Environment variable configuration template
├── README.md                   # Comprehensive project documentation
│
├── modules/
│   ├── lyrics_generator.py     # Gemini AI & offline linguistic rhyme engine
│   ├── music_generator.py      # Chord arrangements, scales, and track orchestration
│   ├── audio_processor.py      # Karplus-Strong, ADSR, DSP synthesis, waveform plotter
│   ├── evaluation.py           # Transparent heuristic memorability & brand fit metrics
│   └── history_manager.py      # Local JSON database storage and CSV/JSON export
│
├── data/
│   └── history.json            # Stored jingle records
│
├── generated_audio/            # Exported .wav commercial jingle files
│
└── assets/                     # Icons and graphics
```

---

## 🚀 Execution Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Configure Environment (Optional)
If you wish to use Gemini AI for enhanced lyrics generation:
```bash
cp .env.example .env
# Open .env and add your GEMINI_API_KEY
```
*Note: If no API key is provided, the application runs automatically in offline fallback mode using its built-in algorithmic engine and offline DSP synthesizer.*

### 3. Launch the Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` (or port `3000` if configured).

---

## 🎯 Key Application Pages
1. **📊 Dashboard**: KPI metrics, genre distribution charts, recent projects, and quick-launch CTA.
2. **🎵 Jingle Generator**: Interactive form with brand metadata, genre/tempo/instrument selection, 1-click brand presets, and multi-variation tabs.
3. **⚖️ Jingle Comparison**: A/B/C audio auditioning, side-by-side metric tables, and multi-axis polar radar charts.
4. **🧠 Memorability & Brand Fit**: Deep-dive diagnostics, brand name repetition counters, oscilloscope waveforms, and frequency spectrograms.
5. **🛡️ Originality & Rights Check**: 7-point pre-publication intellectual property checklist and lyric trope scanner.
6. **📂 History & Library**: Filterable library with inline audio players, WAV downloads, and CSV/JSON export.
7. **ℹ️ About Project**: Academic report, system workflow flowchart, and DSP mathematical formulations.

---

## 📜 Ethical AI & Copyright Notice
JingleGen AI creates original lyrics and synthesizes custom melodies. Automated checks cannot guarantee legal copyright clearance under intellectual property laws. Commercial broadcasts should undergo human legal review.
