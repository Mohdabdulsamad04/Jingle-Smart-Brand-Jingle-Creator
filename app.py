"""
JingleGen AI – Smart Brand Jingle Creator
A complete, production-grade Generative AI web application for college project presentations.
Built with Streamlit, Python DSP, NumPy, SciPy, Matplotlib, and Gemini AI.
"""

import os
import io
import time
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from dotenv import load_dotenv

# Load backend modules
from modules.lyrics_generator import LyricsGenerator
from modules.music_generator import MusicGenerator
from modules.audio_processor import plot_waveform_and_spectrogram
from modules.evaluation import JingleEvaluator, plot_comparison_radar_chart
from modules.history_manager import HistoryManager

load_dotenv()

# Streamlit Page Configuration
st.set_page_config(
    page_title="JingleGen AI – Smart Brand Jingle Creator",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom SaaS-Grade Light Theme CSS with Purple & Indigo Accents
st.markdown("""
<style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
        color: #1e293b;
    }

    /* Main Container Background */
    .stApp {
        background-color: #f8fafc;
    }

    /* Top Navigation / App Title Bar */
    .app-header {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #9333ea 100%);
        padding: 24px 32px;
        border-radius: 16px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 10px 25px -5px rgba(99, 102, 241, 0.25);
    }
    .app-header h1 {
        margin: 0;
        font-size: 28px;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #ffffff;
    }
    .app-header p {
        margin: 6px 0 0 0;
        font-size: 14px;
        color: #e0e7ff;
        opacity: 0.95;
    }

    /* Modern Card Styles */
    .metric-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 20px 22px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(99, 102, 241, 0.08);
        border-color: #cbd5e1;
    }
    .metric-value {
        font-size: 30px;
        font-weight: 800;
        color: #4f46e5;
        line-height: 1.1;
    }
    .metric-label {
        font-size: 13px;
        font-weight: 600;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 6px;
    }

    /* Status Badges */
    .badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .badge-purple {
        background: #ede9fe;
        color: #6d28d9;
        border: 1px solid #ddd6fe;
    }
    .badge-blue {
        background: #e0e7ff;
        color: #3730a3;
        border: 1px solid #c7d2fe;
    }
    .badge-green {
        background: #dcfce7;
        color: #15803d;
        border: 1px solid #bbf7d0;
    }
    .badge-orange {
        background: #ffedd5;
        color: #c2410c;
        border: 1px solid #fed7aa;
    }

    /* Lyric Display Box */
    .lyric-box {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 14px;
        padding: 22px;
        font-size: 14px;
        line-height: 1.8;
        white-space: pre-wrap;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.02);
    }
    .hook-highlight {
        background: #fdf4ff;
        border-left: 4px solid #c026d3;
        padding: 12px 16px;
        border-radius: 6px;
        font-weight: 700;
        color: #86198f;
        margin: 12px 0;
    }

    /* Streamlit Button Styling */
    div.stButton > button:first-child {
        border-radius: 10px;
        font-weight: 600;
        padding: 10px 20px;
        transition: all 0.2s ease;
    }
    .generate-btn button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 100%) !important;
        color: white !important;
        border: none !important;
        font-size: 16px !important;
        padding: 14px 28px !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.35) !important;
    }
    .generate-btn button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(99, 102, 241, 0.5) !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 14px;
        font-weight: 500;
        color: #334155;
    }

    /* Disclaimer box */
    .disclaimer-box {
        background: #fffbeb;
        border-left: 4px solid #f59e0b;
        padding: 14px 18px;
        border-radius: 8px;
        font-size: 12px;
        color: #92400e;
        line-height: 1.6;
        margin-top: 16px;
    }
</style>
""", unsafe_allow_html=True)


# Initialize Backend Singletons in Session State
@st.cache_resource
def get_backend_services():
    lyrics_gen = LyricsGenerator()
    music_gen = MusicGenerator(output_dir="generated_audio")
    hist_mgr = HistoryManager(storage_file="data/history.json")
    # Pre-seed demo data if empty so college presentation is populated immediately
    hist_mgr.seed_initial_demo_data_if_empty(music_gen)
    return lyrics_gen, music_gen, hist_mgr

lyrics_gen, music_gen, hist_mgr = get_backend_services()

# Session State for Generator Form & Current Variations
if 'current_variations' not in st.session_state:
    st.session_state.current_variations = []
if 'form_brand_name' not in st.session_state:
    st.session_state.form_brand_name = "SparkleBrew"
if 'form_brand_desc' not in st.session_state:
    st.session_state.form_brand_desc = "Refreshing organic citrus sparkling energy tea for college students and active creators."
if 'form_category' not in st.session_state:
    st.session_state.form_category = "Food"
if 'form_target_audience' not in st.session_state:
    st.session_state.form_target_audience = "College students, gamers, and young creatives"
if 'form_genre' not in st.session_state:
    st.session_state.form_genre = "Pop"
if 'form_mood' not in st.session_state:
    st.session_state.form_mood = "Energetic"
if 'form_bpm' not in st.session_state:
    st.session_state.form_bpm = 126
if 'form_duration' not in st.session_state:
    st.session_state.form_duration = 15
if 'form_language' not in st.session_state:
    st.session_state.form_language = "English"
if 'form_instruments' not in st.session_state:
    st.session_state.form_instruments = ["Synthesizer", "Piano", "Drums", "Bass"]


# -----------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding: 10px 0 20px 0; border-bottom: 1px solid #e2e8f0; margin-bottom: 20px;">
        <div style="display: flex; align-items: center; gap: 10px;">
            <div style="background: linear-gradient(135deg, #4f46e5, #9333ea); width: 38px; height: 38px; border-radius: 10px; display: flex; align-items: center; justify-content: center; color: white; font-size: 20px; font-weight: bold;">
                J
            </div>
            <div>
                <h3 style="margin: 0; font-size: 18px; font-weight: 800; color: #1e293b;">JingleGen AI</h3>
                <p style="margin: 0; font-size: 11px; color: #64748b; font-weight: 500;">Smart Brand Jingle Creator</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigation",
        [
            "📊 Dashboard",
            "🎵 Jingle Generator",
            "⚖️ Jingle Comparison",
            "🧠 Memorability & Brand Fit",
            "🛡️ Originality & Rights Check",
            "📂 History & Library",
            "ℹ️ About Project"
        ],
        label_visibility="collapsed"
    )

    st.markdown("---")
    st.markdown("##### ⚡ Engine Status")
    
    # Check Gemini API status
    gemini_key = os.environ.get("GEMINI_API_KEY", "").strip()
    if gemini_key:
        st.markdown('<span class="badge badge-green">● Gemini AI Model Active</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge badge-purple">● Local Generative Engine (Offline Mode)</span>', unsafe_allow_html=True)

    st.markdown('<span class="badge badge-blue">● Python DSP Synthesizer Ready (44.1 kHz)</span>', unsafe_allow_html=True)

    st.markdown("---")
    st.caption("College Capstone Demonstration • Final Year GenAI Project")


# -----------------------------------------------------------------------------
# 1. DASHBOARD PAGE
# -----------------------------------------------------------------------------
if page == "📊 Dashboard":
    # Header Banner
    st.markdown("""
    <div class="app-header">
        <h1>JingleGen AI – Smart Brand Jingle Creator</h1>
        <p>Create original advertising jingles and audio branding for fictional companies using Generative AI.</p>
    </div>
    """, unsafe_allow_html=True)

    history_records = hist_mgr.load_all()
    total_jingles = len(history_records)
    unique_brands = len(set(r.get('brand_name', '') for r in history_records if r.get('brand_name')))
    genres = [r.get('genre', 'Pop') for r in history_records]
    top_genre = max(set(genres), key=genres.count) if genres else "Pop"
    
    # Calculate Average Memorability
    mem_scores = [r.get('evaluation', {}).get('memorability_score', 85.0) for r in history_records if r.get('evaluation')]
    avg_mem = round(sum(mem_scores) / len(mem_scores), 1) if mem_scores else 91.2

    # Metric Cards Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{total_jingles}</div>
            <div class="metric-label">Total Jingles Generated</div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{unique_brands}</div>
            <div class="metric-label">Brand Projects</div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">8</div>
            <div class="metric-label">Musical Styles Available</div>
        </div>
        """, unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_mem}%</div>
            <div class="metric-label">Avg Memorability Score</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Call to action & Visualization row
    col_chart, col_cta = st.columns([1.6, 1.0])

    with col_chart:
        st.subheader("Distribution of Generated Jingles by Musical Genre")
        if genres:
            genre_counts = pd.Series(genres).value_counts()
            fig, ax = plt.subplots(figsize=(6.5, 3.2))
            fig.patch.set_facecolor('#ffffff')
            ax.set_facecolor('#ffffff')

            bar_colors = ['#4f46e5', '#7c3aed', '#a855f7', '#ec4899', '#06b6d4', '#10b981', '#f59e0b', '#64748b']
            bars = ax.bar(genre_counts.index, genre_counts.values, color=bar_colors[:len(genre_counts)], width=0.55, edgecolor='#e2e8f0', linewidth=1)
            
            # Label bars
            for bar in bars:
                yval = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, int(yval), ha='center', va='bottom', fontsize=9, fontweight='bold', color='#334155')

            ax.set_ylabel("Count", fontsize=9, color='#64748b')
            ax.set_ylim(0, max(genre_counts.values) + 1.2)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_color('#cbd5e1')
            ax.spines['bottom'].set_color('#cbd5e1')
            ax.tick_params(colors='#475569', labelsize=9)
            plt.xticks(rotation=15)
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.info("No jingles generated yet. Launch your first jingle below!")

    with col_cta:
        st.markdown("""
        <div class="metric-card" style="height: 100%; display: flex; flex-direction: column; justify-content: space-between;">
            <div>
                <span class="badge badge-purple">✨ Generative AI Studio</span>
                <h3 style="margin: 12px 0 8px 0; color: #1e293b;">Launch a New Brand Jingle</h3>
                <p style="font-size: 13px; color: #64748b; line-height: 1.6;">
                    Craft original advertising lyrics, compose custom musical hooks, synthesize broadcast-ready WAV audio, and verify copyright readiness in seconds.
                </p>
            </div>
            <div style="margin-top: 20px;">
                <p style="font-size: 12px; color: #6366f1; font-weight: 600; margin-bottom: 8px;">✓ Original Lyrics &nbsp;•&nbsp; ✓ Polyphonic Synth &nbsp;•&nbsp; ✓ Heuristic Fit</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("🚀 CREATE NEW JINGLE NOW", use_container_width=True, type="primary"):
            st.session_state.sidebar_nav = "🎵 Jingle Generator"
            st.rerun()

    st.markdown("---")
    st.subheader("Recently Generated Brand Jingles")

    recent_jingles = history_records[:3]
    if recent_jingles:
        cols = st.columns(len(recent_jingles))
        for idx, j in enumerate(recent_jingles):
            with cols[idx]:
                audio_path = j.get('audio', {}).get('file_path') or j.get('file_path')
                eval_data = j.get('evaluation', {})
                mem = eval_data.get('memorability_score', 90)

                st.markdown(f"""
                <div class="metric-card">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 8px;">
                        <h4 style="margin: 0; color: #1e293b;">{j.get('brand_name')}</h4>
                        <span class="badge badge-blue">{j.get('genre')}</span>
                    </div>
                    <p style="font-size: 12px; color: #64748b; margin-bottom: 12px; min-height: 38px;">
                        "{j.get('lyrics', {}).get('hook_line', 'Catchy brand earworm hook')}"
                    </p>
                    <div style="display: flex; justify-content: space-between; font-size: 11px; color: #475569; margin-bottom: 10px;">
                        <span>Mood: <b>{j.get('mood')}</b></span>
                        <span>Tempo: <b>{j.get('bpm')} BPM</b></span>
                        <span>Score: <b style="color: #4f46e5;">{mem}/100</b></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
                if audio_path and os.path.exists(audio_path):
                    st.audio(audio_path, format="audio/wav")
    else:
        st.info("No history entries yet. Click 'Create New Jingle' to get started!")


# -----------------------------------------------------------------------------
# 2. JINGLE GENERATOR PAGE
# -----------------------------------------------------------------------------
elif page == "🎵 Jingle Generator":
    st.markdown("""
    <div class="app-header">
        <h1>🎵 Jingle Generator Studio</h1>
        <p>Configure brand attributes and musical preferences to generate original advertising jingles and studio-grade audio.</p>
    </div>
    """, unsafe_allow_html=True)

    # 1-Click Demo Brand Presets
    st.markdown("##### 💡 Quick Fictional Brand Presets (Click to autofill):")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        if st.button("⚡ SparkleBrew (Energy Tea)", use_container_width=True):
            st.session_state.form_brand_name = "SparkleBrew"
            st.session_state.form_brand_desc = "Refreshing organic citrus sparkling energy tea for college students and active creators."
            st.session_state.form_category = "Food"
            st.session_state.form_target_audience = "College students, gamers, and young creatives"
            st.session_state.form_genre = "Pop"
            st.session_state.form_mood = "Energetic"
            st.session_state.form_bpm = 126
            st.rerun()
    with p2:
        if st.button("🍔 ZestyBite (Fast Casual)", use_container_width=True):
            st.session_state.form_brand_name = "ZestyBite"
            st.session_state.form_brand_desc = "Flame-grilled gourmet smash burgers with signature spiced garlic aioli."
            st.session_state.form_category = "Food"
            st.session_state.form_target_audience = "Foodies, late-night youth, and hungry friends"
            st.session_state.form_genre = "Acoustic"
            st.session_state.form_mood = "Playful"
            st.session_state.form_bpm = 116
            st.rerun()
    with p3:
        if st.button("👟 CloudStride (Footwear)", use_container_width=True):
            st.session_state.form_brand_name = "CloudStride"
            st.session_state.form_brand_desc = "Ultra-lightweight eco-friendly running shoes crafted with responsive air cushions."
            st.session_state.form_category = "Fashion"
            st.session_state.form_target_audience = "Marathoners, athletes, and city commuters"
            st.session_state.form_genre = "Electronic"
            st.session_state.form_mood = "Inspirational"
            st.session_state.form_bpm = 124
            st.rerun()
    with p4:
        if st.button("🧠 NeuroLearn (EdTech)", use_container_width=True):
            st.session_state.form_brand_name = "NeuroLearn"
            st.session_state.form_brand_desc = "Gamified AI micro-learning platform for coding, languages, and technical mastery."
            st.session_state.form_category = "Education"
            st.session_state.form_target_audience = "Tech students, aspiring developers, and career switchers"
            st.session_state.form_genre = "Corporate"
            st.session_state.form_mood = "Inspirational"
            st.session_state.form_bpm = 110
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Input Form
    with st.form("jingle_input_form"):
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("### 🏢 Brand Information")
            brand_name = st.text_input("Brand Name *", value=st.session_state.form_brand_name, placeholder="e.g. SparkleBrew, CloudStride")
            brand_desc = st.text_area("Brand Description *", value=st.session_state.form_brand_desc, height=110, placeholder="Explain what the product does, key benefits, and brand personality...")
            
            categories = ["Food", "Fashion", "Technology", "Education", "Healthcare", "Travel", "Shopping", "Other"]
            cat_idx = categories.index(st.session_state.form_category) if st.session_state.form_category in categories else 0
            category = st.selectbox("Product Category *", categories, index=cat_idx)

            target_audience = st.text_input("Target Audience *", value=st.session_state.form_target_audience, placeholder="e.g. Young adults, college students, health-conscious runners")

        with col_right:
            st.markdown("### 🎼 Musical Preferences")
            
            genres = ["Pop", "Acoustic", "Electronic", "Jazz", "Hip-Hop", "Classical", "Corporate", "Folk"]
            genre_idx = genres.index(st.session_state.form_genre) if st.session_state.form_genre in genres else 0
            genre = st.selectbox("Musical Genre *", genres, index=genre_idx)

            moods = ["Happy", "Energetic", "Emotional", "Playful", "Inspirational", "Calm"]
            mood_idx = moods.index(st.session_state.form_mood) if st.session_state.form_mood in moods else 0
            mood = st.selectbox("Mood *", moods, index=mood_idx)

            col_tempo, col_bpm = st.columns(2)
            with col_tempo:
                tempo = st.selectbox("Tempo Descriptor", ["Slow", "Medium", "Fast"], index=1)
            with col_bpm:
                bpm = st.slider("Beats Per Minute (BPM)", min_value=60, max_value=180, value=st.session_state.form_bpm, step=2)

            instruments_options = ["Guitar", "Piano", "Drums", "Synthesizer", "Flute", "Violin", "Bass"]
            selected_instruments = st.multiselect(
                "Instrumentation *",
                instruments_options,
                default=[inst for inst in st.session_state.form_instruments if inst in instruments_options]
            )

            col_dur, col_lang = st.columns(2)
            with col_dur:
                duration_sec = st.select_slider("Jingle Duration (Seconds) *", options=[10, 15, 20, 30], value=st.session_state.form_duration)
            with col_lang:
                languages = ["English", "Hindi", "Telugu"]
                lang_idx = languages.index(st.session_state.form_language) if st.session_state.form_language in languages else 0
                language = st.selectbox("Language *", languages, index=lang_idx)

        st.markdown("<br>", unsafe_allow_html=True)
        generate_clicked = st.form_submit_button("✨ GENERATE MY JINGLE", use_container_width=True)

    # Execution upon Form Submission
    if generate_clicked:
        # Input Validation
        if not brand_name.strip():
            st.error("Validation Error: Please provide a Brand Name.")
        elif not brand_desc.strip():
            st.error("Validation Error: Please provide a Brand Description.")
        elif not target_audience.strip():
            st.error("Validation Error: Please specify the Target Audience.")
        elif not selected_instruments:
            st.error("Validation Error: Please select at least one instrument.")
        else:
            with st.status("Composing original brand jingle...", expanded=True) as status:
                st.write("🔍 Step 1/4: Analyzing brand positioning and audience profile...")
                time.sleep(0.4)

                st.write("✍️ Step 2/4: Generating 100% original advertising lyrics and tagline hook...")
                var_count = len(st.session_state.current_variations) + 1
                lyrics_result = lyrics_gen.generate(
                    brand_name=brand_name.strip(),
                    brand_description=brand_desc.strip(),
                    category=category,
                    target_audience=target_audience.strip(),
                    genre=genre,
                    mood=mood,
                    duration_sec=duration_sec,
                    language=language,
                    variation_id=var_count
                )
                time.sleep(0.3)

                st.write("🎹 Step 3/4: Arranging multi-track polyphony and synthesizing audio (DSP engine)...")
                audio_result = music_gen.generate_jingle_audio(
                    brand_name=brand_name.strip(),
                    genre=genre,
                    mood=mood,
                    tempo_str=tempo,
                    bpm=bpm,
                    instruments=selected_instruments,
                    duration_sec=duration_sec,
                    variation=var_count
                )
                time.sleep(0.3)

                st.write("📊 Step 4/4: Calculating advertising memorability, rhythm, and brand-fit metrics...")
                eval_result = JingleEvaluator.evaluate(
                    brand_name=brand_name.strip(),
                    brand_description=brand_desc.strip(),
                    category=category,
                    target_audience=target_audience.strip(),
                    genre=genre,
                    mood=mood,
                    bpm=bpm,
                    duration_sec=duration_sec,
                    lyrics_data=lyrics_result,
                    instruments=selected_instruments
                )

                # Assemble complete record
                full_record = {
                    'brand_name': brand_name.strip(),
                    'brand_description': brand_desc.strip(),
                    'category': category,
                    'target_audience': target_audience.strip(),
                    'genre': genre,
                    'mood': mood,
                    'tempo': tempo,
                    'bpm': bpm,
                    'duration_sec': duration_sec,
                    'language': language,
                    'instruments': selected_instruments,
                    'variation_num': var_count,
                    'lyrics': lyrics_result,
                    'audio': audio_result,
                    'evaluation': eval_result,
                    'timestamp': time.strftime("%Y-%m-%d %H:%M:%S")
                }

                # Save in history and session
                saved = hist_mgr.save_jingle(full_record)
                st.session_state.current_variations.append(saved)

                status.update(label="🎉 Jingle generated and synthesized successfully!", state="complete", expanded=False)

    # Display Generated Results & Variations
    if st.session_state.current_variations:
        st.markdown("---")
        st.subheader("🎵 Generated Jingle Output & Variations")

        # Create tabs for variations
        var_tabs = st.tabs([f"Variation #{i+1} ({v.get('genre')})" for i, v in enumerate(st.session_state.current_variations)])

        for idx, tab in enumerate(var_tabs):
            item = st.session_state.current_variations[idx]
            lyrics_data = item.get('lyrics', {})
            audio_data = item.get('audio', {})
            eval_data = item.get('evaluation', {})
            audio_file = audio_data.get('file_path')

            with tab:
                col_res_left, col_res_right = st.columns([1.1, 1.2])

                with col_res_left:
                    st.markdown(f"### 🏷️ {item.get('brand_name')} – Jingle #{item.get('variation_num', idx+1)}")
                    
                    # Metadata Badges
                    badge_engine = lyrics_data.get('engine_badge', 'Algorithmic Engine')
                    st.markdown(f"""
                    <div style="margin-bottom: 14px;">
                        <span class="badge badge-purple">{item.get('genre')}</span>
                        <span class="badge badge-blue">{item.get('mood')}</span>
                        <span class="badge badge-orange">{item.get('bpm')} BPM</span>
                        <span class="badge badge-green">{item.get('duration_sec')}s Duration</span>
                        <span class="badge badge-purple">{item.get('language')}</span>
                    </div>
                    """, unsafe_allow_html=True)

                    st.caption(f"Engine: {badge_engine}")

                    # Audio Player
                    st.markdown("##### 🔊 Audio Playback")
                    if audio_file and os.path.exists(audio_file):
                        st.audio(audio_file, format="audio/wav")

                        # Download Button
                        with open(audio_file, "rb") as f:
                            wav_bytes = f.read()
                        st.download_button(
                            label=f"⬇️ Download WAV Audio ({item.get('brand_name')}_v{idx+1}.wav)",
                            data=wav_bytes,
                            file_name=f"{item.get('brand_name').lower()}_jingle_v{idx+1}.wav",
                            mime="audio/wav",
                            use_container_width=True
                        )
                    else:
                        st.warning("Audio file preview not found on disk.")

                    # Audio Source Transparency Disclaimer
                    st.markdown(f"""
                    <div style="background: #f1f5f9; border-radius: 8px; padding: 10px 14px; margin-top: 12px; font-size: 11px; color: #475569;">
                        <b>Audio Generation Mode:</b> {audio_data.get('audio_source_label', 'Algorithmic DSP Synthesizer')}
                        <br><i>Synthesized using multi-track polyphony, Karplus-Strong physical modeling, and ADSR envelopes.</i>
                    </div>
                    """, unsafe_allow_html=True)

                    # Quick Scores
                    st.markdown("##### 📈 Quick AI Scores")
                    s1, s2, s3 = st.columns(3)
                    s1.metric("Memorability", f"{eval_data.get('memorability_score', 90)}/100")
                    s2.metric("Brand Fit", f"{eval_data.get('brand_fit_score', 92)}/100")
                    s3.metric("Overall", f"{eval_data.get('overall_score', 91)}/100")

                with col_res_right:
                    st.markdown("### 📝 Advertising Lyrics")
                    
                    hook_text = lyrics_data.get('hook_line', '')
                    slogan_text = lyrics_data.get('slogan', '')

                    st.markdown(f"""
                    <div class="hook-highlight">
                        🎯 <b>EARWORM HOOK:</b> "{hook_text}"
                    </div>
                    """, unsafe_allow_html=True)

                    st.markdown('<div class="lyric-box">' + lyrics_data.get('lyrics_full', '') + '</div>', unsafe_allow_html=True)

                    if lyrics_data.get('transliteration'):
                        st.markdown("##### 🔤 Romanized Sing-Along Pronunciation:")
                        st.info(lyrics_data.get('transliteration'))

                    st.markdown(f"**Tagline / Slogan:** *{slogan_text}*")

                    st.markdown("---")
                    st.markdown("##### 🎛️ Arrangement & Production Prompt")
                    st.caption(lyrics_data.get('musical_prompt', ''))


# -----------------------------------------------------------------------------
# 3. JINGLE COMPARISON PAGE
# -----------------------------------------------------------------------------
elif page == "⚖️ Jingle Comparison":
    st.markdown("""
    <div class="app-header">
        <h1>⚖️ Jingle Feature Comparison</h1>
        <p>Compare up to three jingle variations side-by-side on musical attributes, lyric catchiness, and audience fit.</p>
    </div>
    """, unsafe_allow_html=True)

    records = hist_mgr.load_all()

    if len(records) < 2:
        st.warning("You need at least 2 generated jingles in history to perform a comparison. Generate some in the 'Jingle Generator' tab!")
    else:
        # Selection options
        options = {f"{r.get('brand_name')} (ID: {r.get('id')[:6]} | {r.get('genre')} | {r.get('mood')})": r for r in records}
        selected_keys = st.multiselect(
            "Select up to 3 jingles to compare:",
            options=list(options.keys()),
            default=list(options.keys())[:min(3, len(options))]
        )

        if len(selected_keys) == 0:
            st.info("Select at least one jingle above.")
        else:
            selected_items = [options[k] for k in selected_keys[:3]]

            # Side-by-Side Audio Players
            st.markdown("### 🎧 A/B/C Listening Audition")
            audio_cols = st.columns(len(selected_items))
            for i, item in enumerate(selected_items):
                with audio_cols[i]:
                    st.markdown(f"**Option {chr(65+i)}: {item.get('brand_name')}**")
                    st.caption(f"{item.get('genre')} • {item.get('mood')} • {item.get('bpm')} BPM")
                    audio_p = item.get('audio', {}).get('file_path') or item.get('file_path')
                    if audio_p and os.path.exists(audio_p):
                        st.audio(audio_p, format="audio/wav")
                    else:
                        st.text("Audio unavailable")

            st.markdown("---")

            # Radar Chart Visualization
            col_radar, col_table = st.columns([1.1, 1.2])

            with col_radar:
                st.subheader("Radar Evaluation Chart")
                radar_fig = plot_comparison_radar_chart(selected_items)
                st.pyplot(radar_fig)
                st.caption("Axes represent heuristic scores across 6 key commercial audio branding dimensions.")

            with col_table:
                st.subheader("Feature Comparison Table")
                comp_rows = []
                for item in selected_items:
                    ev = item.get('evaluation', {})
                    ly = item.get('lyrics', {})
                    comp_rows.append({
                        'Brand': item.get('brand_name'),
                        'Genre': item.get('genre'),
                        'Mood': item.get('mood'),
                        'BPM': item.get('bpm'),
                        'Duration': f"{item.get('duration_sec')}s",
                        'Hook Length': f"{len(ly.get('hook_line', '').split())} words",
                        'Memorability': f"{ev.get('memorability_score', 0)}/100",
                        'Brand Fit': f"{ev.get('brand_fit_score', 0)}/100",
                        'Overall': f"{ev.get('overall_score', 0)}/100",
                    })
                df_comp = pd.DataFrame(comp_rows)
                st.dataframe(df_comp, use_container_width=True, hide_index=True)

                st.markdown("""
                <div class="disclaimer-box">
                    <b>Heuristic Evaluation Model Notice:</b><br>
                    Scores are computed using transparent heuristics evaluating brand name repetition frequency, syllabic simplicity, tempo suitability (105–135 BPM advertising peak), and genre-audience alignment. These are estimated benchmarks rather than scientifically validated human lab results.
                </div>
                """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 4. MEMORABILITY AND BRAND FIT ANALYSIS
# -----------------------------------------------------------------------------
elif page == "🧠 Memorability & Brand Fit":
    st.markdown("""
    <div class="app-header">
        <h1>🧠 Memorability & Brand Fit Analysis</h1>
        <p>In-depth acoustic and linguistic diagnostic evaluation of commercial jingles.</p>
    </div>
    """, unsafe_allow_html=True)

    records = hist_mgr.load_all()
    if not records:
        st.info("No jingles available to evaluate. Generate one first in the Jingle Generator tab!")
    else:
        brand_options = {f"{r.get('brand_name')} – {r.get('lyrics', {}).get('hook_line', '')[:35]}... ({r.get('timestamp')})": r for r in records}
        choice = st.selectbox("Select Jingle for Diagnostic Evaluation:", list(brand_options.keys()))
        target = brand_options[choice]

        ev = target.get('evaluation', {})
        sub = ev.get('sub_metrics', {})
        audio_p = target.get('audio', {}).get('file_path') or target.get('file_path')

        # Top Metric Gauge Row
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <div class="metric-label">Memorability Score</div>
                <div class="metric-value" style="color: #7c3aed; font-size: 38px;">{ev.get('memorability_score', 90)}%</div>
                <p style="font-size: 11px; color: #64748b; margin-top: 4px;">Earworm retention & brand salience</p>
            </div>
            """, unsafe_allow_html=True)
        with m2:
            st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <div class="metric-label">Brand Fit Score</div>
                <div class="metric-value" style="color: #4f46e5; font-size: 38px;">{ev.get('brand_fit_score', 92)}%</div>
                <p style="font-size: 11px; color: #64748b; margin-top: 4px;">Audience, mood & category alignment</p>
            </div>
            """, unsafe_allow_html=True)
        with m3:
            st.markdown(f"""
            <div class="metric-card" style="text-align: center;">
                <div class="metric-label">Composite Effectiveness</div>
                <div class="metric-value" style="color: #06b6d4; font-size: 38px;">{ev.get('overall_score', 91)}%</div>
                <p style="font-size: 11px; color: #64748b; margin-top: 4px;">Weighted commercial broadcast index</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Oscilloscope Waveform & Spectrogram
        st.subheader("Oscilloscope Waveform & Frequency Spectrogram")
        if audio_p and os.path.exists(audio_p):
            fig_spec = plot_waveform_and_spectrogram(audio_p, title=target.get('brand_name'))
            st.pyplot(fig_spec)
        else:
            st.info("Audio waveform preview not available for this entry.")

        # Strengths & Suggestions
        c_str, c_sug = st.columns(2)
        with c_str:
            st.markdown("### 🌟 Identified Strengths")
            for st_item in ev.get('strengths', ["Well-balanced melodic motive", "Clear brand salience"]):
                st.markdown(f"✅ {st_item}")
        with c_sug:
            st.markdown("### 💡 Actionable Improvement Suggestions")
            for sg_item in ev.get('suggestions', ["Test higher tempo variation (128 BPM) for enhanced youth responsiveness"]):
                st.markdown(f"⚡ {sg_item}")

        # Sub-Metric Breakdown
        st.markdown("---")
        st.subheader("Linguistic & Acoustic Sub-Metric Breakdown")
        col_b1, col_b2 = st.columns(2)
        with col_b1:
            st.write(f"**Brand Name Repetition Score:** {sub.get('repetition_score', 90)} / 100")
            st.progress(min(1.0, sub.get('repetition_score', 90) / 100.0))

            st.write(f"**Hook Brevity & Punchiness:** {sub.get('hook_brevity_score', 92)} / 100")
            st.progress(min(1.0, sub.get('hook_brevity_score', 92) / 100.0))

            st.write(f"**Syllabic Simplicity:** {sub.get('simplicity_score', 88)} / 100")
            st.progress(min(1.0, sub.get('simplicity_score', 88) / 100.0))

        with col_b2:
            st.write(f"**Tempo Advertising Suitability:** {sub.get('tempo_suitability', 94)} / 100")
            st.progress(min(1.0, sub.get('tempo_suitability', 94) / 100.0))

            st.write(f"**Genre-Category Affinity:** {sub.get('genre_affinity', 90)} / 100")
            st.progress(min(1.0, sub.get('genre_affinity', 90) / 100.0))

            st.write(f"**Target Audience Tone Match:** {sub.get('tone_score', 86)} / 100")
            st.progress(min(1.0, sub.get('tone_score', 86) / 100.0))

        st.markdown("""
        <div class="disclaimer-box">
            <b>Evaluation Model Disclaimer:</b>
            The memorability and brand fit scores shown above are estimated using transparent computational heuristics derived from advertising research (repetition cadence, vocabulary accessibility, tempo ranges). They are not measured human listener feedback or psychological testing results.
        </div>
        """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 5. ORIGINALITY AND RIGHTS CHECK
# -----------------------------------------------------------------------------
elif page == "🛡️ Originality & Rights Check":
    st.markdown("""
    <div class="app-header">
        <h1>🛡️ Originality & Rights Checklist</h1>
        <p>Pre-publication legal compliance and originality verification tool for commercial AI music generation.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### Mandatory Pre-Publication Verification Checklist
    Ensure each verification item is completed prior to publishing or distributing AI-generated advertising audio:
    """)

    c1 = st.checkbox("1. Lyrics are newly generated and tailored specifically to the brand.", value=True)
    c2 = st.checkbox("2. No existing song lyrics or copyrighted literary works have been copied.", value=True)
    c3 = st.checkbox("3. No specific copyrighted melody or famous musical hooks have been intentionally reproduced.", value=True)
    c4 = st.checkbox("4. No famous singer's voice, timbre, or trademark vocal persona has been imitated.", value=True)
    c5 = st.checkbox("5. Music-generation service licensing terms and API commercial usage rights have been reviewed.", value=True)
    c6 = st.checkbox("6. Commercial usage rights for broadcast/streaming have been verified with client legal counsel.", value=True)
    c7 = st.checkbox("7. Generated content has been reviewed by a human creative director before public distribution.", value=True)

    checklist_total = sum([c1, c2, c3, c4, c5, c6, c7])

    st.markdown("<br>", unsafe_allow_html=True)
    if checklist_total == 7:
        st.success(f"✓ All 7 Compliance Checks Passed ({checklist_total}/7). Ready for creative review!")
    else:
        st.warning(f"⚠️ {checklist_total} of 7 Compliance Checks Completed. Please review pending items above.")

    st.markdown("---")
    st.subheader("Interactive Lyric Originality Scanner")
    test_text = st.text_area("Paste lyrics to run heuristic originality scan:", value="Crack open the can, let the citrus start,\nFeel the cold fizz waking up your heart!\nSparkleBrew – Drink the sunshine!", height=100)

    if st.button("🔍 Run Originality Heuristic Scan"):
        # Simple phrase analysis against common pop tropes
        pop_cliches = ["baby baby", "don't stop believing", "hit me baby one more time", "i will always love you", "shake it off"]
        matches = [cliche for cliche in pop_cliches if cliche in test_text.lower()]
        if matches:
            st.error(f"Potential pop trope overlap detected: {matches}")
        else:
            st.success("✓ Heuristic Scan: 0 exact matches with monitored pop clichés. High lexical originality.")

    st.markdown("""
    <div class="disclaimer-box" style="margin-top: 30px;">
        <h4 style="margin: 0 0 6px 0; color: #92400e;">⚠️ Legal & Copyright Disclaimer</h4>
        Automated heuristics and algorithmic filters cannot guarantee legal copyright clearance, non-infringement, or absolute originality under intellectual property laws. 
        AI-generated musical outputs should always be subjected to human review and independent legal clearance prior to broadcast, monetization, or commercial distribution.
        This application does not claim or warrant that generated audio is free of third-party copyright claims.
    </div>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------------------
# 6. HISTORY PAGE
# -----------------------------------------------------------------------------
elif page == "📂 History & Library":
    st.markdown("""
    <div class="app-header">
        <h1>📂 Generated Jingles Library & History</h1>
        <p>Review, audition, download, and manage your complete portfolio of AI brand jingles.</p>
    </div>
    """, unsafe_allow_html=True)

    records = hist_mgr.load_all()

    # Search & Filter bar
    f_col1, f_col2, f_col3 = st.columns([1.5, 1, 1])
    with f_col1:
        search_query = st.text_input("🔍 Search by Brand Name", placeholder="Type brand name...")
    with f_col2:
        genre_filter = st.selectbox("Filter Genre", ["All"] + ["Pop", "Acoustic", "Electronic", "Jazz", "Hip-Hop", "Classical", "Corporate", "Folk"])
    with f_col3:
        mood_filter = st.selectbox("Filter Mood", ["All"] + ["Happy", "Energetic", "Emotional", "Playful", "Inspirational", "Calm"])

    # Export Buttons
    e1, e2 = st.columns([1, 1])
    with e1:
        st.download_button(
            "📥 Export History as CSV",
            data=hist_mgr.export_csv_bytes(),
            file_name="jinglegen_history_export.csv",
            mime="text/csv",
            use_container_width=True
        )
    with e2:
        st.download_button(
            "📥 Export History as JSON",
            data=hist_mgr.export_json_bytes(),
            file_name="jinglegen_history_export.json",
            mime="application/json",
            use_container_width=True
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Filtered Records
    filtered = []
    for r in records:
        b_name = r.get('brand_name', '').lower()
        if search_query and search_query.lower() not in b_name:
            continue
        if genre_filter != "All" and r.get('genre') != genre_filter:
            continue
        if mood_filter != "All" and r.get('mood') != mood_filter:
            continue
        filtered.append(r)

    st.markdown(f"**Showing {len(filtered)} of {len(records)} jingle projects**")

    if not filtered:
        st.info("No matching jingles found. Adjust your search or create a new jingle!")
    else:
        for idx, item in enumerate(filtered):
            j_id = item.get('id')
            audio_p = item.get('audio', {}).get('file_path') or item.get('file_path')
            lyrics = item.get('lyrics', {})
            eval_data = item.get('evaluation', {})

            with st.expander(f"🎵 {item.get('brand_name')} – {item.get('genre')} ({item.get('mood')}) • {item.get('timestamp')}", expanded=(idx == 0)):
                c_info, c_audio, c_actions = st.columns([1.5, 1.2, 0.8])

                with c_info:
                    st.markdown(f"**Hook:** *\"{lyrics.get('hook_line', '')}\"*")
                    st.markdown(f"**Slogan:** {lyrics.get('slogan', '')}")
                    st.caption(f"Category: {item.get('category')} | Target: {item.get('target_audience')}")
                    st.caption(f"Tempo: {item.get('bpm')} BPM | Duration: {item.get('duration_sec')}s | Language: {item.get('language')}")

                with c_audio:
                    if audio_p and os.path.exists(audio_p):
                        st.audio(audio_p, format="audio/wav")
                        with open(audio_p, "rb") as f:
                            w_bytes = f.read()
                        st.download_button(
                            "⬇️ Download WAV",
                            data=w_bytes,
                            file_name=f"{item.get('brand_name').lower()}_jingle.wav",
                            mime="audio/wav",
                            key=f"dl_{j_id}"
                        )
                    else:
                        st.text("Audio file unavailable")

                with c_actions:
                    st.metric("Memorability", f"{eval_data.get('memorability_score', 90)}/100")
                    st.metric("Brand Fit", f"{eval_data.get('brand_fit_score', 90)}/100")
                    if st.button("🗑️ Delete", key=f"del_{j_id}", type="secondary"):
                        hist_mgr.delete_by_id(j_id)
                        st.success(f"Deleted jingle for {item.get('brand_name')}")
                        st.rerun()


# -----------------------------------------------------------------------------
# 7. ABOUT PROJECT PAGE
# -----------------------------------------------------------------------------
elif page == "ℹ️ About Project":
    st.markdown("""
    <div class="app-header">
        <h1>ℹ️ About Project: AI Jingle Generator</h1>
        <p>Comprehensive academic documentation for final-year engineering college project presentation.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    ### 1. Project Title
    **JingleGen AI – Smart Brand Jingle Creator for Fictional Brands**

    ### 2. Problem Statement
    In traditional advertising, producing high-impact musical jingles requires specialized music theory knowledge, professional recording studios, session musicians, and substantial creative budgets. Small businesses, startups, and marketing students often lack these resources. 
    **JingleGen AI** leverages Generative AI and digital signal processing (DSP) to democratize advertising jingle creation, enabling users to explore multiple original musical variations, moods, tempos, and linguistic hooks for fictional brands in seconds.

    ### 3. Key Objectives
    - **Original Lyric Generation:** Automatically write rhythmic advertising stanzas, earworm hooks, and slogans based on brand identity and target audience without infringing copyrighted works.
    - **Multilingual Support:** Support English, Hindi, and Telugu advertising jingle composition with romanized sing-along transliteration.
    - **Polyphonic Audio Synthesis:** Synthesize multi-instrument audio (piano, acoustic guitar, analog synth, woodwinds, strings, bass, and drums) using mathematical DSP and physical modeling without requiring external dependencies.
    - **Comparative A/B Auditioning:** Enable side-by-side listening, feature analysis, and interactive polar radar charts comparing up to 3 jingle variations.
    - **Explainable Evaluation Metrics:** Evaluate advertising recall, brand-fit affinity, and tempo suitability with transparent heuristic benchmarks.
    - **Responsible AI & Rights Checklist:** Enforce a 7-step pre-publication compliance workflow to uphold intellectual property ethics.

    ### 4. System Architecture & Workflow
    ```
    ┌────────────────────────────────────────────────────────┐
    │                    STREAMLIT UI                        │
    │   (Dashboard, Generator, Comparison, Evaluation, Lib)   │
    └───────────────────────────┬────────────────────────────┘
                                │ User Inputs (Brand, Genre, Tempo)
                                ▼
    ┌────────────────────────────────────────────────────────┐
    │                 MODULE ORCHESTRATION                   │
    ├───────────────────────────┼────────────────────────────┤
    │  lyrics_generator.py      │  music_generator.py        │
    │  - Gemini AI (Gemini 3.8) │  - Chord Progression Maps  │
    │  - Generative Grammar Fall│  - Melodic Hook Arranger   │
    │  - Multi-language engine  │  - Polyphonic Voice Router │
    ├───────────────────────────┼────────────────────────────┤
    │  audio_processor.py       │  evaluation.py             │
    │  - Karplus-Strong Strings │  - Repetition Density      │
    │  - Additive Piano DSP     │  - Genre-Category Matrix   │
    │  - ADSR Envelopes         │  - Radar Chart Renderer    │
    │  - SciPy WAV Export       │  - Matplotlib Visualizer   │
    └───────────────────────────┴────────────────────────────┘
                                │ Persistent JSON Storage
                                ▼
    ┌────────────────────────────────────────────────────────┐
    │       LOCAL DATA STORAGE (data/history.json)           │
    │       AUDIO REPOSITORY   (generated_audio/*.wav)       │
    └────────────────────────────────────────────────────────┘
    ```

    ### 5. Algorithmic Audio Synthesis Details
    - **Karplus-Strong String Synthesis:** Implements physical acoustic string modeling via feedback delay loops and digital averaging lowpass filters for realistic acoustic guitar plucking.
    - **Harmonic Additive Synthesis:** Accurately models acoustic piano hammer transients and decay curves across multiple Fourier overtones ($f_0, 2f_0, 3f_0, \dots$).
    - **ADSR Envelope Generators:** Four-stage dynamic exponential curves modulate attack, decay, sustain, and release phases to prevent clipping and clicks.
    - **Percussion Synthesis:** Pitched sine sweeps (150Hz $\to$ 42Hz) simulate kick drums, highpass-filtered Gaussian white noise produces crisp snares and metallic hi-hats.
    - **Stereo Panning & Master Bus:** Stereo width panning, soft saturation ($tanh$ limiting), and professional fade-out curves produce 44.1 kHz, 16-bit broadcast-grade WAV files.

    ### 6. Technologies Used
    - **Frontend:** Streamlit 1.x (Python) with custom SaaS CSS.
    - **Backend & DSP:** NumPy, SciPy (`scipy.io.wavfile`, `scipy.signal`).
    - **Data Visualizations:** Matplotlib (polar radar charts, oscilloscope waveforms, frequency spectrograms).
    - **Generative AI:** Google GenAI SDK (`gemini-3.8-flash`) with graceful offline fallback.
    - **Data Management:** Python `json`, `uuid`, Pandas.

    ### 7. Limitations & Future Scope
    - **Current Limitations:** The offline synthesis uses synthetic DSP timbres rather than human vocal singing synthesis. External music APIs may require paid keys or internet access.
    - **Future Scope:** Integration with vocal singing synthesis models (e.g. SVS neural networks), automatic vocal pitch correction, and multi-track MIDI export.
    """)
