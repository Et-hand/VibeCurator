"""
app.py  —  FREQUENCE
────────────────────
Run with:  streamlit run app.py
Requires:  music.db in the same directory, .env with Spotify credentials,
           or env vars SPOTIPY_CLIENT_ID / SPOTIPY_CLIENT_SECRET set.
"""

import random
from datetime import datetime
from urllib.parse import urlparse, parse_qs

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

from music_engine import (
    # db
    get_connection, init_db, load_songs_df, get_available_genres,
    filter_songs_by_genres, build_scaler,
    # spotify
    get_spotify_client, get_auth_url, exchange_code_for_token,
    play_song_on_spotify, pause_spotify, set_volume,
    get_currently_playing, get_active_device,
    # sequencer
    get_next_song, get_recommendations,
    # rankings
    get_leaderboard, record_comparison, pick_comparison_partner,
    get_or_create_rating,
    # logging
    log_listening_event, mark_skipped, update_play_time,
)

load_dotenv()

# ─────────────────────────────────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FREQUENCE",
    page_icon="🎛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────
#  STYLES
# ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=Space+Mono:ital,wght@0,400;0,700;1,400&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500&display=swap');

:root {
    --bg:      #070709;
    --surface: #0f0f12;
    --card:    #141418;
    --card2:   #1a1a1f;
    --border:  #252528;
    --border2: #2e2e33;
    --green:   #00e87a;
    --green2:  #00b85f;
    --amber:   #f5a623;
    --red:     #ff4560;
    --blue:    #3d9eff;
    --muted:   #48484f;
    --sub:     #777780;
    --text:    #e2e2e8;
    --white:   #ffffff;
}

html, body, [class*="css"] {
    background-color: var(--bg) !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
}
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.75rem 2.5rem 5rem !important; max-width: 1440px !important; }
section[data-testid="stSidebar"] { display: none; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: transparent !important;
    border-bottom: 1px solid var(--border) !important;
    gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.18em !important;
    color: var(--muted) !important;
    background: transparent !important;
    border: none !important;
    padding: 0.8rem 1.6rem !important;
    text-transform: uppercase !important;
    transition: color 0.2s !important;
}
.stTabs [aria-selected="true"] {
    color: var(--green) !important;
    border-bottom: 2px solid var(--green) !important;
}
.stTabs [data-baseweb="tab-panel"] {
    background: transparent !important;
    padding: 1.75rem 0 !important;
}

/* ── Buttons ── */
.stButton > button {
    font-family: 'Space Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    background: transparent !important;
    color: var(--green) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 2px !important;
    padding: 0.55rem 1.1rem !important;
    transition: all 0.15s ease !important;
    width: 100% !important;
}
.stButton > button:hover {
    border-color: var(--green) !important;
    color: var(--bg) !important;
    background: var(--green) !important;
}
.stButton > button.primary {
    background: var(--green) !important;
    color: var(--bg) !important;
    border-color: var(--green) !important;
    font-weight: 700 !important;
}

/* ── Inputs ── */
.stSelectbox > div > div,
.stMultiSelect > div > div {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
    color: var(--text) !important;
}
.stTextInput > div > div > input {
    background: var(--card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 2px !important;
    color: var(--text) !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
}
.stSlider > div > div > div { background: var(--border) !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 3px; height: 3px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }

/* ── Waveform ── */
@keyframes wv {
    0%,100% { transform: scaleY(0.25); }
    50%      { transform: scaleY(1.0); }
}
.waveform { display:flex; align-items:center; gap:2px; height:20px; }
.wv-bar {
    width:2.5px; border-radius:2px;
    background: var(--green);
    animation: wv 0.9s ease-in-out infinite;
    transform-origin: center;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  CACHED RESOURCES
# ─────────────────────────────────────────────────────────────────
@st.cache_resource
def _get_conn():
    c = get_connection()
    init_db(c)
    return c

@st.cache_resource
def _get_scaler(_conn):
    songs = load_songs_df(_conn)
    return build_scaler(songs)          # (scaler, all_songs_scaled, feature_cols)

conn                              = _get_conn()
_scaler, all_songs_scaled, FCOLS  = _get_scaler(conn)
AVAILABLE_GENRES                  = get_available_genres(conn)


# ─────────────────────────────────────────────────────────────────
#  SESSION STATE
# ─────────────────────────────────────────────────────────────────
_defaults = {
    "sp":                    None,       # spotipy client
    "sp_token":              None,       # raw token dict
    "current_song_id":       None,
    "current_cover_url":     None,
    "recently_played":       [],
    "recently_played_artists": [],
    "session_id":            f"s_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
    "compare_a":             None,
    "compare_b":             None,
    "energy_dir":            "maintain",
    "active_genres":         [],
    "spotify_status":        "",
    "volume":                80,
    "song_start_time":       None,
}
for k, v in _defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Restore cached Spotify client on rerun
if st.session_state.sp is None:
    st.session_state.sp = get_spotify_client()


# ─────────────────────────────────────────────────────────────────
#  SPOTIFY OAuth CALLBACK HANDLER
# ─────────────────────────────────────────────────────────────────
def _handle_oauth_callback():
    """Check query params for Spotify OAuth code and exchange it."""
    try:
        params = st.query_params
        code   = params.get("code", None)
        if code and st.session_state.sp is None:
            token = exchange_code_for_token(code)
            if token:
                import spotipy
                st.session_state.sp       = spotipy.Spotify(auth=token["access_token"])
                st.session_state.sp_token = token
                st.session_state.spotify_status = "Connected to Spotify ✓"
                st.query_params.clear()
                st.rerun()
    except Exception:
        pass

_handle_oauth_callback()


# ─────────────────────────────────────────────────────────────────
#  HELPERS
# ─────────────────────────────────────────────────────────────────

def _active_genres() -> list[str]:
    return st.session_state.active_genres or AVAILABLE_GENRES


def _play(song_row: pd.Series):
    """Set current song, optionally trigger Spotify playback."""
    prev_id = st.session_state.current_song_id

    # Log play time for previous song
    if prev_id and st.session_state.song_start_time:
        elapsed = (datetime.now() - st.session_state.song_start_time).total_seconds()
        update_play_time(conn, prev_id, elapsed)

    st.session_state.current_song_id   = song_row["id"]
    st.session_state.song_start_time   = datetime.now()
    st.session_state.current_cover_url = song_row.get("album_cover_url", None)

    log_listening_event(conn, song_row["id"], st.session_state.session_id)

    rp  = st.session_state.recently_played
    rpa = st.session_state.recently_played_artists
    rp.append(song_row["id"])
    rpa.append(song_row["artist"])
    if len(rp)  > 50: rp.pop(0)
    if len(rpa) > 5:  rpa.pop(0)

    if st.session_state.sp:
        uri, cover, status = play_song_on_spotify(
            st.session_state.sp, song_row["name"], song_row["artist"]
        )
        if cover:
            st.session_state.current_cover_url = cover
        if status == "NO_DEVICE":
            st.session_state.spotify_status = "⚠ No active Spotify device. Open Spotify on any device first."
        elif status == "NOT_FOUND":
            st.session_state.spotify_status = f"⚠ '{song_row['name']}' not found on Spotify."
        elif status == "OK":
            st.session_state.spotify_status = f"▶ Playing on Spotify"
        else:
            st.session_state.spotify_status = f"⚠ {status}"
    else:
        st.session_state.spotify_status = "Spotify not connected — connect above to enable playback."


def _next_song():
    if not st.session_state.current_song_id:
        # Start with a random song from active genres
        genres = _active_genres()
        if genres:
            ph = ",".join(["?"] * len(genres))
            row = pd.read_sql(
                f"SELECT * FROM songs WHERE track_genre IN ({ph}) ORDER BY RANDOM() LIMIT 1",
                conn, params=genres
            )
        else:
            row = pd.read_sql("SELECT * FROM songs ORDER BY RANDOM() LIMIT 1", conn)
        if not row.empty:
            _play(row.iloc[0])
        return

    nxt = get_next_song(
        conn, all_songs_scaled, FCOLS,
        st.session_state.current_song_id,
        _active_genres(),
        st.session_state.recently_played,
        st.session_state.recently_played_artists,
        st.session_state.energy_dir,
    )
    if nxt is not None:
        _play(nxt)


def _skip():
    if st.session_state.current_song_id:
        mark_skipped(conn, st.session_state.current_song_id)
    _next_song()


def _open_rank_dialog():
    if not st.session_state.current_song_id:
        return
    genre = _current_genre()
    partner = pick_comparison_partner(conn, st.session_state.current_song_id, genre)
    if partner:
        st.session_state.compare_a = st.session_state.current_song_id
        st.session_state.compare_b = partner


def _current_genre() -> str:
    """Return a single representative genre for ranking the current song."""
    if not st.session_state.current_song_id:
        return "all"
    row = pd.read_sql(
        "SELECT track_genre FROM songs WHERE id=?",
        conn, params=[st.session_state.current_song_id]
    )
    if row.empty or pd.isna(row.iloc[0]["track_genre"]):
        return "all"
    return row.iloc[0]["track_genre"]


def _song_info(song_id: str) -> pd.Series | None:
    row = pd.read_sql("SELECT * FROM songs WHERE id=?", conn, params=[song_id])
    return row.iloc[0] if not row.empty else None


# ─────────────────────────────────────────────────────────────────
#  REUSABLE UI COMPONENTS
# ─────────────────────────────────────────────────────────────────

def _stat_pill(label: str, value: str) -> str:
    return f"""<div style="font-family:'Space Mono',monospace;font-size:0.6rem;
        letter-spacing:0.08em;color:var(--sub);background:var(--surface);
        border:1px solid var(--border);border-radius:2px;padding:0.28rem 0.55rem;
        display:inline-block;text-transform:uppercase;">
        {label}&nbsp;<span style="color:var(--green);font-weight:700;">{value}</span>
    </div>"""


def _bar(pct: float, color: str = "var(--green)", height: int = 3) -> str:
    return f"""
    <div style="background:var(--border);border-radius:2px;height:{height}px;width:100%;margin:0.2rem 0;">
        <div style="width:{int(pct*100)}%;height:{height}px;border-radius:2px;
                    background:{color};transition:width 0.5s ease;"></div>
    </div>"""


def _section(label: str) -> str:
    return f"""<div style="font-family:'Space Mono',monospace;font-size:0.6rem;
        letter-spacing:0.2em;color:var(--muted);text-transform:uppercase;
        margin-bottom:1rem;padding-bottom:0.4rem;border-bottom:1px solid var(--border);">
        {label}</div>"""


def _card(content: str, accent: bool = False) -> str:
    border = "border-left:3px solid var(--green);" if accent else ""
    return f"""<div style="background:var(--card);border:1px solid var(--border);
        border-radius:3px;padding:1.5rem;{border}">{content}</div>"""


def _album_cover(url: str | None, size: int = 180) -> str:
    if url:
        return f"""<img src="{url}" width="{size}" height="{size}"
            style="border-radius:3px;object-fit:cover;
                   box-shadow:0 8px 32px rgba(0,0,0,0.6);display:block;"/>"""
    # Placeholder grid
    return f"""<div style="width:{size}px;height:{size}px;background:var(--card2);
        border:1px solid var(--border);border-radius:3px;display:flex;
        align-items:center;justify-content:center;">
        <span style="font-size:2rem;opacity:0.2;">♪</span></div>"""


def _mini_cover(url: str | None, size: int = 40) -> str:
    if url:
        return f"""<img src="{url}" width="{size}" height="{size}"
            style="border-radius:2px;object-fit:cover;flex-shrink:0;"/>"""
    return f"""<div style="width:{size}px;height:{size}px;background:var(--card2);
        border:1px solid var(--border);border-radius:2px;flex-shrink:0;display:flex;
        align-items:center;justify-content:center;font-size:0.7rem;opacity:0.3;">♪</div>"""


def _waveform(n: int = 16) -> str:
    bars = "".join(
        f'<div class="wv-bar" style="height:{random.randint(6,18)}px;'
        f'animation-delay:{i*0.055:.2f}s;"></div>'
        for i in range(n)
    )
    return f'<div class="waveform">{bars}</div>'


# ─────────────────────────────────────────────────────────────────
#  HEADER
# ─────────────────────────────────────────────────────────────────
h_left, h_right = st.columns([4, 2])

with h_left:
    st.markdown("""
    <div style="display:flex;align-items:baseline;gap:1rem;margin-bottom:0.15rem;">
        <h1 style="font-family:'Bebas Neue',sans-serif;font-size:2.6rem;
                   letter-spacing:0.12em;color:var(--white);margin:0;">FREQUENCE</h1>
        <span style="font-family:'Space Mono',monospace;font-size:0.55rem;
                     color:var(--muted);letter-spacing:0.2em;text-transform:uppercase;">
        </span>
    </div>
    """, unsafe_allow_html=True)

with h_right:
    sp_connected = st.session_state.sp is not None
    n_songs  = pd.read_sql("SELECT COUNT(*) AS n FROM songs", conn).iloc[0]["n"]
    n_ranked = pd.read_sql("SELECT COUNT(*) AS n FROM rankings", conn).iloc[0]["n"]

    if not sp_connected:
        auth_url = get_auth_url()
        st.markdown(f"""
        <div style="text-align:right;margin-top:0.4rem;">
            <a href="{auth_url}" target="_self" style="
                font-family:'Space Mono',monospace;font-size:0.65rem;
                letter-spacing:0.12em;text-transform:uppercase;
                background:var(--green);color:var(--bg);
                border:none;border-radius:2px;padding:0.5rem 1rem;
                text-decoration:none;font-weight:700;display:inline-block;">
                ⟳ CONNECT SPOTIFY
            </a>
            <div style="font-family:'Space Mono',monospace;font-size:0.55rem;
                        color:var(--muted);margin-top:0.4rem;letter-spacing:0.1em;">
                {n_songs:,} TRACKS · {n_ranked} RANKED
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        try:
            user = st.session_state.sp.me()
            uname = user.get("display_name", "Connected")
        except Exception:
            uname = "Connected"
        st.markdown(f"""
        <div style="text-align:right;margin-top:0.4rem;">
            <div style="font-family:'Space Mono',monospace;font-size:0.65rem;
                        color:var(--green);letter-spacing:0.1em;">
                ● {uname.upper()}
            </div>
            <div style="font-family:'Space Mono',monospace;font-size:0.55rem;
                        color:var(--muted);margin-top:0.3rem;letter-spacing:0.1em;">
                {n_songs:,} TRACKS · {n_ranked} RANKED
            </div>
        </div>
        """, unsafe_allow_html=True)

st.markdown("<div style='height:0.25rem'></div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────
#  GENRE SELECTOR  (always visible)
# ─────────────────────────────────────────────────────────────────
with st.expander("🎛  GENRE FILTER", expanded=True):
    g_col1, g_col2 = st.columns([5, 1])
    with g_col1:
        selected_genres = st.multiselect(
            "Select genres",
            options=AVAILABLE_GENRES,
            default=st.session_state.active_genres or [],
            placeholder="All genres (no filter)",
            label_visibility="collapsed",
        )
        st.session_state.active_genres = selected_genres
    with g_col2:
        if st.button("CLEAR", key="clear_genres"):
            st.session_state.active_genres = []
            st.rerun()

    if selected_genres:
        count_q = ",".join(["?"] * len(selected_genres))
        filtered_count = pd.read_sql(
            f"SELECT COUNT(*) AS n FROM songs WHERE track_genre IN ({count_q})",
            conn, params=selected_genres
        ).iloc[0]["n"]
        st.markdown(
            f'<div style="font-family:\'Space Mono\',monospace;font-size:0.6rem;'
            f'color:var(--sub);margin-top:0.4rem;">'
            f'{filtered_count:,} songs in selected genres</div>',
            unsafe_allow_html=True
        )

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────
#  MAIN TABS
# ─────────────────────────────────────────────────────────────────
tab_play, tab_rank, tab_discover, tab_library = st.tabs([
    "⬡  NOW PLAYING", "◈  RANKINGS", "◎  DISCOVER", "⊞  LIBRARY"
])


# ══════════════════════════════════════════════════════════════════
#  TAB 1  —  NOW PLAYING
# ══════════════════════════════════════════════════════════════════
with tab_play:

    p_left, p_right = st.columns([5, 3], gap="large")

    # ── LEFT: player ──
    with p_left:

        if st.session_state.current_song_id:
            song = _song_info(st.session_state.current_song_id)
            if song is not None:
                rating    = get_or_create_rating(conn, song["id"], _current_genre())
                lb        = get_leaderboard(conn, _current_genre())
                score_txt = "—"
                if not lb.empty:
                    row = lb[lb["song_id"] == song["id"]]
                    if not row.empty:
                        score_txt = f"{row.iloc[0]['score_0_10']:.1f}"

                cover_url = st.session_state.current_cover_url or song.get("album_cover_url")
                energy    = float(song.get("energy", 0.5))

                # ── Album cover + info row ──
                cov_col, info_col = st.columns([1, 2], gap="medium")
                with cov_col:
                    st.markdown(_album_cover(cover_url, size=170), unsafe_allow_html=True)
                with info_col:
                    st.markdown(f"""
                    <div style="padding-top:0.25rem;">
                        <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:0.6rem;">
                            {_waveform(14)}
                            <span style="font-family:'Space Mono',monospace;font-size:0.55rem;
                                         color:var(--muted);letter-spacing:0.18em;
                                         text-transform:uppercase;">PLAYING</span>
                        </div>
                        <div style="font-family:'Bebas Neue',sans-serif;font-size:2.4rem;
                                    line-height:1.0;letter-spacing:0.04em;color:var(--white);
                                    margin-bottom:0.3rem;">{song['name']}</div>
                        <div style="font-family:'Space Mono',monospace;font-size:0.72rem;
                                    color:var(--green);letter-spacing:0.1em;
                                    text-transform:uppercase;margin-bottom:1rem;">
                            {song['artist']}
                        </div>
                        <div style="display:flex;align-items:center;gap:0.5rem;
                                    margin-bottom:0.4rem;">
                            <span style="font-family:'Space Mono',monospace;font-size:0.55rem;
                                         color:var(--sub);letter-spacing:0.1em;">ENERGY</span>
                            <span style="font-family:'Space Mono',monospace;font-size:0.55rem;
                                         color:var(--green);">{int(energy*100)}%</span>
                        </div>
                        {_bar(energy, height=4)}
                        <div style="display:flex;gap:0.5rem;flex-wrap:wrap;margin-top:0.75rem;">
                            {_stat_pill('BPM', f"{song['bpm']:.0f}")}
                            {_stat_pill('KEY', str(song.get('camelot_key','—')))}
                            {_stat_pill('SCORE', score_txt)}
                            {_stat_pill('GENRE', str(song.get('track_genre','—')))}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.markdown(_card("""
            <div style="text-align:center;padding:2rem 0;">
                <div style="font-family:'Bebas Neue',sans-serif;font-size:3.5rem;
                            color:var(--border2);letter-spacing:0.1em;">NO SIGNAL</div>
                <div style="font-family:'Space Mono',monospace;font-size:0.65rem;
                            color:var(--muted);margin-top:0.5rem;letter-spacing:0.15em;">
                    PRESS PLAY TO START A SESSION
                </div>
            </div>
            """), unsafe_allow_html=True)

        # ── Status bar ──
        if st.session_state.spotify_status:
            color = "var(--green)" if "▶" in st.session_state.spotify_status else "var(--amber)"
            if "⚠" in st.session_state.spotify_status:
                color = "var(--amber)"
            st.markdown(
                f'<div style="font-family:\'Space Mono\',monospace;font-size:0.6rem;'
                f'color:{color};letter-spacing:0.08em;margin:0.6rem 0;">'
                f'{st.session_state.spotify_status}</div>',
                unsafe_allow_html=True
            )

        st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

        # ── Transport controls ──
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            if st.button("▶  PLAY / NEXT", key="btn_next", use_container_width=True):
                _next_song(); st.rerun()
        with c2:
            if st.button("⏸  PAUSE", key="btn_pause", use_container_width=True):
                if st.session_state.sp:
                    pause_spotify(st.session_state.sp)
                    st.session_state.spotify_status = "⏸ Paused"
                st.rerun()
        with c3:
            if st.button("⊕  RANK THIS", key="btn_rank", use_container_width=True):
                _open_rank_dialog(); st.rerun()
        with c4:
            if st.button("✕  SKIP", key="btn_skip", use_container_width=True):
                _skip(); st.rerun()

        # ── Volume + energy arc ──
        st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)
        v_col, e_col = st.columns(2)
        with v_col:
            st.markdown(_section("VOLUME"), unsafe_allow_html=True)
            vol = st.slider("vol", 0, 100, st.session_state.volume,
                            label_visibility="collapsed", key="vol_slider")
            if vol != st.session_state.volume:
                st.session_state.volume = vol
                if st.session_state.sp:
                    set_volume(st.session_state.sp, vol)
        with e_col:
            st.markdown(_section("SESSION ARC"), unsafe_allow_html=True)
            direction = st.select_slider(
                "arc", options=["cool", "maintain", "build"],
                value=st.session_state.energy_dir,
                label_visibility="collapsed", key="arc_slider"
            )
            st.session_state.energy_dir = direction

        # ── Inline ranking comparison ──
        if st.session_state.compare_a and st.session_state.compare_b:
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            st.markdown(_section("RANK THIS SONG"), unsafe_allow_html=True)

            sa = _song_info(st.session_state.compare_a)
            sb = _song_info(st.session_state.compare_b)

            if sa is not None and sb is not None:
                ca_col, vs_col, cb_col = st.columns([5, 1, 5])

                def _compare_card(s: pd.Series, cover: str | None) -> str:
                    return f"""
                    <div style="background:var(--card2);border:1px solid var(--border2);
                                border-radius:3px;padding:1.25rem;text-align:center;">
                        <div style="display:flex;justify-content:center;margin-bottom:0.75rem;">
                            {_album_cover(cover, size=80)}
                        </div>
                        <div style="font-family:'Bebas Neue',sans-serif;font-size:1.4rem;
                                    letter-spacing:0.04em;line-height:1.1;">{s['name']}</div>
                        <div style="font-family:'Space Mono',monospace;font-size:0.6rem;
                                    color:var(--green);letter-spacing:0.1em;margin-top:0.3rem;
                                    text-transform:uppercase;">{s['artist']}</div>
                        <div style="font-family:'Space Mono',monospace;font-size:0.55rem;
                                    color:var(--muted);margin-top:0.5rem;">
                            {s['bpm']:.0f} BPM · {s.get('camelot_key','—')}
                        </div>
                    </div>"""

                with ca_col:
                    st.markdown(_compare_card(sa, sa.get("album_cover_url")), unsafe_allow_html=True)
                    if st.button("A IS BETTER", key="vote_a", use_container_width=True):
                        record_comparison(conn, sa["id"], sb["id"], _current_genre())
                        st.session_state.compare_a = None
                        st.session_state.compare_b = None
                        st.rerun()

                with vs_col:
                    st.markdown("""
                    <div style="display:flex;align-items:center;justify-content:center;
                                height:100%;font-family:'Bebas Neue',sans-serif;
                                font-size:1.4rem;color:var(--muted);">VS</div>
                    """, unsafe_allow_html=True)

                with cb_col:
                    st.markdown(_compare_card(sb, sb.get("album_cover_url")), unsafe_allow_html=True)
                    if st.button("B IS BETTER", key="vote_b", use_container_width=True):
                        record_comparison(conn, sb["id"], sa["id"], _current_genre())
                        st.session_state.compare_a = None
                        st.session_state.compare_b = None
                        st.rerun()

                if st.button("SAME / TIE", key="vote_tie", use_container_width=True):
                    record_comparison(conn, sa["id"], sb["id"], _current_genre(), draw=True)
                    st.session_state.compare_a = None
                    st.session_state.compare_b = None
                    st.rerun()

    # ── RIGHT: Up Next queue ──
    with p_right:
        st.markdown(_section("UP NEXT"), unsafe_allow_html=True)

        if st.session_state.current_song_id:
            try:
                preview, temp_id = [], st.session_state.current_song_id
                temp_rp  = list(st.session_state.recently_played)
                temp_rpa = list(st.session_state.recently_played_artists)

                for _ in range(5):
                    nxt = get_next_song(
                        conn, all_songs_scaled, FCOLS,
                        temp_id, _active_genres(),
                        temp_rp, temp_rpa, st.session_state.energy_dir
                    )
                    if nxt is None: break
                    preview.append(nxt)
                    temp_rp.append(nxt["id"])
                    temp_rpa.append(nxt["artist"])
                    if len(temp_rp)  > 50: temp_rp.pop(0)
                    if len(temp_rpa) > 5:  temp_rpa.pop(0)
                    temp_id = nxt["id"]

                for i, s in enumerate(preview):
                    opacity = 1.0 - i * 0.15
                    cover   = s.get("album_cover_url", None)
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:0.75rem;
                                padding:0.6rem 0;border-bottom:1px solid #1c1c20;
                                opacity:{opacity:.2f};">
                        {_mini_cover(cover, 38)}
                        <div style="flex:1;min-width:0;">
                            <div style="font-size:0.8rem;color:var(--text);
                                        white-space:nowrap;overflow:hidden;
                                        text-overflow:ellipsis;">{s['name']}</div>
                            <div style="font-family:'Space Mono',monospace;font-size:0.58rem;
                                        color:var(--sub);margin-top:0.1rem;">{s['artist']}</div>
                        </div>
                        <div style="font-family:'Space Mono',monospace;font-size:0.55rem;
                                    color:var(--muted);text-align:right;flex-shrink:0;">
                            {s['bpm']:.0f}<br>{s.get('camelot_key','—')}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            except Exception:
                st.markdown(
                    '<div style="color:var(--muted);font-size:0.8rem;">Queue unavailable</div>',
                    unsafe_allow_html=True
                )
        else:
            st.markdown(
                '<div style="color:var(--muted);font-family:\'Space Mono\',monospace;'
                'font-size:0.7rem;">Press Play to start a session</div>',
                unsafe_allow_html=True
            )


# ══════════════════════════════════════════════════════════════════
#  TAB 2  —  RANKINGS
# ══════════════════════════════════════════════════════════════════
with tab_rank:

    # Genre selector for leaderboard view
    rank_genres = ["__all__"] + AVAILABLE_GENRES
    rank_labels = ["All Genres (Cross-Genre)"] + AVAILABLE_GENRES
    rb_col, _ = st.columns([2, 3])
    with rb_col:
        rank_view_idx = st.selectbox(
            "View rankings for",
            options=rank_genres,
            format_func=lambda x: "All Genres (Cross-Genre)" if x == "__all__" else x.upper(),
            label_visibility="collapsed",
            key="rank_genre_select"
        )

    lb = get_leaderboard(conn, rank_view_idx if rank_view_idx != "__all__" else None)

    if lb.empty:
        st.markdown(_card("""
        <div style="text-align:center;padding:2.5rem 0;">
            <div style="font-family:'Bebas Neue',sans-serif;font-size:3rem;
                        color:var(--border2);letter-spacing:0.1em;">NO RANKINGS YET</div>
            <div style="font-family:'Space Mono',monospace;font-size:0.65rem;
                        color:var(--muted);margin-top:0.5rem;">
                PLAY SONGS AND USE ⊕ RANK THIS TO BUILD YOUR LEADERBOARD
            </div>
        </div>
        """), unsafe_allow_html=True)
    else:
        lb_left, lb_right = st.columns([3, 1], gap="large")

        with lb_left:
            st.markdown(_section(
                "CROSS-GENRE LEADERBOARD (PLAY-TIME WEIGHTED)"
                if rank_view_idx == "__all__"
                else f"{rank_view_idx.upper()} LEADERBOARD"
            ), unsafe_allow_html=True)

            for i, row in lb.iterrows():
                rank      = lb.index.get_loc(i) + 1
                bar_pct   = row["score_0_10"] / 10
                medal     = {1:"🥇",2:"🥈",3:"🥉"}.get(rank, f"{rank:02d}")
                cover_url = row.get("album_cover_url", None)

                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:0.9rem;
                            padding:0.7rem 0.5rem;border-bottom:1px solid var(--border);
                            border-radius:2px;">
                    <div style="font-family:'Space Mono',monospace;font-size:0.65rem;
                                color:{'var(--amber)' if rank<=3 else 'var(--muted)'};
                                width:28px;text-align:center;flex-shrink:0;">{medal}</div>
                    {_mini_cover(cover_url, 42)}
                    <div style="flex:1;min-width:0;">
                        <div style="font-size:0.85rem;color:var(--text);
                                    white-space:nowrap;overflow:hidden;text-overflow:ellipsis;">
                            {row['name']}</div>
                        <div style="font-family:'Space Mono',monospace;font-size:0.6rem;
                                    color:var(--sub);margin-top:0.1rem;">{row['artist']}</div>
                    </div>
                    <div style="font-family:'Space Mono',monospace;font-size:0.58rem;
                                color:var(--muted);flex-shrink:0;text-align:right;
                                min-width:60px;">
                        {row.get('track_genre','—')}<br>
                        <span style="color:var(--sub)">{row['comparison_count']} comp.</span>
                    </div>
                    <div style="width:70px;flex-shrink:0;">
                        {_bar(bar_pct)}
                    </div>
                    <div style="font-family:'Space Mono',monospace;font-size:0.8rem;
                                color:var(--green);flex-shrink:0;width:32px;
                                text-align:right;font-weight:700;">
                        {row['score_0_10']:.1f}
                    </div>
                </div>
                """, unsafe_allow_html=True)

        with lb_right:
            st.markdown(_section("STATS"), unsafe_allow_html=True)

            n_comps = pd.read_sql("SELECT COUNT(*) AS n FROM comparisons", conn).iloc[0]["n"]
            avg_sig = lb["sigma"].mean() if not lb.empty else 0
            w       = pd.read_sql(
                "SELECT COUNT(*) AS n FROM rankings", conn
            ).iloc[0]["n"]

            def _big_stat(val: str, label: str, color: str = "var(--green)") -> str:
                return f"""
                <div style="margin-bottom:1.25rem;">
                    <div style="font-family:'Bebas Neue',sans-serif;font-size:2.8rem;
                                color:{color};line-height:1;">{val}</div>
                    <div style="font-family:'Space Mono',monospace;font-size:0.55rem;
                                color:var(--muted);letter-spacing:0.12em;">{label}</div>
                </div>"""

            weights = {
                "content": 0.80 if w < 50 else (0.25 if w >= 200 else 0.80 - 0.55*(w-50)/150),
                "collab":  0.15 if w < 50 else (0.50 if w >= 200 else 0.15 + 0.20*(w-50)/150),
                "bias":    0.05 if w < 50 else (0.25 if w >= 200 else 0.05 + 0.20*(w-50)/150),
            }

            st.markdown(f"""
            {_card(
                _big_stat(str(len(lb)), "RANKED SONGS") +
                _big_stat(str(n_comps), "COMPARISONS") +
                _big_stat(f"{avg_sig:.1f}", "AVG UNCERTAINTY", "var(--amber)")
            )}
            """, unsafe_allow_html=True)

            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            st.markdown(_section("MODEL WEIGHTS"), unsafe_allow_html=True)

            for label, val in weights.items():
                st.markdown(f"""
                <div style="margin-bottom:0.6rem;">
                    <div style="display:flex;justify-content:space-between;
                                font-family:'Space Mono',monospace;font-size:0.58rem;
                                color:var(--sub);margin-bottom:0.2rem;">
                        <span>{label.upper()}</span>
                        <span style="color:var(--green)">{val*100:.0f}%</span>
                    </div>
                    {_bar(val)}
                </div>
                """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  TAB 3  —  DISCOVER
# ══════════════════════════════════════════════════════════════════
with tab_discover:

    d_left, d_right = st.columns([3, 1], gap="large")

    with d_left:
        st.markdown(_section("RECOMMENDED FOR YOU"), unsafe_allow_html=True)

        rc1, rc2 = st.columns([1, 5])
        with rc1:
            if st.button("↻  REFRESH", key="refresh_recs"):
                st.rerun()

        recs = get_recommendations(
            conn, all_songs_scaled, FCOLS,
            _active_genres(), n=15
        )

        if recs.empty:
            st.markdown(_card("""
            <div style="text-align:center;padding:1.5rem 0;">
                <div style="font-family:'Space Mono',monospace;font-size:0.65rem;color:var(--muted);">
                    RANK MORE SONGS TO UNLOCK PERSONALISED RECOMMENDATIONS
                </div>
            </div>
            """), unsafe_allow_html=True)
        else:
            for i, row in recs.iterrows():
                idx    = recs.index.get_loc(i) + 1
                cover  = row.get("album_cover_url", None)

                play_key = f"play_rec_{idx}"
                r1, r2 = st.columns([10, 1])
                with r1:
                    st.markdown(f"""
                    <div style="display:flex;align-items:center;gap:0.85rem;
                                padding:0.65rem 0;border-bottom:1px solid var(--border);">
                        <div style="font-family:'Space Mono',monospace;font-size:0.58rem;
                                    color:var(--muted);width:22px;flex-shrink:0;
                                    text-align:right;">{idx:02d}</div>
                        {_mini_cover(cover, 42)}
                        <div style="flex:1;min-width:0;">
                            <div style="font-size:0.85rem;color:var(--text);
                                        white-space:nowrap;overflow:hidden;
                                        text-overflow:ellipsis;">{row['name']}</div>
                            <div style="font-family:'Space Mono',monospace;font-size:0.58rem;
                                        color:var(--sub);margin-top:0.1rem;">{row['artist']}</div>
                        </div>
                        <div style="font-family:'Space Mono',monospace;font-size:0.58rem;
                                    color:var(--muted);flex-shrink:0;text-align:right;">
                            {row['bpm']:.0f} BPM<br>
                            <span style="color:var(--sub)">{row.get('track_genre','—')}</span>
                        </div>
                        <div style="font-family:'Space Mono',monospace;font-size:0.72rem;
                                    color:var(--green);flex-shrink:0;width:42px;
                                    text-align:right;font-weight:700;">
                            {row['hybrid']:.3f}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                with r2:
                    if st.button("▶", key=play_key):
                        _play(row)
                        st.rerun()

    with d_right:
        st.markdown(_section("HOW IT WORKS"), unsafe_allow_html=True)
        w = pd.read_sql("SELECT COUNT(*) AS n FROM rankings", conn).iloc[0]["n"]
        next_milestone = 50 if w < 50 else (200 if w < 200 else None)
        progress_note = (
            f"Rank {next_milestone - w} more songs to shift toward collaborative filtering."
            if next_milestone else "Full collaborative mode active."
        )
        _how_it_works_card = _card(
            f'<div style="font-family:\'Space Mono\',monospace;font-size:0.62rem;'
            f'color:var(--sub);line-height:1.9;">'
            f'Scores combine content similarity, ranking history, and '
            f'collaborative signals.<br><br>'
            f'{progress_note}<br><br>'
            f'Only songs matching your selected genres are recommended.'
            f'</div>'
        )
        st.markdown(_how_it_works_card, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════
#  TAB 4  —  LIBRARY
# ══════════════════════════════════════════════════════════════════
with tab_library:

    lib_left, lib_right = st.columns([3, 1], gap="large")

    with lib_left:
        st.markdown(_section("SEARCH LIBRARY"), unsafe_allow_html=True)
        query = st.text_input(
            "search", placeholder="ARTIST OR TRACK NAME...",
            label_visibility="collapsed", key="lib_search"
        )

        if query:
            results = pd.read_sql("""
                SELECT id, name, artist, bpm, energy, camelot_key,
                       track_genre, album_cover_url
                FROM songs
                WHERE name LIKE ? OR artist LIKE ?
                LIMIT 50
            """, conn, params=[f"%{query}%", f"%{query}%"])

            if results.empty:
                st.markdown(
                    '<div style="font-family:\'Space Mono\',monospace;font-size:0.7rem;'
                    'color:var(--muted);">NO RESULTS</div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div style="font-family:\'Space Mono\',monospace;font-size:0.58rem;'
                    f'color:var(--muted);margin-bottom:0.75rem;">'
                    f'{len(results)} RESULTS</div>',
                    unsafe_allow_html=True
                )
                for _, row in results.iterrows():
                    cover = row.get("album_cover_url", None)
                    r1, r2 = st.columns([10, 1])
                    with r1:
                        st.markdown(f"""
                        <div style="display:flex;align-items:center;gap:0.8rem;
                                    padding:0.6rem 0;border-bottom:1px solid var(--border);">
                            {_mini_cover(cover, 40)}
                            <div style="flex:1;min-width:0;">
                                <div style="font-size:0.82rem;color:var(--text);
                                            white-space:nowrap;overflow:hidden;
                                            text-overflow:ellipsis;">{row['name']}</div>
                                <div style="font-family:'Space Mono',monospace;font-size:0.58rem;
                                            color:var(--sub);margin-top:0.1rem;">{row['artist']}</div>
                            </div>
                            <div style="display:flex;gap:0.4rem;flex-shrink:0;flex-wrap:wrap;
                                        justify-content:flex-end;">
                                {_stat_pill('BPM',f"{row['bpm']:.0f}")}
                                {_stat_pill('KEY',str(row.get('camelot_key','—')))}
                                {_stat_pill('GENRE',str(row.get('track_genre','—')))}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
                    with r2:
                        if st.button("▶", key=f"lib_play_{row['id']}"):
                            _play(row)
                            st.rerun()
        else:
            st.markdown(_card("""
            <div style="text-align:center;padding:3rem 0;">
                <div style="font-family:'Bebas Neue',sans-serif;font-size:2.5rem;
                            color:var(--border2);letter-spacing:0.1em;">SEARCH THE LIBRARY</div>
                <div style="font-family:'Space Mono',monospace;font-size:0.62rem;
                            color:var(--muted);margin-top:0.5rem;">
                    TYPE ABOVE TO FIND ANY TRACK OR ARTIST
                </div>
            </div>
            """), unsafe_allow_html=True)

    with lib_right:
        st.markdown(_section("LIBRARY STATS"), unsafe_allow_html=True)

        total  = pd.read_sql("SELECT COUNT(*) AS n FROM songs", conn).iloc[0]["n"]
        played = pd.read_sql(
            "SELECT COUNT(DISTINCT song_id) AS n FROM listening_events", conn
        ).iloc[0]["n"]

        _stats_card = _card(
            f'<div style="margin-bottom:1.25rem;">'
            f'<div style="font-family:\'Bebas Neue\',sans-serif;font-size:2.6rem;'
            f'color:var(--green);line-height:1;">{total:,}</div>'
            f'<div style="font-family:\'Space Mono\',monospace;font-size:0.55rem;'
            f'color:var(--muted);letter-spacing:0.12em;">TOTAL TRACKS</div></div>'
            f'<div><div style="font-family:\'Bebas Neue\',sans-serif;font-size:2.6rem;'
            f'color:var(--amber);line-height:1;">{played}</div>'
            f'<div style="font-family:\'Space Mono\',monospace;font-size:0.55rem;'
            f'color:var(--muted);letter-spacing:0.12em;">PLAYED THIS SESSION</div></div>'
        )
        st.markdown(_stats_card, unsafe_allow_html=True)

        # Genre breakdown
        all_songs_df = load_songs_df(conn)
        if "track_genre" in all_songs_df.columns:
            st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)
            st.markdown(_section("GENRE BREAKDOWN"), unsafe_allow_html=True)

            gc = all_songs_df["track_genre"].value_counts().head(10)
            max_n = gc.max()

            genre_rows = ""
            for genre_name, count in gc.items():
                pct = count / max_n
                genre_rows += f"""
                <div style="margin-bottom:0.6rem;">
                    <div style="display:flex;justify-content:space-between;
                                font-family:'Space Mono',monospace;font-size:0.58rem;
                                color:var(--sub);margin-bottom:0.2rem;">
                        <span>{str(genre_name).upper()}</span>
                        <span style="color:var(--green)">{count:,}</span>
                    </div>
                    {_bar(pct)}
                </div>"""

            st.markdown(_card(genre_rows), unsafe_allow_html=True)
