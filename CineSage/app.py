import os
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

# ---------------- Page setup ----------------
st.set_page_config(
    page_title="CineSage | Movie Extractor",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------- Modern cinematic styling ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@400;500;600;700&display=swap');

:root { --bg: #080b14; --panel: rgba(20, 25, 43, .78); --line: rgba(255,255,255,.11); --muted: #a7b0c5; --accent: #9b87f5; }
.stApp {
    background:
      radial-gradient(ellipse at 12% 5%, rgba(117, 87, 232, .20), transparent 34%),
      radial-gradient(ellipse at 88% 22%, rgba(36, 190, 190, .11), transparent 30%),
      linear-gradient(145deg, #080b14 0%, #0c1020 55%, #101329 100%);
    color: #f4f5fb;
    font-family: 'DM Sans', sans-serif;
}
.stApp:before {
    content: ""; position: fixed; inset: 0; pointer-events: none; z-index: 0;
    background-image: radial-gradient(rgba(255,255,255,.13) .7px, transparent .7px);
    background-size: 28px 28px; opacity: .16;
}
.block-container { max-width: 1100px; padding-top: 2rem; padding-bottom: 4rem; position: relative; z-index: 1; }
h1, h2, h3 { font-family: 'Space Grotesk', sans-serif !important; letter-spacing: -.035em; }
.hero { text-align:center; padding: 2.1rem 1rem 1.5rem; animation: rise .8s ease-out both; }
.eyebrow { display:inline-block; color:#c4b8ff; border:1px solid rgba(155,135,245,.35); background:rgba(155,135,245,.10); border-radius:999px; padding:7px 13px; font-size:.78rem; letter-spacing:.12em; text-transform:uppercase; }
.hero-title { font-family:'Space Grotesk',sans-serif; font-size:clamp(2.5rem, 6vw, 4.5rem); line-height:1.04; font-weight:700; margin:1rem 0 .8rem; color:#fff; }
.gradient-text { background:linear-gradient(90deg,#c9bcff,#8ce7df); -webkit-background-clip:text; background-clip:text; color:transparent; }
.hero-sub { color:#aeb6ca; max-width:650px; margin:0 auto; font-size:1.05rem; line-height:1.7; }
.panel { background:linear-gradient(145deg,rgba(24,30,51,.86),rgba(13,17,31,.83)); border:1px solid var(--line); border-radius:22px; padding:1.35rem; box-shadow:0 18px 55px rgba(0,0,0,.22); backdrop-filter:blur(14px); animation: rise .65s ease-out both; }
.panel-label { color:#c4b8ff; font-size:.77rem; letter-spacing:.13em; text-transform:uppercase; font-weight:700; margin-bottom:.45rem; }
div[data-testid="stTextArea"] textarea { background:rgba(5,8,18,.72)!important; color:#f4f5fb!important; border:1px solid rgba(255,255,255,.13)!important; border-radius:14px!important; line-height:1.65!important; }
div[data-testid="stTextArea"] textarea:focus { border-color:#9b87f5!important; box-shadow:0 0 0 2px rgba(155,135,245,.16)!important; }
.stButton > button { width:100%; border:0; border-radius:13px; padding:.78rem 1rem; color:#101020; font-weight:700; background:linear-gradient(100deg,#c4b5fd,#8ce7df); transition:transform .2s ease, filter .2s ease; }
.stButton > button:hover { transform:translateY(-2px); filter:brightness(1.06); color:#101020; }
.result-card { background:rgba(22,28,47,.72); border:1px solid var(--line); border-radius:17px; padding:1.1rem 1.2rem; margin:.45rem 0; animation: rise .55s ease-out both; }
.result-label { color:#aab4cb; font-size:.77rem; letter-spacing:.09em; text-transform:uppercase; margin-bottom:.35rem; }
.result-value { color:#f7f7ff; font-size:1.02rem; line-height:1.65; overflow-wrap:anywhere; }
.cast-chip { display:inline-block; margin:4px 5px 4px 0; padding:7px 11px; border:1px solid rgba(155,135,245,.28); border-radius:999px; background:rgba(155,135,245,.10); color:#e4deff; font-size:.9rem; }
.footer { text-align:center; color:#77819b; font-size:.8rem; padding-top:2.4rem; }
div[data-testid="stAlert"] { border-radius:13px; }
@keyframes rise { from { opacity:0; transform:translateY(12px); } to { opacity:1; transform:translateY(0); } }
@media (prefers-reduced-motion: reduce) { *, *:before, *:after { animation:none!important; transition:none!important; } }
</style>
""", unsafe_allow_html=True)

# ---------------- Header ----------------
st.markdown("""
<div class="hero">
  <div class="eyebrow">✦ AI-powered movie intelligence</div>
  <div class="hero-title">Every movie has a<br><span class="gradient-text">story to extract.</span></div>
  <p class="hero-sub">Turn a movie description into structured, easy-to-read information using LangChain and OpenAI.</p>
</div>
""", unsafe_allow_html=True)

# ---------------- Load API key ----------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

# ---------------- Output schema ----------------
class MovieInfo(BaseModel):
    movie_name: str = Field(description="Name of the movie")
    director: str = Field(description="Movie director")
    release_year: int = Field(description="Release year")
    genre: str = Field(description="Movie genre")
    main_cast: list[str] = Field(description="Main actors")
    summary: str = Field(description="Short movie summary")

# ---------------- Input panel ----------------
left, right = st.columns([1.08, 0.92], gap="large")

with left:
    st.markdown('<div class="panel"><div class="panel-label">01 / Your input</div><h3 style="margin-top:0">Movie description</h3><p style="color:#a7b0c5">Paste a paragraph. The model will extract the details defined in your schema.</p></div>', unsafe_allow_html=True)
    description = st.text_area(
        "Movie description",
        height=250,
        placeholder="Paste your movie paragraph here...",
        label_visibility="collapsed",
        key="movie_description",
    )
    extract = st.button("✦  Extract movie information", use_container_width=True)

with right:
    st.markdown("""
    <div class="panel">
      <div class="panel-label">02 / What you'll get</div>
      <h3 style="margin-top:0">A clear movie profile</h3>
      <div class="result-card"><div class="result-label">Movie details</div><div class="result-value">Title · Director · Release year · Genre</div></div>
      <div class="result-card"><div class="result-label">Cast</div><div class="result-value">Main actors in a readable list</div></div>
      <div class="result-card"><div class="result-label">Synopsis</div><div class="result-value">A concise summary generated from your description</div></div>
      <p style="color:#8f9ab3;font-size:.85rem;margin:.9rem 0 0">Structured with your existing Pydantic schema.</p>
    </div>
    """, unsafe_allow_html=True)

# ---------------- Extract ----------------
if extract:
    if not api_key:
        st.error("OPENAI_API_KEY .env file mein nahi mili! Please check your .env file.")
    elif not description.strip():
        st.warning("Please enter a movie description first.")
    else:
        try:
            model = ChatOpenAI(
                model="gpt-4.1-mini",
                api_key=api_key,
                temperature=0.4,
            )
            structured_model = model.with_structured_output(MovieInfo)
            with st.spinner("Reading the story and extracting details..."):
                response = structured_model.invoke(
                    "Extract movie information from this description. "
                    "Use only the provided description and do not invent missing facts. "
                    "For fields not provided, use 'Not provided' where the schema allows it.\n\n"
                    + description
                )

            st.markdown("---")
            st.markdown('<div class="panel-label">03 / Extracted result</div>', unsafe_allow_html=True)
            st.markdown(f'<h2 style="margin-top:.2rem">{response.movie_name}</h2>', unsafe_allow_html=True)

            c1, c2, c3 = st.columns(3)
            for col, label, value in [
                (c1, "Director", response.director),
                (c2, "Release year", str(response.release_year)),
                (c3, "Genre", response.genre),
            ]:
                with col:
                    st.markdown(
                        f'<div class="result-card"><div class="result-label">{label}</div><div class="result-value">{value}</div></div>',
                        unsafe_allow_html=True,
                    )

            st.markdown('<div class="result-card"><div class="result-label">Main cast</div>' +
                        ''.join(f'<span class="cast-chip">{actor}</span>' for actor in response.main_cast) +
                        '</div>', unsafe_allow_html=True)

            st.markdown('<div class="result-card"><div class="result-label">Summary</div>' +
                        f'<div class="result-value">{response.summary}</div></div>', unsafe_allow_html=True)

            with st.expander("View structured JSON"):
                st.code(response.model_dump_json(indent=2), language="json")

        except Exception as e:
            st.error(f"Could not extract movie information: {e}")

st.markdown('<div class="footer">CINESAGE · Built with Streamlit, LangChain & OpenAI</div>', unsafe_allow_html=True)
