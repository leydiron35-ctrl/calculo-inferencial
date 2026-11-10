"""App de chat con Claude para Streamlit Community Cloud."""
import os

import anthropic
import streamlit as st

st.set_page_config(page_title="Chat con Claude", page_icon="💬", layout="centered")

MODELOS = ["claude-sonnet-5-5", "claude-opus-5-5", "claude-haiku-5-5"]


def obtener_api_key() -> str | None:
    """Lee la clave desde st.secrets; si no existe, usa la variable de entorno."""
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except (KeyError, FileNotFoundError):
        return os.getenv("ANTHROPIC_API_KEY")


@st.cache_resource
def crear_cliente(api_key: str) -> anthropic.Anthropic:
    return anthropic.Anthropic(api_key=api_key)


api_key = obtener_api_key()
if not api_key:
    st.error(
        "No se encontró `ANTHROPIC_API_KEY`. Configúrala en `.streamlit/secrets.toml` "
        "(local) o en *Settings → Secrets* (Streamlit Cloud)."
    )
    st.stop()

client = crear_cliente(api_key)

# ---------- Barra lateral ----------
with st.sidebar:
    st.header("⚙️ Configuración")
    modelo = st.selectbox("Modelo", MODELOS)
    system_prompt = st.text_area(
        "Instrucciones del sistema",
        value="Eres un asistente útil. Responde en español de forma clara y concisa.",
        height=120,
    )
    max_tokens = st.slider("Máximo de tokens", 256, 4096, 1024, step=256)
    temperatura = st.slider("Temperatura", 0.0, 1.0, 0.7, step=0.1)
    if st.button("🗑️ Nueva conversación", use_container_width=True):
        st.session_state.mensajes = []
        st.rerun()

# ---------- Estado e historial ----------
st.title("💬 Chat con Claude")
if "mensajes" not in st.session_state:
    st.session_state.mensajes = []

for m in st.session_state.mensajes:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])

# ---------- Entrada del usuario ----------
if prompt := st.chat_input("Escribe tu mensaje..."):
    st.session_state.mensajes.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        try:
            with client.messages.stream(
                model=modelo,
                max_tokens=max_tokens,
                temperature=temperatura,
                system=system_prompt,
                messages=st.session_state.mensajes,
            ) as stream:
                respuesta = st.write_stream(stream.text_stream)
            st.session_state.mensajes.append({"role": "assistant", "content": respuesta})
        except anthropic.AuthenticationError:
            st.error("La API key no es válida. Revisa tus secrets.")
        except anthropic.RateLimitError:
            st.warning("Límite de peticiones alcanzado. Espera un momento e inténtalo de nuevo.")
        except anthropic.APIError as e:
            st.error(f"Error de la API de Anthropic: {e}")
