import streamlit as st
import tempfile
import os
import time
import base64
import logging
from io import BytesIO
import logging
import logging.handlers
import queue
import threading
import time
import urllib.request
import os
from collections import deque
from pathlib import Path
from typing import List

import av
import numpy as np
import pydub
import streamlit as st
from twilio.rest import Client

from streamlit_webrtc import WebRtcMode, webrtc_streamer

HERE = Path(__file__).parent

logger = logging.getLogger(__name__)


# This code is based on https://github.com/streamlit/demo-self-driving/blob/230245391f2dda0cb464008195a470751c01770b/streamlit_app.py#L48  # noqa: E501
def download_file(url, download_to: Path, expected_size=None):
    # Don't download the file twice.
    # (If possible, verify the download using the file length.)
    if download_to.exists():
        if expected_size:
            if download_to.stat().st_size == expected_size:
                return
        else:
            st.info(f"{url} is already downloaded.")
            if not st.button("Download again?"):
                return

    download_to.parent.mkdir(parents=True, exist_ok=True)

    # These are handles to two visual elements to animate.
    weights_warning, progress_bar = None, None
    try:
        weights_warning = st.warning("Downloading %s..." % url)
        progress_bar = st.progress(0)
        with open(download_to, "wb") as output_file:
            with urllib.request.urlopen(url) as response:
                length = int(response.info()["Content-Length"])
                counter = 0.0
                MEGABYTES = 2.0 ** 20.0
                while True:
                    data = response.read(8192)
                    if not data:
                        break
                    counter += len(data)
                    output_file.write(data)

                    # We perform animation by overwriting the elements.
                    weights_warning.warning(
                        "Downloading %s... (%6.2f/%6.2f MB)"
                        % (url, counter / MEGABYTES, length / MEGABYTES)
                    )
                    progress_bar.progress(min(counter / length, 1.0))
    # Finally, we remove these visual elements by calling .empty().
    finally:
        if weights_warning is not None:
            weights_warning.empty()
        if progress_bar is not None:
            progress_bar.empty()


# This code is based on https://github.com/whitphx/streamlit-webrtc/blob/c1fe3c783c9e8042ce0c95d789e833233fd82e74/sample_utils/turn.py
@st.cache_data  # type: ignore
def get_ice_servers():
    """Use Twilio's TURN server because Streamlit Community Cloud has changed
    its infrastructure and WebRTC connection cannot be established without TURN server now.  # noqa: E501
    We considered Open Relay Project (https://www.metered.ca/tools/openrelay/) too,
    but it is not stable and hardly works as some people reported like https://github.com/aiortc/aiortc/issues/832#issuecomment-1482420656  # noqa: E501
    See https://github.com/whitphx/streamlit-webrtc/issues/1213
    """

    # Ref: https://www.twilio.com/docs/stun-turn/api
    try:
        account_sid = os.environ["TWILIO_ACCOUNT_SID"]
        auth_token = os.environ["TWILIO_AUTH_TOKEN"]
    except KeyError:
        logger.warning(
            "Twilio credentials are not set. Fallback to a free STUN server from Google."  # noqa: E501
        )
        return [{"urls": ["stun:stun.l.google.com:19302"]}]

    client = Client(account_sid, auth_token)

    token = client.tokens.create()

    return token.ice_servers



def main():


def app_sst(model_path: str, lm_path: str, lm_alpha: float, lm_beta: float, beam: int):

# Wrap OpenAI import in try/except
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError as e:
    st.error(f"Failed to import OpenAI library: {e}")
    st.info("Please make sure the openai package is installed with: `pip install openai>=1.1.0`")
    OPENAI_AVAILABLE = False
    OpenAI = None

# Import streamlit-webrtc for audio recording
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
    import av
    WEBRTC_AVAILABLE = True
except ImportError:
    st.warning("streamlit-webrtc is not installed. Please install it with: `pip install streamlit-webrtc`")
    WEBRTC_AVAILABLE = False

# Initialize OpenAI client
client = None

def init_openai_client():
    api_key = st.session_state.api_key if 'api_key' in st.session_state else None
    if api_key and api_key.startswith('sk-'):
        try:
            # For OpenAI v1.x
            return OpenAI(api_key=api_key)
        except Exception as e:
            st.error(f"Error initializing OpenAI client: {str(e)}")
            return None
    return None

# Audio processing class if webrtc is available
if WEBRTC_AVAILABLE:
    class AudioProcessor:
        def __init__(self):
            self.audio_frames = []
            self.recording = False
            self.recorded_audio = None

        def recv(self, frame):
            if self.recording:
                self.audio_frames.append(frame.to_ndarray())
            return frame

        def start_recording(self):
            self.recording = True
            self.audio_frames = []

        def stop_recording(self):
            self.recording = False
            if self.audio_frames:
                # Convert frames to a single audio file
                try:
                    import numpy as np
                    # Concatenate all audio frames
                    audio_data = np.concatenate(self.audio_frames, axis=0)
                    # Create wav file in memory
                    import scipy.io.wavfile as wavfile
                    wav_bytes = BytesIO()
                    wavfile.write(wav_bytes, 16000, audio_data.astype(np.int16))
                    wav_bytes.seek(0)
                    self.recorded_audio = wav_bytes.read()
                    return self.recorded_audio
                except Exception as e:
                    st.error(f"Error processing audio: {e}")
                    return None
            return None

# App title and description
st.title("Fantasy Character Chat")
st.markdown("Have conversations with characters from a high fantasy world!")

# Add debugging information - will be visible only in development
debug_mode = True
if debug_mode:
    st.write("### Debug Information")
    import sys
    st.write(f"Python version: {sys.version}")
    st.write("Environment information:")
    try:
        import pkg_resources
        st.write("Installed packages:")
        packages = sorted([f"{pkg.key}=={pkg.version}" for pkg in pkg_resources.working_set])
        for package in packages:
            if any(name in package.lower() for name in ['openai', 'streamlit', 'requests', 'webrtc']):
                st.write(f"- {package}")
    except Exception as e:
        st.write(f"Error checking packages: {e}")

    if 'api_key' in st.session_state and st.session_state.api_key:
        key = st.session_state.api_key
        if key.startswith('sk-proj-'):
            key_type = "Project-based API key"
        elif key.startswith('sk-org-'):
            key_type = "Organization API key"
        else:
            key_type = "Standard API key"
        masked_key = key[:7] + "..." + key[-4:] if key else "Not set"
        st.write(f"API key status: {masked_key} ({key_type})")

    # Check OpenAI client
    if OPENAI_AVAILABLE:
        st.write("OpenAI library is available ✓")
        if client:
            st.write("OpenAI client initialized successfully ✓")
            try:
                # Just a simple test to see if credentials are working
                test_response = client.models.list()
                st.write("API credentials working ✓")
            except Exception as e:
                st.write(f"API credentials issue: {str(e)}")
        else:
            st.write("OpenAI client NOT initialized ✗")
    else:
        st.write("OpenAI library is NOT available ✗")

    # Check WebRTC
    if WEBRTC_AVAILABLE:
        st.write("streamlit-webrtc is available ✓")
    else:
        st.write("streamlit-webrtc is NOT available ✗")

    st.write("---")

# Show main API key input if not configured
if 'api_key' not in st.session_state or not st.session_state.api_key:
    st.write("### ⚙️ Configuration")
    st.warning("⚠️ Please enter your OpenAI API key to continue")

    # Main API key input
    api_key = st.text_input(
        "OpenAI API Key:",
        type="password",
        help="Your key stays in your browser and is never stored",
        placeholder="sk-... or sk-proj-..."
    )

    is_valid_key = api_key and api_key.startswith('sk-')

    if is_valid_key:
        st.session_state.api_key = api_key
        client = init_openai_client()
        if client:
            st.success("✅ API key configured successfully!")
            st.rerun()
else:
    # Key is configured, initialize client
    client = init_openai_client()

    # Voice settings
    voice_col1, voice_col2 = st.columns([1, 3])
    with voice_col1:
        st.write("##### Voice:")
    with voice_col2:
        voice = st.selectbox(
            "Select character voice:",
            ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
            index=0,
            label_visibility="collapsed"
        )

    # Add a small reset key option
    if st.button("🔑 Change API Key"):
        st.session_state.api_key = None
        st.rerun()

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'character_name' not in st.session_state:
    st.session_state.character_name = ""

if 'character_description' not in st.session_state:
    st.session_state.character_description = ""

if 'chat_started' not in st.session_state:
    st.session_state.chat_started = False

if 'audio_processor' not in st.session_state and WEBRTC_AVAILABLE:
    st.session_state.audio_processor = AudioProcessor()

# Character selection section (only shown before chat starts and after API key is provided)
if not st.session_state.chat_started and 'api_key' in st.session_state and st.session_state.api_key:
    st.header("Choose Your Character")

    role_input = st.text_input(
        "What character would you like to talk to?",
        placeholder="inn bartender, grizzled warrior, mysterious wizard..."
    )

    # Only enable the button if there's input
    button_disabled = not role_input
    start_button = st.button("Generate Character", disabled=button_disabled, type="primary")

    if start_button and role_input and client:
        with st.spinner("🧙 Creating your character..."):
            try:
                res = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {
                            "role": "user",
                            "content": f"Write a character description for a character with a role of {role_input} in a high fantasy setting. Make sure the description includes specific character traits and a unique personality to make the interactions with the character interesting. Include details such as their temperament, race, background, and current setting. Keep things concise and within a short paragraph. Begin with the character's name between <name></name> tags."
                        }
                    ]
                )

                role_description = res.choices[0].message.content
                try:
                    name = role_description.split("<name>")[-1].split("</name>")[0]
                    cleaned_description = role_description.replace("<name>", "").replace("</name>", "")
                except:
                    # Fallback if tags aren't used properly
                    name = f"The {role_input.title()}"
                    cleaned_description = role_description

                st.session_state.character_name = name
                st.session_state.character_description = cleaned_description

                # Initialize the chat with system message
                st.session_state.messages = [
                    {"role": "system", "content": f"You are a roleplayer in a high fantasy setting having a conversation with the user. Your role is described below. Keep your responses creative but short.\n\n<role>{cleaned_description}</role>"}
                ]

                st.session_state.chat_started = True
                st.rerun()

            except Exception as e:
                error_message = str(e)
                st.error(f"Error creating character: {error_message}")
                if "API key" in error_message:
                    st.warning("⚠️ Your OpenAI API key appears to be invalid. Please check that you've entered it correctly.")
                elif "quota" in error_message.lower():
                    st.warning("⚠️ You may have reached your OpenAI API quota limit.")
                elif "rate" in error_message.lower():
                    st.warning("⚠️ Rate limit exceeded. Please wait a moment and try again.")
                st.write("Detailed error:", error_message)

# Chat interface (only shown after character is generated)
if st.session_state.chat_started:
    # Display character information with visual styling
    st.header(f"💬 Chatting with {st.session_state.character_name}")

    with st.expander("📜 Character Background", expanded=False):
        st.markdown(f"*{st.session_state.character_description}*")

    # Display chat messages in a container with a light border
    st.markdown("---")
    chat_container = st.container()
    with chat_container:
        # Only display user and assistant messages (not the system prompt)
        for message in [m for m in st.session_state.messages if m["role"] not in ["system", "developer"]]:
            role = "user" if message["role"] == "user" else "assistant"
            with st.chat_message(role, avatar="🧙‍♂️" if role == "assistant" else None):
                st.write(message["content"])

    # Input section with both audio and text options
    st.markdown("---")

    # WebRTC audio section if available
    recorded_audio = None
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("🎤 Voice Message")


        # https://github.com/mozilla/DeepSpeech/releases/tag/v0.9.3
        MODEL_URL = "https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.pbmm"  # noqa
        LANG_MODEL_URL = "https://github.com/mozilla/DeepSpeech/releases/download/v0.9.3/deepspeech-0.9.3-models.scorer"  # noqa
        MODEL_LOCAL_PATH = HERE / "models/deepspeech-0.9.3-models.pbmm"
        LANG_MODEL_LOCAL_PATH = HERE / "models/deepspeech-0.9.3-models.scorer"

        download_file(MODEL_URL, MODEL_LOCAL_PATH, expected_size=188915987)
        download_file(LANG_MODEL_URL, LANG_MODEL_LOCAL_PATH, expected_size=953363776)

        lm_alpha = 0.931289039105002
        lm_beta = 1.1834137581510284
        beam = 100

        webrtc_ctx = webrtc_streamer(
            key="speech-to-text",
            mode=WebRtcMode.SENDONLY,
            audio_receiver_size=1024,
            rtc_configuration={"iceServers": get_ice_servers()},
            media_stream_constraints={"video": False, "audio": True},
        )

        status_indicator = st.empty()

        if not webrtc_ctx.state.playing:
            return

        status_indicator.write("Loading...")
        text_output = st.empty()
        stream = None

        while True:
            if webrtc_ctx.audio_receiver:
                if stream is None:
                    from deepspeech import Model

                    model = Model(model_path)
                    model.enableExternalScorer(lm_path)
                    model.setScorerAlphaBeta(lm_alpha, lm_beta)
                    model.setBeamWidth(beam)

                    stream = model.createStream()

                    status_indicator.write("Model loaded.")

                sound_chunk = pydub.AudioSegment.empty()
                try:
                    audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
                except queue.Empty:
                    time.sleep(0.1)
                    status_indicator.write("No frame arrived.")
                    continue

                status_indicator.write("Running. Say something!")

                for audio_frame in audio_frames:
                    sound = pydub.AudioSegment(
                        data=audio_frame.to_ndarray().tobytes(),
                        sample_width=audio_frame.format.bytes,
                        frame_rate=audio_frame.sample_rate,
                        channels=len(audio_frame.layout.channels),
                    )
                    sound_chunk += sound

                if len(sound_chunk) > 0:
                    sound_chunk = sound_chunk.set_channels(1).set_frame_rate(
                        model.sampleRate()
                    )
                    buffer = np.array(sound_chunk.get_array_of_samples())
                    stream.feedAudioContent(buffer)
                    text = stream.intermediateDecode()
                    text_output.markdown(f"**User:** {text}")
            else:
                status_indicator.write("AudioReciver is not set. Abort.")
                break

        user_message = text
        if user_message.strip():  # Only process non-empty messages
            # Display user message
            with st.chat_message("user"):
                st.write(user_message)

            # Add to conversation history
            st.session_state.messages.append({"role": "user", "content": user_message})

            # Generate response
            with st.spinner(f"🧠 {st.session_state.character_name} is thinking..."):
                res = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=st.session_state.messages,
                )
                response = res.choices[0].message.content

            # Display assistant response
            with st.chat_message("assistant", avatar="🧙‍♂️"):
                st.write(response)

            # Add to conversation history
            st.session_state.messages.append({"role": "assistant", "content": response})

            # Generate speech
            with st.spinner("🔊 Generating voice response..."):
                try:
                    audio_response = client.audio.speech.create(
                        model="tts-1",
                        voice=voice,
                        input=response,
                    )

                    # Play the audio
                    st.audio(audio_response.content, format="audio/mp3", start_time=0)
                except Exception as e:
                    st.warning(f"Could not generate speech (but the text response is available): {str(e)}")

    # Add a reset button at the bottom of the chat
    if st.button("🔄 Start a new conversation"):
        st.session_state.messages = []
        st.session_state.character_name = ""
        st.session_state.character_description = ""
        st.session_state.chat_started = False
        st.rerun()

# Add metrics tracking to sidebar
if 'messages_count' not in st.session_state:
    st.session_state.messages_count = 0

# Update metrics when chat is active
if st.session_state.chat_started:
    message_count = len([m for m in st.session_state.messages if m["role"] in ["user", "assistant"]])
    if message_count != st.session_state.messages_count:
        st.session_state.messages_count = message_count
