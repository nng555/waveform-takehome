import streamlit as st
import tempfile
import os
import time
import base64
import logging
from io import BytesIO

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
    if WEBRTC_AVAILABLE:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.subheader("🎤 Voice Message")

            # WebRTC streamer for audio
            webrtc_ctx = webrtc_streamer(
                key="speech-to-text",
                mode=WebRtcMode.SENDONLY,
                audio_receiver_size=256,
                client_settings=ClientSettings(
                    media_stream_constraints={"audio": True, "video": False},
                ),
                rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            )

            # Recording controls
            if webrtc_ctx.state.playing:
                if st.button("Start Recording"):
                    st.session_state.audio_processor.start_recording()
                    st.info("Recording... Press 'Stop Recording' when finished.")

                if st.button("Stop Recording"):
                    recorded_audio = st.session_state.audio_processor.stop_recording()
                    if recorded_audio:
                        st.success("Recording completed!")
                        # Save to temp file for processing
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                            tmp.write(recorded_audio)
                            st.session_state.audio_file_path = tmp.name
                    else:
                        st.error("No audio was recorded.")
    else:
        st.warning("Audio recording is not available. Please use text input instead.")

    # Text input section
    st.subheader("⌨️ Text Message")
    text_input = st.text_input(
        "Type your message:",
        key="text_input",
        placeholder="What would you like to say?"
    )

    # Helper text
    if WEBRTC_AVAILABLE:
        st.caption("You can either record audio by clicking the microphone, or type your message in the text box.")
    else:
        st.caption("Please type your message in the text box.")

    # Process recorded audio if available
    if 'audio_file_path' in st.session_state and client:
        try:
            # Transcribe audio
            audio_file = open(st.session_state.audio_file_path, "rb")
            transcription = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            os.unlink(st.session_state.audio_file_path)  # Clean up temp file
            del st.session_state.audio_file_path

            user_message = transcription.text
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

                st.rerun()  # Refresh the UI

        except Exception as e:
            st.error(f"Error processing audio: {str(e)}")
            if 'audio_file_path' in st.session_state:
                try:
                    os.unlink(st.session_state.audio_file_path)
                    del st.session_state.audio_file_path
                except:
                    pass
            st.info("💡 Try typing your message instead if audio recording isn't working on your device.")

    # Process text input
    elif text_input and client:
        user_message = text_input

        # Reset the text input field before processing to prevent duplicate submissions
        current_input = user_message
        st.session_state.text_input = ""

        # Display user message
        with st.chat_message("user"):
            st.write(current_input)

        # Add to conversation history
        st.session_state.messages.append({"role": "user", "content": current_input})

        # Generate response
        with st.spinner(f"🧠 {st.session_state.character_name} is thinking..."):
            try:
                res = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=st.session_state.messages,
                    max_tokens=300  # Keep responses reasonably short
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
                        st.warning(f"Could not generate speech (but you can still read the response): {str(e)}")
            except Exception as e:
                st.error(f"Error generating response: {str(e)}")
                st.session_state.messages.pop()  # Remove the user message if we couldn't get a response

        st.rerun()  # Refresh the UI after processing

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
