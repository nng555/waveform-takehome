import streamlit as st
import logging
from typing import Optional, Dict, List, Any

# Dependency imports with proper error handling
try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    OpenAI = None

try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, ClientSettings
    import av
    WEBRTC_AVAILABLE = True
except ImportError:
    WEBRTC_AVAILABLE = False


# -----------------------------------------------------------------------------
# Configuration and Setup
# -----------------------------------------------------------------------------
def initialize_session_state():
    """Initialize all session state variables."""
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    if 'character_name' not in st.session_state:
        st.session_state.character_name = ""

    if 'character_description' not in st.session_state:
        st.session_state.character_description = ""

    if 'chat_started' not in st.session_state:
        st.session_state.chat_started = False

    if 'messages_count' not in st.session_state:
        st.session_state.messages_count = 0

    if 'audio_processed' not in st.session_state:
        st.session_state.audio_processed = False

    if 'pending_audio' not in st.session_state:
        st.session_state.pending_audio = None


def init_openai_client() -> Optional[OpenAI]:
    """Initialize the OpenAI client with the API key from session state."""
    api_key = st.session_state.get('api_key')

    if api_key and api_key.startswith('sk-'):
        try:
            return OpenAI(api_key=api_key)
        except Exception as e:
            st.error(f"Error initializing OpenAI client: {str(e)}")
            return None
    return None


def display_debug_info():
    """Display debug information (only shown in debug mode)."""
    st.write("### Debug Information")
    import sys
    st.write(f"Python version: {sys.version}")

    # Environment information
    try:
        import pkg_resources
        st.write("Installed packages:")
        packages = sorted([f"{pkg.key}=={pkg.version}" for pkg in pkg_resources.working_set])
        for package in packages:
            if any(name in package.lower() for name in ['openai', 'streamlit', 'requests', 'webrtc']):
                st.write(f"- {package}")
    except Exception as e:
        st.write(f"Error checking packages: {e}")

    # API key information
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
                # Simple test to verify credentials
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


# -----------------------------------------------------------------------------
# Character Generation and Chat Functions
# -----------------------------------------------------------------------------
def generate_character(client: OpenAI, role_input: str) -> tuple:
    """Generate a fantasy character based on the role input."""
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

        return name, cleaned_description
    except Exception as e:
        raise Exception(f"Error generating character: {str(e)}")


def process_chat_response(client: OpenAI, messages: List[Dict[str, str]], voice: str) -> tuple:
    """Process the chat response and generate audio if needed."""
    import base64
    from io import BytesIO

    try:
        # Generate text response
        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        response = res.choices[0].message.content

        # Generate speech
        try:
            # Truncate long responses for TTS to avoid errors
            tts_input = response
            if len(response) > 4000:  # OpenAI TTS has input length limits
                tts_input = response[:4000] + "..."

            # Get speech from OpenAI
            audio_response = client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=tts_input,
                response_format="mp3",
            )

            # Convert to base64 string for reliable session state storage
            if hasattr(audio_response, 'content') and audio_response.content:
                # Convert bytes to base64 string
                audio_b64 = base64.b64encode(audio_response.content).decode('utf-8')
                # Wrap in data URL format that st.audio can use directly
                audio_data = f"data:audio/mp3;base64,{audio_b64}"
                return response, audio_data
            else:
                st.warning("Received empty audio content from OpenAI")
                return response, None

        except Exception as e:
            st.warning(f"Could not generate speech: {str(e)}")
            return response, None

    except Exception as e:
        raise Exception(f"Error processing chat response: {str(e)}")


def handle_audio_input(client: OpenAI, audio_value: bytes) -> str:
    """Process audio input and return the transcription."""
    try:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_value
        )
        return transcript.text
    except Exception as e:
        st.error(f"Error transcribing audio: {str(e)}")
        return ""


# -----------------------------------------------------------------------------
# UI Components
# -----------------------------------------------------------------------------
def render_api_key_input():
    """Render the API key input section."""
    st.write("### ⚙️ Configuration")
    st.warning("⚠️ Please enter your OpenAI API key to continue")

    api_key = st.text_input(
        "OpenAI API Key:",
        type="password",
        help="Your key stays in your browser and is never stored",
        placeholder="sk-... or sk-proj-..."
    )

    is_valid_key = api_key and api_key.startswith('sk-')
    if is_valid_key:
        st.session_state.api_key = api_key
        return True
    return False


def render_voice_settings():
    """Render voice settings selection."""
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
    return voice


def render_character_selection(client: OpenAI):
    """Render the character selection UI."""
    st.header("Choose Your Character")

    role_input = st.text_input(
        "What character would you like to talk to?",
        placeholder="inn bartender, grizzled warrior, mysterious wizard..."
    )

    # Only enable the button if there's input
    button_disabled = not role_input
    start_button = st.button("Generate Character", disabled=button_disabled, type="primary")

    if start_button and role_input:
        with st.spinner("🧙 Creating your character..."):
            try:
                name, description = generate_character(client, role_input)

                st.session_state.character_name = name
                st.session_state.character_description = description

                # Initialize the chat with system message
                st.session_state.messages = [
                    {"role": "system", "content": f"You are a roleplayer in a high fantasy setting having a conversation with the user. Your role is described below. Keep your responses creative but short.\n\n<role>{description}</role>"}
                ]

                st.session_state.chat_started = True
                # Reset audio processing flags
                st.session_state.audio_processed = False
                st.session_state.pending_audio = None
                return True

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
                return False
    return False


def render_chat_interface(client: OpenAI, voice: str):
    """Render the chat interface."""
    # Display character information
    st.header(f"💬 Chatting with {st.session_state.character_name}")

    with st.expander("📜 Character Background", expanded=False):
        st.markdown(f"*{st.session_state.character_description}*")

    # Display chat messages first
    if st.session_state.messages != []:
        st.markdown("---")
        st.subheader("Conversation")
    chat_container = st.container()
    with chat_container:
        # Only display user and assistant messages (not the system prompt)
        for message in [m for m in st.session_state.messages if m["role"] not in ["system", "developer"]]:
            role = "user" if message["role"] == "user" else "assistant"
            with st.chat_message(role, avatar="🧙‍♂️" if role == "assistant" else None):
                st.write(message["content"])

    # Input section below the chat
    st.markdown("---")

    # Play pending audio if available (after handling text input)
    if st.session_state.pending_audio is not None:
        try:
            # Use a more reliable approach to play audio
            with st.spinner("Loading audio response..."):
                st.audio(
                    st.session_state.pending_audio,
                    format="audio/mp3",
                    start_time=0
                )
                st.session_state.pending_audio = None
        except Exception as e:
            st.error(f"Error playing audio: {str(e)}")
            st.session_state.pending_audio = None

    # Audio input handling - process only if not already processed
    if not st.session_state.audio_processed:
        audio_value = st.audio_input(f"Record your message to {st.session_state.character_name}")

        if audio_value:
            # Get transcription
            with st.spinner("Transcribing your message..."):
                user_message = handle_audio_input(client, audio_value)

            if user_message.strip():
                # Add to conversation history
                st.session_state.messages.append({"role": "user", "content": user_message})

                # Generate response
                with st.spinner(f"🧠 {st.session_state.character_name} is thinking..."):
                    response, audio_content = process_chat_response(client, st.session_state.messages, voice)

                    # Add to conversation history
                    st.session_state.messages.append({"role": "assistant", "content": response})

                    # Store audio in session state
                    st.session_state.pending_audio = audio_content

                    # Mark as processed to prevent reprocessing on rerun
                    st.session_state.audio_processed = True

                    # Force a rerun to update the chat display
                    st.rerun()
    else:
        # Reset audio processing flag when no input is pending
        st.audio_input(f"Record your message to {st.session_state.character_name}", label_visibility="visible")
        st.session_state.audio_processed = False

    # Reset button
    if st.button("🔄 Start a new conversation"):
        st.session_state.messages = []
        st.session_state.character_name = ""
        st.session_state.character_description = ""
        st.session_state.chat_started = False
        st.session_state.audio_processed = False
        st.session_state.pending_audio = None
        st.rerun()


# -----------------------------------------------------------------------------
# Main Application
# -----------------------------------------------------------------------------
def main():
    # App title and description
    st.title("Fantasy Character Chat")
    st.markdown("Have conversations with characters from a high fantasy world!")

    # Initialize session state
    initialize_session_state()

    # Debug mode (optional)
    debug_mode = False
    if debug_mode:
        display_debug_info()

    # API key configuration
    client = None
    if 'api_key' not in st.session_state or not st.session_state.api_key:
        if render_api_key_input():
            client = init_openai_client()
            if client:
                st.success("✅ API key configured successfully!")
                st.rerun()
    else:
        # Key is configured, initialize client
        client = init_openai_client()

        # Select voice
        voice = render_voice_settings()

        # Option to change API key
        if st.button("🔑 Change API Key"):
            st.session_state.api_key = None
            st.rerun()

        # Character selection or chat interface
        if not st.session_state.chat_started:
            if client and render_character_selection(client):
                st.rerun()
        else:
            render_chat_interface(client, voice)

            # Update metrics when chat is active
            message_count = len([m for m in st.session_state.messages if m["role"] in ["user", "assistant"]])
            if message_count != st.session_state.messages_count:
                st.session_state.messages_count = message_count


if __name__ == "__main__":
    main()
