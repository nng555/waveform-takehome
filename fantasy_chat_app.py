import streamlit as st
from openai import OpenAI
import tempfile
import os
from io import BytesIO
import time

# Initialize OpenAI client
client = None

def init_openai_client():
    api_key = st.session_state.api_key
    if api_key and api_key.startswith('sk-'):
        return OpenAI(api_key=api_key)
    return None

# App title and description
st.title("Fantasy Character Chat")
st.markdown("Have conversations with characters from a high fantasy world!")

# API Key input (in sidebar for cleaner UI)
with st.sidebar:
    st.header("Settings")
    api_key = st.text_input("Enter your OpenAI API Key:", type="password",
                           help="Your key stays in your browser and is never stored")
    if api_key:
        st.session_state.api_key = api_key
        client = init_openai_client()

    st.subheader("Voice Settings")
    voice = st.selectbox("Character voice:",
                        ["alloy", "echo", "fable", "onyx", "nova", "shimmer"],
                        index=0)

# Initialize session state variables
if 'messages' not in st.session_state:
    st.session_state.messages = []

if 'character_name' not in st.session_state:
    st.session_state.character_name = ""

if 'character_description' not in st.session_state:
    st.session_state.character_description = ""

if 'chat_started' not in st.session_state:
    st.session_state.chat_started = False

# Character selection section (only shown before chat starts)
if not st.session_state.chat_started:
    st.header("Choose Your Character")

    role_input = st.text_input(
        "What character would you like to talk to?",
        placeholder="inn bartender, grizzled warrior, mysterious wizard..."
    )

    start_button = st.button("Generate Character")

    if start_button and role_input and client:
        with st.spinner("Creating your character..."):
            try:
                res = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "developer", "content": "You are a helpful assistant."},
                        {
                            "role": "user",
                            "content": f"Write a character description for a character with a role of {role_input} in a high fantasy setting. Make sure the description includes specific character traits and a unique personality to make the interactions with the character interesting. Include details such as their temperament, race, background, and current setting. Keep things concise and within a short paragraph. Begin with the character's name between <name></name> tags."
                        }
                    ]
                )

                role_description = res.choices[0].message.content
                name = role_description.split("<name>")[-1].split("</name>")[0]

                cleaned_description = role_description.replace("<name>", "").replace("</name>", "")

                st.session_state.character_name = name
                st.session_state.character_description = cleaned_description

                # Initialize the chat with system message
                st.session_state.messages = [
                    {"role": "developer", "content": f"You are a roleplayer in a high fantasy setting having a conversation with the user. Your role is described below. Keep your responses creative but short.\n\n<role>{cleaned_description}</role>"}
                ]

                st.session_state.chat_started = True
                st.experimental_rerun()

            except Exception as e:
                st.error(f"Error creating character: {str(e)}")
                if "API key" in str(e):
                    st.warning("Please check your OpenAI API key.")

# Chat interface (only shown after character is generated)
if st.session_state.chat_started:
    # Display character information
    st.header(f"Chatting with {st.session_state.character_name}")

    with st.expander("Character Description", expanded=False):
        st.write(st.session_state.character_description)

    # Display chat messages
    chat_container = st.container()
    with chat_container:
        # Only display user and assistant messages (not the system prompt)
        for message in [m for m in st.session_state.messages if m["role"] != "developer"]:
            with st.chat_message(message["role"]):
                st.write(message["content"])

    # Audio recording
    audio_bytes = st.audio_recorder(
        pause_threshold=2.0,
        sample_rate=24000,
        key="audio_recorder"
    )

    st.caption("Click the microphone to record your message, then wait or click again to stop.")

    # Text input as fallback
    text_input = st.text_input("Or type your message:", key="text_input")

    # Process audio input
    if audio_bytes and client:
        with st.spinner("Processing your message..."):
            # Save audio to temp file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
                tmp.write(audio_bytes)
                tmp_path = tmp.name

            try:
                # Transcribe audio
                audio_file = open(tmp_path, "rb")
                transcription = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=audio_file
                )
                os.unlink(tmp_path)  # Clean up temp file

                user_message = transcription.text
                if user_message.strip():  # Only process non-empty messages
                    # Display user message
                    with st.chat_message("user"):
                        st.write(user_message)

                    # Add to conversation history
                    st.session_state.messages.append({"role": "user", "content": user_message})

                    # Generate response
                    with st.spinner(f"{st.session_state.character_name} is thinking..."):
                        res = client.chat.completions.create(
                            model="gpt-4o-mini",
                            messages=st.session_state.messages,
                        )
                        response = res.choices[0].message.content

                    # Display assistant response
                    with st.chat_message("assistant"):
                        st.write(response)

                    # Add to conversation history
                    st.session_state.messages.append({"role": "assistant", "content": response})

                    # Generate speech
                    with st.spinner("Generating voice response..."):
                        audio_response = client.audio.speech.create(
                            model="tts-1",
                            voice=voice,
                            input=response,
                        )

                        # Play the audio
                        st.audio(audio_response.content, format="audio/mp3", start_time=0)

            except Exception as e:
                st.error(f"Error processing audio: {str(e)}")

    # Process text input
    elif text_input and client:
        user_message = text_input

        # Add to conversation history
        st.session_state.messages.append({"role": "user", "content": user_message})

        # Generate response
        with st.spinner(f"{st.session_state.character_name} is thinking..."):
            res = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=st.session_state.messages,
            )
            response = res.choices[0].message.content

        # Add to conversation history
        st.session_state.messages.append({"role": "assistant", "content": response})

        # Generate speech
        with st.spinner("Generating voice response..."):
            audio_response = client.audio.speech.create(
                model="tts-1",
                voice=voice,
                input=response,
            )

            # Play the audio
            st.audio(audio_response.content, format="audio/mp3", start_time=0)

        # Clear the text input
        st.session_state.text_input = ""
        st.experimental_rerun()

# Reset button
with st.sidebar:
    if st.button("Start New Conversation"):
        st.session_state.messages = []
        st.session_state.character_name = ""
        st.session_state.character_description = ""
        st.session_state.chat_started = False
        st.experimental_rerun()
