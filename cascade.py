from openai import OpenAI
client = OpenAI()

import argparse
from utils import *

if __name__ == "__main__":
    role = input("You have been dropped into a high fantasy world with a large cast of all kinds of wild characters to have conversations with. Type the role of the character you'd like to converse with (e.g. inn bartender, grizzled warrior, mysterious wizard):\n\n")

    res = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "developer", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f"Write a character description for a character with a role of {role} in a high fantasy setting. Make sure the description includes specific character traits and a unique personality to make the interactions with the character interesting. Include details such as their temperament, race, background, and current setting. Keep things concise and within a short paragraph. Begin with the character's name between <name></name> tags."
            }
        ]
    )

    role_description = res.choices[0].message.content
    name = role_description.split("<name>")[-1].split("</name>")[0]

    role_description = role_description.replace("<name>", "")
    role_description = role_description.replace("</name>", "")

    print(f"You are now chatting with {name}. Press enter to start recording, then press enter again to stop.\n")

    curr_chat = [
        {"role": "developer", "content": f"You are a roleplayer in a high fantasy setting having a conversation with the user. Your role is described below. Keep your responses creative but short.\n\n<role>{role_description}</role>"},
    ]

    while True:
        record_audio()
        audio_file = open("output.wav", "rb")
        transcription = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file
        )

        print(f"User: {transcription.text}\n")

        curr_chat.append(
            {"role": "user", "content": transcription.text}
        )

        res = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=curr_chat,
        )

        response = res.choices[0].message.content

        print(f"{name}: {response}")

        curr_chat.append(
            {'role': 'assistant', 'content': response}
        )

        p = pyaudio.PyAudio()
        stream = p.open(format=8,
                        channels=1,
                        rate=24_000,
                        output=True)

        with client.audio.speech.with_streaming_response.create(
            model="tts-1",
            voice="alloy",
            input=response,
            response_format="pcm",
        ) as res:
            for chunk in res.iter_bytes(1024):
                stream.write(chunk)

