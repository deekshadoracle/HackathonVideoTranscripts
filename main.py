import speech_recognition as sr
from moviepy import VideoFileClip
from pydub import AudioSegment
import os
import json
from ai_enrich import enrich_with_groq, test_ai

# Global result list
results = []

# Configuration
chunk_length_ms = 60000  # 1-minute chunks


def transcribe_video(video_path):
    print(f"\nProcessing video: {video_path}")

    audio_path = "temp_audio.wav"

    # Step 1: Extract audio
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(audio_path)
    except Exception as e:
        print(f"Error extracting audio from {video_path}: {e}")
        return

    # Step 2: Transcribe
    recognizer = sr.Recognizer()
    try:
        audio = AudioSegment.from_wav(audio_path)
    except Exception as e:
        print(f"Error loading audio: {e}")
        return

    chunks = [audio[i:i + chunk_length_ms] for i in range(0, len(audio), chunk_length_ms)]
    print(f"Transcribing {len(chunks)} chunks...")

    full_transcript = ""

    for i, chunk in enumerate(chunks):
        chunk_name = f"chunk{i}.wav"
        chunk.export(chunk_name, format="wav")
        with sr.AudioFile(chunk_name) as source:
            audio_data = recognizer.record(source)
            try:
                text = recognizer.recognize_google(audio_data)
                full_transcript += text + "\n"
            except sr.UnknownValueError:
                full_transcript += "[Unrecognized audio]\n"
            except sr.RequestError as e:
                full_transcript += f"[API error: {e}]\n"
        os.remove(chunk_name)

    os.remove(audio_path)

    results.append({
        "video_path": video_path,
        "transcript": full_transcript.strip()
    })
    print(f"Finished: {video_path}")


def process_all_videos(root_folder):
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.lower().endswith(".mp4"):
                full_path = os.path.join(dirpath, filename)
                transcribe_video(full_path)


def main(folder_path):
    print(f"Starting transcription in folder: {folder_path}")
    process_all_videos(folder_path)

    print("\nAll transcripts completed.")
    print(f"Total videos processed: {len(results)}")

    with open("all_transcripts.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)




if __name__ == "__main__":
    # main("C:/Users/rohit chakravorty/OneDrive - Oracle Corporation/Recordings/Video Clips/hackathon/agents/select/test_code")
    enrich_with_groq("all_transcripts.json")
    # test_ai()
