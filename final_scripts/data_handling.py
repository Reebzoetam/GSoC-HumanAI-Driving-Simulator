import os
import tempfile
import ffmpeg
import whisper
import pandas as pd
from pydub import AudioSegment, silence
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import tkinter as tk
from tkinter import filedialog, messagebox
import sys
import shutil

nltk.download('vader_lexicon')
MAX_SEGMENT_DURATION = 5.0
sentiment_analyzer = SentimentIntensityAnalyzer()

def extract_audio(video_path, output_dir):
    if output_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, "audios")
    os.makedirs(output_dir, exist_ok=True)

    base_name = os.path.splitext(os.path.basename(video_path))[0]
    audio_path = os.path.join(output_dir, f"{base_name}.wav")

    # for more dynamic audio extraction
    if video_path.lower().endswith('.wav'):
        print("File is already a WAV — copying without reprocessing.")
        shutil.copy(video_path, audio_path)
        return audio_path

    try:
        ffmpeg.input(video_path).output(
            audio_path,
            format='wav',
            acodec='pcm_s16le',
            ar='16000' 
        ).run(overwrite_output=True)
    except ffmpeg.Error as e:
        print("Error during audio extraction:", e)
        return None

    return audio_path

def transcribe_audio(chunks, timestamps):
    model = whisper.load_model('medium')
    
    transcripts = {
        'start': [],
        'segments': [],
    }

    for chunk, timestamp in zip(chunks, timestamps):
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            temp_audio_path = temp_file.name
            chunk.export(temp_audio_path, format='wav')

            result = model.transcribe(
            temp_audio_path, 
            language='en', 
            fp16=False, 
            initial_prompt="This is audio between six participants in a simulated driving environment as they collaborate and communicate as a team to navigate through the simulation.", 
            condition_on_previous_text=False
            )

            print(result)
            if not result.get('text', '').strip():
                print("Skipping empty transcription.")
                continue
            # may need to write a script to adjust this dynamically as needed depending on video quality
            if result['segments'][0]['no_speech_prob'] < 0.7:
                transcripts['segments'].append(result['text'])
                transcripts['start'].append(timestamp)
            os.remove(temp_audio_path)
    print(transcripts)
    return transcripts


def process_video(video_file, output_path):
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = extract_audio(video_file, temp_dir)
        chunks, timestamps = segment_audio(audio_path)
        transcript = transcribe_audio(chunks, timestamps)
        sentiment_data = analyze_segment_sentiment(transcript)
    for data in sentiment_data:
        print(data)
        print(f"At {data['timestamp']:.2f}s: {data['text']}")
        print(f"Sentiment: {data['sentiment']} (Score: {data['score']:.2f})")
    df = convert_dataframe(sentiment_data, video_file, output_path)
    print(df)
    return transcript

def segment_audio(audio_path):

    MAX_CHUNK_LENGTH = 5 * 1000
    overlap = 0

    audio = AudioSegment.from_wav(audio_path)
    nonsilent_ranges = silence.detect_nonsilent(
    audio,
    min_silence_len=300, 
    # to detect silence levels, will need to consider the volume levels of future audio clips, 
    # or find a way to normalise volume
    # this video has very low volume at times, but idk about future videos
    silence_thresh=audio.dBFS - 20, 
    )

    final_chunks = []
    timestamps = []
    print(nonsilent_ranges)
    for start_ms, end_ms in nonsilent_ranges:
        print(start_ms, end_ms)
        if start_ms >=200:
            overlap = 200
        else:
            overlap = 0
        chunk_start = start_ms - overlap
        chunk = audio[chunk_start:end_ms]
        # other potential chunking options include calling silence.detect_nonsilent again
        # and increasing the sensitvity to silence instead of every 5 seconds
        if len(chunk) > MAX_CHUNK_LENGTH:
            for i in range(0, len(chunk), MAX_CHUNK_LENGTH - overlap):
                subchunk = chunk[i:i + MAX_CHUNK_LENGTH]
                actual_start = chunk_start + i
                final_chunks.append(subchunk)
                timestamps.append(actual_start / 1000.0)
        else:
            final_chunks.append(chunk)
            timestamps.append(chunk_start / 1000.0)

    print(f"Total chunks: {len(final_chunks)}")
    print(f"Timestamps (s): {[round(ts, 2) for ts in timestamps]}")

    return final_chunks, timestamps

def get_sentiment_label(compound):
    # typical threshold values according to VADER sentiment documentation are usually 0.5, -0.5
    # but these are adjusted to 0.2 and -0.2 based on results from previous runs
    if compound >= 0.2:
        return "positive"
    elif compound <= -0.2:
        return "negative"
    else:
        return "neutral"

def analyze_segment_sentiment(transcript_dict):
    print("Analyzing sentiment...")
    sentiment_results = []

    for i in range(len(transcript_dict['start'])):
        start_time = transcript_dict['start'][i]
        text = transcript_dict['segments'][i]
        sentiment = sentiment_analyzer.polarity_scores(text)
        compound = sentiment['compound']
        sentiment_label = get_sentiment_label(compound)
        
        sentiment_results.append({
            "timestamp": start_time,
            "text": text,
            "sentiment": sentiment_label,
            "score": compound
        })

    return sentiment_results

def convert_dataframe(sentiment_results, video_name, output_path):
    print("Converting to DataFrame...")

    csv_filename = os.path.join(output_path, f"{video_name[-8:-4]}_analysis.csv")

    df = pd.DataFrame(sentiment_results)

    if os.path.exists(csv_filename):
        df.to_csv(csv_filename, mode='a', header=False, index=False)
    else:
        df.to_csv(csv_filename, index=False)

def process_multiple_videos(video_folder, output_path):
    transcripts = {}
    try:
        video_files = [f for f in os.listdir(video_folder)]
            
        for file in video_files:
            video_path = os.path.join(video_folder, file)
            try:
                transcripts[file] = process_video(video_path, output_path)
                print(f"Processed: {file}")
            except Exception as e:
                print(f"Error processing {file}: {e}")
                    
    except Exception as e:
        print(f"Error accessing the folder {video_folder}: {e}")
            
    return transcripts, output_path

def select_input():
    root = tk.Tk()
    root.withdraw()

    choice = messagebox.askyesno("Select Input Type", "Do you want to select a folder?\nClick 'No' to choose a file instead.")

    if choice:
        path = filedialog.askdirectory(title="Select a Folder")
        selection = "folder"
    else:
        selection = "file"
        path = filedialog.askopenfilename(title="Select a File")

    messagebox.showinfo("Output Folder", "Now select a folder where you want to save the output files.")
    output_path = filedialog.askdirectory(title="Select Output Folder")
    if not output_path:
        print("No output folder selected.")
        sys.exit(0)

    if path:
        if selection == "folder":
            print("Input selected:", path)
            print("Output folder selected:", output_path)
            process_multiple_videos(path, output_path)
        else:
            print("Input selected:", path)
            print("Output folder selected:", output_path)
            process_video(path, output_path)
    else:
        print("No selection made.")
        sys.exit(0)

if __name__ == '__main__':
    folder_name = select_input()
