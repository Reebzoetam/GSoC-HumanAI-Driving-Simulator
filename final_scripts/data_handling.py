import os
import tempfile
import ffmpeg
import whisper
from transformers import pipeline
import pandas as pd
from pydub import AudioSegment, silence

model = whisper.load_model('medium')
MAX_SEGMENT_DURATION = 5.0
sentiment_pipeline = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment")

def extract_audio(video_path, output_dir):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "audios")
    os.makedirs(output_dir, exist_ok=True)
    
    audio_path = os.path.join(output_dir, os.path.basename(video_path).replace('.mp4', '.wav'))
    ffmpeg.input(video_path).output(audio_path, format='wav', acodec='pcm_s16le', ar='16000').run(overwrite_output=True)
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


def process_video(video_file):
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = extract_audio(video_file, temp_dir)
        chunks, timestamps = segment_audio(audio_path)
        transcript = transcribe_audio(chunks, timestamps)
        sentiment_data = analyze_segment_sentiment(transcript)
    for data in sentiment_data:
        print(data)
        print(f"At {data['timestamp']:.2f}s: {data['text']}")
        print(f"Sentiment: {data['sentiment']} (Confidence: {data['score']:.2f})")
    df = convert_dataframe(sentiment_data, video_file)
    print(df)
    return transcript

def segment_audio(audio_path):

    MAX_CHUNK_LENGTH = 5 * 1000
    overlap = 0

    audio = AudioSegment.from_wav(audio_path)
    nonsilent_ranges = silence.detect_nonsilent(
    audio,
    min_silence_len=500, 
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

def analyze_segment_sentiment(transcript_dict):
    print("Analyzing sentiment...")
    print(len(transcript_dict['segments']))
    print(len(transcript_dict['start']))
    sentiment_results = []

    conversion = {
        "LABEL_0": "negative",
        "LABEL_1": "neutral",
        "LABEL_2": "positive"
    }

    for i in range(len(transcript_dict['start'])):
        start_time = transcript_dict['start'][i]
        text = transcript_dict['segments'][i]
        sentiment = sentiment_pipeline(text)[0]
        sentiment_results.append({
            "timestamp": start_time,
            "text": text,
            "sentiment": conversion.get(sentiment['label'], "unknown"),  # 0 = negative, 1 = neutral, 2 = positive
            "score": sentiment['score']  
        })

    return sentiment_results

def convert_dataframe(sentiment_results, video_name, first_sentiment_results=None):
    print("Converting to DataFrame...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_folder = os.path.join(script_dir, "transcripts")
    os.makedirs(results_folder, exist_ok=True)

    #consider also: videos that are longer than the initial max length of the first video
    if first_sentiment_results:
        first_timestamps = list(first_sentiment_results.keys())

        for entry in sentiment_results:
            closest_timestamp = min(first_timestamps, key=lambda t: abs(t - entry["timestamp"]))
            entry["timestamp"] = closest_timestamp 
    else:
        #taking name of first video OR first 4 and last 4 charas.. ayways
        csv_filename = os.path.join(results_folder, f"{video_name[-8:-4]}_analysis.csv")

    df = pd.DataFrame(sentiment_results)
    
    if os.path.exists(csv_filename):
        df.to_csv(csv_filename, mode='a', header=False, index=False) 
    else:
        df.to_csv(csv_filename, index=False)


def process_multiple_videos(video_folder):
    transcripts = {}
    try:
        video_files = [f for f in os.listdir(video_folder) if f.endswith('.mp4')]
            
        for file in video_files:
            video_path = os.path.join(video_folder, file)
            try:
                transcripts[file] = process_video(video_path)
                print(f"Processed: {file}")
            except Exception as e:
                print(f"Error processing {file}: {e}")
                    
    except Exception as e:
        print(f"Error accessing the folder {video_folder}: {e}")
            
    return transcripts

if __name__ == '__main__':
    # insert function to convert file to string of video.mp4 files eventually
    #rename as needed
    folder_name = "video_files"
    script_dir = os.path.dirname(os.path.abspath(__file__))
    folder_path = os.path.join(script_dir, folder_name)
    transcripts = process_multiple_videos(folder_path)
