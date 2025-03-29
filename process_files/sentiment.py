import os
import tempfile
import ffmpeg
import whisper
from transformers import pipeline
import tensorflow as tf
import keras
import pandas as pd

model = whisper.load_model('medium')
transcript_dict = {}
MAX_SEGMENT_DURATION = 5.0
sentiment_pipeline = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment")

def extract_audio(video_path, output_dir):
    audio_path = os.path.join(output_dir, os.path.basename(video_path).replace('.mp4', '.wav'))
    ffmpeg.input(video_path).output(audio_path, format='wav', acodec='pcm_s16le', ar='16000').run(overwrite_output=True)
    return audio_path

def transcribe_audio(audio_path):
    result = model.transcribe(audio_path, word_timestamps=True)
    return result

def process_video(video_file):
    with tempfile.TemporaryDirectory() as temp_dir:
        audio_path = extract_audio(video_file, temp_dir)
        transcript = transcribe_audio(audio_path)
        store_segments(transcript)
        segment_audio(audio_path, temp_dir)
        sentiment_data = analyze_segment_sentiment(transcript_dict)
    for timestamp, data in sentiment_data.items():
        print(f"At {timestamp:.2f}s: {data['text']}")
        print(f"  Sentiment: {data['sentiment']} (Confidence: {data['score']:.2f})")
    df = convert_dataframe(sentiment_data, video_file)
    print(df)
    return transcript

# since whisper returns the transcribed text by segments already, we will just store them in a dictionary and
# segment the audio based on those timestamps
def store_segments(transcript):
    global transcript_dict
    print("Storing segments...")
    for segment in transcript['segments']:
        start_time = segment['start']
        end_time = segment['end']
        text = segment['text']
        words = segment['words']

        if end_time - start_time <= MAX_SEGMENT_DURATION:
            transcript_dict[start_time] = text
        else:
            sub_text = []
            sub_start = start_time

            for word in words:
                sub_text.append(word['word'])
                if word['end'] - sub_start >= MAX_SEGMENT_DURATION:
                    transcript_dict[sub_start] = " ".join(sub_text)
                    sub_text = []
                    sub_start = word['end']
            if sub_text:
                transcript_dict[sub_start] = " ".join(sub_text)
    print(transcript_dict)

def segment_audio(audio_path, output_dir):
    global transcript_dict
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir, "audio_chunks")
    os.makedirs(output_dir, exist_ok=True)  # Ensure the directory exists
    
    audio_filename = os.path.basename(audio_path).replace('.wav', '')
    start_times = sorted(transcript_dict.keys())  # Sorted list of segment start times
    
    for i, start_time in enumerate(start_times):
        end_time = start_times[i + 1] if i + 1 < len(start_times) else None
        output_file = os.path.join(output_dir, f"{audio_filename}_segment_{i + 1}.wav")
        
        if end_time:
            ffmpeg.input(audio_path, ss=start_time, to=end_time).output(output_file, format='wav').run(overwrite_output=True)
        else:
            ffmpeg.input(audio_path, ss=start_time).output(output_file, format='wav').run(overwrite_output=True)
        
        print(f"Saved segment: {output_file} [{start_time:.2f}s - {end_time if end_time else 'end'}s]")

def analyze_segment_sentiment(transcript_dict):
    sentiment_results = {}

    for start_time, text in transcript_dict.items():
        sentiment = sentiment_pipeline(text)[0]
        sentiment_results[start_time] = {
            "text": text,
            "sentiment": sentiment['label'],  # 0 = negative, 1 = neutral, 2 = positive
            "score": sentiment['score']  
        }

    return sentiment_results

def convert_dataframe(sentiment_results, video_name, first_sentiment_results=None):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_folder = os.path.join(script_dir, "transcripts")
    os.makedirs(results_folder, exist_ok=True)

    #rename csv file so that it takes the last 4 digits of the first video name and the last 4 digits of the last video name
    csv_filename = os.path.join(results_folder, f"{video_name}_analysis.csv")
    # convert sentiment_results to a list of dictionaries so that the row follows per timestamp 
    data = [
        {
            "timestamp": timestamp,
            "text": details["text"],
            "sentiment": details["sentiment"],
            "score": details["score"]
        }
        for timestamp, details in sentiment_results.items()
    ]

    if first_sentiment_results:
        first_timestamps = list(first_sentiment_results.keys())

        for entry in data:
            closest_timestamp = min(first_timestamps, key=lambda t: abs(t - entry["timestamp"]))
            entry["timestamp"] = closest_timestamp 

    df = pd.DataFrame(data)
    
    if os.path.exists(csv_filename):
        df.to_csv(csv_filename, mode='a', header=False, index=False) 
    else:
        df.to_csv(csv_filename, index=False)


def process_multiple_videos(video_urls):
    transcripts = {}
    for file in video_files:
        try:
            transcripts[file] = process_video(file)
            print(f"Processed: {file}")
        except Exception as e:
            print(f"Error processing {file}: {e}")
    return transcripts

if __name__ == '__main__':
    # insert function to convert file to string of video.mp4 files eventually
    video_files = ['/Users/rebeccatam/Downloads/GSoC/Experimenter_CREW_999_1_All_1731617801.mp4']
    transcripts = process_multiple_videos(video_files)
    #for link, transcript in transcripts.items():
        #print(f"Transcript for {link}:")
        #print(transcript)

#TODO: generate multiple file reading system to process multiple files at once
# ensure that file paths are compatible with any users' system
# get rid of the fp16 warning omg
# ensure that documentation is written for all libraries imported
