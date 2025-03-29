import os
import tempfile
import ffmpeg
import whisper

model = whisper.load_model('medium')

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
        print_segments(transcript)
    return transcript

def print_segments(transcript):
    print('printing segments')
    for segments in transcript['segments']:
        print(f"[{segments['start']:.2f}s - {segments['end']:.2f}s]: {segments['text']}")

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
    for link, transcript in transcripts.items():
        print(f"Transcript for {link}:")
        print(transcript)
