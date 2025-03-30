import pytest
from transcribe import transcript_dict, MAX_SEGMENT_DURATION
from sentiment import analyze_segment_sentiment
import os
import pandas as pd

#test for segments < 5s requirement
def test_segment_lengths():
    for start_time, text in sorted(transcript_dict.items()):
        segment_keys = sorted(transcript_dict.keys())
        index = segment_keys.index(start_time)
        end_time = segment_keys[index + 1] if index + 1 < len(segment_keys) else None
        
        if end_time:
            duration = end_time - start_time
        else:
            duration = MAX_SEGMENT_DURATION
        
        assert duration <= MAX_SEGMENT_DURATION, f"Segment from {start_time:.2f}s exceeds max duration ({duration:.2f}s)"

# test for correct csv output
def test_analyze_segment_sentiment(sample_transcript):
    video_name = "test_video"
    csv_filename = f"/Users/rebeccatam/Downloads/GSoC/transcripts/{video_name}.csv"

    df = analyze_segment_sentiment(sample_transcript, video_name)

    assert os.path.exists(csv_filename), f"CSV file {csv_filename} was not created!"

    saved_df = pd.read_csv(csv_filename)

    expected_columns = ["start_time", "text", "sentiment", "score"]
    assert list(saved_df.columns) == expected_columns, "CSV columns do not match expected format!"

    assert saved_df["start_time"].dtype in [float, int], "start_time should be a float or int!"
    assert saved_df["text"].dtype == object, "text should be a string!"
    assert saved_df["score"].dtype in [float, int], "score should be a float or int!"

if __name__ == "__main__":
    pytest.main()
