import pandas as pd
import os
import matplotlib.pyplot as plt

def create_time_buckets(transcript_path):
    df = pd.read_csv(transcript_path)
    df = df.dropna(subset=["timestamp"])
    df["text"] = df["text"].str.strip()
    # filter out empty text rows
    df = df[df["text"].notna() & (df["text"] != '')]
    max_time = int(df.iloc[-1]["timestamp"])

    time_buckets = {}
    for i in range(0, max_time + 5, 5):
        bucket_start = i
        bucket_end = i + 5
        bucket_minutes = bucket_start // 60
        remaining_seconds1 = bucket_start % 60
        bucket_minutes2 = bucket_end // 60
        remaining_seconds2 = bucket_end % 60

        time_buckets[f"{bucket_minutes:02d}:{remaining_seconds1:02d}-{bucket_minutes2:02d}:{remaining_seconds2:02d}"] = []
        for timestamp, details in df.iloc[1:].iterrows():
            if bucket_start <= details["timestamp"] < bucket_end:
                time_buckets[f"{bucket_minutes:02d}:{remaining_seconds1:02d}-{bucket_minutes2:02d}:{remaining_seconds2:02d}"].append(details["text"])
            elif timestamp > bucket_end:
                break
    print(time_buckets)

    data = {
        "time_bucket": [],
        "word_count": []
    }
    for bucket, phrases in time_buckets.items():
        data["time_bucket"].append(bucket)
        word_count = sum(len(phrase.split()) for phrase in phrases)
        data["word_count"].append(word_count)

    print(time_buckets.items())

    df = pd.DataFrame(data)

    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_folder = os.path.join(script_dir, "transcripts")
    os.makedirs(results_folder, exist_ok=True)

    csv_filename = os.path.join(results_folder, f"{transcript_path}_histogram.csv")
    df.to_csv(csv_filename, index=False)
    print(f"Saved time bucket data to {csv_filename}")

if __name__ == '__main__':
    transcript_path = '/Users/rebeccatam/Downloads/GSoC/transcripts/7801_analysis.csv'
    create_time_buckets(transcript_path)
