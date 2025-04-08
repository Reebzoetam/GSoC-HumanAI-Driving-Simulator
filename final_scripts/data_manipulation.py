import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import tkinter as tk
from tkinter import filedialog, messagebox
import sys

def create_time_buckets(transcript_path, output_path):
    df = pd.read_csv(transcript_path)
    df = df.dropna(subset=["timestamp"])
    df["text"] = df["text"].str.strip()
    # filter out empty text rows
    df = df[df["text"].notna() & (df["text"] != '')]
    df = df.sort_values(by="timestamp").reset_index(drop=True)
    max_time = int(df.iloc[-1]["timestamp"])

    data = {
        "time_bucket": [],
        "word_count": [],
        #padding 
        "positive_count": [0],
        "neutral_count": [0],
        "negative_count": [0], 
        "positive_conf": [0],
        "neutral_conf": [0],
        "negative_conf": [0], 
    }

    time_buckets = {}
    for i in range(0, max_time + 5, 5):
        bucket_start = i
        bucket_end = i + 5
        bucket_minutes = bucket_start // 60
        remaining_seconds1 = bucket_start % 60
        bucket_minutes2 = bucket_end // 60
        remaining_seconds2 = bucket_end % 60

        time_buckets[f"{bucket_minutes:02d}:{remaining_seconds1:02d}-{bucket_minutes2:02d}:{remaining_seconds2:02d}"] = []

        positive_count = 0
        neutral_count = 0
        negative_count = 0
        positive_conf = 0
        neutral_conf = 0
        negative_conf = 0

        #NOTE TO FUTURE REBECCA: timestamp represents the row number NOT THE TIME NOT THE TIME I WASTED 2 HOURS ON THIS
        for timestamp, details in df.iterrows():
            if bucket_start <= details["timestamp"] < bucket_end:
                time_buckets[f"{bucket_minutes:02d}:{remaining_seconds1:02d}-{bucket_minutes2:02d}:{remaining_seconds2:02d}"].append(details["text"])

                if details["sentiment"] == "positive":
                    positive_count += 1
                    positive_conf += details["score"]
                elif details["sentiment"] == "neutral":
                    neutral_count += 1
                    neutral_conf += details["score"]
                elif details["sentiment"] == "negative":
                    negative_count += 1
                    negative_conf += details["score"]

            elif details["timestamp"] >= bucket_end:
                if positive_count + neutral_count + negative_count != 0:
                    positive_conf /= (positive_count + neutral_count + negative_count)
                    neutral_conf /= (positive_count + neutral_count + negative_count)
                    negative_conf /= (positive_count + neutral_count + negative_count)

                data["positive_count"].append(positive_count)
                data["neutral_count"].append(neutral_count)
                data["negative_count"].append(negative_count)
                data["positive_conf"].append(positive_conf)
                data["neutral_conf"].append(neutral_conf)
                data["negative_conf"].append(negative_conf)

                positive_count = 0
                neutral_count = 0
                negative_count = 0
                positive_conf = 0
                neutral_conf = 0
                negative_conf = 0
                break
    
    for bucket, phrases in time_buckets.items():
        data["time_bucket"].append(bucket)
        word_count = sum(len(phrase.split()) for phrase in phrases)
        data["word_count"].append(word_count)

    data["positive_count"].append(0)
    data["neutral_count"].append(0)
    data["negative_count"].append(0)
    data["positive_conf"].append(0)
    data["neutral_conf"].append(0)
    data["negative_conf"].append(0)

    df = pd.DataFrame(data)

    csv_filename = os.path.join(output_path, f"{transcript_path}_histogram.csv")
    df.to_csv(csv_filename, index=False)
    print(f"Saved time bucket data to {csv_filename}")
    return csv_filename

def plot_histogram(csv_filename, output_path):
    df = pd.read_csv(csv_filename)
    plt.figure(figsize=(20, 6))
    plt.bar(df["time_bucket"], df["word_count"])
    plt.xlabel("Time Buckets")
    plt.ylabel("Word Count")
    plt.title("Word Count per Time Bucket")
    plt.xticks(rotation=45)
    plt.tight_layout()

    plt.savefig(os.path.join(output_path, "histogram_plot.png"), format="png", bbox_inches="tight")

    plt.show()

def plot_sentiment(csv_filename, output_path):
    df = pd.read_csv(csv_filename)
    df['time_bucket'] = df['time_bucket'].astype(str)

    def normalize(series):
        return 0.2 + 0.8 * (series - series.min()) / (series.max() - series.min() + 1e-9)

    df['positive_alpha'] = normalize(df['positive_conf'])
    df['neutral_alpha'] = normalize(df['neutral_conf'])
    df['negative_alpha'] = normalize(df['negative_conf'])

    plt.figure(figsize=(14, 7))

    cmap_positive = cm.get_cmap('Greens')
    cmap_negative = cm.get_cmap('Reds')
    cmap_neutral = cm.get_cmap('Greys')

    for i, row in df.iterrows():
        plt.bar(row['time_bucket'], row['positive_count'], 
                color=cmap_positive(row['positive_alpha']), alpha=row['positive_alpha'], label="Positive" if i == 0 else "")
        
        plt.bar(row['time_bucket'], row['negative_count'], bottom=row['positive_count'], 
                color=cmap_negative(row['negative_alpha']), alpha=row['negative_alpha'], label="Negative" if i == 0 else "")
        
        plt.bar(row['time_bucket'], row['neutral_count'], 
                bottom=row['positive_count'] + row['negative_count'], 
                color=cmap_neutral(row['neutral_alpha']), alpha=row['neutral_alpha'], label="Neutral" if i == 0 else "")

    plt.xlabel("Time Buckets")
    plt.ylabel("Sentiment Count")
    plt.title("Sentiment Classification per Time Bucket")

    plt.xticks(rotation=45)

    cbar_ax = plt.gca().inset_axes([1.02, 0.2, 0.03, 0.4])
    sm = cm.ScalarMappable(cmap=cmap_positive, norm=mcolors.Normalize(vmin=0.2, vmax=1))
    plt.colorbar(sm, cax=cbar_ax, label="", ticks=[0.2, 0.6, 1])

    cbar_ax2 = plt.gca().inset_axes([1.07, 0.2, 0.03, 0.4])
    sm2 = cm.ScalarMappable(cmap=cmap_negative, norm=mcolors.Normalize(vmin=0.2, vmax=1))
    plt.colorbar(sm2, cax=cbar_ax2, label="", ticks=[0.2, 0.6, 1])

    cbar_ax3 = plt.gca().inset_axes([1.12, 0.2, 0.03, 0.4])
    sm3 = cm.ScalarMappable(cmap=cmap_neutral, norm=mcolors.Normalize(vmin=0.2, vmax=1))
    plt.colorbar(sm3, cax=cbar_ax3, label="Intensity", ticks=[0.2, 0.6, 1])

    plt.tight_layout()

    plt.savefig(os.path.join(output_path, "sentiment_plot.png"), format="png", bbox_inches="tight")
    plt.show()

def select_input():
    root = tk.Tk()
    root.withdraw()

    messagebox.showinfo("Input File", "Select a csv file to process.")
    path = filedialog.askopenfilename(title="Select a csv file to process")

    messagebox.showinfo("Output Folder", "Now select a folder where you want to save the output files.")
    output_path = filedialog.askdirectory(title="Select Output Folder")
    if not output_path:
        print("No output folder selected.")
        sys.exit(0)

    if path:
        data_points = create_time_buckets(path, output_path)
        plot_histogram(data_points, output_path)
        plot_sentiment(data_points, output_path)
    else:
        print("No selection made.")
        sys.exit(0)

if __name__ == '__main__':
    select_input()
