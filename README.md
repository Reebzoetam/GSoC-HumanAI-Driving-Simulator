# GSoC-HumanAI-Driving-Simulator

## A brief introduction

This github respository serves as the deliverables for the Data Handling & Analysis and Data Understanding & Manipulation tasks assigned by HumanAI as a screener for their Communication Analysis Tool for Human-AI Interaction Driving Simulator Experiments – Screening Test Google Summer of Code Project.

The first file, [data_handling.py](/final_scripts/data_handling.py), takes in a folder of .mp4 video files and converts them into a csv file containing timestamped transcriptions with sentiment values of positive, negative, or neutral attached to each phrase.

The second file, [data_manipulation.py](/final_scripts/data_manipulation.py), takes in the previously generated csv file, using its timestamp, text, sentiment, and confidence score values to associate word count and sentiment value with time buckets of 5 seconds (ex: 00:05-00:10). This data was then graphed in two ways. 

plot_histogram() generates a histogram of the number of words/strings that appear within each 5 second time bucket, as visualised below using data from an example .mp4 video. ![Saved histogram plot of word counts at each time bucket](/results/plots/histogram_plot.png)

plot_sentiment() generates a stacked bar graph of the number of sentiment counts of each type (green = positive, grey = neutral, red = negative) within each 5 second time bucket, as visualised using the plot below. The intensity of each bar colour shows the confidence score of the sentiment analysis obtained from the model.
![Saved stacked bar graph plot of sentiment counts at each time bucket](/results/plots/sentiment_plot.png)

## How to run the files

1) Ensure you have all necessary libaries and dependencies before running the code in this repository. The required libraries can be found [here](/requirements.txt).
2) Download data_handling.py and data_manipulation.py from the [final_scripts folder](/final_scripts/) into the same directory. 
3) Download/copy the folder of .mp4 videos you want to process into the same directory as data_handling.py and data_manipulation.py. You should name the folder "video_files" if you don't want to directly edit the 'folder_name' variable in data_handling.py. Otherwise, feel free to change 'folder_name' into whatever your folder is already named.
4) Run data_handling.py first to obtain a csv file from your folder of .mp4 videos with all the necessary data to run data_manipulation.py. This csv file will be named after the last 4 characters of your first video file and end with "_analysis.csv". This will appear in a newly generated folder in the same directory called "transcripts". data_handling.py also automatically prints the contents of the transcript and the timestamps into the output for your convenience. .wav files for each corresponding .mp4 video file will also appear in a newly generated folder in the same directory called "audios".
5) Run data_manipulation.py. This will take your newly generated csv file and create another csv file in the "transcripts" folder containing the time_bucket, word_count... negative_conf values. The histogram and stacked bar chart plots mentioned previously will also appear as a pop up.

## Repository documentation
This was built in python using ffmpeg for video to audio conversion, openapi's whisper for audio transcription, huggingface's transformer model for sentiment analysis, and matplotlib for data visualisation. More details about the various libraries and apis used can be found in the requirements.txt file [here](/requirements.txt).

All information on testing can be found in the testing folder. In particular, the testing documentation can be found 
[here](/testing/testing_docu.txt).
