# GSoC-HumanAI-Driving-Simulator

## A Brief Introduction

This github respository serves as the deliverables for the Data Handling & Analysis and Data Understanding & Manipulation tasks assigned by HumanAI as a screener for their Communication Analysis Tool for Human-AI Interaction Driving Simulator Experiments – Screening Test Google Summer of Code Project.

The first file, [data_handling.py](/final_scripts/data_handling.py), takes in a folder of .mp4 video files, converts them into a .wav audio file, chunks them using pydub's AudioSegment based on periods of silence in the audio, and converts the chunks into a csv file containing timestamped transcriptions with sentiment values of positive, negative, or neutral attached to each phrase. These sentiment values also had a 'score' attached to them, depending on how intensely positive/negative/neutral the sentiment was.

The second file, [data_manipulation.py](/final_scripts/data_manipulation.py), takes in the previously generated csv file, using its timestamp, text, sentiment, and confidence score values to associate word count and sentiment value with time buckets of 5 seconds (ex: 00:05-00:10). This data was then graphed in two ways:

1) plot_histogram() generates a histogram of the number of words/strings that appear within each 5 second time bucket, as visualised below using data from an example .mp4 video. ![Saved histogram plot of word counts at each time bucket](/results/plots/histogram_plot2.png)

2) plot_sentiment() generates a stacked bar graph of the number of sentiment counts of each type (green = positive, grey = neutral, red = negative) within each 5 second time bucket, as visualised using the plot below. The intensity of each bar colour shows the intensity of the sentiment analysis obtained from the model. For example, if a bar was green, the closer the chunk the bar was representing to an intense positive sentiment, the more intensely green the bar would be, as seen on the gradient scales to the right of this image.
![Saved stacked bar graph plot of sentiment counts at each time bucket](/results/plots/sentiment_plot2.png)

## QoL features
- Upon running the code, a simple GUI will pop up to select input files/folder and output folder directly from your machine
- Multi-file processing for multiple files in the same folder
- Multiple file format conversion to .wav files that include .mp4, .mov, .avi, .mp3, .wav, .avi, etc.

## How to Run the Files

1) Ensure you have all necessary libaries and dependencies before running the code in this repository. The required libraries can be found [here](/requirements.txt).
2) Download data_handling.py and data_manipulation.py from the [final_scripts folder](/final_scripts/) into the same directory.
3) Ensure that your desired folders to contain your outputs exist.
4) Run data_handling.py first to obtain a csv file from your folder of .mp4 videos with all the necessary data to run data_manipulation.py. You will be prompted to select an input file/folder and an output folder where your csv file will be placed. This csv file will be named after the last 4 characters of your first video file and end with "_analysis.csv". data_handling.py also automatically prints the contents of the transcript and the timestamps into the output for your convenience. 
5) Run data_manipulation.py. You will be prompted to select an input csv file and an output folder where your csv file and plots will be placed. The new csv file will contain time_bucket, word_count... negative_conf values. The histogram and stacked bar chart plots mentioned previously will also appear as a pop up.

## Challenges

The primary challenge presented in this screener was ensuring proper data manipulation in data_manipulation.py. A great number of hours were needed to debug the dictionary/list parsing steps and ensure that the proper indexed values were passed on in proper order to the dataframe. There were many print statements that had to be used to diagnose the issue as the number of entries in data[time_bucket] and data[positive_count] were often misaligned, leading to an inability to use dataframes. I hope to be able to program additional diagnostic tests and scripts to diganose data misalignments in the future for cases like these to further improve my debugging capabilities.

Additionally, ensuring that the data was presented in a readable plot was also a challenge. I have only primarily used matplotlib to plot data directly, so exploring new matplotlib features such as the colour map for the first time presented an interesting learning curve. I hope to be able to improve my skills with programming data presentation in the future.

Finally, a great deal of time was also invested into fine tuning the parameters for chunking based on silence, whisper's transcription, and sentiment analysis. This involved tweaking multiple parameters such as the silence threshold, silence length, providing a prompt for whisper, changing the threshold for no_speech_prob with whisper's transcription, and changing the threshold values for positive/negative/neutral sentiment identifications based on previous results.

## Repository Documentation
This was built in python using ffmpeg for video to audio conversion, pydub for chunking based on silence, openapi's whisper for audio transcription, Vader Sentiment for more fine-grained sentiment analysis, and matplotlib for data visualisation. More details about the various libraries and apis used can be found in the requirements.txt file [here](/requirements.txt).

All information on testing can be found in the testing folder. In particular, the testing documentation can be found 
[here](/testing/testing_docu.txt).

You can view expected output examples, such as the transcript and plots in the [results folder](/results).
