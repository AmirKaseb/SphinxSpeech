
# SphinxSpeech : Speech Recognition Model for Egyptian Dialect by Metanoia Labs Team

## Table of Contents

- [Overview](#overview)
  - [Framework Used](#framework-used)
  - [Why Nemo?](#Why-Nemo-?)
- [Training](#training)
  - [1. Dataset Preprocessing](#1-Dataset-Preprocessing)
  - [2. Model Training](#2-Model-Training)
    - [Why Subword Tokenizer?](#Why-Subword-Tokenizer)
    - [Building a custom subword tokenizer](#building-a-custom-subword-tokenizer)
    - [Why We Chose MFCC for Feature Extraction](#why-we-chose-mfcc-for-feature-extraction)
    - [Model Architecture](#Model-Architecture)
      - [Citrinet Model Overview](#Citrinet-Model-Overview)
      - [Model parameters tuning](#model-parameters-tuning)
    - [Why We Chose BPE?](#Why-We-Chose-BPE-?)
    - [Model Improvements](#Model-Imporvement)
  - [3. Model Interference & Deployment](#3-model-interference--deployment)
- [Phase 2: Speaker Recognition & Diarization](#phase-2-speaker-recognition--diarization)
- [Contributors](#contributors)

## Overview

Recognizing speech in Egyptian Arabic presents unique challenges due to its specific sounds, words, and grammar compared to standard Arabic. Our project, SphinxSpeech, aims to build a speech recognition model specifically for Egyptian Arabic, developed for the **MTC AIC-2 Competition**.

![Example Image](https://i.imgur.com/vMtlVrO.png)


### Framework Used 
To streamline development and improve performance, we explored various frameworks which were  ( **ESPnet, DeepSpeech, SpeechBrain, Kaldi and Nemo** ) instead of building a neural network from scratch, which can be time-consuming. After careful consideration, we selected **NVIDIA NeMo**.

**NVIDIA NeMo Framework** is a scalable and cloud-native generative AI framework built for researchers and PyTorch developers working on Large Language Models (LLMs), Multimodal Models (MMs), Automatic Speech Recognition (ASR), Text to Speech (TTS), and Computer Vision (CV) domains. It is designed to help you efficiently create, customize, and deploy new generative AI models.

![Example Image](https://docscontent.nvidia.com/dims4/default/9a9e8bc/2147483647/strip/true/crop/1540x867+10+0/resize/1000x563!/quality/90/?url=https%3A%2F%2Fk3-prod-nvidia-docs.s3.us-west-2.amazonaws.com%2Fbrightspot%2F87%2F69%2F41a21a6b4d95aee54d030b998733%2Fnemo-overview.png)


### Why Nemo ?
We chose Nemo for several compelling reasons:


-   **Arabic Support:** NeMo supports Arabic well, crucial for our tasks.
-   **Clear Documentation:** NeMo offers excellent tutorials and documentation, which helped us prototype and develop quickly.
-   **Active Community:** It has a strong community for support and troubleshooting.
-   **User-Friendly:** NeMo's user interface and notebooks allowed us to integrate and deploy models efficiently.

These factors made NeMo the best choice for SphinxSpeech, ensuring high-performance speech recognition tailored for Egyptian Arabic.

## Training 
The Training Process are divided into 3 Main Steps :-

1. **Dataset  Preprocessing** 
2. **Model Training**
3. **Interference & Deployment of the Model** 

### 1- Dataset Preprocessing
We Used  the dataset provided by the competition ( MTC-ASR-Dataset-16K ) which was sampled at 16K :-
- 100 hours of training data  
- 3 hours of adapt data (1.5 hours Clean and 1.5 hours noisy )
````Linux
MTC-ASR-Dataset-16K
 |
 ├── train 
 │   ├── train_sample_0.wav
 │   ├── train_sample_1.wav
 │   └── ...
 ├── adapt
 │   ├── adapt_sample_clean_0.wav
 │   ├── adapt_sample_clean_1.wav
 │   └── ...
 ├── train.csv
 ├── adapt.csv
````
 The CSV  contains Two Columns ( Audio , Transcript )  separated by comma , In order to make data fit with Nemo Framework   we need to make manifests that is used by Nemo Architecture.
 ````csv
 audio,transcript
adapt_sample_0_clean,شوفلنا المشوار ده يا حج
adapt_sample_1_clean,لأ للأسف دكتوره واحده بس بتعمل العمليه ديت عندنا في المحافظه
adapt_sample_2_clean,والراجل تبصله يعني إبن زمنه
adapt_sample_3_clean,و أنت كيف عرفته أبترل يا عمي
adapt_sample_4_clean,ميعرفوش حاجه عن السوبر أه غير إنه لب
..........,.........etc
 ````
 
 These manifests will contain metadata for each audio file, formatted such that each line corresponds to one audio sample. Each line must include the path to the audio file, its duration, and the corresponding transcript. Here's an example:
````json
{"audio_filepath": "path/to/audio.wav", "duration": 3.45, "text": "this is a nemo tutorial"}
````
We Have Created a Scipt to convert format from MTC-Dataset CSV Format to Nemo JSON Dataset format
````python
import csv
import json
import os
import wave
# Define input and output file paths
input_csv =  'adapt.csv'
audio_folder =  r'adapt'
output_json =  'formatted_train_dataset.json'
# Function to get the duration of a WAV file
def  get_wav_duration(wav_path):
try:
with wave.open(wav_path, 'r') as wav_file:
frames = wav_file.getnframes()
rate = wav_file.getframerate()
duration = frames /  float(rate)
return duration
except  Exception  as e:
print(f"Error reading {wav_path}: {e}")
return  None

# List all files in the audio folder
all_files = os.listdir(audio_folder)
print(f"Files in '{audio_folder}': {all_files}")

# Initialize a list to hold the formatted data
formatted_data = []
print("Starting to process the CSV file...")

# Read the CSV file and process each row
with  open(input_csv, mode='r', encoding='utf-8') as csvfile:
csvreader = csv.DictReader(csvfile)
row_count =  sum(1  for row in csvreader) # Get the total number of rows
csvfile.seek(0) # Reset the reader to the beginning of the file
next(csvreader) # Skip the header row
for i, row in  enumerate(csvreader, start=1):

# Get the audio filename and transcript from the CSV
audio_filename = row['audio'].strip() # Remove any leading/trailing whitespace
transcript = row['transcript']

# Append .WAV extension if missing
if  not audio_filename.lower().endswith('.wav'):
audio_filename +=  '.wav'

# Construct the full path to the audio file
audio_filepath = os.path.join(audio_folder, audio_filename)

# Debugging: Print the full path of the audio file
print(f"Processing file {audio_filepath}")

# Check if the file exists
if  not os.path.isfile(audio_filepath):
print(f"File not found: {audio_filepath}")
continue

# Get the duration of the audio file
duration = get_wav_duration(audio_filepath)
if duration is  None:
print(f"Skipping file {audio_filepath} due to error in reading duration.")
continue

# Create a dictionary in the desired format
formatted_entry = {
"audio_filepath": audio_filepath,
"duration": duration,
"text": transcript
}

# Add the formatted entry to the list
formatted_data.append(formatted_entry)
print(f"Processed {i}/{row_count} rows")

# Write the formatted data to a JSON file
try:
with  open(output_json, mode='w', encoding='utf-8') as jsonfile:
json.dump(formatted_data, jsonfile, indent=4, ensure_ascii=False)
print(f"Formatted data has been saved to {output_json}")
except  Exception  as e:
print(f"Error writing to {output_json}: {e}")
print("Processing complete.")
````
- In the end we divided the train folder into two manifests **final_train.json** and **final_test.json**  with ratio 90% to 10 % , you can find the dataset manifests in folder **manifests**
  
###  2- Model Training :-
The First step include  making a subword tokenizer which  is essential for Automatic Speech Recognition (ASR) ,  Extensive research in Neural Machine Translation and Language Modeling has demonstrated that subword tokenization not only reduces the length of tokenized representations, making sentences shorter and more manageable for models to learn, but also enhances the accuracy of token predictions.

#### Why  Subword Tokenizer ?

We previously emphasized that subword tokenization is essential for Automatic Speech Recognition (ASR), not just a desirable feature. In another tutorials , we used the Connectionist Temporal Classification (CTC) loss function to train the model, but this loss function has some limitations:

1.  **Conditional Independence of Tokens**: Generated tokens are conditionally independent of each other. This means the probability of predicting the character "ا" after "اهل#" is independent of the previous token, making it possible to predict any other token unless the model has future context.
2.  **Sequence Length Constraints**: The length of the generated target sequence must be shorter than that of the source sequence, which can limit the model's effectiveness.

By implementing subword tokenization, we address these limitations, thereby improving the model's performance and prediction accuracy.

####  Building a custom subword tokenizer
- First Downloading Nemo Framework and all libraries required for the training 
````python
"""
You can run either this notebook locally (if you have all the dependencies and a GPU) or on Google Colab.
Instructions for setting up Colab are as follows:
1. Open a new Python 3 notebook.
2. Import this notebook from GitHub (File -> Upload Notebook -> "GITHUB" tab -> copy/paste GitHub URL)
3. Connect to an instance with a GPU (Runtime -> Change runtime type -> select "GPU" for hardware accelerator)
4. Run this cell to set up dependencies.
5. Restart the runtime (Runtime -> Restart Runtime) for any upgraded packages to take effect
NOTE: User is responsible for checking the content of datasets and the applicable licenses and determining if suitable for the intended use.
"""
# Install dependencies
!pip install wget
!apt-get install sox libsndfile1 ffmpeg
!pip install text-unidecode
!pip install matplotlib>=3.3.2
## Install NeMo
BRANCH = 'r2.0.0rc0'
!python -m pip install git+https://github.com/NVIDIA/NeMo.git@$BRANCH#egg=nemo_toolkit[all]
## Grab the config we'll use in this example
!mkdir configs
!wget -P configs/ https://raw.githubusercontent.com/NVIDIA/NeMo/$BRANCH/examples/asr/conf/citrinet/config_bpe.yaml
"""
Remember to restart the runtime for the kernel to pick up any upgraded packages (e.g. matplotlib)!
Alternatively, you can uncomment the exit() below to crash and restart the kernel, in the case
that you want to use the "Run All Cells" (or similar) option.
"""
# exit()
````
- Then Running the tokenizer script 
````python
!python ./scripts/process_asr_text_tokenizer.py \
--manifest="{data_dir}/an4/train_manifest.json" \
--data_root="{data_dir}/tokenizers/an4/" \
--vocab_size=64\
--tokenizer="spe" \
--no_lower_case \
--spe_type="unigram" \
--log
````
The script requires several important arguments:

-   **--manifest or --data_file**: Specify either the path to an ASR manifest file (`--manifest`) or a file with text data on separate lines (`--data_file`). Multiple files can be concatenated using commas.
    
-   **--data_root**: The output directory where the tokenizers will be placed. Subdirectories will be created if they don't exist.
    
-   **--vocab_size**: Size of the tokenizer vocabulary. Larger vocabularies can include whole words but will increase the decoder size proportionally.
    
-   **--tokenizer**: Choose either `spe` (Google SentencePiece) or `wpe` (HuggingFace BERT WordPiece).
    
-   **--no_lower_case**: When enabled, creates separate tokens for upper and lower case characters. By default, text is converted to lower case before tokenization.
    
-   **--spe_type**: Specifies the type of SentencePiece tokenization (unigram, bpe, char, word). Defaults to `bpe`.
    
-   **--spe_character_coverage**: Determines the proportion of the original vocabulary to cover. Default is 1.0 for small vocabularies; 0.9995 is suggested for larger vocabularies (e.g., Japanese, Mandarin).
    
-   **--spe_sample_size**: Use a sampled dataset if it's too large. A negative value (default = -1) uses the entire dataset.
    
-   **--spe_train_extremely_large_corpus**: Use this flag if training on very large datasets to avoid memory issues. It may silently fail if out of RAM.
    
-   **--log**: Enable logging messages.

#### Why We Chose Unigram and SPE Tokenization ?
##### Advantages of Unigram Tokenization:

1.  **Efficiency**: Captures common subword units efficiently, crucial for the nuances of the Egyptian dialect.
2.  **Flexibility**: Balances character and word-level representations for better handling of frequent and rare words.
3.  **Improved Accuracy**: Captures morphological features accurately, enhancing overall recognition performance.


##### Advantages of SentencePiece (SPE):

1.  **Language-Agnostic**: Adapts well to the unique characteristics of Egyptian Arabic.
2.  **End-to-End Tokenization**: Integrates seamlessly with neural network models, enhancing the ASR pipeline.
3.  **Consistency**: Ensures consistent preprocessing, crucial for high accuracy.
4.  **Scalability**: Efficiently handles large datasets, essential for training  ASR models.

Choosing Unigram and SPE leverages their strengths to enhance the accuracy, efficiency, and scalability of SphinxSpeech for Egyptian dialect speech recognition.

- At the end we had our tokinzer in **tokenizer_spe_unigram_v64** Folder with voab file and tokenizer Model.


#### Why We Chose MFCC for Feature Extraction

We go for Mel-Frequency Cepstral Coefficients (MFCC) to maker our Egyptian dialect speech recognition for a few key reasons:

1- **Effective Speech Representation**: MFCC effectively captures essential speech details critical for accurate recognition of Egyptian Arabic.

2- **Noise Robustness**: It handles variations in pronunciation and background noise, ensuring reliable performance in diverse environments.

3- **Efficiency**: MFCC strikes a balance between computational efficiency and quality, making it suitable for real-time speech processing.

4- **Proven Method** : Widely used in speech processing, MFCC is supported by established tools and research, streamlining our development process.

Choosing MFCC aligns with our goal of creating a robust speech recognition system tailored specifically for Egyptian Arabic, ensuring accurate and efficient performance in real-world applications.

![Example Image](https://img1.daumcdn.net/thumb/R1280x0/?scode=mtistory2&fname=https%3A%2F%2Fk.kakaocdn.net%2Fdn%2FTsu71%2FbtqETBgoxsP%2F7rgu73Uyc3isPddR9q1ZOK%2Fimg.png)


#### Model Architecture :-
- We will use a Citrinet model to demonstrate the effectiveness of subword tokenization for both training and inference. Citrinet, similar to QuartzNet in architecture, employs subword tokenization along with 8x subsampling and Squeeze-and-Excitation mechanisms. This combination achieves high transcription accuracy while maintaining efficient inference through non-autoregressive decoding.
#####  Citrinet Model Overview:-

![Example Image](https://docs.nvidia.com/deeplearning/nemo/archives/nemo-100rc1/user-guide/docs/_images/citrinet_vertical.png)

The Citrinet model, which we utilize for Egyptian dialect speech recognition, is designed to achieve high transcription accuracy while maintaining efficient inference. It features several key components:

1.  **Prolog**:
    
    -   The initial convolutional layer with Batch Normalization (BN) and ReLU activation, which processes the input features with a stride of 2 to reduce the temporal dimension.
2.  **Block 1**:
    
    -   A series of convolutional layers, each followed by BN and ReLU activation, repeated `R` times. This block further refines the features extracted from the input.
3.  **Block B**:
    
    -   Another series of convolutional layers, similar to Block 1, but with a stride of 4 for more aggressive downsampling. This block is also repeated `B` times to deepen the network.
4.  **Epilog**:
    
    -   Final convolutional layers with BN and ReLU activation, leading to a 1x1 convolution that adjusts the output channels to match the vocabulary size.
5.  **Squeeze-and-Excitation Mechanism**:
    
    -   Integrated within the blocks to adaptively recalibrate channel-wise feature responses by modeling interdependencies between channels, enhancing the model's representational power.
6.  **CTC (Connectionist Temporal Classification)**:
    
    -   The output layer uses the CTC loss function to align input features with target transcriptions, facilitating training without the need for pre-segmented data.
7.  **Non-Autoregressive Decoding**:
    
    -   Citrinet employs non-autoregressive decoding for efficient inference, allowing it to predict outputs without sequential dependencies.

By leveraging subword tokenization, 8x subsampling, and Squeeze-and-Excitation, Citrinet achieves strong accuracy in transcriptions while ensuring efficient and scalable inference, making it a robust choice for Egyptian dialect ASR tasks.

##### Model parameters tuning 
- After Several Tries , We tried a lot of different combinations of parameters until we reached the best results
````yaml
name: &name "ContextNet5x1"
sample_rate: &sample_rate 16000
repeat: &repeat 1
dropout: &dropout 0.0
separable: &separable true

model:
  train_ds:
    manifest_filepath: ???
    sample_rate: 16000
    batch_size: 64
    trim_silence: True
    max_duration: 16.7
    shuffle: True
    num_workers: 8
    pin_memory: true
    # tarred datasets
    is_tarred: false
    tarred_audio_filepaths: null
    shard_strategy: "scatter"
    shuffle_n: 2048
    # bucketing params
    bucketing_strategy: "synced_randomized"
    bucketing_batch_size: null

  validation_ds:
    manifest_filepath: ???
    sample_rate: 16000
    batch_size: 32
    shuffle: False
    num_workers: 8
    pin_memory: true

  tokenizer:
    dir: ???  # path to directory which contains either tokenizer.model (bpe) or vocab.txt (for wpe)
    type: ???  # Can be either bpe or wpe

  preprocessor:
    _target_: nemo.collections.asr.modules.AudioToMelSpectrogramPreprocessor
    normalize: "per_feature"
    window_size: 0.02
    sample_rate: *sample_rate
    window_stride: 0.01
    window: "hann"
    features: &n_mels 64
    n_fft: 512
    frame_splicing: 1
    dither: 0.00001

  spec_augment:
    _target_: nemo.collections.asr.modules.SpectrogramAugmentation
    rect_freq: 50
    rect_masks: 5
    rect_time: 120

  encoder:
    _target_: nemo.collections.asr.modules.ConvASREncoder
    feat_in: *n_mels
    activation: relu
    conv_mask: true

    jasper:
      - filters: 128
        repeat: 1
        kernel: [11]
        stride: [1]
        dilation: [1]
        dropout: *dropout
        residual: true
        separable: *separable
        se: true
        se_context_size: -1

      - filters: 256
        repeat: *repeat
        kernel: [13]
        stride: [1]
        dilation: [1]
        dropout: *dropout
        residual: true
        separable: *separable
        se: true
        se_context_size: -1

      - filters: 256
        repeat: *repeat
        kernel: [15]
        stride: [1]
        dilation: [1]
        dropout: *dropout
        residual: true
        separable: *separable
        se: true
        se_context_size: -1

      - filters: 256
        repeat: *repeat
        kernel: [17]
        stride: [1]
        dilation: [1]
        dropout: *dropout
        residual: true
        separable: *separable
        se: true
        se_context_size: -1

      - filters: 256
        repeat: *repeat
        kernel: [19]
        stride: [1]
        dilation: [1]
        dropout: *dropout
        residual: true
        separable: *separable
        se: true
        se_context_size: -1

      - filters: 256
        repeat: 1
        kernel: [21]
        stride: [1]
        dilation: [1]
        dropout: 0.0
        residual: false
        separable: *separable
        se: true
        se_context_size: -1

      - filters: &enc_feat_out 1024
        repeat: 1
        kernel: [1]
        stride: [1]
        dilation: [1]
````
# Phase 2: Speaker Recognition & Diarization

## Table of Contents

1. [Introduction](#introduction)
2. [Pipeline Configuration](#pipeline-configuration)
   - [Pipeline Name](#pipeline-name)
   - [Key Components](#key-components)
     - [Clustering Method](#clustering-method)
     - [Embedding Model](#embedding-model)
     - [Segmentation Model](#segmentation-model)
3. [Inference Final Script](#Inference-Final-Script)

## Introduction

The PyAnnote diarization pipeline is an advanced system designed for segmenting audio recordings and identifying different speakers. This document provides an overview of the pipeline’s configuration, highlighting the technical decisions that enhance its performance.

## Pipeline Configuration

### Pipeline Name

![Example Image](https://i.imgur.com/shrWXYh.png)


- **Name**: `pyannote.audio.pipelines.SpeakerDiarization`
  
  **Rationale**: This pipeline is specifically developed for the task of speaker diarization, which involves separating and identifying speakers in audio recordings. Its design is optimized for handling the complexities of speaker variability and recording conditions.

### Key Components

#### Clustering Method

![Example Image](https://tech.skit.ai/assets/images/posts/speaker-diarization/diarization_clustering.jpg)


- **Method**: `AgglomerativeClustering`
  
  **Rationale**: Agglomerative Clustering is used due to its ability to dynamically group speech segments based on similarity without requiring a fixed number of clusters. This flexibility is essential for adapting to diverse audio data and accurately identifying speaker boundaries.

#### Embedding Model



- **Model**: `pyannote-3.1-offline/vox.bin`
  
  **Rationale**: The `vox.bin` embedding model is chosen for its capability to generate high-quality speaker embeddings. These embeddings provide detailed speaker features, which are critical for distinguishing between different speakers with high accuracy.

#### Segmentation Model

![Example Image](https://huggingface.co/collinbarnwell/pyannote-segmentation-30/resolve/main/example.png)

- **Model**: `pyannote-3.1-offline/seg.bin`
  
  **Rationale**: The `seg.bin` model is employed to detect speech boundaries effectively. Precise segmentation is fundamental for ensuring that each speaker’s contributions are accurately isolated, which facilitates accurate clustering and identification.

## Inference Final Script

- There is a notebook contain all the dependecies and required libraries for the inference you would find it in ``\FinalInferenceNoteBook`` directory.

```python
import os
import json
import torch
from nemo.collections.asr.models import EncDecCTCModelBPE
from pyannote.audio import Pipeline
from pydub import AudioSegment
import numpy as np

# Initialize the ASR model
asr_model = EncDecCTCModelBPE.restore_from(restore_path="/content/MetanoiaLabsModel.nemo")
asr_model.eval()
"""
Load the ASR model from a pretrained checkpoint. This model is used for transcribing audio segments. 
The `eval()` method sets the model to evaluation mode.
"""

# Initialize the diarization pipeline
pipeline = Pipeline.from_pretrained("pyannote-3.1-offline/config.yaml")
"""
Load the speaker diarization pipeline from a pretrained configuration. This pipeline is used to segment 
the audio into different speaker turns.
"""

# Directory containing WAV files
audio_dir = "./"
"""
Specify the directory where the input WAV files are located.
"""

# Directory to save the JSON files
output_dir = "./Finaljsonfiles"
"""
Specify the directory where the output JSON files will be saved.
"""

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)
"""
Create the output directory if it does not already exist to store the resulting JSON files.
"""

# List all WAV files in the directory
audio_files = [os.path.join(audio_dir, f) for f in os.listdir(audio_dir) if f.endswith('.wav')]
"""
List all WAV files in the specified directory.
"""

# Function to transcribe a segment
def transcribe_segment(audio_segment):
    """
    Transcribe a given audio segment using the ASR model.

    Parameters:
        audio_segment (AudioSegment): The audio segment to be transcribed.

    Returns:
        str: The transcribed text.
    """
    with torch.no_grad():
        # Convert the audio segment to a tensor
        audio_tensor = torch.from_numpy(np.array(audio_segment.get_array_of_samples(), dtype=np.float32))
        # Perform transcription
        return asr_model.transcribe([audio_tensor], batch_size=1)[0]

# Process each audio file
for audio_file in audio_files:
    audio_id = os.path.basename(audio_file).split('.')[0]
    """
    Extract the base name of the audio file (without extension) to use as an identifier for the JSON file.
    """

    try:
        # Load the audio file using pydub
        audio = AudioSegment.from_wav(audio_file)

        # Diarize the audio file
        diarization = pipeline(audio_file)
        """
        Perform speaker diarization on the audio file to segment it into different speaker turns.
        """

        # Prepare a list to store JSON output
        json_output = []

        # Extract and transcribe each segment
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            start_time = turn.start
            end_time = turn.end

            # Check if the segment duration is valid
            if end_time <= start_time:
                print(f"Skipping invalid segment from {start_time} to {end_time} in {audio_file}")
                continue
            """
            Skip segments where the end time is not greater than the start time, which indicates an invalid segment.
            """

            # Extract the audio segment using pydub
            segment_audio = audio[start_time * 1000:end_time * 1000]  # pydub works in milliseconds

            # Transcribe the audio segment
            text = transcribe_segment(segment_audio)

            # Append the transcription to the JSON output
            json_output.append({
                "start": start_time,
                "end": end_time,
                "speaker": speaker,
                "text": text
            })
    
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        continue
    """
    Handle exceptions during processing and continue with the next file if an error occurs.
    """

    # Save the JSON output to a file
    json_file = os.path.join(output_dir, f"{audio_id}.json")
    with open(json_file, 'w', encoding='utf-8') as f:
        json.dump(json_output, f, ensure_ascii=False, indent=4)
    """
    Save the JSON output for the current audio file to a JSON file in the specified output directory.
    """

    print(f"Processed and saved {json_file}")

print("All files processed.")

````

## Contributors
Metanoia Labs Team:
- [Amir Kasseb](https://www.linkedin.com/in/amirkasseb/)
- [Islam Ibrahim](https://www.linkedin.com/in/islamibr29/)
- [Perihane Tarek](https://www.linkedin.com/in/perihane-tarek/)
- [Mohamed Tarek](https://www.linkedin.com/in/mohamed-tarek-a67275249/)

Supervision Acknowledge:
- [Bassem Mokhtar](https://www.linkedin.com/in/bassem-mokhtar-8251a512/)
- [Omar Adel](https://www.linkedin.com/in/omarxadel/)
