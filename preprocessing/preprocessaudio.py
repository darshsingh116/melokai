import os
import librosa
import numpy as np
from .audio import *
from dotenv import load_dotenv
from sklearn.preprocessing import MinMaxScaler
from joblib import Parallel, delayed
load_dotenv()
archive_path = os.getenv('ARCHIVE_PATH')
module_path = os.getenv('MODULE_PATH')

# import ray


def preprocess_and_save_audio(df):
    # Directory where processed files will be saved
    processed_dir = os.path.join(os.getcwd(), 'processed-audio')
    
    # Check if the directory exists, if not, create it
    if not os.path.exists(processed_dir):
        os.makedirs(processed_dir)
    
    for index, row in df.iterrows():
        tp=[]
        file_path = os.path.join(archive_path, "train", row['folder'], f"{row['audio']}.osu")
        with open(file_path, 'r', encoding='utf-8') as file: 
            current_section = None
            # inherit data
            for line in file:
                line = line.strip()
                if line.startswith('['):
                    # Handle section headers
                    current_section = line[1:-1]
                    continue
                
                if current_section == 'HitObjects' and line:
                    continue
                
                if current_section == 'TimingPoints' and line:
                    if(line == ""):
                        continue
                    tp = line.split(',')
                    break

                if ':' in line:
                    continue
            

        beatlen = float(tp[1])
        first_beat_time = int(float(tp[0]))
        if int(float(tp[0]))<(beatlen/8):
            first_beat_time=beatlen+int(float(tp[0]))

        # Construct the file path from the folder and audio fields
        file_path = os.path.join(archive_path,"train", row['folder'], "audio.opus")
        
        # Load the audio file
        try:
            y, sr = librosa.load(file_path, sr=22000)
            # # Get the tempo (BPM) and beat frames
            # tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)

            # # Convert beat frames to time (in seconds)
            # beat_times = librosa.frames_to_time(beat_frames, sr=sr)

            # # Convert first beat time to milliseconds
            # first_beat_time_ms = beat_times[0] * 1000 if len(beat_times) > 0 else None

            # # Output the results
            # print(f"Estimated Tempo (BPM): {tempo}")
            # if first_beat_time_ms is not None:
            #     print(f"First Beat Time: {first_beat_time_ms:.2f} milliseconds")
            # else:
            #     print("No beats detected.")
            # print(y)
            # print(sr)

            # Calculate the chunk duration in samples
            sr=22000
            chunk_duration_samples = int((beatlen * sr)/(4000))  # beatlen is in ms, sr is in samples per second

            # List to hold extracted features
            
            mfcc_array = []
            chunk_midpoint_time_array = []
            audio_length_ms = (len(y) / sr) * 1000
            n_chunks = int((audio_length_ms - first_beat_time)/(beatlen/4))
            # Loop through chunks
            for i in range(n_chunks):
                # Calculate the start and end of the current chunk
                chunk_midpoint_time = first_beat_time + i * (beatlen/4)  # Midpoint of the chunk in ms
                chunk_midpoint_sample = int(chunk_midpoint_time * sr / 1000)  # Convert time to sample index
                chunk_start = max(0, chunk_midpoint_sample - (chunk_duration_samples // 2))
                chunk_end = min(len(y), chunk_midpoint_sample + (chunk_duration_samples // 2))

                # Slice the audio for this chunk
                audio_chunk = y[chunk_start:chunk_end]
                chunk_length = audio_chunk.shape[0]


                # Compute MFCCs for this chunk
                # mfcc = librosa.feature.mfcc(y=audio_chunk, sr=sr, n_mfcc=13, n_fft=(chunk_end-chunk_start), hop_length=(chunk_end-chunk_start), n_mels=40)
                mfcc = librosa.feature.mfcc(y=audio_chunk,n_fft=chunk_length, hop_length=chunk_length, sr=sr, n_mfcc=80, n_mels=128)
                mfcc = np.mean(mfcc, axis=1)
                # show_data(data)
                
                mfcc = np.array(mfcc)
                mfcc = np.transpose(mfcc)
                # print('mfcc shape ', mfcc.shape)
                # # Compute Chroma features for this chunk
                # chroma = librosa.feature.chroma_cqt(y=audio_chunk, sr=sr, hop_length=0, n_chroma=12)

                # # Combine MFCC and Chroma features
                # combined_features = np.concatenate((mfcc, chroma), axis=0)

                # # Transpose to have time frames as rows
                # combined_features = np.transpose(combined_features)

                # # Append to the features list
                # features.append(combined_features)
                mfcc_array.append(mfcc)
                chunk_midpoint_time_array.append(chunk_midpoint_time)
                
                
        except FileNotFoundError:
            print(f"File {file_path} not found.")
            continue



       


        # # Define custom frame size and hop length
        # frame_size = 221  # Frame size (n_fft)
        # hop_length = 220  # Hop length (number of samples between successive frames)
        # #doing above will create 1mff per 10ms audio as sampling rate is 22000 and hop size is 220
        
        # # Compute MFCCs with the specified frame size and hop length
        # mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, n_fft=frame_size, hop_length=hop_length,n_mels=40)
        # # Extract Chroma features
        # chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=hop_length, n_chroma=12)
        
        # # Combine features
        # combined_features = np.concatenate((mfcc, chroma), axis=0)

        # combined_features = np.transpose(combined_features)
        
        # # Convert to float32 to save memory
        # combined_features = combined_features.astype(np.float32)
        # print(len(combined_features))
        # print(combined_features.shape)
        # # print(len(mfcc))
        # # print("~~~~")
        # # chunks = create_chunks_from_mfcc(mfcc)

        features = []

        scaler = MinMaxScaler(feature_range=(0, 1))
        scaler = scaler.fit(mfcc_array)
        mfcc_array = scaler.transform(mfcc_array)
        print(mfcc_array.shape)
        for chunk_midpoint_time, mfcc in zip(chunk_midpoint_time_array, mfcc_array):
            new_entry = (int(chunk_midpoint_time), mfcc)
            features.append(new_entry)

        # Construct the save path
        save_path = os.path.join(processed_dir, f"{row['audio']}-a.npy")
        
        # Save the MFCC features as a .npy file
        # features = np.array(features)
        features = np.array(features, dtype=object)
        # print(features.shape)
        # for f in features:
        #     print(f)
        np.save(save_path, features)
        # print(f"Saved: {save_path}")


# @ray.remote
def process_audio(row, processed_dir):
    # List to hold extracted features
    mfcc_array = []
    chunk_midpoint_time_array = []
    try:
        tp=[]
        file_path = os.path.join(archive_path, "train", row['folder'], f"{row['audio']}.osu")
        with open(file_path, 'r', encoding='utf-8') as file: 
            current_section = None
            # inherit data
            for line in file:
                line = line.strip()
                if line.startswith('['):
                    # Handle section headers
                    current_section = line[1:-1]
                    continue
                
                if current_section == 'HitObjects' and line:
                    continue
                
                if current_section == 'TimingPoints' and line:
                    if(line == ""):
                        continue
                    tp = line.split(',')
                    break

                if ':' in line:
                    continue
            

        beatlen = float(tp[1])
        first_beat_time = int(float(tp[0]))
        if int(float(tp[0]))<(beatlen/8):
            first_beat_time=beatlen+int(float(tp[0]))

        # Construct the file path from the folder and audio fields
        file_path = os.path.join(archive_path,"train", row['folder'], "audio.opus")
        
        # Load the audio file
        try:
            y, sr = librosa.load(file_path, sr=22000)
            # Calculate the chunk duration in samples
            sr=22000
            chunk_duration_samples = int((beatlen * sr)/(4000))  # beatlen is in ms, sr is in samples per second

            
            audio_length_ms = (len(y) / sr) * 1000
            n_chunks = int((audio_length_ms - first_beat_time)/(beatlen/4))
            # Loop through chunks
            for i in range((n_chunks-1)):
                # Calculate the start and end of the current chunk
                chunk_midpoint_time = first_beat_time + i * (beatlen/4)  # Midpoint of the chunk in ms
                chunk_midpoint_sample = int(chunk_midpoint_time * sr / 1000)  # Convert time to sample index
                chunk_start = int(max(0, chunk_midpoint_sample - (chunk_duration_samples // 2)))
                chunk_end = int(min(len(y), chunk_midpoint_sample + (chunk_duration_samples // 2)))

                # Slice the audio for this chunk
                audio_chunk = y[chunk_start:chunk_end]
                chunk_length = audio_chunk.shape[0]


                # Compute MFCCs for this chunk
                # mfcc = librosa.feature.mfcc(y=audio_chunk, sr=sr, n_mfcc=13, n_fft=(chunk_end-chunk_start), hop_length=(chunk_end-chunk_start), n_mels=40)
                mfcc = librosa.feature.mfcc(y=audio_chunk,n_fft=chunk_length, hop_length=chunk_length, sr=sr, n_mfcc=80, n_mels=128)
                mfcc = np.mean(mfcc, axis=1)
                # show_data(data)
                
                mfcc = np.array(mfcc)
                mfcc = np.transpose(mfcc)
                # print('mfcc shape ', mfcc.shape)
                # # Compute Chroma features for this chunk
                # chroma = librosa.feature.chroma_cqt(y=audio_chunk, sr=sr, hop_length=0, n_chroma=12)

                # # Combine MFCC and Chroma features
                # combined_features = np.concatenate((mfcc, chroma), axis=0)

                # # Transpose to have time frames as rows
                # combined_features = np.transpose(combined_features)

                # # Append to the features list
                # features.append(combined_features)
                mfcc_array.append(mfcc)
                chunk_midpoint_time_array.append(chunk_midpoint_time)
        except FileNotFoundError:
            print(f"File {file_path} not found.")
    except FileNotFoundError:
            print(f"File {file_path} not found.")

    features = []
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler = scaler.fit(mfcc_array)
    mfcc_array = scaler.transform(mfcc_array)
    print(mfcc_array.shape)
    for chunk_midpoint_time, mfcc in zip(chunk_midpoint_time_array, mfcc_array):
        new_entry = (int(chunk_midpoint_time), mfcc)
        features.append(new_entry)

    # Construct the save path
    save_path = os.path.join(processed_dir, f"{row['audio']}-a.npy")
    
    # Save the MFCC features as a .npy file
    # features = np.array(features)
    features = np.array(features, dtype=object)
    # print(features.shape)
    # for f in features:
    #     print(f)
    np.save(save_path, features)
    # print(f"Saved: {save_path}")

from multiprocessing import Pool, cpu_count
from tqdm import tqdm
from functools import partial

# def process_audio_batch(rows):
#     processed_dir = os.path.join(os.getcwd(), 'processed-audio')
#     for row in rows:
#         index, row_data = row  # Unpack the tuple
#         process_audio(row_data,processed_dir)
#         pass

# def preprocess_and_save_audio_in_parallel(df):
#     processed_dir = os.path.join(os.getcwd(), 'processed-audio')
#     os.makedirs(processed_dir, exist_ok=True)

#     # Convert the DataFrame to a list of rows
#     tasks = list(df.iterrows())

#     batch_size = max(1, len(tasks) // 8)  # Adjust the batch size based on the number of cores

#     # Create batches of tasks
#     batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]

#     # Use Pool to parallelize the processing
#     with Pool(8) as pool:
#         # Use map to wrap the iterable and process batches
#         pool.map(process_audio_batch, batches)




# def preprocess_and_save_audio_in_parallel(df):
#     # ray.init()
#     # Directory where processed files will be saved
#     processed_dir = os.path.join(os.getcwd(), 'processed-audio')
    
#     # Check if the directory exists, if not, create it
#     os.makedirs(processed_dir, exist_ok=True)
    
    
#     processed_df = ray.get([process_audio.remote(row, processed_dir) for index, row in df.iterrows()])

def preprocess_and_save_audio_in_parallel(df):
    # Directory where processed files will be saved
    processed_dir = os.path.join(os.getcwd(), 'processed-audio')
    
    # Check if the directory exists, if not, create it
    os.makedirs(processed_dir, exist_ok=True)
    
    # Use joblib's Parallel and delayed to parallelize the processing
    Parallel(n_jobs=-1)(delayed(process_audio)(row, processed_dir) for index, row in df.iterrows())


# Example usage with your DataFrame
# df = pd.read_csv('your_data.csv')  # Load your DataFrame if not already loaded
# preprocess_and_save_audio(df)


