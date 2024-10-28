import os
import tensorflow as tf
import numpy as np
import pandas as pd
from dotenv import load_dotenv

np.set_printoptions(precision=10, suppress=True)

#setting data path
load_dotenv()
project_path = os.getenv('PROJECT_PATH')

def load_df_from_csv():
    # Define the path to the CSV file in the processed metadata folder
    load_path = os.path.join(project_path,"processed-metadata", "processed-metadata.csv")
    
    # Check if the file exists before attempting to load it
    if os.path.exists(load_path):
        # Load the DataFrame from the CSV file
        df = pd.read_csv(load_path)
        print(f"DataFrame loaded from: {load_path}")
        return df
    else:
        print(f"File not found: {load_path}")
        return None


def load_all_beatmaps_from_df(df):
    # List to hold all loaded beatmap data
    all_beatmaps = []
    
    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Extract the audio filename from the current row
        audio_filename = row['audio']
        
        # Define the path to the saved beatmap file based on the audio filename
        load_path = os.path.join(project_path,"processed-beatmaps", f"{audio_filename}-b.npy")
        
        # Load the NumPy array from disk with allow_pickle=True
        if os.path.exists(load_path):
            try:
                beatmap_data = np.load(load_path, allow_pickle=True)
                all_beatmaps.append(beatmap_data)
            except ValueError as e:
                print(f"Error loading {load_path}: {e}")
        else:
            print(f"File not found: {load_path}")
    
    return all_beatmaps



def load_all_beatmaps_and_audios_from_df(df):
    # Lists to hold all loaded beatmap and audio data
    all_beatmaps = []
    all_audios = []
    itr = 1
    total = len(df)
    # Iterate over each row in the DataFrame
    for index, row in df.iterrows():
        # Extract the audio filename from the current row
        audio_filename = row['audio']
        
        # Define the paths to the saved beatmap and audio files
        beatmap_path = os.path.join(project_path,"processed-beatmaps", f"{audio_filename}-b.npy")
        audio_path = os.path.join(project_path,"processed-audio", f"{audio_filename}-a.npy")
        
        # Check if both files exist and load them
        beatmap_data = None
        audio_data = None

        if os.path.exists(beatmap_path):
            try:
                beatmap_data = np.load(beatmap_path, allow_pickle=True)
            except ValueError as e:
                print(f"Error loading {beatmap_path}: {e}")
        
        if os.path.exists(audio_path):
            try:
                audio_data = np.load(audio_path, allow_pickle=True)
            except ValueError as e:
                print(f"Error loading {audio_path}: {e}")

        # If both files exist, append the data to their respective lists
        if beatmap_data is not None and audio_data is not None:
            all_beatmaps.append(beatmap_data)
            all_audios.append(audio_data)
        else:
            if beatmap_data is None:
                print(f"Beatmap file not found or failed to load: {beatmap_path}")
            if audio_data is None:
                print(f"Audio file not found or failed to load: {audio_path}")

        print(f"{itr}/{total}")
        itr += 1
    return all_beatmaps, all_audios




def process_audio_and_beatmap_for_model_single(metadata):
    # Define paths to the beatmap and audio files
    beatmap_path = os.path.join(project_path, "processed-beatmaps", f"{metadata['audio']}-b.npy")
    audio_path = os.path.join(project_path, "processed-audio", f"{metadata['audio']}-a.npy")

    # Load beatmap and audio data if files exist
    if not os.path.exists(beatmap_path):
        print(f"Beatmap file not found: {beatmap_path}")
        return None, None

    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        return None, None

    try:
        beatmap_data = np.load(beatmap_path, allow_pickle=True)
        audio_data = np.load(audio_path, allow_pickle=True)
    except ValueError as e:
        print(f"Error loading files: {e}")
        return None, None
    # for h in beatmap_data:
    #     print(h)
    # Initialize the processed lists
    processed_audio = []
    processed_beatmap = []

    # Iterate over each timeframe (20ms chunks with 10ms overlap)
    num_chunks = len(audio_data)
    beatmap_itr = 0

    input_footer = [float(metadata["bpm"]),float(metadata["total_length"])]+[float(x) for x in metadata[-9:]]

    for i in range(num_chunks):
        # fix_under_ten_ms = False
        if (beatmap_itr > 0) and beatmap_itr < len(beatmap_data):
            if beatmap_data[beatmap_itr][2] < beatmap_data[beatmap_itr-1][2]:
                raise ValueError(f"PARSING ERROR SOMEWHERE : Beatmap timestamp {beatmap_data[beatmap_itr][2]}ms is smaller than previous which is {beatmap_data[beatmap_itr-1][2]} in map {metadata['folder']}/{metadata['audio']}.osu")
        
        timestamp = i * 10  # Calculate the timestamp for this chunk in ms
        
        if (len(processed_beatmap) > 1) and (beatmap_itr > 0) and (beatmap_itr < len(beatmap_data)):
            if ((beatmap_data[beatmap_itr][2] - beatmap_data[beatmap_itr-1][2]) < 10) or (timestamp > beatmap_data[beatmap_itr][2]):
                if np.array_equal(processed_beatmap[-2], np.zeros_like(beatmap_data[0])):
                    processed_beatmap[-2] = processed_beatmap[-1]
                    processed_beatmap[-1] = beatmap_data[beatmap_itr]
                    # fix_under_ten_ms = True
                    # print(f"fixing under 10ms parts,beatmap_itr {beatmap_itr}, beatmap_data[beatmap_itr][2] {beatmap_data[beatmap_itr][2]} , beatmap_data[beatmap_itr-1][2] {beatmap_data[beatmap_itr-1][2]} ,  timestamp {timestamp}")
                    beatmap_itr += 1
                else:
                    a = (beatmap_data[beatmap_itr][2]- beatmap_data[beatmap_itr-1][2]) < 10
                    b = (timestamp >= beatmap_data[beatmap_itr][2])
                    raise  ValueError(f"Error on line 146 dataReadFromDisk.py Beatmap timestamp {beatmap_data[beatmap_itr][2]}ms in map {metadata['folder']}/{metadata['audio']}.osu {a} or {b}")


        # Add timestamp to the start of the audio chunk and footer on back
        audio_part = np.array(audio_data[i])
        audio_part = audio_part.flatten()
        audio_part = np.insert(audio_part,0,timestamp)
        audio_chunk = np.concatenate((audio_part, input_footer))
        processed_audio.append(audio_chunk)

        # Check for the timestamp of the current beatmap data
        if beatmap_itr < len(beatmap_data):
            beatmap_timestamp = beatmap_data[beatmap_itr][2]  # Assuming the timestamp is at index 3

            # If the audio chunk's timestamp overtakes the next beatmap timestamp, raise an error
            if beatmap_timestamp < timestamp and len(beatmap_data) > (beatmap_itr+1):
                raise ValueError(f"Audio timestamp {timestamp}ms has overtaken beatmap timestamp {beatmap_timestamp}ms in map {metadata['folder']}/{metadata['audio']}.osu")

            # Find the corresponding beatmap data within the 10ms window
            elif timestamp <= beatmap_timestamp < (timestamp + 10):
                beatmap_chunk = beatmap_data[beatmap_itr]
                # print(f"in beatmap_itr += 1 , timestamp is {timestamp} and beatmap time stamp is {beatmap_timestamp}")
                beatmap_itr += 1  # Move to the next beatmap entry
            else:
                # print(f"adding chunk with zero audio timestamp is {timestamp} and beatmap time stamp is {beatmap_timestamp}")
                beatmap_chunk = np.zeros_like(beatmap_data[0])  # Default to zeros if no match is found
        else:
            # print("beatmap finished adding zeroes")
            beatmap_chunk = np.zeros_like(beatmap_data[0])


        # if not fix_under_ten_ms:
        processed_beatmap.append(beatmap_chunk)

    return np.array(processed_audio), np.array(processed_beatmap)


def create_dataset(processed_audio_list, processed_beatmap_list, hyper_params_list):

    # Generator function defined inside create_dataset
    def data_generator():
        for audio, beatmap, params in zip(processed_audio_list, processed_beatmap_list, hyper_params_list):
            yield (
                tf.convert_to_tensor(audio, dtype=tf.float32),
                tf.convert_to_tensor(beatmap, dtype=tf.float32),
                tf.convert_to_tensor(params, dtype=tf.float32)  # Convert params to tensor (1D array)
            )

    # Create TensorFlow dataset
    dataset = tf.data.Dataset.from_generator(
        data_generator,
        output_signature=(
            tf.TensorSpec(shape=[60000, 14], dtype=tf.float32),  # Variable length sequence with feature dimension 142
            tf.TensorSpec(shape=[60000, 4], dtype=tf.float32),   # Variable length sequence with feature dimension 13
            tf.TensorSpec(shape=[11], dtype=tf.float32),       # 1D array of hyperparameters (params) with variable length
        )
    )

    return dataset


# def create_dataset(audio_features, beatmap_outputs, hyper_params):
#     # Convert to RaggedTensor
#     ragged_audio_features = tf.ragged.constant(audio_features)
#     ragged_beatmap_outputs = tf.ragged.constant(beatmap_outputs)

#     # Create dataset with RaggedTensor
#     dataset = tf.data.Dataset.from_tensor_slices((ragged_audio_features, ragged_beatmap_outputs, hyper_params))
#     return dataset

def padding_and_masking(data, dim,max_length):
    """Pad information to a fixed length."""
    # print(data.shape)
    # Create an array of 1s with shape (100, 1)
    extra_column = np.ones((len(data), 1))

    # Concatenate the extra_column to the original data array along the last axis
    expanded_data = np.concatenate((data, extra_column), axis=1)

    # print(expanded_data.shape)
    # Create a single array of zeros with dimension 13
    single_array = np.zeros((dim+1))

    # Create a list of 100 such arrays
    padded_list = np.array([single_array for _ in range((max_length-len(data)))])
    # print(padded_list.shape)
    padded_hitobjects = np.concatenate((expanded_data, padded_list), axis=0)

    return padded_hitobjects

def process_audio_and_beatmap_for_model_all_and_save(df,chunk_size=2000):
    # Initialize empty arrays to store processed data
    processed_audio_list = []
    processed_beatmap_list = []

    save_path = os.path.join(project_path,"model-processed-data")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    itr=0
    # Iterate through the DataFrame
    for index, row in df.iterrows():
        processed_audio, processed_beatmap = process_audio_and_beatmap_for_model_single(row)
        # Append the processed data to lists
        processed_audio_list.append(processed_audio)
        processed_beatmap_list.append(processed_beatmap)

        # Save periodically to avoid memory overflow
        if (itr + 1) % chunk_size == 0:
            
            # # Convert lists to numpy arrays
            # audio_array = np.array(processed_audio_list)
            # beatmap_array = np.array(processed_beatmap_list)

            # # Save to disk
            # np.save(f'{save_path}/processed_audio_chunk_{itr//chunk_size}.npy', audio_array)
            # np.save(f'{save_path}/processed_beatmap_chunk_{itr//chunk_size}.npy', beatmap_array)
            # Example data

            input_data = [np.array(song) for song in processed_audio_list]
            output_data = [np.array(song) for song in processed_beatmap_list]

            # Create a TensorFlow dataset
            dataset = tf.data.Dataset.from_tensor_slices((input_data, output_data))
            dataset = dataset.batch(1)  # Batch size of 1 for demonstration

            # Custom filename and extension
            filename = f"processed_dataset_{itr//chunk_size}"
            # extension = ".tfdata"  # Custom extension

            # # Define the save path (just the directory part)
            save_path = os.path.join(save_path, filename)

            # Ensure the directory exists
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            # Save the dataset
            tf.data.experimental.save(dataset, save_path)

            # Clear lists to free up memory
            processed_audio_list = []
            processed_beatmap_list = []

        itr+=1
    # Save remaining data
    if processed_audio_list and processed_beatmap_list:
        
        # audio_array = np.array(processed_audio_list)
        # beatmap_array = np.array(processed_beatmap_list)
        # np.save(f'{save_path}/processed_audio_chunk_final.npy', audio_array)
        # np.save(f'{save_path}/processed_beatmap_chunk_final.npy', beatmap_array)

        # input_data = [np.array(song) for song in processed_audio_list]
        # output_data = [np.array(song) for song in processed_beatmap_list]

        # # Create a TensorFlow dataset
        # dataset = tf.data.Dataset.from_tensor_slices((input_data, output_data))
        # dataset = dataset.batch(1)  # Batch size of 1 for demonstration

        # # Custom filename and extension
        # filename = f"processed_beatmap_chunk_final.npy"
        # # extension = ".tfdata"  # Custom extension

        # # # Define the save path (just the directory part)
        # save_path = os.path.join(save_path, filename)

        # # Ensure the directory exists
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)

        # # Save the dataset
        # tf.data.experimental.save(dataset, save_path)

        # Generator function
    
        # Create dataset
        dataset = create_dataset(processed_audio_list, processed_beatmap_list)

        # Optionally, batch the dataset if needed
        dataset = dataset.batch(1)

        return dataset,itr,processed_audio_list,processed_beatmap_list

    

    




def new_process_audio_and_beatmap_for_model_all_and_save(df,chunk_size=2000):
    # Initialize empty arrays to store processed data
    input_audio_list = []
    # output_beatmap_list = []
    processed_beatmap_list = []
    hyper_params_list = []

    save_path = os.path.join(project_path,"model-processed-data")
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    itr=0
    # Iterate through the DataFrame
    for index, row in df.iterrows():
        try:
            input_audio, processed_beatmap,params  = new_process_audio_and_beatmap_for_model_single(row)
            # Append the processed data to lists
            input_audio_list.append(input_audio)
            # output_beatmap_list.append(output_beatmap)
            processed_beatmap_list.append(processed_beatmap)
            hyper_params_list.append(params)

        except Exception as e :
            print(e)
            print("Nan or not float found in params")
    # Save remaining data
    if input_audio_list and processed_beatmap_list:
        
        # audio_array = np.array(processed_audio_list)
        # beatmap_array = np.array(processed_beatmap_list)
        # np.save(f'{save_path}/processed_audio_chunk_final.npy', audio_array)
        # np.save(f'{save_path}/processed_beatmap_chunk_final.npy', beatmap_array)

        # input_data = [np.array(song) for song in processed_audio_list]
        # output_data = [np.array(song) for song in processed_beatmap_list]

        # # Create a TensorFlow dataset
        # dataset = tf.data.Dataset.from_tensor_slices((input_data, output_data))
        # dataset = dataset.batch(1)  # Batch size of 1 for demonstration

        # # Custom filename and extension
        # filename = f"processed_beatmap_chunk_final.npy"
        # # extension = ".tfdata"  # Custom extension

        # # # Define the save path (just the directory part)
        # save_path = os.path.join(save_path, filename)

        # # Ensure the directory exists
        # if not os.path.exists(save_path):
        #     os.makedirs(save_path)

        # # Save the dataset
        # tf.data.experimental.save(dataset, save_path)

        # Generator function
    
        # Create dataset
        # dataset = create_dataset(processed_audio_list, processed_beatmap_list,hyper_params_list)

        # Optionally, batch the dataset if needed
        # dataset = dataset.batch(1)

        return input_audio_list,processed_beatmap_list,hyper_params_list




def new_process_audio_and_beatmap_for_model_single(metadata):
    # Define paths to the beatmap and audio files
    beatmap_path = os.path.join(project_path, "processed-beatmaps", f"{metadata['audio']}-b.npy")
    audio_path = os.path.join(project_path, "processed-audio", f"{metadata['audio']}-a.npy")

    # Load beatmap and audio data if files exist
    if not os.path.exists(beatmap_path):
        print(f"Beatmap file not found: {beatmap_path}")
        return None, None, None

    if not os.path.exists(audio_path):
        print(f"Audio file not found: {audio_path}")
        return None, None, None

    try:
        beatmap_data = np.load(beatmap_path, allow_pickle=True)
        audio_data = np.load(audio_path, allow_pickle=True)
    except ValueError as e:
        print(f"Error loading files: {e}")
        return None, None, None

    processed_beatmap = []
    params = [float(metadata["bpm"]), float(metadata["total_length"])] + [float(x) for x in metadata[-9:]]
    params = np.array(params)
    for p in params:
        if isinstance(p, float) and not np.isnan(p):
            pass
        else:
            raise ValueError("Nan or not float found in params in single func")

    import math
    from bisect import bisect_left, bisect_right

    """
    Maps beatmap_data to audio_data, producing processed_beatmap with fields:
    [x, y, timestamp, object_type]

    object_type:
        0: nothing
        1: circle
        2: slider start
        3: slider middle/end or intermediates
        4: spinner point
        5: spinner intermediate
    """
    margin = 30

    # Helper function to find the nearest timestamp within a margin
    def find_nearest_timestamp(target_ts, audio_timestamps, margin):
        idx = bisect_left(audio_timestamps, target_ts)
        candidates = []
        
        # Collect the candidates: one before and one at/after target_ts
        if idx < len(audio_timestamps):
            candidates.append(audio_timestamps[idx])  # The first candidate >= target_ts
        if idx > 0:
            candidates.append(audio_timestamps[idx - 1])  # The first candidate < target_ts

        nearest = None
        min_diff = margin + 1  # Initialize with a value greater than the allowed margin

        # Check the candidates for the nearest timestamp within the margin
        for ts in candidates:
            diff = abs(ts - target_ts)
            if diff <= margin and diff < min_diff:
                min_diff = diff
                nearest = ts

        # If no candidate was within the margin, return the next available timestamp
        if nearest is None:
            if idx < len(audio_timestamps):
                nearest = audio_timestamps[idx]  # Return the next available timestamp after target_ts
            elif idx > 0:
                nearest = audio_timestamps[idx - 1]  # Return the last timestamp if no next exists

        return nearest
    
    def generate_intermediate_coordinates(processed_beatmap, index1, index2):
        """
        Generates a list of intermediate [x, y] coordinates between two points in the processed_beatmap.

        Parameters:
        - processed_beatmap (list or np.ndarray): The processed beatmap data.
        Each element should be in the format [x, y, timestamp, object_type].
        - index1 (int): The index of the first point in processed_beatmap.
        - index2 (int): The index of the second point in processed_beatmap.

        Returns:
        - List[List[float, float]]: A list of intermediate [x, y] coordinates.
        Returns an empty list if there are no intermediates.
        """
        # Ensure index1 is less than index2
        if index1 > index2:
            index1, index2 = index2, index1

        # Calculate the number of intermediate points (n)
        n = index2 - index1 - 1

        # If there are no intermediate points, return an empty list
        if n <= 0:
            return []

        # Extract x and y coordinates for both points
        x1, y1 = processed_beatmap[index1][0], processed_beatmap[index1][1]
        x2, y2 = processed_beatmap[index2][0], processed_beatmap[index2][1]

        intermediates = []

        # Calculate intermediate coordinates
        for i in range(1, n + 1):
            # Calculate the proportion for interpolation
            proportion = i / (n + 1)

            # Interpolate x and y coordinates
            xi = x1 + proportion * (x2 - x1)
            yi = y1 + proportion * (y2 - y1)

            # Optionally, round the coordinates to integers if needed
            xi = round(xi)
            yi = round(yi)

            intermediates.append([xi, yi])

        return intermediates



    

    # Step 1: Preprocess audio_data
    # Sort audio_data by timestamp
    audio_data_sorted = sorted(audio_data, key=lambda x: x[0])
    audio_timestamps = [row[0] for row in audio_data_sorted]

    # Create a mapping from timestamp to list of indices in audio_data_sorted
    timestamp_to_indices = {}
    for idx, row in enumerate(audio_data_sorted):
        ts = row[0]
        if ts in timestamp_to_indices:
            timestamp_to_indices[ts].append(idx)
        else:
            timestamp_to_indices[ts] = [idx]

    # Step 2: Initialize processed_beatmap with [0, 0, timestamp, 0]
    processed_beatmap = [[0, 0, ts, 0] for ts in audio_timestamps]

    # Step 3: Iterate through beatmap_data and map to processed_beatmap
    for hitobj in beatmap_data:
        x, y, starttime, endtime, list_of_points = hitobj

        if starttime == endtime:
            # Circle
            nearest_ts = find_nearest_timestamp(starttime, audio_timestamps, margin)
            if nearest_ts is not None:
                for idx in timestamp_to_indices[nearest_ts]:
                    processed_beatmap[idx] = [x, y, nearest_ts, 1]

        elif list_of_points:
            # Slider
            # Step 3.1: Remove consecutive duplicate points
            unique_points = [list_of_points[0]]
            for point in list_of_points[1:]:
                if point != unique_points[-1]:
                    unique_points.append(point)
            num_unique_points = len(unique_points)
            if num_unique_points < 2:
                continue

            # Step 3.2: Calculate distances between consecutive unique points
            distances = []
            total_distance = 0.0
            for i in range(1, num_unique_points):
                p1 = unique_points[i - 1]
                p2 = unique_points[i]
                dx = p2[0] - p1[0]
                dy = p2[1] - p1[1]
                dist = math.hypot(dx, dy)
                distances.append(dist)
                total_distance += dist

            # Step 3.3: Calculate cumulative distances
            cumulative_distances = []
            cum_dist = 0.0
            for dist in distances:
                cum_dist += dist
                cumulative_distances.append(cum_dist)

            # Step 3.4: Calculate slider point timestamps
            total_delta = endtime - starttime
            num_segments = num_unique_points - 1
            slider_point_times = []
            slider_point_times.append(starttime)

            if total_distance == 0:
                # Distribute slider_point_times evenly over the duration
                if num_segments > 0:
                    delta_per_segment = total_delta / num_segments
                    for i in range(1, num_unique_points):
                        slider_ts = starttime + round(delta_per_segment * i)
                        slider_point_times.append(slider_ts)
            else:
                for cum_dist in cumulative_distances:
                    ratio = cum_dist / total_distance
                    slider_ts = starttime + round(total_delta * ratio)
                    slider_point_times.append(slider_ts)

            # raise ValueError(f"slider_point_times = {slider_point_times}")
            indexList = []
            for i, slider_ts in enumerate(slider_point_times):
                slider_end_ts = find_nearest_timestamp(slider_ts, audio_timestamps, margin)
                if slider_end_ts is not None:
                    for idx in timestamp_to_indices[slider_end_ts]:
                        processed_beatmap[idx] = [unique_points[i][0], unique_points[i][1], slider_end_ts, 3]
                        indexList.append(idx)
            slider_start_ts = find_nearest_timestamp(starttime, audio_timestamps, margin)
            if slider_start_ts is not None:
                for idx in timestamp_to_indices[slider_start_ts]:
                    processed_beatmap[idx][3] = 2

            # raise ValueError(f"indexList = {indexList}")

            for i in range(len(indexList)-1):
                intermidiate_points_for_SL = generate_intermediate_coordinates(processed_beatmap,indexList[i],indexList[i+1])
                for n in range(indexList[i+1]-indexList[i]-1):
                    processed_beatmap[indexList[i]+n+1][0] = intermidiate_points_for_SL[n][0]
                    processed_beatmap[indexList[i]+n+1][1] = intermidiate_points_for_SL[n][1]
                    processed_beatmap[indexList[i]+n+1][3] = 3



            

        else:
            # Spinner
            spinner_start_ts = find_nearest_timestamp(starttime, audio_timestamps, margin)
            spinner_end_ts = find_nearest_timestamp(endtime, audio_timestamps, margin)
            if spinner_start_ts is not None:
                for idx in timestamp_to_indices[spinner_start_ts]:
                    processed_beatmap[idx] = [x, y, spinner_start_ts, 4]
            if spinner_end_ts is not None and spinner_end_ts != spinner_start_ts:
                for idx in timestamp_to_indices[spinner_end_ts]:
                    processed_beatmap[idx] = [x, y, spinner_end_ts, 4]

            # Assign object_type = 5 for spinner intermediates
            if spinner_start_ts is not None and spinner_end_ts is not None and spinner_end_ts > spinner_start_ts:
                start_idx = bisect_left(audio_timestamps, spinner_start_ts)
                end_idx = bisect_right(audio_timestamps, spinner_end_ts)
                for idx in range(start_idx, end_idx):
                    current_ts = audio_timestamps[idx]
                    if spinner_start_ts < current_ts < spinner_end_ts:
                        processed_beatmap[idx] = [x, y, current_ts, 5]

    # Extract input_audio
    input_audio = [a[1] for a in audio_data_sorted]

    return np.array(input_audio), np.array(processed_beatmap), params