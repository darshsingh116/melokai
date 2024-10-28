import os
import numpy as np

def load_mfcc_data_from_npy_folder(folder_path):
    """
    Load MFCC data stored as .npy files from a specified folder and store them in memory.
    
    Parameters:
    - folder_path: str, path to the folder containing .npy files.
    
    Returns:
    - mfcc_data: list of numpy arrays, each containing MFCC features from a .npy file.
    """
    mfcc_data = []
    
    # List all files in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.npy'):
            file_path = os.path.join(folder_path, filename)
            
            # Load the .npy file
            data = np.load(file_path)
            
            # Append the loaded data to the list
            mfcc_data.append(data)
    
    return mfcc_data


import numpy as np

def create_chunks_from_mfcc(song_mfcc):
    # """
    # Create overlapping chunks from MFCC data for a single song.
    
    # Parameters:
    # - song_mfcc: numpy array, containing transposed MFCC features (time steps in rows, coefficients in columns).
    # - chunk_size_ms: int, size of each chunk in milliseconds.
    # - overlap_ms: int, overlap between consecutive chunks in milliseconds.
    # - sr: int, sample rate of the audio (default 22050 Hz).
    
    # Returns:
    # - chunks: numpy array, containing overlapping chunks for the song.
    # """

    '''above explanataion is wrong'''
    
    # # Calculate the number of rows for the chunk size and overlap
    # rows_per_ms = sr / 1000  # number of samples per millisecond
    # chunk_size = int(chunk_size_ms * rows_per_ms)  # number of samples for the chunk size
    # overlap_size = int(overlap_ms * rows_per_ms)  # number of samples for the overlap



    # Let's assume the following values:

    # Sampling rate (sr) = 22000 Hz
    # Frame size (FS) = 221 samples
    # Hop length (HL) = 220 samples
    # Number of MFCC coefficients (n_mfcc) = 13
    # MFCCs per second = 22000/220 = 100
    # MFCCs for 10ms = 1


    chunk_size =  10
    overlap_size =  9
    
    num_rows = song_mfcc.shape[0]  # number of time steps (since the data is transposed)
    # print(song_mfcc.shape)
    chunks = []
    print(f"num_rows is {num_rows}")
    
    # Create chunks with the specified size and overlap
    for start in range(0, num_rows - int(chunk_size) + 1, int(chunk_size - overlap_size)):
        end = start + chunk_size
        chunk = song_mfcc[start:int(end), :]  # Slicing rows for time steps, keeping all columns (coefficients)
        chunks.append(chunk)
    
    # print("create_chunks_from_mfcc")
    # print(len(chunks))
    return np.array(chunks)


def create_overlapping_chunks(mfcc_data, sr=22050):
    """
    Create overlapping chunks from MFCC data for each song in the list.
    
    Parameters:
    - mfcc_data: list of numpy arrays, each containing MFCC features for a song.
    - chunk_size_ms: int, size of each chunk in milliseconds.
    - overlap_ms: int, overlap between consecutive chunks in milliseconds.
    - sr: int, sample rate of the audio (default 22050 Hz).
    
    Returns:
    - song_chunks: list of numpy arrays, each containing overlapping chunks for a song.
    """
    songs = []
    
    for song_mfcc in mfcc_data:
        chunks = create_chunks_from_mfcc(song_mfcc, sr)
        songs.append(chunks)
    
    return songs