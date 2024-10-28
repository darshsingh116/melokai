import math
from typing import List, Tuple
import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv

#setting data path
load_dotenv()
archive_path = os.getenv('ARCHIVE_PATH')


class OsuBeatmap:
    def __init__(self,hit_objects, timing_points  ,audio, split, folder, beatmapset_id, beatmap_id, approved, total_length, hit_length, version,
                 file_md5, diff_size, diff_overall, diff_approach, diff_drain, mode, count_normal, count_slider,
                 count_spinner, submit_date, approved_date, last_update, artist, artist_unicode, title, title_unicode,
                 creator, creator_id, bpm, source, tags, genre_id, language_id, favourite_count, rating, storyboard,
                 video, download_unavailable, audio_unavailable, playcount, passcount, packs, max_combo, diff_aim,
                 diff_speed, difficultyrating ,sliderMultiplier):
        self.hit_objects = hit_objects
        self.timing_points = timing_points 
        self.audio = audio
        self.split = split
        self.folder = folder
        self.beatmapset_id = beatmapset_id
        self.beatmap_id = beatmap_id
        self.approved = approved
        self.total_length = total_length
        self.hit_length = hit_length
        self.version = version
        self.file_md5 = file_md5
        self.diff_size = diff_size
        self.diff_overall = diff_overall
        self.diff_approach = diff_approach
        self.diff_drain = diff_drain
        self.mode = mode
        self.count_normal = count_normal
        self.count_slider = count_slider
        self.count_spinner = count_spinner
        self.submit_date = submit_date
        self.approved_date = approved_date
        self.last_update = last_update
        self.artist = artist
        self.artist_unicode = artist_unicode
        self.title = title
        self.title_unicode = title_unicode
        self.creator = creator
        self.creator_id = creator_id
        self.bpm = bpm
        self.source = source
        self.tags = tags
        self.genre_id = genre_id
        self.language_id = language_id
        self.favourite_count = favourite_count
        self.rating = rating
        self.storyboard = storyboard
        self.video = video
        self.download_unavailable = download_unavailable
        self.audio_unavailable = audio_unavailable
        self.playcount = playcount
        self.passcount = passcount
        self.packs = packs
        self.max_combo = max_combo
        self.diff_aim = diff_aim
        self.diff_speed = diff_speed
        self.difficultyrating = difficultyrating
        self.sliderMultiplier = sliderMultiplier

    def __repr__(self):
        return f"OsuBeatmap({self.title} - {self.artist})"


    
def process_hitobject(hitobject: str, uninherited_timingpointvar: int, inherited_timingpointvar: int, last_inherited_timingpointvar: int, timing_points: List[str],sm:str) -> Tuple[List[str], int, int]:
    parts = hitobject.split(',')
    hit_type = int(parts[3])
    sm = float(sm)

    # Convert hit_type to a binary string
    binary_representation = bin(hit_type)[-4:]

    # Check the last three bits
    last_three_bits = binary_representation.zfill(3)  # Ensure it's always 3 bits long

    # Find indices of '1' bits
    indices_of_ones = [i for i, bit in enumerate(reversed(last_three_bits)) if bit == '1']

    #calc corresponding timing point
    timestamp = int(parts[2])
    timingpoint_for_this_hitobject = []
    
    lastIndexFlag = False

    uninherited_temp_ptr = inherited_timingpointvar
    while (int(round(float(timing_points[uninherited_temp_ptr][0]))) <= timestamp):
        if float(timing_points[uninherited_temp_ptr][1])>0:
            uninherited_timingpointvar = uninherited_temp_ptr
        uninherited_temp_ptr += 1
        if uninherited_temp_ptr == len(timing_points):
            break
    
    inherited_timingpoint_temp_var = inherited_timingpointvar
    while (int(round(float(timing_points[inherited_timingpoint_temp_var][0]))) <= timestamp):
        if float(timing_points[inherited_timingpoint_temp_var][1])<0:
            last_inherited_timingpointvar = inherited_timingpointvar
            inherited_timingpointvar = inherited_timingpoint_temp_var
        inherited_timingpoint_temp_var += 1
        if inherited_timingpoint_temp_var == len(timing_points):
            break

    uninherited_timingpoint_for_this_hitobject = timing_points[uninherited_timingpointvar]
    if last_inherited_timingpointvar == -1:
        timingpoint_for_this_hitobject = [timestamp,-100,0,0,0,0,0,0]
    elif inherited_timingpointvar == len(timing_points):
         timingpoint_for_this_hitobject = timing_points[last_inherited_timingpointvar]
    elif float(timing_points[inherited_timingpointvar][1])<0:
        timingpoint_for_this_hitobject = timing_points[inherited_timingpointvar]
    else:
        timingpoint_for_this_hitobject = timing_points[last_inherited_timingpointvar]
        
    timingpoint_for_this_hitobject_with_posVal = list(timingpoint_for_this_hitobject)
    timingpoint_for_this_hitobject_with_posVal[1] = str(abs(float(timingpoint_for_this_hitobject_with_posVal[1])))

    footerListWithTimingPointsData = [float(value) for value in timingpoint_for_this_hitobject_with_posVal] + [float(value) for value in uninherited_timingpoint_for_this_hitobject]
    
    if 0 in indices_of_ones:
        parts = [int(parts[0]),int(parts[1]),int(parts[2]),int(parts[2]),[]]
        return [parts,uninherited_timingpointvar,inherited_timingpointvar,last_inherited_timingpointvar]
    
    elif 1 in indices_of_ones:
        # Handle hit objects of type 2 or 6 (slider)
        # slider_data = parts[5]
            #calc slider duration
        beatlen=float(timing_points[uninherited_timingpointvar][1])
        svm = (100.0/abs(float(timingpoint_for_this_hitobject[1])))
        length = float(parts[7])
        slider_duration_acc_to_len = ((length * beatlen)/(sm * 100 * svm))
        return [split_slider(hitobject,slider_duration_acc_to_len),uninherited_timingpointvar,inherited_timingpointvar,last_inherited_timingpointvar]
    
    elif 3 in indices_of_ones:
        # Handle hit objects of type 8 or 12
        return [[int(parts[0]),int(parts[1]),int(parts[2]),int(parts[5]),[]],uninherited_timingpointvar,inherited_timingpointvar,last_inherited_timingpointvar]
    else:
        # print(hit_type)
        # print(hitobject)
        raise ValueError("Anomoly where hit type is other tan 1,5,2,6,8,12 ... in beatmap.py line 141.")  # Return as-is if no special handling    


    
def parse_osu_file(file_path) -> Tuple[OsuBeatmap , List[str]]:
    with open(file_path, 'r', encoding='utf-8') as file:
        data = {}
        hit_objects = []
        timing_points  = []
        current_section = None
        uninherited_timingpointvar = 0
        inherited_timingpointvar = 0
        sliderMultiplier = 0
        last_inherited_timingpointvar = -1
        hyperParamFooter = ["","","","","","","","","",]
        # inherit data
        for line in file:
            line = line.strip()
            if line.startswith('['):
                # Handle section headers
                current_section = line[1:-1]
                continue
            
            if current_section == 'HitObjects' and line:
                objs,uninherited_timingpointvar,inherited_timingpointvar,last_inherited_timingpointvar = process_hitobject(line,uninherited_timingpointvar,inherited_timingpointvar,last_inherited_timingpointvar,timing_points,sliderMultiplier)
                # print(objs)
                hit_objects.append(objs)
                continue
            
            if current_section == 'TimingPoints' and line:
                if(line == ""):
                    continue
                tp = line.split(',')
                # print("Before conv ",tp)
                timing_points.append(tp)
                # print("After conversion:", timing_points)

            if ':' in line:
                key, value = line.split(':', 1)
                data[key.strip()] = value.strip()
                if key.strip() == 'StackLeniency':
                    hyperParamFooter[0] = (value.strip())
                elif key.strip() == 'DistanceSpacing':
                    hyperParamFooter[1] = (value.strip())
                elif key.strip() == 'BeatDivisor':
                    hyperParamFooter[2] = (value.strip())
                elif key.strip() == 'HPDrainRate':
                    hyperParamFooter[3] = (value.strip())
                elif key.strip() == 'CircleSize':
                    hyperParamFooter[4] = (value.strip())
                elif key.strip() == 'OverallDifficulty':
                    hyperParamFooter[5] = (value.strip())
                elif key.strip() == 'ApproachRate':
                    hyperParamFooter[6] = (value.strip())
                elif key.strip() == 'SliderTickRate':
                    hyperParamFooter[7] = (value.strip())
                elif key.strip() == 'SliderMultiplier':
                    sliderMultiplier = (value.strip())
                    hyperParamFooter[8]=sliderMultiplier
        
    # print("herererere")
    # Create an instance of OsuBeatmap using the parsed data
    return [OsuBeatmap(
        hit_objects=hit_objects,
        timing_points = timing_points, 
        audio=data.get('AudioFilename', ''),
        split='',  # Handle this if necessary
        folder='',  # Handle this if necessary
        beatmapset_id=int(data.get('BeatmapSetID', 0)),
        beatmap_id=int(data.get('BeatmapID', 0)),
        approved=int(data.get('Approved', 0)),  # This might be absent in the provided format
        total_length=int(data.get('PreviewTime', 0)),  # osu! files might use 'PreviewTime' or similar
        hit_length=int(data.get('HitLength', 0)),
        version=data.get('Version', ''),
        file_md5='',  # This might be absent in the provided format
        diff_size=float(data.get('CircleSize', 0)),
        diff_overall=float(data.get('OverallDifficulty', 0)),
        diff_approach=float(data.get('ApproachRate', 0)),
        diff_drain=float(data.get('HPDrainRate', 0)),
        mode=int(data.get('Mode', 0)),
        count_normal=0,  # This might be calculated or parsed separately
        count_slider=0,  # This might be calculated or parsed separately
        count_spinner=0,  # This might be calculated or parsed separately
        submit_date='',  # osu! files might not contain submit date
        approved_date='',  # osu! files might not contain approved date
        last_update='',  # osu! files might not contain last update date
        artist=data.get('Artist', ''),
        artist_unicode=data.get('ArtistUnicode', ''),
        title=data.get('Title', ''),
        title_unicode=data.get('TitleUnicode', ''),
        creator=data.get('Creator', ''),
        creator_id=0,  # This might be absent in the provided format
        bpm=float(data.get('BPM', 0)),
        source=data.get('Source', ''),
        tags=data.get('Tags', ''),
        genre_id=int(data.get('Genre', 0)),
        language_id=int(data.get('Language', 0)),
        favourite_count=0,  # This might be absent in the provided format
        rating=0.0,  # This might be absent in the provided format
        storyboard=bool(int(data.get('Storyboard', 0))),
        video=bool(int(data.get('Video', 0))),
        download_unavailable=bool(int(data.get('DownloadUnavailable', 0))),
        audio_unavailable=bool(int(data.get('AudioUnavailable', 0))),
        playcount=0,  # This might be absent in the provided format
        passcount=0,  # This might be absent in the provided format
        packs=data.get('Packs', ''),
        max_combo=int(data.get('MaxCombo', 0)),
        diff_aim=float(data.get('DifficultyAim', 0)),
        diff_speed=float(data.get('DifficultySpeed', 0)),
        difficultyrating=float(data.get('DifficultyRating', 0)),
        sliderMultiplier=float(data.get('SliderMultiplier', 0))
    ),hyperParamFooter]





# Function to split slider into segments
def split_slider(slider_data: str, slider_duration_acc_to_len: float):
    # print(slider_data)
    # Parse slider_data
    parts = slider_data.split(',')
    path_data = parts[5][2:]

    start_point = tuple(map(int, parts[:2]))
    # Calculate path points
    path_points = [start_point] + [tuple(map(int, p.split(':'))) for p in path_data.split('|')]

    hitobject = [int(parts[0]),int(parts[1]),int(parts[2]),int(int(parts[2])+slider_duration_acc_to_len),path_points]
    return hitobject




from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
# Define a thread-safe counter
progress_lock = threading.Lock()
progress_counter = 0

def process_file(index, row):
    try:
        file_path = os.path.join(archive_path, "train", row['folder'], f"{row['audio']}.osu")
        audio_file_path = os.path.join("processed-audio", f"{row['audio']}-a.npy")
        
        if os.path.exists(file_path) and os.path.exists(audio_file_path):
            # Parse the osu! file
            data, hyperParamFooter = parse_osu_file(file_path)
            
            # Add the new columns to the row
            row_dict = row.to_dict()
            row_dict['StackLeniency'] = hyperParamFooter[0]
            row_dict['DistanceSpacing'] = hyperParamFooter[1]
            row_dict['BeatDivisor'] = hyperParamFooter[2]
            row_dict['HPDrainRate'] = hyperParamFooter[3]
            row_dict['CircleSize'] = hyperParamFooter[4]
            row_dict['OverallDifficulty'] = hyperParamFooter[5]
            row_dict['ApproachRate'] = hyperParamFooter[6]
            row_dict['SliderTickRate'] = hyperParamFooter[7]
            row_dict['SliderMultiplier'] = hyperParamFooter[8]

            # # # Convert the row to a DataFrame and append to new_df
            # new_df = pd.concat([new_df, pd.DataFrame([row])], ignore_index=True)
            # print("hyperParamFooter")
        
            # Convert the data to a NumPy array
            data_array = np.array(data.hit_objects, dtype=object)
        
            # Define the save path in the processed folder
            save_path = os.path.join("processed-beatmaps", f"{row['audio']}-b.npy")
        
            # Save the NumPy array to disk
            np.save(save_path, data_array)
            # print(data.hit_objects)
            # for h in data.hit_objects:
            #     print(h)
            
            # Increment the progress counter in a thread-safe manner
            with progress_lock:
                global progress_counter
                progress_counter += 1
                print(f"Processed {progress_counter} files")

            return [True,row_dict]
        else:
            print(f"File not found: {file_path}")
            return [False,row]
        
    except Exception as e:
        logging.error(f"Error processing row {index}: {e}")
        print(f"Error processing row {index}: {e}")
        return [False,row]

import os
import pandas as pd

def load_osu_files_from_df_notparallel(df):
    os.makedirs("processed-beatmaps", exist_ok=True)
    results = []
    error_count = 0
    error_rows = []

    for index, row in df.iterrows():
        try:
            result = process_file(index, row)
            if result[0]:
                results.append(result[1])
            else:
                error_count += 1
                error_rows.append(result[1])
        except Exception as e:
            # Handle unexpected exceptions
            error_count += 1
            error_rows.append({'index': index, 'error': str(e)})

    new_df = pd.DataFrame(results)    
    # Save metadata also for retrieval later
    save_df_as_csv(new_df)

import logging
# Set up logging
logging.basicConfig(filename='error_log.txt', level=logging.DEBUG, format='%(asctime)s - %(message)s')

def load_osu_files_from_df(df):
    os.makedirs("processed-beatmaps", exist_ok=True)
    results = []
    error_count = 0
    error_rows = []

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_file, index, row) for index, row in df.iterrows()]

        for future in as_completed(futures):
            result = future.result()
            if result[0]:
                results.append(result[1])
            else:
                error_count += 1
                error_rows.append(result[1])

    new_df = pd.DataFrame(results)
    

    # Save the successfully processed data
    save_df_as_csv(new_df)
    
    # Log summary of errors
    print(f"Total errors encountered: {error_count}")


def save_df_as_csv(df):
    # Ensure the processed metadata folder exists
    os.makedirs("processed-metadata", exist_ok=True)
    
    # Define the save path in the processed metadata folder
    save_path = os.path.join("processed-metadata", "processed-metadata.csv")
    
    # Save the DataFrame as a CSV file
    df.to_csv(save_path, index=False)
    print(f"DataFrame saved as a CSV at: {save_path}")