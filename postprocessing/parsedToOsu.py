import os
from typing import List, Tuple
import numpy as np
import math
from collections import Counter

def unparse_osu_data(processed_beatmap: np.ndarray,metadata: dict,hyper_params: dict): 
    """
    Reconstructs an osu! beatmap file from processed beatmap data, metadata, hyperparameters, and SliderMultiplier changes.

    :param processed_beatmap: NumPy array of shape (N, 4) where each row is [x, y, timestamp, object_type]
    :param metadata: Dictionary containing metadata fields like Title, Artist, BPM, etc.
    :param hyper_params: Dictionary containing hyperparameters:
                         {
                             "StackLeniency": float,
                             "DistanceSpacing": float,
                             "BeatDivisor": int,
                             "HPDrainRate": float,
                             "CircleSize": float,
                             "OverallDifficulty": float,
                             "ApproachRate": float,
                             "SliderTickRate": float,
                             "SliderMultiplier": float  # Default SliderMultiplier
                         }
    :param slider_multiplier_changes: List of tuples [(timestamp1, SliderMultiplier1), (timestamp2, SliderMultiplier2), ...]
    :param output_path: Path to save the reconstructed .osu file.
    """
    print(metadata)
   
    # Extract BPM from metadata
    try:
        bpm = float(metadata.get('bpm', 120.0))  # Default BPM = 120 if not provided
    except ValueError:
        print("value error in bpm")
        bpm = 120.0  # Default BPM if invalid value is provided

    # Calculate beat length in milliseconds
    beat_length = 60000 / bpm  # beat_length in ms

    # Create base TimingPoints list with base timing point (uninherited)
    
    # timing_points = [
    #     [str(0), str(beat_length), "4", "2", "1", "100", "1", "0"]  # Base timing point
    # ]


    # Initialize osu! sections
    general_section = (
        "[General]\n"
        f"AudioFilename: {metadata.get('AudioFilename', 'audio.mp3')}\n"
        f"AudioLeadIn: 0\n"
        f"PreviewTime: {metadata.get('PreviewTime', 58020)}\n"
        f"Countdown: 0\n"
        f"SampleSet: Soft\n"
        f"StackLeniency: {hyper_params.get('StackLeniency', 0.7)}\n"
        f"Mode: {metadata.get('Mode', 0)}\n"
        f"LetterboxInBreaks: 0\n"
        f"SpecialStyle: 0\n"
        f"WidescreenStoryboard: 0\n"
        "\n"
    )

    metadata_section = (
        "[Metadata]\n"
        f"Title: {metadata.get('Title', 'Unknown Title')}\n"
        f"TitleUnicode: {metadata.get('TitleUnicode', '')}\n"
        f"Artist: {metadata.get('Artist', 'Unknown Artist')}\n"
        f"ArtistUnicode: {metadata.get('ArtistUnicode', '')}\n"
        f"Creator: {metadata.get('Creator', 'Unknown Creator')}\n"
        f"Version: {metadata.get('Version', 'Easy')}\n"
        f"Source: {metadata.get('Source', '')}\n"
        f"Tags: {metadata.get('Tags', '')}\n"
        f"BeatmapID: {metadata.get('BeatmapID', 0)}\n"
        f"BeatmapSetID: {metadata.get('BeatmapSetID', 0)}\n"
        "\n"
    )

   
    default_slider_multiplier = float(hyper_params.get('SliderMultiplier', 1.82))  # Default SliderMultiplier

    difficulty_section = (
        "[Difficulty]\n"
        f"HPDrainRate: {hyper_params.get('HPDrainRate', 5.0)}\n"
        f"CircleSize: {hyper_params.get('CircleSize', 4.0)}\n"
        f"OverallDifficulty: {hyper_params.get('OverallDifficulty', 5.0)}\n"
        f"ApproachRate: {hyper_params.get('ApproachRate', 5.0)}\n"
        f"SliderMultiplier: {default_slider_multiplier}\n"
        f"SliderTickRate: {hyper_params.get('SliderTickRate', 1.4)}\n"
        f"DistanceSpacing: {hyper_params.get('DistanceSpacing', 1.2)}\n"
        f"BeatDivisor: {hyper_params.get('BeatDivisor', 4)}\n"
        "\n"
    )

    events_section = "[Events]\n"
    # Placeholder for Events; can be extended based on actual data
    events_section += "\n"

    timingpoints_section = "[TimingPoints]\n"
    timingpoints_section += f"{str(processed_beatmap[0][2])},{str(beat_length)},4,2,1,100,1,0\n"
    hitobjects_section = "[HitObjects]\n"

    # Parsing processed_beatmap to reconstruct HitObjects
    i = 0
    N = len(processed_beatmap)
    # print(f"hi1{N}")
    while i < N:
        # print(i)
        obj = processed_beatmap[i]
        x, y, timestamp, object_type = obj

        if object_type == 1:
            # Circle
            hitobject = f"{int(x)},{int(y)},{int(timestamp)},1,0,0:0:0:0:\n"
            hitobjects_section += hitobject
            i += 1

        elif object_type == 2:
            # print("slider!")
            # Slider Start
            # slider_start = obj
            slider_points = []
            slider_points.append(processed_beatmap[i])
            j=i+1
            while (j < N) and (processed_beatmap[j][3] == 3):
                slider_points.append(processed_beatmap[j])
                j += 1
            
            if len(slider_points) < 2:
                raise ValueError("slider less than 2")
            # print(f"slider point len = {len(slider_points)}")

            # Assuming slider_points is already defined as a list of data where
            # each data's 0th and 1st positions are x and y coordinates respectively.
            # Example:
            # slider_points = [
            #     (1, 2, other_data),
            #     (4, 5, other_data),
            #     (6, 7, other_data),
            #     (4, 5, other_data),
            #     (1, 2, other_data)
            # ]

            # Step 1: Count frequencies of each (x, y) pair
            freq_counter = Counter((point[0], point[1]) for point in slider_points)

            # Step 2: Extract unique points in their original order
            unique_points = []
            seen = set()
            for point in slider_points:
                xy = (point[0], point[1])
                if xy not in seen:
                    unique_points.append(point)  # Append the entire point (including additional data)
                    seen.add(xy)
            # print(f"uniq points len = {len(unique_points)}")
            # Step 3: Determine if there are any repeats
            if any(freq > 1 for freq in freq_counter.values()):
                # There are repeating points
                if len(unique_points) < 3:
                    # Not enough unique points to have internal repeats
                    n = 1
                else:
                    # Extract frequencies of internal unique points (excluding first and last)
                    internal_freqs = [freq_counter[tuple(pt)] for pt in unique_points[1:-1]]
                    if internal_freqs:
                        n = min(internal_freqs)
                    else:
                        n = 1
            else:
                # No repeating points
                n = 1
            
            # # Step 4: Assign frequencies
            # # Assign frequency of 1 to first and last unique points
            # # Assign frequency of n to all other unique points
            # result_freq_table = {}
            # for i, pt in enumerate(unique_points):
            #     if i == 0 or i == len(unique_points) - 1:
            #         result_freq_table[pt] = 1
            #     else:
            #         result_freq_table[pt] = n




            ###################### CALC SLIDER POINTS    
    
            key_points = [unique_points[0]]  # Always include the first point
            tolerance=1e-6
            for i in range(1, len(unique_points) - 1):
                prev_point = key_points[-1]
                current_point = unique_points[i]
                next_point = unique_points[i + 1]
                
                # Calculate the area of the triangle formed by prev_point, current_point, next_point
                area = abs((current_point[0] - prev_point[0]) * (next_point[1] - prev_point[1]) -
                        (current_point[1] - prev_point[1]) * (next_point[0] - prev_point[0]))
                
                if area > tolerance:
                    key_points.append(current_point)
                # If area <= tolerance, the current_point is colinear and can be skipped
            
            key_points.append(unique_points[-1])
            # print(f"uniq points = {key_points}")
            if len(unique_points) == 2:
                slider_type = "L"
            elif len(unique_points) == 3:
                slider_type = "P"
            else:
                slider_type = "B"

            curve_points = ""
            for i in range(len(key_points)-1):
                curve_points = curve_points + "|" + f"{key_points[i+1][0]}:{key_points[i+1][1]}"
            
            

            # print(f"curve points {curve_points}")
            # Initialize total distance
            total_distance = 0.0

            # Iterate through the unique_points to calculate distances between consecutive points
            for i in range(1, len(unique_points)):
                # Extract x and y coordinates of the previous point
                x1, y1 = unique_points[i - 1][0], unique_points[i - 1][1]
                
                # Extract x and y coordinates of the current point
                x2, y2 = unique_points[i][0], unique_points[i][1]
                
                # Calculate Euclidean distance between the two points
                distance = math.hypot(x2 - x1, y2 - y1)
                
                # Add the distance to the total distance
                total_distance += distance    

            # print(f"total dis {total_distance}")

            ####CALC TIMING POINT
            # Calculate Beat Length in milliseconds
            beatlen = 60000.0 / bpm  # Beat Length (ms per beat)
            slider_duration = float(slider_points[-1][2]) - float(slider_points[0][2])
            print(f"for {float(slider_points[0][2])} time the ending is {float(slider_points[-1][2])} the beatlen is {beatlen} and sm is {sm}")
            # Calculate Slider Velocity Multiplier (svm)
            sm = default_slider_multiplier
            svm = (total_distance * beatlen) / (sm * 100.0 * slider_duration)
            if svm == 0:
                svm = 1
            timingpoint_for_this_hitobject = -1*(100.0/svm)
            timing_point = f"{slider_points[0][2]},{timingpoint_for_this_hitobject},4,1,0,50,1,0\n"
            timingpoints_section += timing_point
            # print(f"timing_point {timing_point}")
            ########
            slider_hitobject = f"{int(x)},{int(y)},{int(timestamp)},2,0,{slider_type}{curve_points},{n},{total_distance}\n"
            hitobjects_section += slider_hitobject
            # print(f"slider_hitobject {slider_hitobject}")
            i = j  # Move past the slider
            # else:
            #     # Slider end not found; treat as a circle
            #     hitobject = f"{int(x)},{int(y)},{int(timestamp)},1,0,0:0:0:0:\n"
            #     hitobjects_section += hitobject
            #     i += 1

        elif object_type == 4:
            # Spinner Start or Spinner Point
            # spinner_start = obj
            j = i + 1
            spinner_points = []
            while j < N and processed_beatmap[j][3] == 5:
                spinner_points.append(processed_beatmap[j])
                j += 1
            # Determine spinner end
            if j < N and processed_beatmap[j][3] == 4:
                # Next spinner starts, current spinner ends before that
                spinner_end = spinner_points[-1] if spinner_points else None
            elif j < N and processed_beatmap[j][3] in [1, 2, 0]:
                # Current spinner ends before a different object
                spinner_end = spinner_points[-1] if slider_points else None
            else:
                raise ValueError("spinner out of bounds")

            if spinner_end is None and len(spinner_points) > 0:
                # Assume the last spinner point is the end
                spinner_end = spinner_points[-1]
                spinner_points = spinner_points[:-1]

            if spinner_end is not None:
                end_x, end_y, end_timestamp, end_object_type = spinner_end
                # Spinner in osu! is represented by type 8 with endTime
                spinner_hitobject = f"{int(x)},{int(y)},{int(timestamp)},8,0,{int(end_timestamp)},0:0:0:0:\n"
                hitobjects_section += spinner_hitobject
                i = j + 1  # Move past the spinner
            else:
                # Spinner end not found; skip
                i += 1

        else:
            # Ignore other object types or treat as nothing
            i += 1
    
    
    # for tp in timing_points:
    #     timingpoints_section += f"{tp}" + '\n'
    timingpoints_section += "\n"
    # print("hi3")
    # Combine all sections
    osu_content = (
        general_section +
        metadata_section +
        difficulty_section +
        events_section +
        timingpoints_section +
        hitobjects_section
    )
    # print("\n\n\n")
    # print(osu_content)
    output_path = "output.osu"

    # Write to the output file
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(osu_content)

    print(f"osu! file has been written to {output_path}")

# # Example Usage
# if __name__ == "__main__":
#     import numpy as np

#     # Example processed_beatmap array
#     # Columns: x, y, timestamp, object_type
#     # object_type: 1 = Circle, 2 = Slider Start, 3 = Slider Intermediate, 4 = Spinner
#     processed_beatmap = np.array([
#         [256, 192, 1000, 1],  # Circle at 1s
#         [300, 200, 2000, 2],  # Slider start at 2s
#         [350, 250, 3000, 3],  # Slider intermediate at 3s
#         [400, 300, 4000, 3],  # Slider intermediate at 4s
#         [450, 350, 5000, 1],  # Circle at 5s
#         [500, 400, 6000, 4],  # Spinner start at 6s
#         [500, 400, 7000, 4],  # Spinner end at 7s
#         [550, 450, 8000, 2],  # Slider start at 8s
#         [600, 500, 9000, 3],  # Slider intermediate at 9s
#         [650, 550, 10000, 3], # Slider intermediate at 10s
#         [700, 600, 11000, 1], # Circle at 11s
#     ])

#     # Example metadata dictionary
#     metadata = {
#         'AudioFilename': 'example.mp3',
#         'PreviewTime': 1000,
#         'Mode': 0,
#         'Title': 'Example Title',
#         'TitleUnicode': '',
#         'Artist': 'Example Artist',
#         'ArtistUnicode': '',
#         'Creator': 'CreatorName',
#         'Version': 'Easy',
#         'Source': 'SourceName',
#         'Tags': 'tag1 tag2',
#         'BeatmapID': 123456,
#         'BeatmapSetID': 654321,
#         'BPM': '180'  # BPM value
#     }

#     # Example hyper_params dictionary
#     hyper_params = {
#         "StackLeniency": 0.7,
#         "DistanceSpacing": 1.2,
#         "BeatDivisor": 4,
#         "HPDrainRate": 5.0,
#         "CircleSize": 4.0,
#         "OverallDifficulty": 5.0,
#         "ApproachRate": 5.0,
#         "SliderTickRate": 1.4,
#         "SliderMultiplier": 1.4  # Default SliderMultiplier
#     }

#     # Example slider_multiplier_changes list
#     slider_multiplier_changes = [
#         (2000, 1.4),  # At 2000ms, SliderMultiplier = 1.4
#         (8000, 1.6),  # At 8000ms, SliderMultiplier = 1.6
#         # Add more tuples as needed: (timestamp, SliderMultiplier)
#     ]

    # # Output path
    # output_path = "output.osu"

    # # Call the unparser function
    # unparse_osu_data(
    #     processed_beatmap=processed_beatmap,
    #     metadata=metadata,
    #     hyper_params=hyper_params,
    #     slider_multiplier_changes=slider_multiplier_changes,
    #     output_path=output_path
    # )
