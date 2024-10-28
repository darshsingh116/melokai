import os
import numpy as np
import pandas as pd

def process_timing_data(data):
    timingpoints = []
    prev_uninherited_str = None
    prev_inherited_str = None
    np.set_printoptions(precision=10, suppress=True)
    # for d in data:
    #     print(d[-16:])
    # print(data[0].shape)

    for array in data:
        if np.array_equal(np.array(array),np.zeros(35)):
            continue
        # if array[3] == 12 or array[3] == 8:
        #     print(array)
        # print(array)
        # Extract the last 16 elements
        last_16_elements = array[-16:]
        
        # Divide into uninherited and inherited timing points
        uninherited_points = last_16_elements[8:]
        inherited_points = last_16_elements[:8]

        # Modify the second value of inherited points
        inherited_points[1] = 0-float(inherited_points[1])

        # Convert elements to strings
        # Convert points to lists of strings
        if uninherited_points[0] < 0:
            uninherited_points[0] = str(inherited_points[0])
        uninherited_str_list = [str(int(value)) for value in uninherited_points]
        inherited_str_list = [str(int(value)) for value in inherited_points]
        inherited_str_list[1] =  str(inherited_points[1])
        uninherited_str_list[1] =  str(uninherited_points[1])
        
        # Join the lists into comma-separated strings
        uninherited_str = uninherited_str_list
        inherited_str = inherited_str_list

        # Append uninherited timing points if different from previous
        if uninherited_str != prev_uninherited_str:
            timingpoints.append(",".join(uninherited_str))
            prev_uninherited_str = uninherited_str

        # Append inherited timing points if different from previous
        # but only if uninherited and inherited are both different
        if inherited_str != prev_inherited_str:
            timingpoints.append(",".join(inherited_str))
            prev_inherited_str = inherited_str

    return timingpoints




def process_hitobject_data(data):
    hitobjects = []
    prev_slider_val = 2 #if 2 then prev was no slider or slider ended
    prev_slider_data = []
    length = 0
    
    for array in data:
        if np.array_equal(np.array(array),np.zeros(35)):
            continue
        # print(array[17:18])
        # print(prev_slider_data)
        hit_type = int(array[3])
        # Convert hit_type to a binary string
        binary_representation = bin(hit_type)[-4:]

        # Check the last three bits
        last_three_bits = binary_representation.zfill(3)  # Ensure it's always 3 bits long

        # Find indices of '1' bits
        indices_of_ones = [i for i, bit in enumerate(reversed(last_three_bits)) if bit == '1']
        # print(indices_of_ones)

        if 0 in indices_of_ones:
            # Handle hit objects of type 1 or 4
            # print("circle")
            hitobjects.append(",".join([str(int(array[0])),str(int(array[1])),str(int(array[2])),str(int(array[3])),str(int(array[4])),f"{int(array[13])}:{int(array[14])}:{int(array[15])}:{int(array[16])}:"]))
            
        
        elif 1 in indices_of_ones:
            # Handle hit objects of type 2 or 6 (slider)
            # print("slider")
            if prev_slider_val == 2:
                slider_type_char = "Z" #junk val
                if int(array[5]) == 1:
                    slider_type_char = "B"
                elif int(array[5]) == 2:
                    slider_type_char = "L"
                elif int(array[5]) == 3:
                    slider_type_char = "P"

                length = array[9]
                prev_slider_data = [str(int(array[0])),str(int(array[1])),str(int(array[2])),str(int(array[3])),str(int(array[4])),f"{slider_type_char}",str(int(array[8])),"sliderLen",f"{int(array[10])}",f"{int(array[11])}:{int(array[12])}",f"{int(array[13])}:{int(array[14])}:{int(array[15])}:{int(array[16])}:"]
                prev_slider_val = int(array[17])
            elif prev_slider_val == 0 and not (int(array[17]) == 2):
                length += array[9]
                prev_slider_data[5] = prev_slider_data[5] + f"|{int(array[6])}:{int(array[7])}" 
                prev_slider_data[8] = prev_slider_data[8] + f"|{int(array[10])}"
                prev_slider_data[9] = prev_slider_data[9] + f"|{int(array[11])}:{int(array[12])}"
                #make more here
                prev_slider_val = int(array[17])

            elif (prev_slider_val == 1) or (int(array[17]) == 2):
                length += array[9]
                prev_slider_data[5] = prev_slider_data[5] + f"|{int(array[6])}:{int(array[7])}" 
                prev_slider_data[8] = prev_slider_data[8] + f"|{int(array[10])}"
                prev_slider_data[9] = prev_slider_data[9] + f"|{int(array[11])}:{int(array[12])}"
                prev_slider_data[7] = str(length/array[8])
                prev_slider_val = int(array[17])
                # print(prev_slider_data)
                if prev_slider_val == 2 :
                    # print("here")
                    #append to hitobjects
                    hitobjects.append(",".join(prev_slider_data))
                    # prev_slider_data = []
                
        
        elif 3 in indices_of_ones:
            # Handle hit objects of type 8 or 12
            # print("spinner")
            hitobjects.append(",".join([str(int(array[0])),str(int(array[1])),str(int(array[2])),str(int(array[3])),str(int(array[4])),str(int(array[18])),f"{int(array[13])}:{int(array[14])}:{int(array[15])}:{int(array[16])}:"]))

        else:
            # print(hit_type)
            # print(hitobject)
            raise ValueError("Anomoly where hit type is other tan 1,5,2,6,8,12 ... in makeBeatmap.py line 99.")  # Return as-is if no special handling    
    return hitobjects


def load_and_print_npy_data(df):
    for index, row in df.iterrows():
        # Construct the full path to the .npy file
        folder_path = os.path.join("processed-beatmaps", f"{row['audio']}-b.npy")
        # Load the .npy file into a NumPy array
        if os.path.exists(folder_path):
            data_array = np.load(folder_path)
            
            # Print the loaded data
            print(f"Loaded data from {folder_path}:")
            # print(data_array)
            timingpoint = np.array(process_timing_data(data_array))
            hitobjects = np.array(process_hitobject_data(data_array))
            # print(timingpoint)
            # for h in hitobjects:
            #     print(h)
            # for t in timingpoint:
            #     print(t)
            # Define file name

            # Define folder and file name
            folder_name = 'aiOutput'
            os.makedirs(folder_name, exist_ok=True)
            filename = os.path.join(folder_name, f"{row['title']}.osu")

            # Open the file in write mode
            with open(filename, 'w') as file:
                # Write the [General] section
                file.write("[General]\n")
                file.write(f"AudioFilename:audio.mp3\n")
                file.write("\n")  # Leave a blank line
                
                # Write the [Difficulty] section
                file.write("[Difficulty]\n")
                file.write(f"HPDrainRate:{row['HPDrainRate']}\n")
                file.write(f"CircleSize:{row['CircleSize']}\n")
                file.write(f"OverallDifficulty:{row['OverallDifficulty']}\n")
                file.write(f"ApproachRate:{row['ApproachRate']}\n")
                file.write(f"SliderMultiplier:{row['SliderMultiplier']}\n")
                file.write(f"SliderTickRate:{row['SliderTickRate']}\n")
                file.write("\n")  # Leave a blank line
                
                # Write the [TimingPoints] section
                file.write("[TimingPoints]\n")
                for point in timingpoint:
                    file.write(f"{point}\n")
                file.write("\n")  # Leave a blank line
                
                # Write the [HitObjects] section
                file.write("[HitObjects]\n")
                for obj in hitobjects:
                    file.write(f"{obj}\n")

            print(f"File '{filename}' has been created.")
        else:
            print(f"File {folder_path} does not exist.")
            return None



def make_beatmap_from_output(row,data_array):
        # print(data_array)
        timingpoint = np.array(process_timing_data(data_array))
        hitobjects = np.array(process_hitobject_data(data_array))
        # print(timingpoint)
        # for h in hitobjects:
        #     print(h)
        # for t in timingpoint:
        #     print(t)
        # Define file name

        # Define folder and file name
        folder_name = 'aiOutput'
        os.makedirs(folder_name, exist_ok=True)
        filename = os.path.join(folder_name, f"output.osu")

        # Open the file in write mode
        with open(filename, 'w') as file:
            # Write the [General] section
            file.write("[General]\n")
            file.write(f"AudioFilename:audio.mp3\n")
            file.write("\n")  # Leave a blank line
            
            # Write the [Difficulty] section
            file.write("[Difficulty]\n")
            file.write(f"HPDrainRate:{row['HPDrainRate']}\n")
            file.write(f"CircleSize:{row['CircleSize']}\n")
            file.write(f"OverallDifficulty:{row['OverallDifficulty']}\n")
            file.write(f"ApproachRate:{row['ApproachRate']}\n")
            file.write(f"SliderMultiplier:{row['SliderMultiplier']}\n")
            file.write(f"SliderTickRate:{row['SliderTickRate']}\n")
            file.write("\n")  # Leave a blank line
            
            # Write the [TimingPoints] section
            file.write("[TimingPoints]\n")
            for point in timingpoint:
                file.write(f"{point}\n")
            file.write("\n")  # Leave a blank line
            
            # Write the [HitObjects] section
            file.write("[HitObjects]\n")
            for obj in hitobjects:
                file.write(f"{obj}\n")

        print(f"File '{filename}' has been created.")


    
