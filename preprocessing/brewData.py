import pandas as pd
from .beatmap import *
from .preprocessaudio import *
import time
from dotenv import load_dotenv
load_dotenv()
module_path = os.getenv('MODULE_PATH')

def brew():
    #note time for audio
    # tic = time.time()
    csv_path = os.path.join(module_path,"metadata.csv")
    #select dataset from metadata
    df=pd.read_csv(csv_path)
    df = df[(df["mode"] == 0) & (df["difficultyrating"]>4.8) & (df["difficultyrating"]< 6) & (df["split"]=="train") & (df['total_length'] < 600)]
    # df = df[df["audio"]=="0013ddfd8bd55fdccc0b253e313b7a60"]

    df = df[0:2]
    # Preprocess and store all audio data tobe trained
    
    # preprocess_and_save_audio_in_parallel(df)
    preprocess_and_save_audio(df)

    #  # End the timer (Toc)
    # toc = time.time()
    # # Calculate and print the elapsed time
    # elapsed_time = toc - tic
    # print(f"Elapsed time to preprocess audio: {elapsed_time:.2f} seconds")


    # # #note time for beatmap
    # tic = time.time()

    # Cook the .osu data
    # load_osu_files_from_df(df)    #uncomment this
    load_osu_files_from_df_notparallel(df)
    #we are saving metadata inside load_osu_siles_from_df func as we are modifying df with addition of hyperparams and then saving


    #  # End the timer (Toc)
    # toc = time.time()

    # # Calculate and print the elapsed time
    # elapsed_time = toc - tic
    # print(f"Elapsed time to preprocess beatmaps: {elapsed_time:.2f} seconds")



