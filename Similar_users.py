import os
import re
import ast
import json
import numpy as np
import pandas as pd
import ast
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from sklearn.cluster import KMeans

signal_columns = ["gsr","rsp","skt","emg_zygo","emg_coru","emg_trap"]

def segment_data(df,window_size=5):
  start_time = df["time"].min()
  df["window"] = ((df["time"] - start_time) // window_size).astype(int)
  df = df.iloc[1:].reset_index(drop=True)
  df.drop(columns=['time'], inplace=True)
  print(df)
  return df

def synchronize_annotations(df_phys,df_va):
    """
    Associate each physiological sample with the most recent
    valence/arousal annotation.

    Physiological sampling rate:
        1000 Hz

    Annotation sampling rate:
        irregular

    IMPORTANT:
        Physiological values are NOT averaged here.
    """

    df_phys = df_phys.copy()
    df_va = df_va.copy()

    df_phys = df_phys.sort_values(
        "time"
    ).reset_index(drop=True)

    df_va = df_va.sort_values(
        "jstime"
    ).reset_index(drop=True)

    df_va = df_va.rename(
        columns={
            "jstime": "time"
        }
    )

    synchronized = pd.merge_asof(
        df_phys,
        df_va[
            [
                "time",
                "valence",
                "arousal"
            ]
        ],
        on="time",
        direction="backward"
    )

    # Remove physiological samples before
    # the first annotation
    synchronized = synchronized.dropna(
        subset=[
            "valence",
            "arousal"
        ]
    ).reset_index(drop=True)

    return synchronized


def load_data(file_path):
    columns = ["time", "ecg","bvp","gsr","rsp","skt","emg_zygo","emg_coru","emg_trap"]
    df = pd.read_csv(file_path, sep="\t", names=columns)
    return df

def load_valence_arousal_data(file_path):
    columns = ["jstime", "valence", "arousal"]
    df = pd.read_csv(file_path, sep="\t", names=columns)
    return df

def create_5sec_windows(df,window_size=5,sampling_rate=1000,signal_columns=signal_columns):

    df = df.copy()

    df = df.sort_values("time").reset_index(drop=True)

    # Time elapsed from beginning
    start_time = df["time"].iloc[0]

    df["elapsed_time"] = (df["time"] - start_time)

    # Window number
    df["window"] = (df["elapsed_time"] // window_size).astype(int)

    expected_samples = (window_size * sampling_rate)

    X = []

    window_records = []

    for window_no, window_df in df.groupby("window",sort=True):

        signal_data = window_df[signal_columns].to_numpy(dtype=np.float32)

        # We require exactly 5000 samples
        if len(signal_data) < expected_samples:

            print(
                f"Skipping window {window_no}: "
                f"{len(signal_data)} samples"
            )

            continue

        # --------------------------------------------------
        # Mean valence/arousal for this window
        # --------------------------------------------------

        valence_mean = window_df["valence"].mean()

        arousal_mean = window_df["arousal"].mean()

        # --------------------------------------------------
        # Opportune label
        # --------------------------------------------------

        valence_abs = abs(valence_mean)

        arousal_abs = abs(arousal_mean)

        if (valence_abs >= 0.5 or arousal_abs >= 0.5):
            label = 1
        else:
            label = 0

        # --------------------------------------------------
        # Store raw physiological window
        # --------------------------------------------------

        X.append(signal_data)

        window_records.append({

            "window": window_no,

            "start_time":
                window_df["time"].iloc[0],

            "end_time":
                window_df["time"].iloc[-1],

            # GSR
            "GSR_mean": window_df["gsr"].mean(),
            "GSR_std": window_df["gsr"].std(),

            #RESP
            "RESP_mean": window_df["rsp"].mean(),
            "RESP_std": window_df["rsp"].std(),

            # Skin Temp
            "Skin_temp_mean": window_df["skt"].mean(),
            "Skin_temp_std": window_df["skt"].std(),

            # EMG
            "EMG_mean": window_df[["emg_zygo","emg_coru","emg_trap"]].mean().mean(),
            "EMG_std": window_df[["emg_zygo","emg_coru","emg_trap"]].std().mean(),


            "valence_mean":
                valence_mean,

            "arousal_mean":
                arousal_mean,

            "valence_abs":
                valence_abs,

            "arousal_abs":
                arousal_abs,

            "label":
                label
        })

    X = np.asarray(X,dtype=np.float32)

    window_info = pd.DataFrame(window_records)

    return X, window_info

def modi_create_physiological_response_profile(df):
    data = np.asarray(df, dtype=float)

    valence = (data[:, -2] >= 0.5).astype(int)
    arousal = (data[:, -1] >= 0.5).astype(int)
    # Match if (High,High), (High,Low), (Low,High)
    match_mask = (
        ((valence == 1) & (arousal == 1)) |
        ((valence == 1) & (arousal == 0))  |
        ((valence == 0)  & (arousal == 1))
    )

    # Non-match = everything else
    non_match_mask = ~match_mask

    # Separate rows
    match_rows = df[match_mask]
    non_match_rows = df[non_match_mask]

    # Convert physiological features (all columns except last two) to float
    match_features = match_rows[:, :-2].astype(float)
    non_match_features = non_match_rows[:, :-2].astype(float)

    # Mean and Std
    match_mean = np.mean(match_features, axis=0) if match_features.size else np.zeros(df.shape[1]-2)
    non_match_mean = np.mean(non_match_features, axis=0) if non_match_features.size else np.zeros(df.shape[1]-2)

    match_std = np.std(match_features, axis=0) if match_features.size else np.zeros(df.shape[1]-2)
    non_match_std = np.std(non_match_features, axis=0) if non_match_features.size else np.zeros(df.shape[1]-2)

    # Concatenate into final profile
    final_array = np.concatenate([match_mean, non_match_mean, match_std, non_match_std])

    return final_array

# main function
user_lst=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
#user_lst=[3]
test_users=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
#test_users=[30]
all_results=[]
file_path="Opportune_moment_project/output/"
similar_user_lst=dict()
for user_t in test_users:
  test_user=f'User{user_t}'
  print(test_user)

  if(user_t==7):
      continue
  
  skip_user=False
  resp_profile=[]
  for user_no in user_lst:
      user_id=f'User{user_no}'
      print("Path exists:", os.path.exists(
          file_path + f"sub{user_no}_merged.csv"
      ))
      df = pd.read_csv(file_path + f"sub{user_no}_merged.csv")
      scaler = MinMaxScaler(feature_range=(-1, 1))
      df[["valence", "arousal"]]= scaler.fit_transform(df[["valence", "arousal"]])
      X, window_info=create_5sec_windows(df)

     
      if(user_no==user_t):


        n_sample=window_info.shape[0]
        best_split = None
        min_samples_per_class = 40
        #min_ratio = 0.70
        start_idx = int(0.7 * n_sample)
        end_idx = int(0.8 * n_sample)

        total_opp=(window_info['label']==1).sum()
        total_nonopp=(window_info['label']==0).sum()


        for split_idx in range(start_idx, end_idx):

            train_labels = window_info.iloc[:split_idx]
            test_labels = window_info.iloc[split_idx:]

            # counts
            train_opp = (train_labels['label'] == 1).sum()
            train_nonopp = (train_labels['label'] == 0).sum()

            test_opp = (test_labels['label'] == 1).sum()
            test_nonopp = (test_labels['label'] == 0).sum()

            valid_counts = (
                train_opp >= min_samples_per_class and
                train_nonopp >= min_samples_per_class )

            
            if valid_counts:
                best_split = split_idx
                break

        print("Selected split:", best_split)

        if best_split is not None:

            part1 = window_info.iloc[:best_split]
            split_percentage = (best_split / n_sample) * 100

            print(f"User {user_t}")
            print(f"Best split index: {best_split}")
            print(f"Total valid samples: {n_sample}")
            print(f"Calibration percentage: {split_percentage:.2f}%")
            train_opp_ratio=train_opp/total_opp
            train_nonopp_ratio=train_nonopp/total_nonopp
            test_opp_ratio=test_opp/total_opp
            test_nonopp_ratio=test_nonopp/total_nonopp

        else:
            print(f"No valid split found for user {user_t}")
            skip_user = True
            break

      
      if(user_no==user_t):
           final_array=modi_create_physiological_response_profile(part1[['GSR_mean', 'RESP_mean', 'Skin_temp_mean', 'EMG_mean','valence_abs','arousal_abs']].values)
      else:
           final_array=modi_create_physiological_response_profile(window_info[['GSR_mean', 'RESP_mean', 'Skin_temp_mean', 'EMG_mean','valence_abs','arousal_abs']].values)
         
      resp_profile.append(np.insert(final_array.astype(object),0,user_id))  

  if skip_user:
    continue
  
  df_profils = pd.DataFrame(
        resp_profile,
        columns=["User", "GSR_match_mean", "Resp_match_mean", "Skin_temp_match_mean", "EMG_match_mean","GSR_unmatch_mean", "Resp_unmatch_mean", "Skin_temp_unmatch_mean", "EMG_unmatch_mean",
        "GSR_match_std", "Resp_match_std", "Skin_temp_match_std", "EMG_match_std","GSR_unmatch_std", "Resp_unmatch_std", "Skin_temp_unmatch_std", "EMG_unmatch_std"
        ]
       )  
  scaler = StandardScaler()
  X_scaled = scaler.fit_transform(df_profils[["GSR_match_mean", "Resp_match_mean", "Skin_temp_match_mean", "EMG_match_mean","GSR_unmatch_mean", "Resp_unmatch_mean", "Skin_temp_unmatch_mean", "EMG_unmatch_mean",
    "GSR_match_std", "Resp_match_std", "Skin_temp_match_std", "EMG_match_std","GSR_unmatch_std", "Resp_unmatch_std", "Skin_temp_unmatch_std", "EMG_unmatch_std"]])

  best_k = 2  # Change this based on the above graphs
  kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
  df_profils["Cluster"] = kmeans.fit_predict(X_scaled)




  # Train and evaluate p-LSTM model for test user
  test_user_id = test_user
  cluster = df_profils[df_profils["User"] == test_user_id]["Cluster"].values[0]

  # Find similar users in same cluster
  similar_users = [u for u in df_profils[df_profils["Cluster"] == cluster]["User"]]

  similar_user_lst[test_user_id]=similar_users
print(similar_user_lst)
with open("Opportune_moment_project/similar_users.json", "w") as f:
    json.dump(similar_user_lst, f, indent=4) 

      