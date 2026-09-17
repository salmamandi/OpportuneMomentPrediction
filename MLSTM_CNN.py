import os
import re
import ast
import json
import numpy as np
import pandas as pd
import ast
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input,
    LSTM,
    Conv1D,
    BatchNormalization,
    Activation,
    GlobalAveragePooling1D,
    Dense,
    Dropout,
    Concatenate
)

from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam


signal_columns = ["GSR", "Resp", "Skin_Temp", "EMG_z", "EMG_c", "EMG_t"]
def segment_data(df,window_size=5):
  start_time = df["time"].min()
  df["window"] = ((df["time"] - start_time) // window_size).astype(int)
  df['prev_valence'] = df['valence'].shift(1)
  df['prev_arousal'] = df['arousal'].shift(1)
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
    columns = ["time", "ECG", "BVP", "GSR", "Resp", "Skin_Temp", "EMG_z", "EMG_c", "EMG_t"]
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
        if len(signal_data) != expected_samples:

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

def add_previous_va(X,window_info):

    window_info = window_info.copy()

    window_info["prev_valence"] = window_info["valence_abs"].shift(1)

    window_info["prev_arousal"] = window_info["arousal_abs"].shift(1)

    # Window 0 has no previous window
    valid = window_info[["prev_valence","prev_arousal"]].notna().all(axis=1)

    X = X[valid.to_numpy()]

    window_info = window_info.loc[valid].reset_index(drop=True)

    return X, window_info   

def fit_scaler(X_train):

    n_samples, n_timesteps, n_channels = (X_train.shape)

    scaler = StandardScaler()

    X_train_2d = X_train.reshape(-1,n_channels)

    scaler.fit(X_train_2d)

    return scaler


def apply_scaler(X,scaler):

    n_samples, n_timesteps, n_channels = (X.shape)

    X_2d = X.reshape(-1,n_channels)

    X_scaled = scaler.transform(X_2d)

    X_scaled = X_scaled.reshape(n_samples,n_timesteps,n_channels)

    return X_scaled.astype(np.float32)   


def build_mlstm_fcn(signal_shape=(5000, 6),feature_dim=128):

  
    # Physiological input 
    signal_input = Input(shape=signal_shape,name="physiological_input")

    
    # LSTM branch
   
    lstm_branch = LSTM(64,return_sequences=False,name="lstm_branch")(signal_input)

    lstm_branch = Dropout(0.5)(lstm_branch)

    
    # FCN branch
   
    x = Conv1D(filters=128,kernel_size=8,padding="same")(signal_input)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv1D(filters=256,kernel_size=5,padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = Conv1D(filters=128,kernel_size=3,padding="same")(x)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)
    x = GlobalAveragePooling1D()(x)

    # MLSTM-FCN feature

    physiological_features = Concatenate(name="mlstm_fcn_features_concat")([lstm_branch,x])

    physiological_features = Dense(feature_dim,activation="relu",name="physiological_features")(physiological_features)

    # Previous valence + arousal
    previous_va_input = Input(shape=(2,),name="previous_va_input")

    va_features = Dense(16,activation="relu",name="previous_va_dense")(previous_va_input)

    # Fusion

    fused = Concatenate( name="fusion")([ physiological_features, va_features])

    fused = Dense(64,activation="relu")(fused)

    fused = Dropout(0.5)(fused)
    fused = Dense(32,activation="relu")(fused)
    fused = Dropout(0.3)(fused)
   
    # Output
  
    output = Dense(1,activation="sigmoid",name="opportune_output")(fused)

    model = Model(inputs=[signal_input,previous_va_input],outputs=output)

    model.compile(optimizer=Adam(learning_rate=1e-4),loss="binary_crossentropy",metrics=["accuracy"])

    return model    

def train_mlstm_fcn(X_train,prev_va_train,y_train,X_val=None,prev_va_val=None,y_val=None,batch_size=8,epochs=50):

    model = build_mlstm_fcn(signal_shape=(X_train.shape[1],X_train.shape[2]))

    # Class weights  
    classes = np.unique(y_train)

    weights = compute_class_weight(class_weight="balanced",classes=classes,y=y_train)

    class_weights = {int(c): float(w)for c, w in zip(classes,weights)}

    print("Class weights:",class_weights)

    # Early stopping
    callbacks = [EarlyStopping(monitor="val_loss",patience=8,restore_best_weights=True)]

    # Validation
 
    validation_data = None

    if (X_val is not None and prev_va_val is not None and y_val is not None):

        validation_data = ([X_val,prev_va_val],y_val)

   
    # Training
  

    history = model.fit([X_train,prev_va_train],
              y_train,
              validation_data=validation_data,
              epochs=epochs,

            batch_size=batch_size,

            class_weight=class_weights,

            callbacks=callbacks,

            shuffle=True,

            verbose=1)
    

    return model, history   

def evaluate_mlstm_fcn(model,X_test,prev_va_test,y_test):

    probabilities = model.predict([X_test,prev_va_test],batch_size=8,verbose=1).ravel()
    predictions = (probabilities >= 0.5).astype(int)
    accuracy = accuracy_score(y_test,predictions)
    precision = precision_score(y_test,predictions,zero_division=0)

    recall = recall_score(y_test,predictions,zero_division=0)

    f1 = f1_score(
        y_test,
        predictions,
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=[0, 1]
    )

    tn, fp, fn, tp = cm.ravel()

    tpr = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    fpr = (
        fp / (fp + tn)
        if (fp + tn) > 0
        else 0
    )

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "TPR": tpr,
        "FPR": fpr,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp
    }

    return results, probabilities, predictions  
    
# main function
user_lst=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
test_users=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30]
all_results=[]
file_path="raw/"

signal_columns = ["GSR", "Resp", "Skin_Temp", "EMG_z", "EMG_c", "EMG_t"]
sequence_length = 3

for user_t in test_users:
  test_user=f'User{user_t}'
  print(test_user)
  
  if(user_t==7):
    continue
 
  skip_user=False
  X_train_list = []
  info_train_list = []
  X_test = None
  info_test = None

  #print(f"Skip user status:{skip_user} for user {user_t}")

  with open("similar_users.json", "r") as f:
    similar_user_lst = json.load(f)
  
  similar_user_lst = ast.literal_eval(similar_user_lst)
  if test_user not in similar_user_lst:
      continue
  
  print(f'number of simialr users:{len(similar_user_lst[test_user])}')
  
  for user_no in similar_user_lst[test_user]:

    user_id=int(re.search(r'(\d+)',user_no).group(1))
    if user_id == 7:
              continue
    #print(f"For user {user_no}")
    
    user_file_path=file_path+"physiological/sub"+str(user_id)+"_DAQ.txt"
    emo_file_path=file_path+"annotations/sub"+str(user_id)+"_joystick.txt"
    df_b = load_data(user_file_path)
    df_a = load_valence_arousal_data(emo_file_path)
    df_mod=synchronize_annotations(df_b,df_a)

    scaler = MinMaxScaler(feature_range=(-1, 1))
    df_mod[["valence", "arousal"]]= scaler.fit_transform(df_mod[["valence", "arousal"]])
    X, window_info=create_5sec_windows(df_mod)
    X,window_info=add_previous_va(X,window_info)

    # for test user
    if(user_id==user_t):


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

      # Target user's train and test data
      X_train_list.append(X[:best_split])
      train_info = window_info.iloc[:best_split].copy()
      #train_info["user_id"] = user_id
      info_train_list.append(train_info)  

      X_test = X[best_split:]
      info_test = window_info.iloc[best_split:].copy()
      #info_test["user_id"] = user_id
      info_test = info_test.reset_index(drop=True) 

    else:

      X_train_list.append(X)
      train_info = window_info.copy()
      #train_info["user_id"] = user_id
      info_train_list.append(train_info)   

  if(skip_user):
    continue
  # finally we have X_train_list & info_train_list. Also X_test & info_test
  X_train = np.concatenate(X_train_list, axis=0)

  window_info_train = pd.concat(info_train_list,ignore_index=True)
  print(X_train.shape)
  print(window_info_train.shape)  

  prev_va_train = window_info_train[["prev_valence", "prev_arousal"]].to_numpy(dtype=np.float32)

  y_train = window_info_train["label"].to_numpy(dtype=np.int32)

  prev_va_test = info_test[["prev_valence", "prev_arousal"]].to_numpy(dtype=np.float32)

  y_test = info_test["label"].to_numpy(dtype=np.int32)
  # create train and validation set
  indices = np.arange(len(y_train))

  train_idx, val_idx = train_test_split(
      indices,
      test_size=0.2,
      random_state=42,
      stratify=y_train
  )

  X_model_train = X_train[train_idx]
  X_val = X_train[val_idx]

  prev_va_model_train = prev_va_train[train_idx]
  prev_va_val = prev_va_train[val_idx]

  y_model_train = y_train[train_idx]
  y_val = y_train[val_idx]

  # scaled train, validation and test data 
  scaler=fit_scaler(X_model_train)
  X_model_train=apply_scaler(X_model_train,scaler)
  X_val=apply_scaler(X_val,scaler)
  X_test=apply_scaler(X_test,scaler)
  # train the model
  model,history=train_mlstm_fcn(X_model_train,prev_va_model_train,y_model_train,X_val,prev_va_val,y_val)
  results, probabilities, predictions=evaluate_mlstm_fcn(model,X_test,prev_va_test,y_test)
  print(results)
  results["user"] = user_t

  results["split_percentage"] = (split_percentage)

  all_results.append(results)

df_results = pd.DataFrame(all_results)
df_results.to_csv("Opportune_moment_results/mlstm_cnn_results.csv",index=False)

print(df_results)