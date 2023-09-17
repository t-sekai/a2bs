import os
import math
import shutil
import numpy as np
import lmdb as lmdb
import torch
import glob
import json
from loguru import logger
from collections import defaultdict
from torch.utils.data import Dataset
import pickle
# import librosa 
import scipy.io.wavfile


class a2bsDataset(Dataset):
    def __init__(self, loader_type = 'train', data_dir = 'datasets', facial_fps=15, audio_fps=16000, facial_length=34, stride=10, speaker_id=True, build_cache=False):
        self.loader_type = loader_type
        self.data_dir = data_dir
        self.out_lmdb_dir = data_dir + "_cache"
        self.facial_fps = facial_fps
        self.audio_fps = audio_fps
        self.facial_length = facial_length
        self.stride = stride
        self.speaker_id = speaker_id
        
        if build_cache:
            self.build_cache()
        
        self.lmdb_env = lmdb.open(self.out_lmdb_dir, readonly=True, lock=False)
        with self.lmdb_env.begin() as txn:
            self.n_samples = txn.stat()["entries"]
            
            
    def build_cache(self):
        if os.path.exists(self.out_lmdb_dir):
            shutil.rmtree(self.out_lmdb_dir)
        
        self.n_out_samples = 0
        audio_file_paths = [name for name in sorted(glob.glob(os.path.join(self.data_dir, '**/*.wav'), recursive=True))]
        facial_file_paths = [name for name in sorted(glob.glob(os.path.join(self.data_dir, '**/*.json'), recursive=True))]
        
            
        map_size = int(1024 * 1024 * 2048 * (self.audio_fps/16000)**3 * 4)  # in 1024 MB
        dst_lmdb_env = lmdb.open(self.out_lmdb_dir, map_size=map_size)
        n_filtered_out = defaultdict(int)
        
        for audio_file in audio_file_paths:
            audio_each_file = []
            facial_each_file = []
            vid_each_file = []
            #vid_each_file = []
            if f'{audio_file[:-4]}.json' not in facial_file_paths:
                print(f'Skipping {audio_file[:-4]}.')
                continue
            else:
                facial_file = f'{audio_file[:-4]}.json'
            
            sr, audio_each_file = scipy.io.wavfile.read(audio_file) # np array
            audio_each_file = audio_each_file[::sr//16000]
            with open(facial_file, 'r') as facial_data_file:
                facial_data = json.load(facial_data_file)
                facial_factor = math.ceil(1/((facial_data['frames'][20]['time'] - facial_data['frames'][10]['time'])/10))//self.facial_fps
                for j, frame_data in enumerate(facial_data['frames']):
                    # 60FPS to 15FPS
                    if j % facial_factor == 0:
                        facial_each_file.append(frame_data['weights']) 
            facial_each_file = np.array(facial_each_file)
            
            if self.speaker_id:
                vid_each_file.append(int(audio_file.split('/')[1]))
            
            self._sample_from_clip(
                dst_lmdb_env,
                audio_each_file, facial_each_file, vid_each_file
                ) 
        dst_lmdb_env.sync()
        dst_lmdb_env.close()

    def __len__(self):
        return self.n_samples

    
    def _sample_from_clip(self, dst_lmdb_env, audio_each_file, facial_each_file, vid_each_file):
        
        audio_start = 0
        facial_start = 0
        
        #print(f"before: {audio_each_file.shape} {facial_each_file.shape}")
        audio_each_file = audio_each_file[facial_start:]
        facial_each_file = facial_each_file[facial_start:]
        #print(f"after: {audio_each_file.shape} {facial_each_file.shape}")
        round_seconds_facial = facial_each_file.shape[0] // self.facial_fps  # assume 1500 frames / 15 fps = 100 s
        round_seconds_audio = len(audio_each_file) // self.audio_fps # assume 16,000,00 / 16,000 = 100 s

  
        clip_s_t, clip_e_t = 0, round_seconds_facial  # Assuming no cleaning, the entire duration is used.
        clip_s_f_audio, clip_e_f_audio = self.audio_fps * clip_s_t, clip_e_t * self.audio_fps # [160,000,90*160,000]
        clip_s_f_facial, clip_e_f_facial = clip_s_t * self.facial_fps, clip_e_t * self.facial_fps # [150,90*15]

        
        # Calculate audio_short_length based on facial data
        audio_short_length = math.floor(self.facial_length / self.facial_fps * self.audio_fps)

        # Calculate num_subdivision based on the stride and facial data
        num_subdivision = math.floor((clip_e_f_facial - clip_s_f_facial - self.facial_length) / self.stride) + 1
        
        sample_audio_list = []
        sample_facial_list = []
        sample_vid_list = []
        
        for i in range(num_subdivision):
            start_idx = clip_s_f_facial + i * self.stride
            fin_idx = start_idx + self.facial_length
            audio_start = clip_s_f_audio + math.floor(i * self.stride * self.audio_fps / self.facial_fps)
            audio_end = audio_start + audio_short_length

            if audio_end > clip_e_f_audio:
                break
            
            sample_audio = audio_each_file[audio_start:audio_end]
            sample_facial = facial_each_file[start_idx:fin_idx]
            sample_vid = np.array(vid_each_file) if vid_each_file != [] else np.array([-1])

            #start_time = start_idx / self.facial_fps
            #end_time = fin_idx / self.facial_fps

            sample_audio_list.append(sample_audio)
            sample_facial_list.append(sample_facial)
            sample_vid_list.append(sample_vid)

        if len(sample_audio_list) > 0:
            with dst_lmdb_env.begin(write=True) as txn:
                for audio, facial, vid in zip(sample_audio_list, sample_facial_list, sample_vid_list):
                    # Serialize the data using pickle
                    serialized_data = pickle.dumps([audio, facial, vid])

                    # Save data
                    k = "{:005}".format(self.n_out_samples).encode("ascii")
                    txn.put(k, serialized_data)
                    self.n_out_samples += 1
        

    def __getitem__(self, idx):
        with self.lmdb_env.begin(write=False) as txn:
            key = "{:005}".format(idx).encode("ascii")
            sample = txn.get(key)

            # Deserialize using pickle
            in_audio, in_facial, vid = pickle.loads(sample)

            in_audio = torch.from_numpy(in_audio).float()
            vid = torch.from_numpy(vid).int()

            if self.loader_type == "test":
                in_facial = torch.from_numpy(in_facial).float()
            else:
                in_facial = torch.from_numpy(in_facial).reshape((in_facial.shape[0], -1)).float()

            return in_audio, in_facial, vid