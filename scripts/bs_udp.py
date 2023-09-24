import time
from pythonosc import udp_client
import numpy as np
import torch
#from tqdm import tqdm
import scipy
import math
import pygame
from SimpleNet import FaceGenerator
import threading

blend =  [
        "browDownLeft",
        "browDownRight",
        "browInnerUp",
        "browOuterUpLeft",
        "browOuterUpRight",
        "cheekPuff",
        "cheekSquintLeft",
        "cheekSquintRight",
        "eyeBlinkLeft",
        "eyeBlinkRight",
        "eyeLookDownLeft",
        "eyeLookDownRight",
        "eyeLookInLeft",
        "eyeLookInRight",
        "eyeLookOutLeft",
        "eyeLookOutRight",
        "eyeLookUpLeft",
        "eyeLookUpRight",
        "eyeSquintLeft",
        "eyeSquintRight",
        "eyeWideLeft",
        "eyeWideRight",
        "jawForward",
        "jawLeft",
        "jawOpen",
        "jawRight",
        "mouthClose",
        "mouthDimpleLeft",
        "mouthDimpleRight",
        "mouthFrownLeft",
        "mouthFrownRight",
        "mouthFunnel",
        "mouthLeft",
        "mouthLowerDownLeft",
        "mouthLowerDownRight",
        "mouthPressLeft",
        "mouthPressRight",
        "mouthPucker",
        "mouthRight",
        "mouthRollLower",
        "mouthRollUpper",
        "mouthShrugLower",
        "mouthShrugUpper",
        "mouthSmileLeft",
        "mouthSmileRight",
        "mouthStretchLeft",
        "mouthStretchRight",
        "mouthUpperUpLeft",
        "mouthUpperUpRight",
        "noseSneerLeft",
        "noseSneerRight",
        "tongueOut"
    ]

def process(net, test_audio):
    facial_length = 34
    facial_fps = 15
    audio_fps = 16000
    pre_frames = 4
    stride = 4
    audio_short_length = math.floor(facial_length / facial_fps * audio_fps)

    audio_start = 0
    audio_end = audio_start + audio_short_length
    face_start = 0
    face_cache = np.zeros((1,(len(test_audio) * facial_fps) // audio_fps ,51))
    #with tqdm(total=(face_cache.shape[1]-facial_length)//stride+1) as pbar:
    while audio_end < len(test_audio):
        in_audio = torch.from_numpy(test_audio[audio_start:audio_end]).view(1,-1).float().cuda()
        pre_face = torch.from_numpy(face_cache[:,face_start:face_start+facial_length,:]).float()
        in_pre_face = pre_face.new_zeros((pre_face.shape[0], pre_face.shape[1], pre_face.shape[2] + 1)).cuda()
        in_pre_face[:, face_start:face_start+pre_frames, :-1] = pre_face[:, face_start:face_start+pre_frames]
        in_pre_face[:, face_start:face_start+pre_frames, -1] = 1 
        #print(in_pre_face.shape, in_audio.shape)
        with torch.no_grad():
            out_face = net(in_pre_face, in_audio)

        face_cache[:,face_start:face_start+facial_length,:] = out_face.cpu().numpy()
    
        face_start += stride
        audio_start = (face_start*audio_fps)//facial_fps
        audio_end = audio_start + audio_short_length
            #pbar.update(1)
    
    # add missing dimension for tongueOut
    tongueOut = np.zeros((face_cache.shape[0],face_cache.shape[1], 1))
    face_cache = np.concatenate((face_cache, tongueOut), axis=2)
    face_cache = np.squeeze(face_cache)
    return face_cache

def send_udp(out_face):
    #outWeight = np.zeros(52)

    ##need to implement get value in
    outWeight = out_face

    outWeight = outWeight * (outWeight > 1.0e-9)

    client = udp_client.SimpleUDPClient('127.0.0.1', 5008)
    osc_array = outWeight.tolist()
    
    fps = 15
    for i in range(len(osc_array)):
        #print(out_face[i].shape)
        for j, out in enumerate(osc_array[i]):
            client.send_message('/' + str(blend[j]), out)
        time.sleep(1/(fps + 9)) # temp
    

def play_audio(audio_file_path):
    pygame.mixer.init()
    pygame.mixer.music.load(audio_file_path)
    pygame.mixer.music.play()
    while pygame.mixer.music.get_busy():
        pass  # Keep the thread running while audio is playing

if __name__ == "__main__":
    # load in audio
    # audio_file = None
    start = time.time()
    ### just testing with cache
    test_audio_file = 'test_audio/out5.wav'
    sr, test_audio_raw = scipy.io.wavfile.read(test_audio_file) # np array
    test_audio = test_audio_raw[::sr//16000] # convert to 16khz
    print('Original sample rate:', sr)

    # load in model
    model_path = 'ckpt_model/simplenet1.pth'
    net = FaceGenerator()
    net.load_state_dict(torch.load(model_path))
    net = net.cuda().eval()

    print('Preprocessing time:', time.time() - start)
    start = time.time()
    
    face_cache = process(net, test_audio)
    
    print('Process time:', time.time() - start)
    start = time.time()
    udp_thread = threading.Thread(target=send_udp, args=(face_cache,))
    udp_thread.daemon = True  # Set the thread as a daemon to allow it to exit when the main program exits
    udp_thread.start()

    audio_thread = threading.Thread(target=play_audio, args=(test_audio_file,))
    audio_thread.daemon = True
    audio_thread.start()

    udp_thread.join()
    audio_thread.join()
   
    print('Send UDP time:', time.time() - start)
    print('done')
