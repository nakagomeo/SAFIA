'''
2019/04/25
Yu Nakagome

Amplitude base SAFIA for TEAI
paper : https://www.jstage.jst.go.jp/article/ast/22/2/22_2_149/_pdf/-char/en

sources must be smaller than  #mic
'''
import numpy as np
import wave
import scipy.signal as signal
from scipy import fromstring, int16
import os
import argparse

def generate_mask(x_array, threshold):
    num_mic = x_array.shape[0]
    num_frames, num_fft = x_array[0].shape
    mask = np.zeros((num_mic,num_frames,num_fft))
    x_power = np.abs(x_array)
    for i in range(num_mic):
        #print(np.array([x_power[i] - x_power[k] > threshold for k in range(num_mic) if k != i ]))
        #mask[i] = np.all(np.array([x_power[i] - x_power[k] > threshold for k in range(num_mic) if k != i ]), axis=0)
        mask[i] = np.all(np.array([20.*np.log10(x_power[i]/x_power[k]) > threshold for k in range(num_mic) if k != i ]), axis=0)

    print("mask one rate : ",[np.sum(mask[k])/(num_frames*num_fft) for k in range(num_mic)])
    return mask

def safia(x_array,threshold= 0.0):
    mask = generate_mask(x_array,threshold)
    y_array = np.multiply(mask,x_array)
    return y_array

def outputSeparateWave(folderName,convData,orgFile):

    wr=wave.open(orgFile, "rb")
    micNum=np.shape(convData)[1]
    for m in range(micNum):
        convData2=convData[:,m].astype(np.int16)
        #print(num_imp[::1])
        fileName=folderName+str(m+1)+".wav"

        print(fileName)
        w = wave.Wave_write(fileName)
        w.setnchannels(1)
        w.setsampwidth(wr.getsampwidth())
        w.setframerate(wr.getframerate())
        w.setnframes(wr.getnframes())
        w.setparams(wr.getparams())
        w.writeframes(convData2)
        w.close()
    wr.close()


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--thres_list",type=list, default = [-10.0,-3.0,-1.0,-0.5,0.0,0.5,1.0,3.0,10.0])
    parser.add_argument("--cut_length",type=int, default = 30000000)
    args = parser.parse_args()

    wav_dir = '/mnt/aoni02/nakagome/tutorial_english/DB/pin_mic/2019_04_16_no1/'
    output_dir = '/mnt/aoni02/nakagome/tutorial_english/safia_output/2019_04_16_no1/'
    wav_list =['ch1.wav','ch2.wav','ch3.wav','ch4.wav','ch5.wav']
    #thres_list = [20.0,15.0,10.0,5.0,3.0,1.0,0.5,0.0,-0.5,-1.0,-3.0,-5.0,-7.0,-10.0,-15.0,-20.0]
    fs = 44100
    nfft = 4096
    nperseg = 4096

    thres_list = args.thres_list
    x_array = []
    for wavFile in wav_list:
        org = wave.open(os.path.join(wav_dir,wavFile), "rb")
        orgData = org.readframes(org.getnframes())
        #from IPython import embed;embed()
        print("Sec : ", float(org.getnframes()) / org.getframerate())
        num_org_data = fromstring(orgData, dtype = int16)
        org.close()
        num_org_data=num_org_data.astype(np.float64)
        print(len(num_org_data))
        num_org_data = num_org_data[:args.cut_length]
        t,f,stft = signal.stft(num_org_data,fs=fs,nperseg=nperseg,nfft=nfft,axis=0)
        x_array.append(stft)
    x_array = np.array(x_array)
    x_array = np.transpose(x_array,[0,2,1])
    #from IPython import embed;embed();exit()
    for t in thres_list:
        print("thres : ",t)
        output_file = output_dir + 'thres_sn_'+str(t)+'_ch'
        y_array = safia(x_array,threshold=t)
        print(y_array.shape)
        t,y_istft = signal.istft(y_array,fs=fs,nperseg=nperseg,nfft=nfft,time_axis=1,freq_axis=2)
        #from IPython import embed;embed();exit()
        y_istft = np.transpose(y_istft)
        outputSeparateWave(output_file,y_istft,os.path.join(wav_dir,wavFile))
