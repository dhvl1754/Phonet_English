#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import numpy as np
import python_speech_features as pyfeat
from scipy.io.wavfile import read
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import resample_poly
from tensorflow import keras
import gc
from matplotlib import cm
import seaborn as sns

from tqdm import tqdm
import speech_recognition as sr
from pydub import AudioSegment
import eng_to_ipa as ipa


# In[2]:


#Defining Phonemes

import numpy as np
import pandas as pd

class Phonological:

    def __init__(self):

        self.list_phonological = {
            'syllabic': ['aj', 'aw', 'ej', 'i', 'o', 'ow', 'u', 'æ', 'ɑ', 'ɔj', 'ə', 'ɚ', 'ɛ', 'ɪ', 'ɫ̩', 'ʊ',"ʉː", "ʉ" ,'m̩', 'n̩'],
            'diphthong': ['aj', 'aw', 'ej', 'ow', 'ɔj'],
            'consonantal': ['b', 'd', 'dʒ', 'f', 'h', 'k', 'kʰ', 'l', 'p', 'pʰ', 's', 't', 'tʃ', 'tʰ', 'v', 'z', 'ð', 'ɡ', 'ɫ', 'ɹ', 'ɾ', 'ʃ', 'ʒ', 'ʔ', 'θ', 'm', 'n', 'ŋ', 'ɱ', 'm̩', 'n̩'],
            'sonorant': ["ʉː", "ʉ",'j', 'w', 'l', 'ɫ', 'ɹ', 'ɾ', 'm', 'n', 'ŋ', 'ɱ', 'aj', 'aw', 'ej', 'i', 'o', 'ow', 'u', 'æ', 'ɑ', 'ɔj', 'ə', 'ɚ', 'ɛ', 'ɪ', 'ɫ̩', 'ʊ', 'm̩', 'n̩'],
            'continuant': ["ʉː", "ʉ",'j', 'w', 'f', 'h', 'l', 's', 'tʃ', 'v', 'z', 'ð', 'ɫ', 'ɹ', 'ɾ', 'ʃ', 'ʒ', 'θ', 'aj', 'aw', 'ej', 'i', 'o', 'ow', 'u', 'æ', 'ɑ', 'ɔj', 'ə', 'ɚ', 'ɛ', 'ɪ', 'ɫ̩', 'ʊ'],
            'nasal': ['m', 'n', 'ŋ', 'ɱ', 'm̩', 'n̩'],
            'voice': ["ʉː", "ʉ",'j', 'w', 'b', 'd', 'dʒ', 'l', 'v', 'z', 'ð', 'ɡ', 'ɫ', 'ɹ', 'ɾ', 'ʒ', 'm', 'n', 'ŋ', 'ɱ', 'aj', 'aw', 'ej', 'i', 'o', 'ow', 'u', 'æ', 'ɑ', 'ɔj', 'ə', 'ɚ', 'ɛ', 'ɪ', 'ɫ̩', 'ʊ', 'm̩', 'n̩'],
            'labial': ['w', 'b', 'f', 'p', 'pʰ', 'v', 'm', 'ɱ', 'm̩'], 
            'round': ["ʉː", "ʉ",'aw', 'o', 'ow', 'u', 'ɔj', 'ʊ'],
            'coronal': ['j', 'd', 'dʒ', 'l', 's', 't', 'tʃ', 'tʰ', 'z', 'ð', 'ɹ', 'ɾ', 'ʃ', 'ʒ', 'θ', 'n', 'n̩'],
            'distributed': ['dʒ', 'ð', 'ɹ', 'ʃ', 'ʒ', 'θ'],
            'anterior': ['d', 'l', 's', 't', 'tʰ', 'z', 'ð', 'ɹ', 'ɾ', 'θ', 'n','n̩'],
            'strident': ['dʒ', 's', 'tʃ', 'z', 'ʃ', 'ʒ'],
            'spreadglottis': ['h', 'kʰ', 'pʰ', 'tʰ'],
            'lateral': ['l', 'ɫ', 'ɫ̩'],
            'long' : ["ʉː"],
            'dorsal': ['j', 'w', 'k', 'kʰ', 'ɡ', 'ɫ', 'ŋ', 'ɫ̩'],
            'high': ["ʉː", "ʉ",'j', 'w', 'aj', 'aw', 'i', 'ow', 'u', 'ɔj', 'ɪ', 'ʊ'],
            'low': ['aj', 'aw', 'æ', 'ɑ'],
            'back': ["ʉː", "ʉ",'w', 'aj', 'aw', 'o', 'ow', 'u', 'ɑ', 'ɔj', 'ʊ'],
            'front': ['ej', 'i', 'æ', 'ɛ', 'ɪ'],
            'tense': ["ʉː", "ʉ",'ej', 'i', 'o', 'ow', 'u'],
            'rhotic': ['ɹ', 'ɾ', 'ɚ'],
            'flap': ['ɾ'],
            'pause': ['SIL','','spn']}

    def get_list_phonological(self):
        return self.list_phonological

    def get_list_phonological_keys(self):
        keys=self.list_phonological.keys()
        return list(keys)

    def get_d1(self):
        keys=self.get_list_phonological_keys()
        dict_1={"xmin":[],"xmax":[],"phoneme":[],"phoneme_code":[]}
        for k in keys:
            dict_1[k]=[]
        return dict_1

    def get_d2(self):
        keys=self.get_list_phonological_keys()
        dict_2={"n_frame":[],"phoneme":[],"phoneme_code":[]}
        for k in keys:
            dict_2[k]=[]
        return dict_2

    def get_list_phonemes(self):
        keys=self.get_list_phonological_keys()
        phon=[]
        for k in keys:
            phon.append(self.list_phonological[k])
        phon=np.hstack(phon)

        return np.unique(phon)


def main():
    phon=Phonological()
    keys=phon.get_list_phonological_keys()
    print(keys)
    d1=phon.get_d1()
    print(d1)
    d2=phon.get_d2()
    print(d2)
    ph=phon.get_list_phonemes()
    print(ph)
    print(len(ph))

if __name__=="__main__":
    main()


# In[10]:


class Phonet_English:

    def __init__(self, phonological_classes):
        
        self.Phon=Phonological()
        self.phonemes=self.Phon.get_list_phonemes()
        self.GRU_size=128
        self.hidden_size=128
        self.lr=0.0001
        self.recurrent_droput_prob=0.0
        self.size_frame=0.025
        self.time_shift=0.01
        self.nfilt=33
        self.len_seq=40
        self.names=self.Phon.get_list_phonological_keys()
        self.num_labels=[2 for j in range(len(self.names))]
        self.nfeat=34
        self.thrplot=0.7
        self.nphonemes=len(self.phonemes)
        if phonological_classes[0]=="all":
            self.keys_val=self.names
        else:
            self.keys_val=phonological_classes
        self.models=self.load_model()
        self.model_phon=self.load_model_phon()
        self.MU, self.STD=self.load_scaler()
        
    def load_model(self):
        input_size=(self.len_seq, self.nfeat)
        model_file="TrainedModel/model.h5"     #----------------Load the model for trained for probablities
        Model=self.model(input_size)
        Model.load_weights(model_file)
        return Model

    def mask_correction(self, posterior, threshold=0.5):
        
        for j in np.arange(1,len(posterior)-1):
            if (posterior[j-1]>=threshold) and (posterior[j]<threshold) and (posterior[j+1]>=threshold):
                posterior[j]=(posterior[j-1]+posterior[j+1])/2
            if (posterior[j-1]<threshold) and (posterior[j]>=threshold) and (posterior[j+1]<threshold):
                posterior[j]=(posterior[j-1]+posterior[j+1])/2
        return posterior



    def load_model_phon(self):
        input_size=(self.len_seq, self.nfeat)
        Model_phonemes="TrainedModel/phonemes_weights.hdf5"   #---------------Load the model for Phoneme recognition
        Model_phon=self.modelp(input_size)
        Model_phon.load_weights(Model_phonemes)
        return Model_phon

    def load_scaler(self):
        file_mu="TrainedModel/mu.npy"                                 #----------------Load the files
        file_std="TrainedModel/std.npy"                               #----------------Load the files
        MU=np.load(file_mu)
        STD=np.load(file_std)

        return MU, STD


    def modelp(self, input_size):
       
        input_data=keras.layers.Input(shape=(input_size))
        x=input_data
        x=keras.layers.BatchNormalization()(x)
        x=keras.layers.Bidirectional(keras.layers.GRU(self.GRU_size, recurrent_dropout=self.recurrent_droput_prob, return_sequences=True, reset_after=True))(x)
        x=keras.layers.Bidirectional(keras.layers.GRU(self.GRU_size, recurrent_dropout=self.recurrent_droput_prob, return_sequences=True, reset_after=True))(x)
        x = keras.layers.TimeDistributed(keras.layers.Dense(self.hidden_size, activation='relu'))(x)
        x = keras.layers.TimeDistributed(keras.layers.Dense(self.nphonemes, activation='softmax'))(x)
        modelGRU=keras.Model(inputs=input_data, outputs=x)
        opt=keras.optimizers.Adam(learning_rate=self.lr)
        modelGRU.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        return modelGRU


    def model(self, input_size):
      
        input_data=keras.layers.Input(shape=(input_size))
        x=input_data
        x=keras.layers.BatchNormalization()(x)
        x=keras.layers.Bidirectional(keras.layers.GRU(self.GRU_size, recurrent_dropout=self.recurrent_droput_prob, return_sequences=True, reset_after=True))(x)
        x=keras.layers.Bidirectional(keras.layers.GRU(self.GRU_size, recurrent_dropout=self.recurrent_droput_prob, return_sequences=True, reset_after=True))(x)
        x=keras.layers.Dropout(0.2)(x)
        x = keras.layers.TimeDistributed(keras.layers.Dense(self.hidden_size, activation='relu'))(x)
        x=keras.layers.Dropout(0.2)(x)
            # multi-task
        xout=[]
        out=[]
        for j in range(len(self.names)):
            xout.append(keras.layers.TimeDistributed(keras.layers.Dense(self.hidden_size, activation='relu'))(x))
            out.append(keras.layers.TimeDistributed(keras.layers.Dense(2, activation='softmax'), name=self.names[j])(xout[-1]))

        modelGRU=keras.Model(inputs=input_data, outputs=out)
        opt=keras.optimizers.Adam(learning_rate=self.lr)
        alphas=list(np.ones(len(self.names))/len(self.names))
        modelGRU.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'], sample_weight_mode="temporal", loss_weights=alphas)
        return modelGRU

    def get_feat(self, signal, fs):
        
        signal=signal-np.mean(signal)
        signal=signal/np.max(np.abs(signal))
        mult = int(fs*self.size_frame*self.len_seq)
        fill = int(self.len_seq*self.time_shift*fs)
        fillv=0.05*np.random.randn(fill)
        signal=np.hstack((signal,fillv))
        Fbank, energy=pyfeat.fbank(signal,samplerate=fs,winlen=self.size_frame,winstep=self.time_shift,
          nfilt=self.nfilt,nfft=512,lowfreq=0,highfreq=None,preemph=0.97)

        energy= np.expand_dims(energy, axis=1)
        feat2=np.concatenate((Fbank,energy),axis=1)
        return np.log10(feat2)

    def number2phoneme(self, seq):

        try:
            phonemes=[self.phonemes[j] for j in seq]

            for j in range(1,len(phonemes)-1):
                if phonemes[j]!=phonemes[j-1] and phonemes[j]!=phonemes[j+1]:
                    phonemes[j]=phonemes[j-1]

            return phonemes
        except:
            return np.nan


    def get_phon_wav(self, audio_file, feat_file="", plot_flag=True):
    
        if audio_file.find('.wav')==-1 and audio_file.find('.WAV')==-1:
            raise ValueError(audio_file+" is not a valid audio file")

        fs, signal=read(audio_file)
        if fs!=16000:
            signal=resample_poly(signal, 16000, fs)
            fs=16000
        feat=self.get_feat(signal,fs)

        nf=int(feat.shape[0]/self.len_seq)

        start=0
        fin=self.len_seq
        Feat=[]
        for j in range(nf):
            featmat_t=feat[start:fin,:]
            Feat.append(featmat_t)
            start=start+self.len_seq
            fin=fin+self.len_seq
        Feat=np.stack(Feat, axis=0)
        Feat=Feat-self.MU
        Feat=Feat/self.STD
        df={}
        dfa={}
        pred_mat_phon=np.asarray(self.model_phon.predict(Feat))
        pred_mat_phon_seq=np.concatenate(pred_mat_phon,0)
        pred_vec_phon=np.argmax(pred_mat_phon_seq,1)

        nf=int(len(signal)/(self.time_shift*fs)-1)
        if nf>len(pred_vec_phon):
            nf=len(pred_vec_phon)
        
        phonemes_list=self.number2phoneme(pred_vec_phon[:nf])

        t2=np.arange(nf)*self.time_shift
        
        df["time"]=t2
        df["phoneme"]=phonemes_list
        dfa["time"]=t2
        dfa["phoneme"]=phonemes_list
#         print(phonemes_list)
        pred_mat=np.asarray(self.models.predict(Feat))

        
        for l, problem in enumerate(self.keys_val):

            index=self.names.index(problem)
            pred_matv=pred_mat[index][:,:,1]
            post=np.hstack(pred_matv)[:nf]
            dfa[problem]=self.mask_correction(post)
        
        if plot_flag:
            self.plot_phonological(feat_file, fs, signal, dfa, phonemes_list, t2)

        dfa=pd.DataFrame(dfa)
        if len(feat_file)>0:
            dfa.to_csv(feat_file, index=False)
        gc.collect()
        return dfa

    def plot_phonological(self, feat_file, fs, signal, dfa, phonemes_list, t2):
        n_plots=int(np.ceil(len(self.keys_val)/4))
        figsize=(6,int(n_plots*3))
        colors = cm.get_cmap('Accent', 5)
        col_order=[0,1,2,3]*n_plots
        plt.figure(figsize=figsize)
        for l, problem in enumerate(self.keys_val):

            if (l==0) or (l==4) or (l==8) or (l==12) or (l==16):
                subp=int(l/4+1)
                plt.subplot(n_plots,1, subp)
                t=np.arange(len(signal))/fs
                signal=signal-np.mean(signal)
                plt.plot(t,signal/np.max(np.abs(signal)), color=colors.colors[4], alpha=0.5)
                plt.grid()

            plt.plot(t2,dfa[problem],  color=colors.colors[col_order[l]], label=problem, linewidth=2)
            ini=t2[0]
            for nu in range(1,len(phonemes_list)):
                if phonemes_list[nu]!=phonemes_list[nu-1] or nu==len(phonemes_list)-1:
                    difft=t2[nu]-ini
                    plt.text(x=ini+difft/2, y=1, s=phonemes_list[nu-1], color="k", fontsize=10)
                    ini=t2[nu]

            plt.xlabel("Time (s)")
            plt.ylabel("Phonological posteriors")
            plt.legend(loc=8, ncol=2)

        plt.tight_layout()
        plt.savefig(feat_file+"post.png")
        plt.show()

    def get_phon_path(self, audio_path, feat_path, plot_flag=False):

        hf=os.listdir(audio_path)
        hf.sort()

        if not os.path.exists(feat_path):
            os.makedirs(feat_path)

        if feat_path[-1]!="/":
            feat_path=feat_path+"/"

        pbar=tqdm(range(len(hf)))

        for j in pbar:
            pbar.set_description("Processing %s" % hf[j])
            audio_file=audio_path+hf[j]
            feat_file=feat_path+hf[j].replace(".wav", ".csv")
            self.get_phon_wav(audio_file, feat_file, plot_flag)


    def get_posteriorgram(self, audio_file):

        if audio_file.find('.wav')==-1 and audio_file.find('.WAV')==-1:
            raise ValueError(audio_file+" is not a valid audio file")

        fs, signal=read(audio_file)
        if fs!=16000:
            raise ValueError(str(fs)+" is not a valid sampling frequency")
        feat=self.get_feat(signal,fs)
        nf=int(feat.shape[0]/self.len_seq)
        start=0
        fin=self.len_seq
        Feat=[]
        for j in range(nf):
            featmat_t=feat[start:fin,:]
            Feat.append(featmat_t)
            start=start+self.len_seq
            fin=fin+self.len_seq
        Feat=np.stack(Feat, axis=0)
        Feat=Feat-self.MU
        Feat=Feat/self.STD

        pred_mat_phon=np.asarray(self.model_phon.predict(Feat))
        pred_mat_phon_seq=np.concatenate(pred_mat_phon,0)
        pred_vec_phon=np.argmax(pred_mat_phon_seq,1)
        nf=int(len(signal)/(self.time_shift*fs)-1)
        phonemes_list=self.number2phoneme(pred_vec_phon[:nf])
        t=np.arange(nf)*self.time_shift
        posteriors=[]
        pred_mat=np.asarray(self.models.predict(Feat))
        for l, problem in enumerate(self.keys_val):
            
            index=self.names.index(problem)
            pred_matv=pred_mat[index][:,:,1]
            post=np.hstack(pred_matv)[:nf]
            posteriors.append(self.mask_correction(post))

        posteriors=np.vstack(posteriors)
        plt.figure()
        plt.imshow(np.flipud(posteriors), extent=[0, t[-1], 0, len(self.keys_val)], aspect='auto')
        plt.xlabel("Time (seconds)")
        plt.ylabel("Phonological class")
        plt.yticks(np.arange(len(self.keys_val))+0.5, self.keys_val)
        ini=t[0]
        for nu in range(1,len(phonemes_list)):
            if phonemes_list[nu]!=phonemes_list[nu-1] or nu==len(phonemes_list)-1:
                difft=t[nu]-ini
                plt.text(x=ini+difft/2, y=19, s="/"+phonemes_list[nu-1]+"/", color="k", fontsize=12)
                ini=t[nu]
        plt.colorbar()
        plt.show()


    def get_PLLR(self, audio_file, feat_file="", projected=True, plot_flag=False):

        df=self.get_phon_wav(audio_file, plot_flag=plot_flag)
        dfPLLR={}
        dfPLLR["time"]=df["time"]
        dfPLLR["phoneme"] = df["phoneme"]
        PLLR=np.zeros((len(df["time"]), len(self.keys_val)))
        post=np.zeros((len(df["time"]), len(self.keys_val)))
        
        for l, problem in enumerate(self.keys_val):

            PLLR[:,l]=np.log10(df[problem]/(1-df[problem]))
            post[:,l]=df[problem]
        if projected:
            N=PLLR.shape[1]
            I=np.identity(N)
            Ones=np.ones((N,N))*1/np.sqrt(N)
            P=I-Ones.T*Ones
            PLLRp=np.matmul(PLLR,P)

        for l, problem in enumerate(self.keys_val):
            if projected:
                dfPLLR[problem]=PLLRp[:,l]
            else:
                dfPLLR[problem]=PLLR[:,l]
        dfPLLR=pd.DataFrame(dfPLLR)
        if len(feat_file)>0:
            dfPLLR.to_csv(feat_file)
        return dfPLLR

