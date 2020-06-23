
import numpy as np
import scipy.io.wavfile
import matplotlib.pyplot as plt
from scipy import signal
import math
import librosa

List1 = []
List2 = []
data1, rate1 = librosa.load(r'90conv.wav')
data2, rate2 = librosa.load(r'drill.wav')

start = 1000000
fivesecframe = 110250

List1 = data1[start:start+fivesecframe]
List2 = data2[100001:210251]
List3 = []

for i in range(0, 110250):
    List3.append((List1[i])+(List2[i]/8))

ar1 = np.array(List3)
librosa.output.write_wav(r'drill222.wav', ar1, 22050)


p = int(8/1000*22050)
data = List3
minEnv = 0
maxEnv = 0
minLPEnv = 0
minHPEnv = 0
maxLPEnv = 0
maxHPEnv = 0
Env = 0
LPEnv = 0
HPEnv = 0
pc = 0.5
N = 5

PauseList = []
ListEnv = []
ListmaxEnv = []
ListminEnv = []
ListLPEnv = []
ListminLPEnv = []
ListmaxLPEnv = []
ListHPEnv = []
ListminHPEnv = []
ListmaxHPEnv = []
GammaT_E = 1-np.exp(-1/4)
GammaT_rd = 1-np.exp(-1/375)

ListFFTed=[]

for k in range(0, 1250): 
    i = int(k*88.2)
    Framedata = data[i:i+p]
    FFTed = []
    
    Env = 0
    LPEnv = 0
    HPEnv = 0
    
    FFTed = np.fft.fft(Framedata, 256, norm="ortho")
    FFTed = abs(FFTed)[:128]
    X = np.linspace(0, rate1/2, 128, endpoint=True)
    
    for t in range(0, 128):
        if t < 23:
            LPEnv = LPEnv + (abs((FFTed[t])))**2
        else:
            HPEnv = HPEnv + (abs((FFTed[t])))**2
        Env = LPEnv + HPEnv

    Env = 10 * math.log(Env,10)
    LPEnv = 10 * math.log(LPEnv,10)
    HPEnv = 10 * math.log(HPEnv,10)

    if ListEnv != []:
        if Env < ListEnv[-1]:
            Env = (1-GammaT_E)*ListEnv[-1] + GammaT_E*Env
        if HPEnv < ListHPEnv[-1]:
            HPEnv = (1-GammaT_E)*ListHPEnv[-1]+GammaT_E*HPEnv
        if LPEnv < ListLPEnv[-1]:
            LPEnv = (1-GammaT_E)*ListLPEnv[-1]+GammaT_E*LPEnv


    if k < 50:
        minEnv = Env
        maxEnv = Env
        minLPEnv = LPEnv
        maxLPEnv = LPEnv
        minHPEnv = HPEnv
        maxHPEnv = HPEnv

    if Env > maxEnv :
        maxEnv = Env
    if LPEnv > maxLPEnv :
        maxLPEnv = LPEnv
    if HPEnv > maxHPEnv :
        maxHPEnv = HPEnv
    if minEnv > Env :
        minEnv = Env
    if minLPEnv > LPEnv :
        minLPEnv = LPEnv
    if minHPEnv > HPEnv :
        minHPEnv = HPEnv
        

    if Env < maxEnv :
        maxEnv = (1-GammaT_rd)*ListmaxEnv[-1] + GammaT_rd*Env
    if Env > minEnv :
        minEnv = (1-GammaT_rd)*ListminEnv[-1] + GammaT_rd*Env
    if HPEnv < maxHPEnv :
        maxHPEnv = (1-GammaT_rd)*ListmaxHPEnv[-1] + GammaT_rd*HPEnv
    if HPEnv > minHPEnv :
        minHPEnv = (1-GammaT_rd)*ListminHPEnv[-1] + GammaT_rd*HPEnv
    if LPEnv < maxLPEnv :
        maxLPEnv = (1-GammaT_rd)*ListmaxLPEnv[-1] + GammaT_rd*LPEnv
    if LPEnv > minLPEnv :
        minLPEnv = (1-GammaT_rd)*ListminLPEnv[-1] + GammaT_rd*LPEnv
    

    Delta = maxEnv - minEnv
    LPDelta = maxLPEnv - minLPEnv
    HPDelta = maxHPEnv - minHPEnv
    
    
    if LPDelta < N and HPDelta < N:
        PauseList.append(True)
        
    elif LPDelta > N:
        if LPEnv - minLPEnv < (pc * LPDelta):
            if HPDelta < N:
                if Env - minEnv < 0.5*Delta:
                    PauseList.append(True)
                else:
                    PauseList.append(False)
            elif HPDelta > 2*N:
                if (HPEnv- minHPEnv) < 2*pc*HPDelta:
                    PauseList.append(True)
                else :
                    PauseList.append(False)
            elif HPEnv-minHPEnv < 0.5*HPDelta:
                PauseList.append(True)
            else :
                if HPEnv-minHPEnv > pc*HPDelta:
                    PauseList.append(False)
                elif LPDelta > 2*N:
                    if LPEnv - minLPEnv < 2*pc*LPDelta:
                        PauseList.append(True)
                    else:
                        PauseList.append(False)
                else:
                    if LPEnv - minLPEnv < 0.5*LPDelta:
                        PauseList.append(True)
                    else:
                        PauseList.append(False)
        else :
            if HPDelta < N:
                PauseList.append(False)
            else :
                if HPEnv-minHPEnv > pc*HPDelta:
                    PauseList.append(False)
                elif LPDelta > 2*N:
                    if LPEnv - minLPEnv < 2*pc*LPDelta:
                        PauseList.append(True)
                    else:
                        PauseList.append(False)
                else:
                    if LPEnv - minLPEnv < 0.5*LPDelta:
                        PauseList.append(True)
                    else:
                        PauseList.append(False)
    elif (HPEnv-minHPEnv) < pc*HPDelta:
        if (Env-minEnv)<0.5*Delta:
            PauseList.append(True)
        else:
            PauseList.append(False)
    else:
        PauseList.append(False)
        
    ListEnv.append(Env)
    ListminEnv.append(minEnv)
    ListmaxEnv.append(maxEnv)
    ListHPEnv.append(HPEnv)
    ListminHPEnv.append(minHPEnv)
    ListmaxHPEnv.append(maxHPEnv)
    ListLPEnv.append(LPEnv)
    ListminLPEnv.append(minLPEnv)
    ListmaxLPEnv.append(maxLPEnv)
    ListFFTed.append(np.log(FFTed))
    
    
time125=[]
for i in range(0,1250):
    time125.append(float(i/250))
plt.plot(X, FFTed)
plt.xlabel('Frequency (Hz)')
plt.show()
plt.plot(time125, PauseList)
plt.xlabel('Time (s)')

check=[]
time5s=[]
for i in range(0, 1250):
    k = 88.2*i
    for t in range(int(k), int(k+88.2)):
        time5s.append(PauseList[i]*(-1)+1)

for i in range(0, fivesecframe):
    check.append(List3[i]*(time5s[i]))
check = np.array(check)

librosa.output.write_wav(r'checking.wav', check, 22050)






