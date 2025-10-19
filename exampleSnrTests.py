import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
import scipy as sc
import scipy.stats as stats
from SignalToNoiseRatio import SnrOnAES128 as SNR

plains = pd.read_csv("plaintexts.csv", header=None).values
powerTraces = pd.read_csv("traces.csv", header=None).values
keys = pd.read_csv("key_forSNR.csv", header=None).values
keys = keys[0] 

snrObj = SNR()

snrObj.SetLeakageModel("HW")
snrObj.SetPlainTexts(plains)
snrObj.SetPowerTraces(powerTraces)
snrObj.SetCorrectKeys(keys)

returnedPlain = snrObj.GetPlainTexts()
returnedTraces = snrObj.GetPowerTraces()
returnedkeys = snrObj.GetCorrectKeys()
returnedLeakageModel = snrObj.GetLeakageModel()

print("returnedPlain.shape=", returnedPlain.shape)
print("returnedTraces.shape=", returnedTraces.shape)
print("returnedkeys.shape=", returnedkeys.shape)
print("returnedLeakageModel=", returnedLeakageModel)

# snrObj.SNRforTargetByte(1,"pooled")
snrObj.SNRforAllBytes("pooled")