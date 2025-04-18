import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from SignalToNoiseRatio import SnrOnAES128 as SNR

plain = pd.read_csv("plaintexts.csv", header=None).values
powerTraces = pd.read_csv("traces.csv", header=None).values
keys = pd.read_csv("key_forSNR.csv", header=None).values
keys = keys[0]

#print(keys)

snrObj = SNR()

snrObj.SetPlainTexts(plain)
snrObj.SetPowerTraces(powerTraces)
snrObj.SetCorrectKeys(keys)

receivedKeys = snrObj.GetCorrectKeys()
print("receivedKeys: ",receivedKeys)

#snrObj.createCorrectHypothesis(15)
# snrObj.calculateHwLabels()
# snrObj.createEmptyGroups()
# snrObj.fillGroups()
#snrObj.groupBasedOnHW()

snrObj.calculateSNR(16)

