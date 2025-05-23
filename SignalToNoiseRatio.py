import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from tqdm import tqdm 
from progress.spinner import MoonSpinner

class SnrOnAES128:
    sboxTable = (
        0x63, 0x7C, 0x77, 0x7B, 0xF2, 0x6B, 0x6F, 0xC5, 0x30, 0x01, 0x67, 0x2B, 0xFE, 0xD7, 0xAB, 0x76,
        0xCA, 0x82, 0xC9, 0x7D, 0xFA, 0x59, 0x47, 0xF0, 0xAD, 0xD4, 0xA2, 0xAF, 0x9C, 0xA4, 0x72, 0xC0,
        0xB7, 0xFD, 0x93, 0x26, 0x36, 0x3F, 0xF7, 0xCC, 0x34, 0xA5, 0xE5, 0xF1, 0x71, 0xD8, 0x31, 0x15,
        0x04, 0xC7, 0x23, 0xC3, 0x18, 0x96, 0x05, 0x9A, 0x07, 0x12, 0x80, 0xE2, 0xEB, 0x27, 0xB2, 0x75,
        0x09, 0x83, 0x2C, 0x1A, 0x1B, 0x6E, 0x5A, 0xA0, 0x52, 0x3B, 0xD6, 0xB3, 0x29, 0xE3, 0x2F, 0x84,
        0x53, 0xD1, 0x00, 0xED, 0x20, 0xFC, 0xB1, 0x5B, 0x6A, 0xCB, 0xBE, 0x39, 0x4A, 0x4C, 0x58, 0xCF,
        0xD0, 0xEF, 0xAA, 0xFB, 0x43, 0x4D, 0x33, 0x85, 0x45, 0xF9, 0x02, 0x7F, 0x50, 0x3C, 0x9F, 0xA8,
        0x51, 0xA3, 0x40, 0x8F, 0x92, 0x9D, 0x38, 0xF5, 0xBC, 0xB6, 0xDA, 0x21, 0x10, 0xFF, 0xF3, 0xD2,
        0xCD, 0x0C, 0x13, 0xEC, 0x5F, 0x97, 0x44, 0x17, 0xC4, 0xA7, 0x7E, 0x3D, 0x64, 0x5D, 0x19, 0x73,
        0x60, 0x81, 0x4F, 0xDC, 0x22, 0x2A, 0x90, 0x88, 0x46, 0xEE, 0xB8, 0x14, 0xDE, 0x5E, 0x0B, 0xDB,
        0xE0, 0x32, 0x3A, 0x0A, 0x49, 0x06, 0x24, 0x5C, 0xC2, 0xD3, 0xAC, 0x62, 0x91, 0x95, 0xE4, 0x79,
        0xE7, 0xC8, 0x37, 0x6D, 0x8D, 0xD5, 0x4E, 0xA9, 0x6C, 0x56, 0xF4, 0xEA, 0x65, 0x7A, 0xAE, 0x08,
        0xBA, 0x78, 0x25, 0x2E, 0x1C, 0xA6, 0xB4, 0xC6, 0xE8, 0xDD, 0x74, 0x1F, 0x4B, 0xBD, 0x8B, 0x8A,
        0x70, 0x3E, 0xB5, 0x66, 0x48, 0x03, 0xF6, 0x0E, 0x61, 0x35, 0x57, 0xB9, 0x86, 0xC1, 0x1D, 0x9E,
        0xE1, 0xF8, 0x98, 0x11, 0x69, 0xD9, 0x8E, 0x94, 0x9B, 0x1E, 0x87, 0xE9, 0xCE, 0x55, 0x28, 0xDF,
        0x8C, 0xA1, 0x89, 0x0D, 0xBF, 0xE6, 0x42, 0x68, 0x41, 0x99, 0x2D, 0x0F, 0xB0, 0x54, 0xBB, 0x16
    )

    def __init__(self):
        self.plainTexts = None
        self.powerTraces = None
        self.correctKeys = None
        self.correctHypothesis = None
        self.HammingWeightLabels = None
        self.groups = None
        self.traceGroupMean ={}
        self.powerTraceSignal = []
        self.powerTraceNoise = []
        self.signalVariance = None
        self.noiseVariance = None
        self.signalToNoiseRatio = None
        self.signalToNoiseRatioAllBytes = []
        self.maxSnrValue = None
        self.progressBarEnabled = False

    def enableProgressBar(self):
        self.progressBarEnabled = True
    
    def disableProgressBar(self):
        self.progressBarEnabled = False

    def MaxSnrValueTargetByte(self, type):
        if type=="Single":
            self.maxSnrValue = np.max(self.signalToNoiseRatio)
        elif type=="All":
            self.maxSnrValue = np.max(self.signalToNoiseRatioAllBytes)

    def plotSNRtargetByte(self,targetByte, singleRun=False):
        self.CheckOutputsDirectory(targetByte)
        self.MaxSnrValueTargetByte("Single")
        fig, ax1 = plt.subplots(1,1,figsize=(12, 5))
        ax1.set_title(f"SNR for Byte Number: {targetByte} using HW Leakage Model")
        ax1.set_xlabel("Power Trace Time Sample")
        ax1.set_ylabel("SNR Value")
        ax1.set_ylim(0, (self.maxSnrValue + (self.maxSnrValue/10)))
        ax1.plot(self.signalToNoiseRatio)
        plt.savefig(f"./Outputs/SNR_forTargetByte_{targetByte}.jpg")
        print(f"INFO: The output file (SNR_forTargetByte_{targetByte}.jpg) from this execution is stored at ./Outputs directory.")
        plt.savefig(f"./Outputs/SNR_forTargetByte_{targetByte}.pdf")
        print(f"INFO: The output file (SNR_forTargetByte_{targetByte}.pdf) from this execution is stored at ./Outputs directory.")
        if singleRun:
            plt.show()
        

    def plotSNRallBytes(self):
        self.CheckOutputsDirectory("All")
        self.MaxSnrValueTargetByte("All")
        fig, axs = plt.subplots(4,4,figsize=(16, 10))
        for row in range(4):
            for col in range(4):
                axs[row, col].plot(self.signalToNoiseRatioAllBytes[(4*row)+col])
                axs[row, col].set_title(f"Byte Number: {((4*row)+col)+1}", size=8)
                axs[row, col].set_ylim(0, (self.maxSnrValue + (self.maxSnrValue/10)))
                axs[row, col].set_xlabel("Power Trace Time Sample", fontsize = 6)
                axs[row, col].set_ylabel("SNR Value", fontsize = 6)
                axs[row, col].tick_params(axis='both', which='major', labelsize=4)
                axs[row, col].tick_params(axis='both', which='minor', labelsize=4)
        fig.suptitle("SNR for All Bytes using HW Leakage Model")
        fig.subplots_adjust(hspace=0.6, wspace=0.2)
        plt.savefig(f"./Outputs/SNR_forAllTargetByte.jpg")
        print(f"INFO: The output file (SNR_forAllTargetByte.jpg) from this execution is stored at ./Outputs directory.")
        plt.savefig(f"./Outputs/SNR_forAllTargetByte.pdf")
        print(f"INFO: The output file (SNR_forAllTargetByte.pdf) from this execution is stored at ./Outputs directory.")
        plt.show()

    def SetCorrectKeys(self, correctKeys):
        self.correctKeys = correctKeys

    def SetPowerTraces(self, powerTraces):
        self.powerTraces = powerTraces

    def SetPlainTexts(self, plainTexts):
        self.plainTexts = plainTexts

    def GetCorrectKeys(self):
        return self.correctKeys
        
    def GetPlainTexts(self):
        return self.plainTexts

    def GetPowerTraces(self):
        return self.powerTraces
    
    def Sbox(self, inp):
        return self.sboxTable[inp]

    def HammingWeight(self, num):
        return bin(num).count("1")

    def HammingDistance(self, num1, num2):
        return self.HammingWeight(num1^num2)
    
    def ClearVariables(self):
        self.correctHypothesis = None
        self.HammingWeightLabels = None
        self.groups = None
        self.traceGroupMean ={}
        self.powerTraceSignal = []
        self.powerTraceNoise = []
        self.signalVariance = None
        self.noiseVariance = None
        self.signalToNoiseRatio = None
    
    def CreateCorrectHypothesis(self, byteNumber):
        self.correctHypothesis = np.zeros(self.plainTexts.shape[0])
        print("*****************************************************************")
        print(f"Creating Correct Hypothesis for Byte Number: {byteNumber+1}")
        with MoonSpinner('Processing…') as bar:
            for i in range(self.plainTexts.shape[0]):
                self.correctHypothesis[i] = self.HammingWeight(self.Sbox(self.correctKeys[byteNumber] ^ self.plainTexts[i][byteNumber]))
                bar.next()
                if(self.progressBarEnabled):
                    self.ProgressBar(i, self.plainTexts.shape[0]-1)
            if(self.progressBarEnabled):
                print("")

    def CalculateHwLabels(self):
        self.HammingWeightLabels = np.unique(self.correctHypothesis) # ===> HammingWeightLabels = [0,1,2,3,4,5,6,7,8]

    def CreateEmptyGroups(self):
        self.groups={}
        for i in range(self.HammingWeightLabels.shape[0]):
            self.groups[i]=[]

    def FillGroups(self):
        print(f"Grouping Power Traces:")
        with MoonSpinner('Processing…') as bar:
            for index, val in enumerate(self.correctHypothesis):
                self.groups[val].append(self.powerTraces[index])
                bar.next()
                if(self.progressBarEnabled):
                    self.ProgressBar(index, len(self.correctHypothesis)-1)
            if(self.progressBarEnabled):
                print("")

    def GroupBasedOnHW(self):
        self.CalculateHwLabels()
        self.CreateEmptyGroups()
        self.FillGroups()

    def CalculateSignal(self):
        print(f"Calculating Signal Values:")
        with MoonSpinner('Processing…') as bar:
            for i in self.HammingWeightLabels:
                self.traceGroupMean[i]=np.mean(self.groups[i], axis=0)
                self.powerTraceSignal.append(self.traceGroupMean[i])
                bar.next()
                if(self.progressBarEnabled):
                    self.ProgressBar(i, len(self.HammingWeightLabels)-1)
            if(self.progressBarEnabled):
                print("")

    def CalculateNoise(self):
        print(f"Calculating Noise Values:")
        with MoonSpinner('Processing…') as bar:
            for i in self.HammingWeightLabels:
                for trace in self.groups[i]:
                    self.powerTraceNoise.append(trace-self.traceGroupMean[i])
                bar.next()
                if(self.progressBarEnabled):
                    self.ProgressBar(i, len(self.HammingWeightLabels)-1)
            if(self.progressBarEnabled):
                print("")

    def CalculateSignalVariance(self):
        print(f"Calculating Signal Variance...")
        self.signalVariance = np.var(self.powerTraceSignal, axis=0)

    def CalculateNoiseVariance(self):
        print(f"Calculating Noise Variance...")
        self.noiseVariance = np.var(self.powerTraceNoise, axis=0)

    def SNRforTargetByte(self, targetByte):
        self.ClearVariables()
        self.CreateCorrectHypothesis(targetByte-1)
        self.GroupBasedOnHW()
        self.CalculateSignal()
        self.CalculateNoise()
        self.CalculateSignalVariance()
        self.CalculateNoiseVariance()
        print(f"Calculating Signal to Noise Ratio...")
        self.signalToNoiseRatio = self.signalVariance/self.noiseVariance
        self.plotSNRtargetByte(targetByte, singleRun=True)
        print("Process Finished Successfully.")
        print("*****************************************************************")

    def SNRforAllBytes(self):
        print("Calculating SNR for all Bytes:")
        for i in range(16):
            self.ClearVariables()
            self.CreateCorrectHypothesis(i)
            self.GroupBasedOnHW()
            self.CalculateSignal()
            self.CalculateNoise()
            self.CalculateSignalVariance()
            self.CalculateNoiseVariance()
            print(f"Calculating Signal to Noise Ratio...")
            self.signalToNoiseRatio = self.signalVariance/self.noiseVariance
            self.signalToNoiseRatioAllBytes.append(self.signalToNoiseRatio)
            self.plotSNRtargetByte(i)
        self.plotSNRallBytes()
        print("Process Finished Successfully.")
        print("*****************************************************************")


    def ProgressBar(self, count_value, total, suffix=''):
        bar_length = 100
        filled_up_Length = int(round(bar_length* count_value / float(total)))
        percentage = round(100.0 * count_value/float(total),1)
        bar = '=' * filled_up_Length + '-' * (bar_length - filled_up_Length)
        sys.stdout.write('[%s] %s%s ...%s\r' %(bar, percentage, '%', suffix))
        sys.stdout.flush()

    def DeleteFilesInDirectory(self, directoryPath, targetByte):
        try:
            files = os.listdir(directoryPath)
            for file in files:
                if targetByte == "All":
                    if file== f"SNR_forAllTargetByte.jpg" or file== f"SNR_forAllTargetByte.pdf":
                        filePath = os.path.join(directoryPath, file)
                        if os.path.isfile(filePath):
                            os.remove(filePath)
                        print(f"INFO: The file ({file}) from the previous execution deleted successfully.")
                if file== f"SNR_forTargetByte_{targetByte}.jpg" or file== f"SNR_forTargetByte_{targetByte}.pdf":
                    filePath = os.path.join(directoryPath, file)
                    if os.path.isfile(filePath):
                        os.remove(filePath)
                    print(f"INFO: The file ({file}) from the previous execution deleted successfully.")
        except OSError:
            print("Err: Error occurred while deleting previous output files.")

    def CheckOutputsDirectory(self, targetByte=None):
        isDirExist = os.path.exists('Outputs')
        if isDirExist:
            self.DeleteFilesInDirectory('./Outputs',targetByte)
        else:
            os.makedirs('Outputs')