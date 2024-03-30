# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 18:25:03 2023

@author: KristianHennings
"""
import json
import math
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np

class LabBenchFile:
    def __init__(self, filename):
        with open(filename, 'r') as file:
            self._data = json.load(file)

    def __getitem__(self, n):
        return self.getData(n)
    
    def getData(self, n):
        return SessionData(self._data, n)
    
    def getIDs(self):
        return self._data['id']
    
    def getNumberOfSessions(self):
        return len(self.getIDs()) 
    
    def describe(self):
        print('SESSIONS [ {n} ]'.format(n = self.getNumberOfSessions()))
        for id in self.getIDs():
            print('{id}'.format(id = id))
    
class SessionData:
    def __init__(self, data, n):
        self._data = data['data'][n]
        self._id = data['id'][n]        
        
    @property
    def id(self):
        return self._id
    
    def __getitem__(self, n):
        return self.getResult(n)

    def getResult(self, id):
        result = self._data.get(id)
        
        if result is None:
            raise ValueError("Did not find result with ID: {:}".format(id))
            
        if result['Type'] == 'ThresholdResult':
            return ThresholdResult(result, self._id)
        else:
            raise ValueError('Unknown result type: {:}'.format(result['Type']))
       
    def describe(self):
        idSpace = max([len(test['ID']) for key, test in self._data.items()]) + 1
        typeSpace = max([len(test['Type']) for key, test in self._data.items()]) + 1
        cNames = ['ID', 'TYPE']        
        print(f'{cNames[0]:<{idSpace}} | {cNames[1]:<{typeSpace}}')
        print((idSpace + typeSpace + 3) * "=")

        for key, test in self._data.items():
            print(f'{test["ID"]:<{idSpace}} | {test["Type"]:<{typeSpace}}')

class ThresholdResult:
    def __init__(self, result, sessionId):
        self._result = result
        self._channels = [ThresholdChannel(c, sessionId, result['ID']) for c in result['Channels']]
        self._sessionId = sessionId
        
        
    def describe(self):
        print('Result keys:')
        print(self._result.keys())
    
    @property
    def Thresholds(self):
        return self._result['THR']
    
    @property
    def Channels(self):
        return self._channels

class ThresholdChannel:
    def __init__(self, channel, sessionId, testId):
        self._channel = channel
        self._sessionId= sessionId
        self._testId = testId
        self._channelId = channel['ID']
        
    def describe(self):
        print('CHANNEL KEYS:')
        print(self._channel.keys()) 
        
        print('FUNCTION')
        print(self._channel['function'].keys())
        
    def betaRange(self):
        function = self._channel['function']
        Imax = self._channel['Imax']
        
        a = function['alpha'][-1]
        b = math.pow(10, function['beta'][-1])
        g = function['gamma'][-1]
        l = function['lambda'][-1]
        pf = Quick(a, b, g, l)
        i25 = pf.ICDF(0.25) * Imax
        i75 = pf.ICDF(0.75) * Imax
        
        return i75 - i25
        
    def plotEstimation(self):
        intensity = np.array(self._channel['intensity'])
        response = np.array(self._channel['response'])
        Imax = self._channel['Imax']
        
        n = np.array(range(0, len(intensity)))
        
        function = self._channel['function']
        alpha = np.array(function['alpha']) * Imax
        alphaLower = np.array(function['alphaLower']) * Imax
        alphaUpper = np.array(function['alphaUpper']) * Imax
        
        a = function['alpha'][-1]
        b = math.pow(10, function['beta'][-1])
        g = function['gamma'][-1]
        l = function['lambda'][-1]
        pf = Quick(a, b, g, l)
        x = np.linspace(0, 1, 100)
        cdf = np.array([pf.CDF(v) for v in x])
        i25 = pf.ICDF(0.25) * Imax
        i50 = pf.ICDF(0.50) * Imax
        i75 = pf.ICDF(0.75) * Imax
                
        fig = plt.figure(figsize=(8, 4))
        gs = gridspec.GridSpec(1, 2, width_ratios=[4, 1])
        axes = [plt.subplot(gs[0]), plt.subplot(gs[1])]
        
        axes[0].scatter(n[response == 0], intensity[response == 0], marker='.',s = 5, color='black')
        axes[0].scatter(n[response == 1], intensity[response == 1], marker='+', color='black')
        axes[0].plot(n, alpha)
        axes[0].fill_between(n, alphaLower, alphaUpper, color='blue', alpha=0.1)
        axes[0].set_ylim(0, Imax)
        axes[0].set_title('Responses')
        axes[0].set_xlabel('Stimulation Number []')
        axes[0].set_ylabel('Intensity (p) [kPa]')
        

        axes[1].plot(cdf, x * Imax, color = 'black')
        axes[1].plot([0, 1], [i50, i50], color ='red')
        axes[1].fill_between([0, 1], [i25, i25], [i75, i75], color='red', alpha=0.1)
        axes[1].set_title(r'$\psi$(p)')
        axes[1].set_xlabel('Probability []')
        
        for ax in axes:
           ax.spines['top'].set_visible(False)
           ax.spines['right'].set_visible(False)
        
        axes[1].set_ylim(0, Imax)
        axes[1].set_xlim(-0.01, 1.01)
        axes[1].set_yticks([])
        axes[1].set_yticklabels([])
        axes[1].spines['left'].set_visible(False)
   
        # Adjust layout for better spacing
        plt.tight_layout()      
        
        plt.savefig('{sid}{tid} Estimation'.format(sid=self._sessionId, tid=self._testId), dpi=600)
        plt.show()
    
    def plotConvergence(self):
        function = self._channel['function']
        alpha = function['alpha']
        alphaLower = function['alphaLower']
        alphaUpper = function['alphaUpper']
        
        beta = function['beta']
        betaLower = function['betaLower']
        betaUpper = function['betaUpper']

        N = len(alpha)
        n = range(0, N)
        
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(4, 6))
        
        axes[0].plot(n, alpha)
        axes[0].fill_between(n, alphaLower, alphaUpper, color='blue', alpha=0.1)
        axes[0].set_title(r'Alpha ($\alpha$)')
        axes[0].set_ylabel(r'$\alpha$ []')

        # Plot on the second subplot
        axes[1].plot(n, beta)
        axes[1].fill_between(n, betaLower, betaUpper, color='blue', alpha=0.1)
        axes[1].set_title(r'Beta ($\beta$)')
        axes[1].set_xlabel('Stimulation Number []')
        axes[1].set_ylabel(r'$log_{10}(\beta)$ []')

        for ax in axes:
           ax.spines['top'].set_visible(False)
           ax.spines['right'].set_visible(False)
           
        # Adjust layout for better spacing
        plt.tight_layout()

        # Show the plot
        plt.savefig('{sid}{tid} Convergence'.format(sid=self._sessionId, tid=self._testId), dpi=600)        
        plt.show()
        

class Quick:
    def __init__(self, a, b, g, l):
        self._alpha = a
        self._beta = b
        self._gamma = g
        self._lambda = l
        
    @property
    def Name(self):
        return "Quick"

    @property
    def Alpha(self):
        return self._alpha
    
    @property
    def Beta(self):
        return self._beta
    
    @property
    def Gamma(self):
        return self._gamma
    
    @property
    def Lambda(self):
        return self._lambda
    
    def F(self, x):
        if x < 0:
            raise ValueError("The Quick definition is not defined for negative x values. If your x values are log transformed you should use the LogQuick psychometric function instead")
        
        return 1 - math.pow(2, - math.pow(x / self.Alpha, self.Beta));
    
    def CDF(self, x):
        return self.Gamma + (1 - self.Gamma - self.Lambda) * self.F(x);
    
    def ICDF(self, x):
        c = (x - self.Gamma)/(1 - self.Gamma - self.Lambda);
        return self.Alpha * math.pow(-math.log(1 - c)/math.log(2), 1/self.Beta)
    