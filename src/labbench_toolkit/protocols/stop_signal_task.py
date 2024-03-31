import math
import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
from labbench_toolkit.psychophysics import Quick
import statistics

class PsiStopSignalTask:
    def __init__(self, result, name = 'Stop Signal Task'):
        annotations = result.Annotations

        self._lowLimit = annotations['sstLowerLimit']
        self._highLimit = annotations['sstHighLimit']
        self._delays = annotations['sstDelays']
        self._responses = annotations['sstAnswer']
        self._alpha = annotations['sstAlpha']
        self._alphaLower = annotations['sstAlphaLower']
        self._alphaUpper = annotations['sstAlphaUpper']
        self._beta = annotations['sstBeta']
        self._betaLower = annotations['sstBetaLower']
        self._betaUpper = annotations['sstBetaUpper']
        self._gtTime = annotations['gtTime']
        self._gtAnswer = annotations['gtAnswer']
        self._lambda = 0.02
        self._gamma = 0.00
        self._sessionId = result.SessionID
        self._displayPlot = False
        self._savePlot = False
        self._name = name
              
    def SaveFiles(self, enable: bool):
        self._savePlot = enable

        return self

    def Display(self, enable: bool):
        self._displayPlot = enable

        return self

    @property
    def ReactionTime(self):
        return statistics.mean(self.GetReactionTimes())

    def GetReactionTimes(self):
        return [time for time, answer in zip(self._gtTime, self._gtAnswer) if answer == 1]

    @property
    def StopSignalReactionTime(self):
        return self.ReactionTime - self.Delay
    
    @property
    def BetaRange(self):
        return self.ICDF(0.25) - self.ICDF(0.75)
    
    @property
    def Delay(self):
        return self.ICDF(0.5)
    
    @property
    def StopSignalDelays(self):
        return self._delays
    
    @property
    def Responses(self):
        return self._responses
    
    def ICDF(self, p):
        a = self._alpha[-1]
        b = math.pow(10, self._beta[-1])
        g = self._gamma
        l = self._lambda
        pf = Quick(a, b, g, l)
        
        return self.transform(pf.ICDF(p))        

    def transform(self, x):
        return (self._highLimit - self._lowLimit) * (1 - x) + self._lowLimit

    def analyse(self):
        intensity = np.array(self._delays)
        response = np.array(self._responses)
        alpha = [self.transform(x) for x in self._alpha]
        alphaLower = [self.transform(x) for x in self._alphaLower]
        alphaUpper = [self.transform(x) for x in self._alphaUpper]
        
        Imax = self._highLimit
        
        n = np.array(range(0, len(intensity)))
               
        a = self._alpha[-1]
        b = math.pow(10, self._beta[-1])
        g = self._gamma
        l = self._lambda
        pf = Quick(a, b, g, l)
        
        x = np.linspace(0, 1, 100)
        cdf = np.array([pf.CDF(v) for v in x])
        i25 = self.transform(pf.ICDF(0.25))
        i50 = self.transform(pf.ICDF(0.50))
        i75 = self.transform(pf.ICDF(0.75))
        
        
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
        axes[0].set_ylabel('Stop Signal Delay ($t_{ssd}$) [ms]')
        

        axes[1].plot(cdf, [self.transform(v) for v in x], color = 'black')
        axes[1].plot([0, 1], [i50, i50], color ='red')
        axes[1].fill_between([0, 1], [i25, i25], [i75, i75], color='red', alpha=0.1)
        axes[1].set_title(r'$\psi$($t_{ssd}$)')
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
        
        if (self._savePlot):
            plt.savefig(self.AnalysisFileName, dpi=600)

        if (self._displayPlot):
            plt.show()

        plt.close()
        
    @property
    def AnalysisFileName(self):
        return f'{self._sessionId} {self._name}.png'
    
    def analyseConvergence(self):
        alpha = [self.transform(x) for x in self._alpha]
        alphaLower = [self.transform(x) for x in self._alphaLower]
        alphaUpper = [self.transform(x) for x in self._alphaUpper]

        
        N = len(alpha)
        n = range(0, N)
        
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(4, 6))
        
        axes[0].plot(n, alpha)
        axes[0].fill_between(n, alphaLower, alphaUpper, color='blue', alpha=0.1)
        axes[0].set_title(r'Alpha ($\alpha$)')
        axes[0].set_ylabel(r'$\alpha$ []')

        # Plot on the second subplot
        axes[1].plot(n, self._beta)
        axes[1].fill_between(n, self._betaLower, self._betaUpper, color='blue', alpha=0.1)
        axes[1].set_title(r'Beta ($\beta$)')
        axes[1].set_xlabel('Stimulation Number []')
        axes[1].set_ylabel(r'$log_{10}(\beta)$ []')

        for ax in axes:
           ax.spines['top'].set_visible(False)
           ax.spines['right'].set_visible(False)
           
        # Adjust layout for better spacing
        plt.tight_layout()

        # Show the plot

        if (self._savePlot):
            plt.savefig(self.ConvergenceFilename, dpi=600)

        if (self._displayPlot):
            plt.show()        

        plt.close()

    @property
    def ConvergenceFilename(self):
        return f'{self._sessionId} {self._name} (Convergence).png'