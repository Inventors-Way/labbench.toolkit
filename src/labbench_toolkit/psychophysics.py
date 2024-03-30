import math

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
    