# -*- coding: utf-8 -*-
"""
Created on Sun Nov 19 18:25:03 2023

@author: KristianHennings
"""
import json
from labbench_toolkit.result import Result
from labbench_toolkit.result import ThresholdResult
from labbench_toolkit.result import EvokedPotentialsResult

class DataFile:
    def __init__(self, filename):
        with open(filename, 'r') as file:
            self._data = json.load(file)

    def __getitem__(self, n):
        return self.getData(n)
    
    def getData(self, n):
        return Session(self._data, n)
    
    def getIDs(self):
        return self._data['id']
    
    def getNumberOfSessions(self):
        return len(self.getIDs()) 
    
    def describe(self):
        print('SESSIONS [ {n} ]'.format(n = self.getNumberOfSessions()))
        for id in self.getIDs():
            print('{id}'.format(id = id))
    
class Session:
    def __init__(self, data, n):
        self._data = data['data'][n]
        self._id = data['id'][n]
        self._resultCreators = {
            'ThresholdResult': lambda result, id : ThresholdResult(result, id),
            'EvokedPotentialsResult': lambda result, id : EvokedPotentialsResult(result, id)
        }        
        
    @property
    def ID(self):
        return self._id
    
    def __getitem__(self, n):
        return self.getResult(n)

    def getResult(self, id):
        result = self._data.get(id)
        
        if result is None:
            raise ValueError("Did not find result with ID: {:}".format(id))
            
        creator = self._resultCreators[result['Type']]

        if creator is None:
            return Result(result, self.ID)

        return creator(result, self.ID)
       
    def describe(self):
        idSpace = max([len(test['ID']) for key, test in self._data.items()]) + 1
        typeSpace = max([len(test['Type']) for key, test in self._data.items()]) + 1
        cNames = ['ID', 'TYPE']

        print(f"SESSION [ {self.ID} ]")        
        print()
        print(f'{cNames[0]:<{idSpace}} | {cNames[1]:<{typeSpace}}')
        print((idSpace + typeSpace + 3) * "=")

        for _, test in self._data.items():
            print(f'{test["ID"]:<{idSpace}} | {test["Type"]:<{typeSpace}}')
            
        print((idSpace + typeSpace + 3) * "=")
        print()

        

       

