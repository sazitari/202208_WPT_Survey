import numpy as np
import pandas as pd
from scipy import optimize, stats
from . import classtools as ct

class MathProcessing():
    @ct.define_properties(readable=("data"))
    def __init__(self, x, y):
        self.__data = pd.DataFrame(data=np.vstack([x,y]).T,columns=['x','y']).astype(np.float)
        self.__xRange = (self.__data.min()['x'], self.__data.max()['x'])
        self.__yRange = (self.__data.min()['y'], self.__data.max()['y'])
        self.__binnType = {
            'r1'    :lambda e,h,l: 2*e-l,
            's1'    :lambda e,h,l: 2*e-h,
            'r2'    :lambda e,h,l: e+h-l,
            's2'    :lambda e,h,l: e-h+l,
            'max'   :lambda e,h,l: h,
            'min'   :lambda e,h,l: l,
            'ave'   :lambda e,h,l: e
        }
        self.__x = self.__data['x'].values
        self.__y = self.__data['y'].values
    @property
    def x(self):
        return self.__data['x'].values
    @property
    def y(self):
        return self.__data['y'].values
    @property
    def xRange(self):
        return (self.__data.min()['x'], self.__data.max()['x'])
    @property
    def yRange(self):
        return (self.__data.min()['y'], self.__data.max()['y'])

    def pearsonr(self):
        return stats.pearsonr(self.__data['x'], self.__data['y'])

    def binning(self, nBin=100, type="ave"):
        data = self.__data.copy()
        data["binArr"], bin = pd.cut(data['x'], nBin, retbins=True, labels=False)
        cenArr = bin[:-1]+np.diff(bin)/2

        newData = pd.DataFrame(columns=['x','y'])
        for i in range(nBin):
            binData = data.query(f"binArr=={i}")
            e,h,l = (np.mean(binData['y']), np.max(binData['y']), np.min(binData['y']))
            newData.loc[i] = [cenArr[i], self.__binnType[type](e,h,l)]
        self.__data = newData

    def curveFit(self, initXY=[0,0], func=lambda param,x,y: y-(param[0]*x+param[1])):
        validData = self.__data.dropna()
        retLq = optimize.leastsq(func, initXY, args=(validData['x'], validData['y']))[0]
        return retLq.copy()

    def applyFunc(self, func, axis='x'):
        self.__data[axis] = self.__data[axis].apply(func)

class Error(Exception):
    pass
