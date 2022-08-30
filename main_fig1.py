import os, sys
import pathlib
import numpy as np
import pandas as pd
import re
from scipy import constants as const
from lib import math_proc
from lib import fig_proc
from lib.units import *



TREND_TYPE = "r1"
BINS_NUMBER = 50
PA_FILE_NAME = 'CMOS_PA_DATA.csv'
PA_DATA_NAMES = ('pub', 'month', 'author', 'freq', 'Psat', 'PAEmax', 'OP1dB', 'PAE_P1dB', 'Gain')
RECT_FILE_NAME = 'RECT_DATA.csv'
RECT_DATA_NAMES = ('num', 'title', 'pub', 'tech', 'freq', 'RL', 'Pin@PCEmax', 'PCEmax', 'Pout@PCEmax', 'PCE@6dBm')

'''
FmW = 24*GHz
AmW = (100*m, 100*m)
FsT = 122*GHz
AsT = (1*m, 3*cm)
Prect = 6+10*np.log10(2**3)
Ant_per_PA = 2**12
'''

atx = 80*cm
arx = 80*cm
R = 100*m
Prect = 6+10*np.log10(2**3)
Ant_per_PA = 2**4

def main():
    '''
    SmW = np.power(AmW, 2)
    SsT = np.power(AsT, 2)
    '''
    Stx = atx**2
    Srx = arx**2

    df_pa = pd.read_csv(PA_FILE_NAME, names=PA_DATA_NAMES, skiprows=1, dtype=str)
    df_pa = df_pa[["freq","Psat","PAEmax"]].applymap(str_to_float)

    df_Psat = df_pa[["freq","Psat"]].dropna()
    coef_Psat, param_Psat = opt_leastsq(x=df_Psat["freq"], y=df_Psat["Psat"], func=lambda x: np.log(x), axis='x', Nbin=BINS_NUMBER, trend_type=TREND_TYPE)

    df_PAE = df_pa[["freq","PAEmax"]].dropna()
    coef_PAE, param_PAE = opt_leastsq(x=df_PAE["freq"], y=df_PAE["PAEmax"], func=lambda y: np.log(y), axis='y', Nbin=BINS_NUMBER, trend_type=TREND_TYPE)

    df_rect = pd.read_csv(RECT_FILE_NAME, names=RECT_DATA_NAMES, skiprows=1, dtype=str)
    df_rect = df_rect[["freq","PCEmax"]].applymap(str_to_float)

    df_PCE = df_rect[["freq","PCEmax"]].dropna()
    coef_PCE, param_PCE = opt_leastsq(x=df_PCE["freq"], y=df_PCE["PCEmax"], func=lambda y: np.log(y), axis='y', Nbin=20, trend_type='r1')

    print(f"Psat: eps={coef_Psat:.3f}, PAEmax: eps={coef_PAE:.3f}, PCEmax: eps={coef_PCE:.3f}")

    df_ims = pd.DataFrame(data=np.linspace(0.1, 300, 1001, endpoint=True), columns=["freq"])
    wl = const.c/(df_ims["freq"]*GHz)
    Dant = wl/2
    df_ims["Tx_Nant"] = (Stx/(Dant**2)).astype(np.uint64)
    df_ims["Tx_Npa"] = (df_ims["Tx_Nant"]/Ant_per_PA).astype(np.uint64)
    df_ims["Tx_PAE"] = np.exp(param_PAE[0]*df_ims["freq"] + param_PAE[1])
    df_ims["Tx_PAE"] = df_ims["Tx_PAE"].where(df_ims["Tx_PAE"]<100, 100)
    df_ims["Tx_Psat"] = param_Psat[0]*np.log(df_ims["freq"]) + param_Psat[1]
    df_ims["Tx_Pdis"] = 10**(df_ims["Tx_Psat"]/10)*1*mW / (df_ims["Tx_PAE"]/100) * df_ims["Tx_Npa"]
    df_ims["Tx_Pt"] = 10**(df_ims["Tx_Psat"]/10)*1*mW * df_ims["Tx_Npa"]
    df_ims["Tx_EIRP"] = 10*np.log10(df_ims["Tx_Pt"]/(1*mW)) + 10*np.log10(df_ims["Tx_Nant"])
    df_ims["NFWPT_eff"] = (1-np.exp(-(Stx*Srx)/((R*wl)**2)))*100
    df_ims["Rx_Pr"] = df_ims["NFWPT_eff"]/100 * df_ims["Tx_Pt"]
    df_ims["Rx_Nant"] = (Srx/(Dant**2)).astype(np.uint64)
    df_ims["Rx_Nrect"] = (df_ims["Rx_Pr"]/(10**(Prect/10)*1*mW)).astype(np.uint64)
    df_ims["Rx_PCE"] = np.exp(param_PCE[0]*df_ims["freq"] + param_PCE[1])
    df_ims["Rx_PCE"] = df_ims["Rx_PCE"].where(df_ims["Rx_PCE"]<100, 100)
    df_ims["Rx_Pgen"] = df_ims["Rx_Pr"] * df_ims["Rx_PCE"]/100
    df_ims["System_eff"] = df_ims["Rx_Pgen"]/df_ims["Tx_Pdis"] * 100

    fig = fig_proc.Figure_Processing((3,2), (14,14))

    fig.add_plot((0,0), df_ims["freq"].values, df_ims["Tx_Nant"].values, label="Tx_Nant", color="b")
    fig.set_axis((0,0), xlabel="freq (GHz)", ylabel="Tx_Nant", yscale="log", ylim=(0,10**9))
    fig.add_plot((0,0), df_ims["freq"].values, df_ims["Tx_Npa"].values, twinx=1, label="Tx_Npa", color="r")
    fig.set_axis((0,0), twinx=1, ylabel="Tx_Npa", yscale="log", ylim=(0,10**9))

    fig.add_plot((0,1), df_ims["freq"].values, df_ims["Rx_Nant"].values, label="Rx_Nant", color="b")
    fig.set_axis((0,1), xlabel="freq (GHz)", ylabel="Rx_Nant", yscale="log", ylim=(0,10**9))
    fig.add_plot((0,1), df_ims["freq"].values, df_ims["Rx_Nrect"].values, twinx=1, label="Rx_Nrect", color="r")
    fig.set_axis((0,1), twinx=1, ylabel="Rx_Nrect", yscale="log", ylim=(0,10**9))

    fig.add_plot((1,0), df_ims["freq"].values, df_ims["Tx_Pt"].values, color="b")
    fig.set_axis((1,0), xlabel="freq (GHz)", ylabel="Tx_Pt (W)")

    fig.add_plot((1,1), df_ims["freq"].values, df_ims["Rx_Pr"].values, label="Rx_Pr", color="b")
    fig.add_plot((1,1), df_ims["freq"].values, df_ims["Rx_Pgen"].values, label="Rx_Pgen", color="r")
    fig.set_axis((1,1), xlabel="freq (GHz)", ylabel="Rx_Pr, Rx_Pgen (W)")
    fig.set_legend((1,1))

    fig.add_plot((2,0), df_ims["freq"].values, df_ims["NFWPT_eff"].values, color="b", label="NF WPT efficiency")
    fig.add_plot((2,0), df_ims["freq"].values, df_ims["Rx_PCE"].values, color="r", label="PCE")
    fig.add_plot((2,0), df_ims["freq"].values, df_ims["Tx_PAE"].values, color="g", label="PAE")
    fig.set_axis((2,0), xlabel="freq (GHz)", ylabel="Efficiency (%)")
    fig.set_legend((2,0))

    fig.add_plot((2,1), df_ims["freq"].values, df_ims["System_eff"].values, color="b")
    fig.set_axis((2,1), xlabel="freq (GHz)", ylabel="System efficiency (%)")


def str_to_float(value):
    try:
        ret = np.nan if str(value).isspace() or not str(value) else float(value)
    except Exception:
        ret = float(re.sub(r"[\(\*\/].*", "", value))
    return ret



def opt_leastsq(x, y, func, axis, Nbin, trend_type):
    fig = fig_proc.Figure_Processing((1,1), (8,4))

    mp = math_proc.MathProcessing(x, y)
    mp.applyFunc(func, axis=axis)

    coef = mp.pearsonr()[0]
    fig.add_plot((0,0), mp.x, mp.y, marker='o', wline=0, color="b", label="applied data")

    mp.binning(Nbin, trend_type)
    fig.add_plot((0,0), mp.x, mp.y, marker='x', wline=0, color="r", label="binned data")

    param = mp.curveFit()
    x_sq = np.linspace(mp.xRange[0], mp.xRange[1], 101, endpoint=True)
    y_sq = param[0]*x_sq + param[1]
    fig.add_plot((0,0), x_sq, y_sq, color="k", label="leastSQ line")

    fig.set_legend((0,0))

    return coef, param



if __name__ == '__main__':
    main()
