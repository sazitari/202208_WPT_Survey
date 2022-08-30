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
CSV_FILE_NAME = 'CMOS_PA_DATA.csv'
DATA_NAMES = ('pub', 'month', 'author', 'freq', 'Psat', 'PAEmax', 'OP1dB', 'PAE_P1dB', 'Gain')

atx = 3*m
arx = 5*cm
R = 40*m
Prect = 6+10*np.log10(2**3)
Ant_per_PA = 2**12
PCE = 40

def main():
    Stx = atx**2
    Srx = arx**2

    fig = fig_proc.Figure_Processing((4,2), (12,12))
    df_csv = pd.read_csv(CSV_FILE_NAME, names=DATA_NAMES, skiprows=1, dtype=str)
    df_num = df_csv[["freq","Psat","PAEmax","OP1dB","PAE_P1dB","Gain"]].applymap(str_to_float)

    df_Psat = df_num[["freq","Psat"]].dropna()
    mp_Psat = math_proc.MathProcessing(df_Psat["freq"], df_Psat["Psat"])
    mp_Psat.applyFunc(lambda x: np.log(x), axis='x')
    coef_Psat = mp_Psat.pearsonr()[0]
    fig.add_plot((0,0), mp_Psat.x, mp_Psat.y, marker='o', wline=0, color="b")

    mp_Psat.binning(BINS_NUMBER, TREND_TYPE)
    fig.add_plot((0,0), mp_Psat.x, mp_Psat.y, marker='x', wline=0, color="r")

    df_PAE = df_num[["freq","PAEmax"]].dropna()
    mp_PAE = math_proc.MathProcessing(df_PAE["freq"], df_PAE["PAEmax"])
    mp_PAE.applyFunc(lambda x: np.log(x), axis='y')
    coef_PAE = mp_PAE.pearsonr()[0]
    fig.add_plot((0,1), mp_PAE.x, mp_PAE.y, marker='o', wline=0, color="b")

    mp_PAE.binning(BINS_NUMBER, TREND_TYPE)
    fig.add_plot((0,1), mp_PAE.x, mp_PAE.y, marker='x', wline=0, color="r")

    param_Psat = mp_Psat.curveFit()
    param_PAE = mp_PAE.curveFit()

    x = np.linspace(mp_Psat.xRange[0], mp_Psat.xRange[1], 100, endpoint=True)
    y = param_Psat[0]*x + param_Psat[1]
    fig.add_plot((0,0), x, y, color="k")

    x = np.linspace(mp_PAE.xRange[0], mp_PAE.xRange[1], 100, endpoint=True)
    y = param_PAE[0]*x + param_PAE[1]
    fig.add_plot((0,1), x, y, color="k")

    print(f"Psat: eps = {coef_Psat:.3f}, PAEmax: eps = {coef_PAE:.3f}")

    df_ims = pd.DataFrame(data=np.linspace(0.1, 300, 1001, endpoint=True), columns=["freq"])
    wl = const.c/(df_ims["freq"]*GHz)
    Dant = wl/2
    df_ims["Tx_Nant"] = (Stx/(Dant**2)).astype(np.uint64)
    df_ims["Tx_Npa"] = (df_ims["Tx_Nant"]/Ant_per_PA).astype(np.uint64)
    df_ims["Tx_PAE"] = np.exp(param_PAE[0]*df_ims["freq"] + param_PAE[1])
    df_ims["Tx_Psat"] = param_Psat[0]*np.log(df_ims["freq"]) + param_Psat[1]
    df_ims["Tx_Pdis"] = 10**(df_ims["Tx_Psat"]/10)*1*mW / (df_ims["Tx_PAE"]/100) * df_ims["Tx_Npa"]
    df_ims["Tx_Pt"] = 10**(df_ims["Tx_Psat"]/10)*1*mW * df_ims["Tx_Npa"]
    df_ims["Tx_EIRP"] = 10*np.log10(df_ims["Tx_Pt"]/(1*mW)) + 10*np.log10(df_ims["Tx_Nant"])
    df_ims["NFWPT_eff"] = (1-np.exp(-(Stx*Srx)/((R*wl)**2)))*100
    df_ims["Rx_Pr"] = df_ims["NFWPT_eff"]/100 * df_ims["Tx_Pt"]
    df_ims["Rx_Nant"] = (Srx/(Dant**2)).astype(np.uint64)
    df_ims["Rx_Nrect"] = (df_ims["Rx_Pr"]/(10**(Prect/10)*1*mW)).astype(np.uint64)
    df_ims["Rx_Pgen"] = df_ims["Rx_Pr"] * PCE/100
    df_ims["System_eff"] = df_ims["Rx_Pgen"]/df_ims["Tx_Pdis"] * 100

    fig.add_plot((1,0), df_ims["freq"].values, df_ims["Tx_Nant"].values, label="Tx_Nant", color="b")
    fig.set_axis((1,0), xlabel="freq (GHz)", ylabel="Tx_Nant", yscale="log", ylim=(0,10**9))
    fig.add_plot((1,0), df_ims["freq"].values, df_ims["Tx_Npa"].values, twinx=1, label="Tx_Npa", color="r")
    fig.set_axis((1,0), twinx=1, ylabel="Tx_Npa", yscale="log", ylim=(0,10**9))

    fig.add_plot((1,1), df_ims["freq"].values, df_ims["Rx_Nant"].values, label="Rx_Nant", color="b")
    fig.set_axis((1,1), xlabel="freq (GHz)", ylabel="Rx_Nant", yscale="log", ylim=(0,10**5))
    fig.add_plot((1,1), df_ims["freq"].values, df_ims["Rx_Nrect"].values, twinx=1, label="Rx_Nrect", color="r")
    fig.set_axis((1,1), twinx=1, ylabel="Rx_Nrect", yscale="log", ylim=(0,10**5))

    fig.add_plot((2,0), df_ims["freq"].values, df_ims["Tx_Pt"].values, color="b")
    fig.set_axis((2,0), xlabel="freq (GHz)", ylabel="Tx_Pt (W)")
    fig.add_plot((2,1), df_ims["freq"].values, df_ims["Rx_Pr"].values, label="Rx_Pr", color="b")
    fig.add_plot((2,1), df_ims["freq"].values, df_ims["Rx_Pgen"].values, label="Rx_Pgen", color="r")
    fig.set_axis((2,1), xlabel="freq (GHz)", ylabel="Rx_Pr, Rx_Pgen (W)")

    fig.add_plot((3,0), df_ims["freq"].values, df_ims["NFWPT_eff"].values, color="b")
    fig.set_axis((3,0), xlabel="freq (GHz)", ylabel="NFWPT efficiency (%)")
    fig.add_plot((3,1), df_ims["freq"].values, df_ims["System_eff"].values, color="b")
    fig.set_axis((3,1), xlabel="freq (GHz)", ylabel="System efficiency (%)")



def str_to_float(value):
    try:
        ret = np.nan if str(value).isspace() or not str(value) else float(value)
    except Exception:
        ret = float(re.sub(r"[\(\*\/].*", "", value))
    return ret


if __name__ == '__main__':
    main()
