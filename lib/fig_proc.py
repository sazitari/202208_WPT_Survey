import numpy as np
import matplotlib.pyplot as plt

class Figure_Processing():
    def __init__(self, size, figsize=None):
        self.create_fig(size, figsize)

    def create_fig(self, size, figsize=None):
        self.size = tuple(size)
        self.fig = plt.figure(figsize=figsize, tight_layout=True)
        self.ax = [[[self.fig.add_subplot(*size,i*size[1]+j+1)] for j in range(size[1])] for i in range(size[0])]

    def add_plot(self, pax, x, y, color=None, marker=None, wline=1, label=None, twinx=0):
        if twinx:
            self.ax[pax[0]][pax[1]].append(self.ax[pax[0]][pax[1]][0].twinx())
        (art,) = self.ax[pax[0]][pax[1]][twinx].plot(x, y, color=color, linewidth=wline, marker=marker, label=label)
        return art

    def set_axis(self, pax, twinx=0, xlabel=None, ylabel=None, xscale="linear", yscale="linear", xlim=[None,None], ylim=[None,None]):
        if xlabel:
            self.ax[pax[0]][pax[1]][twinx].set(xlabel=xlabel)
        if ylabel:
            self.ax[pax[0]][pax[1]][twinx].set(ylabel=ylabel)
        if xscale:
            self.ax[pax[0]][pax[1]][twinx].set_xscale(xscale)
        if yscale:
            self.ax[pax[0]][pax[1]][twinx].set_yscale(yscale)
        if xlim[0]:
            self.ax[pax[0]][pax[1]][twinx].set_xlim(left=xlim[0])
        if xlim[1]:
            self.ax[pax[0]][pax[1]][twinx].set_xlim(right=xlim[1])
        if ylim[0]:
            self.ax[pax[0]][pax[1]][twinx].set_ylim(bottom=ylim[0])
        if ylim[1]:
            self.ax[pax[0]][pax[1]][twinx].set_ylim(top=ylim[1])

    def set_legend(self, pax, twinx=0, art=[], label=[], bbox_to_anchor=(0.5,0,0.5,1), loc="lower right", space=0):
        self.ax[pax[0]][pax[1]][twinx].legend(handles=art, labels=label, bbox_to_anchor=bbox_to_anchor, loc=loc, borderaxespad=space)

    def show_fig(self):
        plt.show()

class Error(Exception):
    pass
