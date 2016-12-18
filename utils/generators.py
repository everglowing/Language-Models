# List of numpy functions that generate an output sequence
import numpy as np

def next_char_char(xdata, extra_data=None):
    ydata = np.copy(xdata)
    ydata[:-1] = xdata[1:]
    ydata[-1] = xdata[0]
    return ydata

def same_char_char(xdata, extra_data=None):
    ydata = np.copy(xdata)
    return ydata

def same_ipa_char(xdata, extra_data=None):
    ydata = np.copy(extra_data)
    return ydata

def next_ipa_ipa(xdata, extra_data=None):
    ydata = np.copy(xdata)
    ydata[:-1] = xdata[1:]
    ydata[-1] = xdata[0]
    return ydata
