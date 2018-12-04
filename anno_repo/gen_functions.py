"""
Annomator
Copyright 2018 Arend Smits.
All rights reserved.  MIT Licence.  
"""

# Python 2.7 
# coding: utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function 

import time
# Python 27 seconds format for compatibility
def time_seconds_format(seconds):
    mins, secs = divmod(seconds, 60)
    hours, mins = divmod(mins, 60)
    if hours > 0:
        time_format = "%d hours %d mins" % (hours, mins)
    elif mins > 0:
        time_format = "%d mins %d secs" % (mins, secs)
    elif secs >= 10:
        time_format = "%0.0d seconds" % (secs)
    else:
        time_format = "%0.03f seconds" % (secs)
    return time_format
