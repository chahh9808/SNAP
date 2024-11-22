import gc
import datetime
import inspect
import sys
import time
import torch
import numpy as np

sys.stdout.close = lambda: None

class TimeTracker(object):
    """
    Class to track latency for adaption.
    """
    global_time_tracker = None

    @classmethod
    def init_tracker(cls, **kwargs):
        cls.global_time_tracker = TimeTracker(**kwargs)

    @classmethod
    def track(cls, avgmeter=None, **kwargs):
        assert avgmeter is not None
        if cls.global_time_tracker is not None:
            cls.global_time_tracker.do_track(avgmeter=avgmeter, **kwargs)
        
    @classmethod        
    def set_timestamp(cls,**kwargs):
        cls.global_time_tracker.timestamp = time.time()
        
    def __init__(self):
        self.begin = True
        self.timestamp = time.time()
    
    def do_track(self, avgmeter):
        """
        Track the latency
        """
        cur_time = time.time() 
        latency = cur_time - self.timestamp
        self.timestamp = cur_time
        avgmeter.update(latency)
    
        