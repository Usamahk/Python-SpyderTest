# -*- coding: utf-8 -*-
"""
Created on Fri Nov 27 19:20:43 2015

@author: Usamahk
"""

def hello(name):
    """Given an object 'name', print 'Hello ' and the object."""
    print("Hello {}".format(name))


i = 42
if __name__ == "__main__":
    hello(i)