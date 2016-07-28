# -*- coding: utf-8 -*-
"""
Created on Fri Jan 30 10:43:27 2015

@author: vanvalen
"""

def format_coord(x,y,sample_image):
    numrows, numcols = sample_image.shape
    col = int(x+0.5)
    row = int(y+0.5)
    if col>= 0 and col<numcols and row>=0 and row<numrows:
        z = sample_image[row,col]
        return 'x=%1.4f, y=%1.4f, z=%1.4f'%(x,y,z)
    else:
        return 'x=%1.4f, y=1.4%f'%(x,y)
        
        