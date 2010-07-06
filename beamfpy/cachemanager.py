"""
cachemanager.py (depreceated)

Part of the beamfpy library: several classes for the implemetation of 
acoustic beamforming

(c) Ennes Sarradj 2007-2010, all rights reserved
ennes.sarradj@gmx.de
"""

from beamfpy import cache_dir
from numpy import *
import os

def limit_cache(siz=20000000000L, dir=cache_dir):
	"""
	limits the size of the cache via cleanup of files long time not accessed:
	siz: maximum size of dir, in Bytes, defaults to 20 GByte
	dir: cache directory, defaults to cache_dir
	"""
	l=[]
	for name in os.listdir(dir):
		s = os.stat(os.path.join(dir,name))
		l.append((name,s.st_size,s.st_atime))
	if len(l)==0:
		return
	l = array(l)
	indices = lexsort(l.T)
	sz = sum(array(l[:,1],dtype=int64))
	print sz
	if sz>siz:
		diff = sz - siz*0.8
		i = 0
		while diff>0:
			name,sz,t = l.take(indices,axis=0)[i]
			try:
				os.remove(os.path.join(dir,name))
				diff -= int(sz)
			except:
				pass
			i+=1
			print diff,name,sz,t
		
if __name__=='__main__':
	limit_cache()
