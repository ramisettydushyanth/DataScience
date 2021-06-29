# -*- coding: utf-8 -*-
"""
Created on Wed Jun  2 22:48:01 2021

@author: ramisedu
"""


from scipy import constants

import scipy

print(constants.liter)

print(constants.pi)

print(constants.micro)

print(constants.speed_of_light)

print(constants.pound)

print(constants.kilo)

print(constants.yotta)    #1e+24
print(constants.zetta)    #1e+21
print(constants.exa)      #1e+18
print(constants.peta)     #1000000000000000.0
print(constants.tera)     #1000000000000.0
print(constants.giga)     #1000000000.0
print(constants.mega)     #1000000.0
print(constants.kilo)     #1000.0
print(constants.hecto)    #100.0
print(constants.deka)     #10.0
print(constants.deci)     #0.1
print(constants.centi)    #0.01
print(constants.milli)    #0.001
print(constants.micro)    #1e-06
print(constants.nano)     #1e-09
print(constants.pico)     #1e-12
print(constants.femto)    #1e-15
print(constants.atto)     #1e-18
print(constants.zepto)    #1e-21


print(constants.kibi)
print(constants.mebi)    #1048576
print(constants.gibi)    #1073741824
print(constants.tebi)    #1099511627776
print(constants.pebi)    #1125899906842624
print(constants.exbi)    #1152921504606846976
print(constants.zebi)    #1180591620717411303424
print(constants.yobi)    #1208925819614629174706176

print(constants.gram)        #0.001
print(constants.metric_ton)  #1000.0
print(constants.grain)       #6.479891e-05
print(constants.lb)          #0.45359236999999997
print(constants.pound)       #0.45359236999999997
print(constants.oz)          #0.028349523124999998
print(constants.ounce)       #0.028349523124999998
print(constants.stone)       #6.3502931799999995
print(constants.long_ton)    #1016.0469088
print(constants.short_ton)   #907.1847399999999
print(constants.troy_ounce)  #0.031103476799999998
print(constants.troy_pound)  #0.37324172159999996
print(constants.carat)       #0.0002
print(constants.atomic_mass) #1.66053904e-27
print(constants.m_u)         #1.66053904e-27
print(constants.u)           #1.66053904e-27

print(constants.degree)     #0.017453292519943295
print(constants.arcmin)     #0.0002908882086657216
print(constants.arcminute)  #0.0002908882086657216
print(constants.arcsec)     #4.84813681109536e-06
print(constants.arcsecond)  #4.84813681109536e-06

print(constants.degree*180)     #0.017453292519943295


print(constants.minute)      #60.0
print(constants.hour)        #3600.0
print(constants.day)         #86400.0
print(constants.week)        #604800.0
print(constants.year)        #31536000.0
print(constants.Julian_year) #31557600.0

print(constants.inch)              #0.0254
print(constants.foot)              #0.30479999999999996
print(constants.yard)              #0.9143999999999999
print(constants.mile)              #1609.3439999999998
print(constants.mil)               #2.5399999999999997e-05
print(constants.pt)                #0.00035277777777777776
print(constants.point)             #0.00035277777777777776
print(constants.survey_foot)       #0.3048006096012192
print(constants.survey_mile)       #1609.3472186944373
print(constants.nautical_mile)     #1852.0
print(constants.fermi)             #1e-15
print(constants.angstrom)          #1e-10
print(constants.micron)            #1e-06
print(constants.au)                #149597870691.0
print(constants.astronomical_unit) #149597870691.0
print(constants.light_year)        #9460730472580800.0
print(constants.parsec)            #3.0856775813057292e+16



print(constants.atm)         #101325.0
print(constants.atmosphere)  #101325.0
print(constants.bar)         #100000.0
print(constants.torr)        #133.32236842105263
print(constants.mmHg)        #133.32236842105263
print(constants.psi)         #6894.757293168361

print(constants.hectare) #10000.0
print(constants.acre)    #4046.8564223999992

print(constants.liter)            #0.001
print(constants.litre)            #0.001
print(constants.gallon)           #0.0037854117839999997
print(constants.gallon_US)        #0.0037854117839999997
print(constants.gallon_imp)       #0.00454609
print(constants.fluid_ounce)      #2.9573529562499998e-05
print(constants.fluid_ounce_US)   #2.9573529562499998e-05
print(constants.fluid_ounce_imp)  #2.84130625e-05
print(constants.barrel)           #0.15898729492799998
print(constants.bbl)              #0.15898729492799998


print(constants.kmh)            #0.2777777777777778
print(constants.mph)            #0.44703999999999994
print(constants.mach)           #340.5
print(constants.speed_of_sound) #340.5
print(constants.knot)           #0.5144444444444445


print(constants.zero_Celsius)      #273.15
print(constants.degree_Fahrenheit) #0.5555555555555556

print(constants.eV)            #1.6021766208e-19
print(constants.electron_volt) #1.6021766208e-19
print(constants.calorie)       #4.184
print(constants.calorie_th)    #4.184
print(constants.calorie_IT)    #4.1868
print(constants.erg)           #1e-07
print(constants.Btu)           #1055.05585262
print(constants.Btu_IT)        #1055.05585262
print(constants.Btu_th)        #1054.3502644888888
print(constants.ton_TNT)       #4184000000.0

print(constants.hp)         #745.6998715822701
print(constants.horsepower) #745.6998715822701

print(constants.dyn)             #1e-05
print(constants.dyne)            #1e-05
print(constants.lbf)             #4.4482216152605
print(constants.pound_force)     #4.4482216152605
print(constants.kgf)             #9.80665
print(constants.kilogram_force)  #9.80665

print(constants.kilo)


from scipy.optimize import root
from math import cos

def equ(x):
    return x+cos(x)

myroot=root(equ,0)

print(myroot)


from scipy.optimize import minimize

def equ(x):
    return x**2+x+2

mymin=minimize(equ,0,method='BFGS')


print(mymin)


import numpy as np
from scipy.sparse import csr_matrix

arr=np.array([0,0,0,9,0,0,4,0,7])

print(csr_matrix(arr))

print(csr_matrix(arr).data)


arr = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 2]])

print(csr_matrix(arr).data)

print(csr_matrix(arr).count_nonzero())

arr = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 2]])

mat = csr_matrix(arr)

print(mat)

arr = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 2]])

mat = csr_matrix(arr)
mat.sum_duplicates()

print(mat)


arr = np.array([[0, 0, 0], [0, 0, 1], [1, 0, 2]])

newarr = csr_matrix(arr).tocsc()

print(newarr.data)


from scipy.sparse.csgraph import connected_components

from scipy.sparse import csr_matrix

import numpy as np

arr = np.array([
  [0, 1, 2],
  [1, 0, 0],
  [2, 0, 0]
])
new_arr=csr_matrix(arr)

print(connected_components(new_arr))



import numpy as np

from scipy.sparse.csgraph import dijkstra

from scipy.sparse import csr_matrix

arr=np.array([[0,1,2],[1,0,0],[2,0,0]])

new_arr=csr_matrix(arr)

print(dijkstra(new_arr,return_predecessors=True,indices=0))


import numpy as np
from scipy.sparse.csgraph import floyd_warshall
from scipy.sparse import csr_matrix

arr = np.array([
  [0, 1, 2],
  [1, 0, 0],
  [2, 0, 0]
])

newarr = csr_matrix(arr)

print(floyd_warshall(newarr, return_predecessors=True))


from scipy.sparse.csgraph import bellman_ford
from scipy.sparse import csr_matrix

arr = np.array([
  [0, -1, 2],
  [1, 0, 0],
  [2, 0, 0]
])

newarr = csr_matrix(arr)

print(bellman_ford(newarr, return_predecessors=True, indices=0))

print(arr.ndim)

from scipy.sparse.csgraph import depth_first_order
from scipy.sparse import csr_matrix

arr = np.array([
  [0, 1, 0, 1],
  [1, 1, 1, 1],
  [2, 1, 1, 0],
  [0, 1, 0, 1]
])

newarr = csr_matrix(arr)

print(depth_first_order(newarr, 1))

import numpy as np
from scipy.sparse.csgraph import breadth_first_order
from scipy.sparse import csr_matrix

arr = np.array([
  [0, 1, 0, 1],
  [1, 1, 1, 1],
  [2, 1, 1, 0],
  [0, 1, 0, 1]
])

newarr = csr_matrix(arr)

print(breadth_first_order(newarr, 1))


from scipy.spatial import Delaunay

import matplotlib.pyplot as plt


points = np.array([
  [2, 4],
  [3, 4],
  [3, 0],
  [2, 2],
  [4, 1]
])

simplices=Delaunay(points).simplices

plt.triplot(points[:,0],points[:,1],simplices)

plt.scatter(points[:,0],points[:,1],color='r')
plt.show()


from scipy.spatial import ConvexHull

import matplotlib.pyplot as plt


points = np.array([
  [2, 4],
  [3, 4],
  [3, 0],
  [2, 2],
  [4, 1],
  [1, 2],
  [5, 0],
  [3, 1],
  [1, 2],
  [0, 2]
])


hull=ConvexHull(points)

hull_points=hull.simplices

plt.scatter(points[:,0],points[:,1])

for simplex in hull_points:
    plt.plot(points[simplex,0],points[simplex,1],'k-')
    
plt.show()

from scipy.spatial import KDTree

points = [(1, -1), (2, 3), (-2, 3), (2, -3)]


kdtree=KDTree(points)


res=kdtree.query((1,1))

print(res)


from scipy.spatial.distance import euclidean



point1=(1,2)

point2=(3,4)


res=euclidean(point1,point2)

print(res)



from scipy.spatial.distance import cityblock

p1 = (1, 0)
p2 = (10, 2)

res = cityblock(p1, p2)

print(res)




from scipy.spatial.distance import cosine

p1 = (1, 0)
p2 = (10, 2)

res = cosine(p1, p2)

print(res)




from scipy.spatial.distance import hamming


p1=(True,False,True)
p2=(False,True,True)

res=hamming(p1,p2)

print(res)



from scipy import io

import numpy as np

arr=np.arange(10)

io.savemat('arr.mat',{'var':arr})

mydata=io.loadmat('arr.mat',squeeze_me=True)
print(mydata)

print(mydata['var'])


from scipy.interpolate import interp1d

import numpy as np

xs=np.arange(10)
ys=2*xs+1

print(xs)
print(ys)

interp_func=interp1d(xs,ys)

arr=interp_func(np.arange(2,3,0.1))

print(arr)



from scipy.interpolate import UnivariateSpline

import numpy as np


xs=np.arange(10)

ys=xs**2+np.sin(xs)+1


interpol_func=UnivariateSpline(xs,ys)

arr=interpol_func(np.arange(2,3,0.1))

print(arr)



from scipy.interpolate import Rbf
import numpy as np

xs = np.arange(10)
ys = xs**2 + np.sin(xs) + 1

interp_func = Rbf(xs, ys)

newarr = interp_func(np.arange(2.1, 3, 0.1))

print(newarr)



import numpy as np

from scipy.stats import ttest_ind


v1=np.random.normal(size=100)
v2=np.random.normal(size=100)

print(v1)

res=ttest_ind(v1,v2).pvalue

print(res)


import numpy as np

from scipy.stats import kstest


v=np.random.normal(size=100)


res=kstest(v,'norm')

print(res)




import numpy as np

from scipy.stats import describe


v=np.random.normal(size=100)


res=describe(v)

print(res)



import numpy as np

from scipy.stats import skew,kurtosis,normaltest


var=np.random.normal(size=100)

res1=normaltest(var).pvalue

print(res1)

print(skew(var))

print(kurtosis(var))

















