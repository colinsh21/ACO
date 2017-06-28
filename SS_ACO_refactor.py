# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 11:20:46 2016

@author: colinsh
"""

#from System_Structure_ACO import Space
from SS_v7_stay import Space
#from SS_v6_iso import Space
#from SS_v5_restructure import Space
#from SS_v4_flow import Space
#from SS_v3_node_rem import Space
import networkx as nx
#import matplotlib.pyplot as plt
#get_ipython().magic(u'matplotlib inline')


#Set run parameters
area = nx.grid_graph(dim=[4,4])
#print area
n_c=10
n_a=10
c={1:{(0,0):{'d':-20,'l':[]},
      (3,3):{'d':10,'l':[(1,1.0)]},
      (1,3):{'d':10,'l':[(1,1.0)]}},
   2:{(3,3):{'d':-10,'l':[(1,1.0)]},
      (2,0):{'d':5,'l':[(2,1.0)]},
       (0,0):{'d':5,'l':[(2,1.0)]}}}
cap={1:[5,10,20],2:[5,10]}
max_c=25
t=30
r=4
a=1.0
b=1.0
ini_p=.5
diss=.2
keep=1
f_size=7

#Initialize space
space=Space(n_col=n_c,
            n_ant=n_a,
            spatial=area,
            components=c,
            capacities=cap,
            max_cap=max_c,
            tests=t,
            removals=r,
            alpha=a,
            beta=b,
            initial_pheromone=ini_p,
            dissipation=diss,
            keep_best=keep,
            figure_size=f_size)

#space=Space()
#Heuristic Tests:
#temp - off
#edge - off
#cap - off

#Functions:
#term - off
#cycle - 1000
count=0
print 'Generations: (generation, convergence criteria, # of failures, # of cycles, # of steps)'
while (space.convergence<10) and (count<25): #
    space.step()
    print '(gen: {}, con: {}, f: {}, cyc: {}, t: {}, s: {}, e: {}),'.format(count,
                                    space.convergence,space.failure,
                                    space.cycle,space.total_steps,
                                    space.start_steps,space.end_steps),
    count+=1
    

print ''
space.vizualize_c_f()    
space.vizualize_pf()
space.vizualize_systems()
        


