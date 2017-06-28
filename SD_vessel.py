# -*- coding: utf-8 -*-
"""
Created on Tue Jan 12 11:20:46 2016

@author: colinsh
"""

#from System_Structure_ACO import Space
from SS_v13_slime import Space
#from SS_v12_paper import Space
#from SS_v11_damage import Space
#from SS_v10_loop_prevention import Space
#from SS_v8_minmax import Space
#from SS_v7_stay import Space
#from SS_v6_iso import Space
#from SS_v5_restructure import Space
#from SS_v4_flow import Space
#from SS_v3_node_rem import Space
import networkx as nx
#import matplotlib.pyplot as plt
#get_ipython().magic(u'matplotlib inline')


#Set run parameters
#Make Ship
area = nx.grid_graph(dim=[8,6])
area.remove_node((0,0))
area.remove_node((0,5))
area.remove_node((0,4))
area.remove_node((1,5))
area.remove_node((1,4))
area.remove_node((4,5))
area.remove_node((4,4))
area.remove_node((5,5))
area.remove_node((5,4))
area.remove_node((6,5))
area.remove_node((6,4))
area.remove_node((7,5))
area.remove_node((7,4))


#area=nx.grid_graph(dim=[5,3,4])
#a_l=nx.spectral_layout(a)
#n_rem=[]
#for n in area:
    #print n
 #   if n[2]>2 and (n[0]==0 or n[0]>=4):
  #      n_rem.append(n)

#print n_rem
#area.remove_nodes_from(n_rem)

#c={1:{(0,1,0):{'d':1,'l':[(1,1.0)]}, #power
#      (3,1,0):{'d':1,'l':[(1,1.0)]},
#      (3,1,3):{'d':1,'l':[(1,1.0)]},
#      (0,1,2):{'d':1,'l':[(1,1.0)]},
#     (4,0,2):{'d':1,'l':[(1,1.0)]},
#      (4,2,2):{'d':1,'l':[(1,1.0)]},
#      (1,1,0):{'d':-6,'l':[(2,1.0)]}},
#   2:{(0,1,0):{'d':1,'l':[(2,1.0)]}, #chill water
#      (3,1,0):{'d':1,'l':[(2,1.0)]},
#      (3,1,3):{'d':-6,'l':[(1,1.0)]},
#      (0,1,2):{'d':1,'l':[(2,1.0)]},
#      (4,0,2):{'d':1,'l':[]},
#      (4,2,2):{'d':1,'l':[]},
#      (1,1,0):{'d':1,'l':[(2,1.0)]}}}

#print area
n_c=10
n_a=10
c={1:{(5,3):{'d':1,'l':[(1,1.0)]},
      (3,3):{'d':1,'l':[(1,1.0)]},
      (1,0):{'d':1,'l':[(1,1.0)]},
      (1,2):{'d':1,'l':[(1,1.0)]},
      (3,5):{'d':1,'l':[(1,1.0)]},
      (4,0):{'d':-6,'l':[(2,1.0)]},
      (5,0):{'d':1,'l':[(1,1.0)]}},
   2:{(5,3):{'d':1,'l':[(2,1.0)]},
      (3,3):{'d':1,'l':[(2,1.0)]},
      (1,0):{'d':1,'l':[(2,1.0)]},
      (1,2):{'d':1,'l':[(2,1.0)]},
      (3,5):{'d':1,'l':[(2,1.0)]},
      (4,0):{'d':1,'l':[(2,1.0)]},
      (5,0):{'d':-6,'l':[(1,1.0)]}}}
      
cap={1:[1,3,6],2:[1,3,6]}
max_c=25
t=50
r=(1,1)
a=1.0
b=0.0
ini_p=1.0
diss=0.1
keep=1
f_size=9

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
while (space.convergence<30) and (count<50): #
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
        


