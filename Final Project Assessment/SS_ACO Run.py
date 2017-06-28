# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 17:28:10 2015

@author: colinsh
"""

from System_Structure_ACO import Space


space=Space(size=[4,4],num_ants=10,num_colonies=10,
            source=[[(0,0)],[(3,3)]],source_magnitude=[[20],[10]],
            sink=[[(3,3),(1,3)],[(0,0),(2,0)]],sink_magnitude=[[10,10],[5,5]], sink_threshold=[[.5,1.0],[1.0,1.0]],
            links=[(1,0,(3,3))],capacities=[[5,10],[5]],edge_capacity=20,percent_removals=1.0,
            dissipation=0.7,initial_pheromone=1.0,initial_termination=1.0,
            alpha=1.0,beta=1.0)
converged=0
i=1
pareto_history=[]
criteria=0
while not converged:
    ant_graphs=space.step()
    
    paretoPoints=[]
    for ant in xrange(len(ant_graphs)):
        paretoPoints.append(ant_graphs[ant][-1])
    
    pareto_history.append(paretoPoints)
    if i>5 and pareto_history[-1]==pareto_history[-2]:
        criteria+=1
    else:
        criteria=0

    print 'Pareto front of generation',i,':',paretoPoints, criteria #'\r',
    if criteria>10:
        converged=1

    i+=1
    if i>50:
        converged=1
        print '\n'

print 'Ant Paths'

#visualization
space.visualize_pareto(pareto_history)
 
for i in xrange(len(pareto_history[-1])):
    score=pareto_history[-1][i]
    space.visualize_system_single(ant_graphs[i][0],space.g,score)