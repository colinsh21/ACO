# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 17:28:10 2015

@author: colinsh
"""

#from System_Structure_ACO import Space
from SS_v6_iso import Space
#from SS_v5_restructure import Space
#from SS_v4_flow import Space
#from SS_v3_node_rem import Space
import networkx as nx
import matplotlib.pyplot as plt
get_ipython().magic(u'matplotlib inline')
#switched node complexity calculation.
#
#g = nx.grid_graph(dim=[3,3,3])
#
#space=Space(graph=g,num_ants=10,num_colonies=10,
#            source=[[(0,0,0)],[(2,2,2)]],source_magnitude=[[20],[10]],
#            sink=[[(2,2,2),(1,2,0)],[(0,0,0),(2,0,1)]],sink_magnitude=[[10,10],[5,5]], sink_threshold=[[.5,1.0],[1.0,1.0]],
#            links=[(1,0,(2,2,2))],capacities=[[5,10],[5]],edge_capacity=20,fraction_removals=15.0/27, removal_trials=20,
#            dissipation=0.2,initial_pheromone=1.0,initial_termination=1.0,local_search_params=(5,2),
#            alpha=1.0,beta=1.0)

def pareto_change(hist1,hist2,e): #hist1 is newer
    change=0    
    if len(hist1)!=len(hist2):
        change=1
    else:
        sum1=[[],[]]
        sum1[0]=sum(i for i,j in hist1)
        sum1[1]=sum(j for i,j in hist1)
        
        sum2=[[],[]]
        sum2[0]=sum(i for i,j in hist2)
        sum2[1]=sum(j for i,j in hist2)
        
        diff=[[],[]]
        diff[0]=abs(sum1[0]-sum2[0])/sum2[0]
        diff[1]=abs(sum1[1]-sum2[1])/sum2[1]
        
        if diff[0]>e or diff[1]>e:
            change=1
    return change

g = nx.grid_graph(dim=[4,4]) #must be undirected

# Testing local capacity pheromone dissipation
space=Space(graph=g,num_ants=10,num_colonies=10,
            source=[[(0,0)],[(3,3)]],source_magnitude=[[20],[10]],
            sink=[[(3,3),(1,3)],[(0,0),(2,0)]],sink_magnitude=[[10,10],[5,5]], 
            sink_threshold=[[.5,1.0],[1.0,1.0]],links=[(1,0,(3,3))],
            capacities=[[5,10],[5,10]],edge_capacity=15,
            fraction_removals=0.20,removal_trials=10,
            dissipation=0.2,initial_pheromone=0.5,initial_termination=0.5,
            local_search_params=(2,2),random=0,
            alpha=1.0,beta=1.0)


converged=0
e=.0001
i=1
pareto_history=[]
gen_history=[]
cycles=[]
failures=[]
t=[]
criteria=0
ant_hist=[]
print 'Generation:',
while not converged:
    ant_graphs,all_ants=space.step()
    ant_hist=list(ant_hist+all_ants)
    histPoints=[]
    for ant in xrange(len(all_ants)):
        #print 'ant',all_ants[ant]
        #print 'ant score',all_ants[ant][-1]
        histPoints.append(all_ants[ant][-1])
    #print histPoints
        
    gen_history.append(histPoints)
    #print gen_history
    
    paretoPoints=[]
    for ant in xrange(len(ant_graphs)):
        paretoPoints.append(ant_graphs[ant][-1])
    #print paretoPoints
    
    pareto_history.append(paretoPoints)
#    if len(pareto_history)>2:
#        change=pareto_change(pareto_history[-1],pareto_history[-2],e)
#    else:
#        change=0
        
    if i>5 and pareto_history[-1]==pareto_history[-2]:  #change==1:
        criteria+=1
    else:
        criteria=0
    
    #print 'Pareto front of generation',i,':',paretoPoints, criteria #'\r',
    if criteria>9:
        converged=1
    print '{},'.format((i,criteria,space.cycle_count,space.ant_failures,space.repeats)),
    cycles.append(space.cycle_count)
    failures.append(space.ant_failures)
    t.append(i)
    i+=1
    if i>50:
        converged=1
        print '\n'

print 'Ant Paths'
#visualization

plt.plot(t,cycles,'-b',label='cycles')
plt.plot(t,failures,'-r',label='failures')
plt.legend()

#opt_ants,score_list=space.ant_ranking(ant_hist)
#final_pareto=[]
#for i in opt_ants:
#    final_pareto.append(score_list[i])


opt_ants,score_list=space.ant_ranking(all_ants)
final_pareto=[]
for i in opt_ants:
    final_pareto.append(score_list[i])
    
space.visualize_pareto(gen_history, pareto_history, final_pareto)

opt_singles=[opt_ants[0]] #start
rep_count=[1]
for ant in opt_ants:
    g=all_ants[ant][0]
    match=0
    for single in opt_singles:
        g_s=all_ants[single][0]
        t_0=nx.is_isomorphic(g_s[0],g[0],node_match=space.nm,edge_match=space.em)
        t_1=nx.is_isomorphic(g_s[1],g[1],node_match=space.nm,edge_match=space.em)
        #print t_0,t_1
        if t_0 and t_1:
            match=1
            ID=single
            continue
    if match==0:
        opt_singles.append(ant)
        rep_count.append(1)
    if match==1:
        i=opt_singles.index(ID)
        #print opt_singles,rep_count,ID,i
        rep_count[i]+=1
        
        
#    for ant in opt_ants:
#        score=score_list[ant]
#        space.visualize_system_single(all_ants[ant][0],
#                                      space.source,space.source_magnitude,
#                                      space.sink,space.sink_magnitude,
#                                      space.g,score,7)
##print opt_singles,rep_count
fig_size=10                                
for ant in opt_singles:
    score=score_list[ant]
    i=opt_singles.index(ant)
    rep_score=(score[0],score[1],rep_count[i])
    space.visualize_system_single(all_ants[ant][0],
                                  space.source,space.source_magnitude,
                                  space.sink,space.sink_magnitude,
                                  space.g,rep_score,fig_size)



#space.visualize_pareto(gen_history, pareto_history, final_pareto)
#fig_size=10
#for ant in opt_ants:
#    score=score_list[ant]
#    #use ant_hist or all_ants depending on update scheme
#    space.visualize_system_single(all_ants[ant][0],
#                                  space.source,space.source_magnitude,
#                                  space.sink,space.sink_magnitude,
#                                  space.g,score,fig_size) 


#space.visualize_pareto(gen_history, pareto_history)
#
#fig_size=10
#for i in xrange(len(pareto_history[-1])):
#    score=pareto_history[-1][i]
#    space.visualize_system_single(ant_graphs[i][0],
#                                  space.source,space.source_magnitude,
#                                  space.sink,space.sink_magnitude,
#                                  space.g,score,fig_size)
    
                       