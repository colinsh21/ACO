# -*- coding: utf-8 -*-
"""
Created on Fri Apr 24 14:21:38 2015

@author: colinsh
"""

# Imports
from System_Structure_ACO import Space


import datetime
import os
import time

# Scientific computing imports
import numpy
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
from operator import itemgetter
import seaborn; seaborn.set()

os.chdir("C:\Users\colinsh\Box Sync\Winter 2015\CMPLXSYS 530\ACO Data")


def get_closest_to_utopia(score_list):
    utopia_pt=(max(score_list,key=itemgetter(0))[0],min(score_list,key=itemgetter(1))[1])
    
    pts = np.asarray(score_list)
    dist_2 = np.sum((pts - utopia_pt)**2, axis=1)
    closest_index=np.argmin(dist_2)
    
    return score_list[closest_index]
       
    
def get_spread(score_list):
    best_obj1=(max(score_list,key=itemgetter(1)))
    best_obj2=(min(score_list,key=itemgetter(1)))
    dist=(abs(best_obj1[0]-best_obj2[0]),abs(best_obj1[1]-best_obj2[1]))
    
    spread=dist[0]*dist[1]
    
    return spread
    
    
def store_data(space,results,history,run_output_path):
    #store parameters
    model_parameters={'size':space.size,
                  'num_ants':space.num_ants,
                  'num_colonies':space.num_colonies,
                  'sources':space.source,
                  'source_mag':space.source_magnitude,
                  'sinks':space.sink,
                  'sink_mag':space.sink_magnitude,
                  'sink_threshold': space.sink_threshold,
                  'links':space.links,
                  'capacities':space.capacities,
                  'edge_capacity':space.edge_capacity,
                  'percent_removal':space.percent_removals,
                  'dissipation':space.dissipation,
                  'initial_pheromone': space.initial_pheromone,
                  'initial_termination': space.initial_termination,
                  'alpha':space.alpha,
                  'beta':space.beta
                  }
        
    #convert to df and save
    model_parameters_df = pandas.DataFrame(model_parameters.items(),
                         columns=["parameter", "value"])
    model_parameters_df.to_csv(os.path.join(run_output_path,"parameters.csv"),index=False)
    
    #convert and save history
    pareto_df=pd.DataFrame(history)
    pareto_df.to_csv(os.path.join(run_output_path,"pareto_history.csv"),index=False)
    
    #convert and save results
    #print results
    results_df=pd.DataFrame([results])
    results_df.columns=["dissipation_rate",
                        "ini_utopia_obj1",
                        "ini_utopia_obj2",
                        "ini_spread",
                        "last_utopia_obj1",
                        "last_utopia_obj2",
                        "last_spread"]
    results_df.to_csv(os.path.join(run_output_path,"run_results.csv"),index=False)
                            
        
def store_all_results(results_list,output_path='output'):
    results_list_df=pd.DataFrame(results_list)
    results_list_df.columns=['dissipation_rate',
                             'ini_utopia_obj1',
                             'ini_utopia_obj2',
                             'ini_spread',
                             'last_utopia_obj1',
                             'last_utopia_obj2',
                             'last_spread']
                                  
    results_list_df.to_csv(os.path.join(output_path,"results.csv"),index=False)
    

def visualize(space,pareto_history,ant_graphs,run_output_path):
    #save front
    space.visualize_pareto(pareto_history)
    plt.savefig(os.path.join(run_output_path, "pareto_history.png"))
    
    #save_systems
    for i in xrange(len(pareto_history[-1])):
        score=pareto_history[-1][i]
        space.visualize_system_single(ant_graphs[i][0],space.g,score)
        run_id="system_{0}.png".format(i+1)
        plt.savefig(os.path.join(run_output_path,run_id))

    
def store(space,results,pareto_history,ant_graphs,output_path='output'):
        
    try:
        os.makedirs(output_path)
    except:
        pass
    
    timestamp_suffix = time.strftime("%Y%m%d")
    
    run_id = 0
    run_output_path = os.path.join(output_path,
                                 "run-{0}-{1}".format(timestamp_suffix,
                                                     run_id))
    # Get a unique run #
    while os.path.exists(run_output_path):
        run_id += 1
        run_output_path = os.path.join(output_path,
                                 "run-{0}-{1}".format(timestamp_suffix,
                                                     run_id))        

    try:
        os.makedirs(run_output_path)
    except:
        pass
    
    store_data(space,results,pareto_history,run_output_path)
    visualize(space,pareto_history,ant_graphs,run_output_path)
    
#Run through dissipation rates

dissipation_list=[.6,.7,.8,.9]
out_p='output'
num_trials=5
results_list=[]
for c in xrange(len(dissipation_list)):
    print 'Running dissipation =',dissipation_list[c]
    for n in xrange(num_trials):
        #initialize with dissipation rate
        s=Space(size=[4,4],num_ants=10,num_colonies=10,
           source=[[(0,0)],[(3,3)]],source_magnitude=[[20],[10]],
           sink=[[(3,3),(1,3)],[(0,0),(2,0)]],sink_magnitude=[[10,10],[5,5]], sink_threshold=[[.5,1.0],[1.0,1.0]],
           links=[(1,0,(3,3))],capacities=[[5,10],[5]],edge_capacity=20,percent_removals=1.0,
           dissipation=dissipation_list[c],initial_pheromone=1.0,initial_termination=1.0,
           alpha=1.0,beta=1.0)
        
        #iterate until coverged
        converged=0
        i=1
        pareto_history=[]
        criteria=0
        while not converged:
            ant_graphs=s.step()
    
            paretoPoints=[]
            for ant in xrange(len(ant_graphs)):
                paretoPoints.append(ant_graphs[ant][-1])
    
            pareto_history.append(paretoPoints)
            if i>5 and pareto_history[-1]==pareto_history[-2]:
                criteria+=1
            else:
                criteria=0
    
            print 'Pareto front of run', c,'trial',n,'generation',i,':',paretoPoints #'\r',
            if criteria>10:
                converged=1
    
            i+=1
            if i>25:
                converged=1
        
        #calculate metrics
        ini_utopia=get_closest_to_utopia(pareto_history[0])
        ini_spread=get_spread(pareto_history[0])
        last_utopia=get_closest_to_utopia(pareto_history[-1])
        last_spread=get_spread(pareto_history[-1])
        
        #save run results
        results=[dissipation_list[c],ini_utopia[0],ini_utopia[1],ini_spread,last_utopia[0],last_utopia[1],last_spread]
        #print results
        results_list.append(results)
        store(s,results,pareto_history,ant_graphs,output_path=out_p)
    
store_all_results(results_list,output_path=out_p)    
    


