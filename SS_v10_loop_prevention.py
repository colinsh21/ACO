# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 09:43:45 2016

@author: colinsh
"""


# Standard imports
import copy
import itertools

# Scientific computing imports
import numpy
import matplotlib.pyplot as plt
import networkx as nx
import pandas
import collections
from capacity_scaling import capacity_scaling 
from relabel_nodes import convert_node_labels_to_integers
from collections import Iterable
from collections import namedtuple
from bisect import bisect
from networkx.algorithms.flow import edmonds_karp
from networkx.algorithms.traversal.depth_first_search import dfs_tree
import networkx.algorithms.isomorphism as iso
from random import choice
from random import sample

# Visualization imports
import matplotlib.cm as cm
#from matplotlib import rc
from mpl_toolkits.axes_grid1 import make_axes_locatable
get_ipython().magic(u'matplotlib inline')

import seaborn; seaborn.set()



class Ant(object):
    """
    Ant class, which encapsulates the entire behavior of an ant.
    """
    def __init__(self,model,ant_id,colony_id,perm,gen):
        """
        Class constructor.
        """
    
        # Set model link and ID
        self.model = model
        self.ant_id = (ant_id,colony_id,perm,gen)
        #Get objective weighting - (c-1)*r+k
        total_ants=float(self.model.n_ant*self.model.n_col)
        #ant_id[0]=ant, ant_id[1]=colony
        this_ant=self.model.n_ant*self.ant_id[1]+self.ant_id[0] #c*a/c+k
        self.w=this_ant/(total_ants-1.0)  
        
    def make_sos(self,ants):
        """
        Design system of sytems, takes in ant list so permuations can be set
        """
#        #Get objective weighting - (c-1)*r+k
#        total_ants=float(self.model.n_ant*self.model.n_col)
#        #ant_id[0]=ant, ant_id[1]=colony
#        this_ant=self.model.n_ant*self.ant_id[1]+self.ant_id[0] #c*a/c+k
#        self.w=this_ant/(total_ants-1.0)        
        
        #Initialize sos solution
        self.sos={}
        
        for sys in self.model.g:
            self.sos[sys]=self.make_s(sys)
        
        self.score=self.score_sos(self.sos)
       
        #Get min and max
        sos_min,score_min=self.s_min(self.sos)
        #Set permuation
        self.set_perm_sos(sos_min,score_min,1,ants) 
        
        sos_max,score_max=self.s_max(self.sos)
        #Set permuation
        self.set_perm_sos(sos_max,score_max,2,ants)            
        
        
                       
    def add_demand(self,g,sys):
        """
        Adds component demand for system.
        """
        g=copy.deepcopy(g)
        for c,val in self.model.components[sys].iteritems():
            g.add_node(c,demand=val['d'])
        return g
            
    def make_s(self,sys):
        """
        Design of single system
        """        
        #Iterate until system generated that meets flow demand
        sat=0
        while not sat:
            #Initialize system solution
            s_sol,total_steps,steps=self.sys_structure(sys)
            s_sol=self.sys_capacity(s_sol,sys)
            #Add demand
            s_sol=self.add_demand(s_sol,sys)
            
            sat,inflow=self.test_s(s_sol,sys)
            if sat==0:
                self.model.failure+=1
        
        self.end_steps=steps 
        self.total_steps=total_steps
        self.start_steps=total_steps-steps
            
        return s_sol
        
    def s_min(self,sos):
        """
        Removes all unused edges in the flow
        """
        sos_c=copy.deepcopy(sos)
        
        #set crossover thresold
        threshold=numpy.exp(-self.model.gen/50.0)
        
        #print 'pre',self.score_sos(sos_c)
        for sys,G in sos_c.iteritems():
            #randomize labels for fair capacity scaling
            #print G.edges()
            G=self.add_demand(G,sys)
            g=convert_node_labels_to_integers(G,ordering='random',label_attribute='old')
            name_map=nx.get_node_attributes(g,'old')
            flowCost, flowDict = nx.capacity_scaling(g,weight=3)
            
            #use flow to get a min cost network
            sys_caps=list(self.model.capacities[sys])
            #print sys,sys_caps
            #remove_bunch=[]
            
            for u,v,d in g.edges_iter(data=True):
                f=flowDict[u][v]
                num=numpy.random.uniform(0,1,1)
                if f==0 and num>=threshold:
                    g.remove_edge(u,v)
#                else:
#                    cap_index=next(x[0] for x in enumerate(sys_caps) if x[1] >= f)
#                    g[u][v]['capacity']=sys_caps[cap_index]
            g=nx.relabel_nodes(g,name_map)
            s,t=self.get_s_t(sys) 
            g=self.prune_graph(g,t,s)
            #print g.edges()
            sos_c[sys]=g

        score=self.score_sos(sos_c)
        #print 'post',score
        return sos_c,score
        
    def s_max(self,sos):
        """
        Make all edges bi-directional
        """ 
        sos_c=copy.deepcopy(sos)
        #set crossover thresold
        threshold=numpy.exp(-self.model.gen/50.0)        
        
        #ebunch=[]
        for sys,g in sos_c.iteritems():
            #print sys
            for u,v,d in g.edges_iter(data=True):
                #print u,v,d
                if d['capacity'] not in self.model.capacities[sys]:
                    print u,v,d,'error'
                num=numpy.random.uniform(0,1,1)
                if ((v,u) not in g.edges()) and num>=threshold:
                    g.add_edge(v,u,capacity=int(d['capacity']))
                    #ebunch.append((v,u,d['capacity']))
                    #ebunch.append((v,u,self.model.capacities[sys][-1]))
            #g.add_weighted_edges_from(ebunch,weight='capacity')
            
        score=self.score_sos(sos_c)
        
        
        return sos_c,score
    
    def set_perm_sos(self,sos_perm,sos_score,perm_id,ants):
        """
        Set an sos permutation and score to its corresponding ant
        ants input is list of all ant objects for the generation
        """
        
        perm_ant_id=list(self.ant_id)
        perm_ant_id[-2]=perm_id
        
        #get ant from id
        a=next((x for x in ants if x.ant_id==tuple(perm_ant_id)), None)
        
        #set sos and score
        a.set_sos(sos_perm,sos_score)        
        
#        print perm_ant_id
#        perm_ant=self.model.ant_by_id(tuple(perm_ant_id))
#        print perm_ant
#        perm_ant.set_sos(sos_perm,score)
        
    
    def set_sos(self,sos,score):
        """
        Set ant sos and score from input
        """
        self.sos=sos
        self.score=score
        
        
    def pheromone_reduction(self,p,sys,key,col,t,dec):
        """
        Reduce pheromone by a factor
        """
        factor=.1
        p[sys][key][col][0][t][dec]*=(1.0-factor)
        p[sys][key][col][1][t][dec]*=(1.0-factor)
        
        return p
        
            
    def sys_structure(self,sys):
        """
        Creates structure of system, sys.
        """    
        col=self.ant_id[1]        
        
        
        #Initialize solution   
        s_sol=nx.DiGraph()
        s,t=self.get_s_t(sys) 
        
        #initialize walk trackers
        terminate=0
        steps=0
        tot_steps=0
        consecutive_no_additions=0
        cycling=0
        sinks_included=0
        ever_cycled=0
        #exit_code=0
        
        #initialize run pheromone dict
        p=copy.deepcopy(self.model.p)
                
        
        d=nx.diameter(self.model.spatial)
        
        
        #initialize start of random walk
        last_nodes={}
        for n in s:
            last_nodes[n]=[n]
        
        for key in last_nodes:
            s_sol.add_node(key)
            
        #start walk
        while not terminate:
            '''
            Ant step is encompassed below
            '''            
            new=0 #was a new node added?
            steps+=1
            tot_steps+=1                      
            for key in last_nodes:
                

                #track next nodes
                last_nodes_next={}
                
                #Get number of branches
                ##Branch pheromones
    
                b_0=list(p[sys][key][col][0]['b'])
                b_1=list(p[sys][key][col][1]['b'])                
                b_ph=self.make_composite_ph(b_0,b_1,self.w)
                #branching heuristic - explore before sinks, exploit after
                #get neighbors already in graph
                n_in=s_sol.out_degree(key)
                factor=1.0
                for i in xrange(len(b_ph)):
                    if sinks_included: #replicate
                        if i != n_in:
                            b_ph[i]=b_ph[i]**self.model.alpha*\
                                factor**self.model.beta
                    else: #avoid
                        if i > (len(b_ph)-n_in):
                            b_ph[i]=b_ph[i]**self.model.alpha*\
                                factor**self.model.beta
                    
                
                #movement heuristic - include sinks???
#                r = numpy.exp(numpy.log(2)/(10.0*d))-1 #rate to double over d*k
#                T=(1+r)**steps #temperature at step x
#                if sinks_included:
#                    b_ph[0]*=T #as steps increases, chance of moving goes down
#                else:
#                    b_ph[0]*=1.0/T #as steps increase, chance of moving goes up
# 
                branches=self.make_decision(b_ph)
                

                            
                ### Inertia term - branching heuristic
                #Terminate walk once all ants are stationary.
                
                neighbors_chosen=[] #list other branches taken out of that node
                
                #If all sinks included can terminate ants
                terminate_ant=0
#                if sinks_included==1:
#                    t_0=self.model.p[sys][key][col][0]['t']
#                    t_1=self.model.p[sys][key][col][1]['t']
#                    t_ph=self.make_composite_ph(t_0,t_1,self.w)
#                    terminate_ant=self.make_decision(t_ph)
#                    if terminate_ant==1: #0 is do not terminate
#                        branches=0 #this ant does not branch
                        
                if branches==0: #Either not leaving or terminated
                    if terminate_ant==0: #not terminated, add to next round
                        next_node=key
                        #record node and predecessor for next step
                        if next_node in last_nodes_next:
                            last_nodes_next[next_node].append(key)
                        else:
                            last_nodes_next[next_node]=[key]
                            
                p=self.pheromone_reduction(p,sys,key,col,'b',branches)
                        
                #continue walk for each chosen branch       
                for branch in xrange(branches):
                    #get neighbors
                    all_p_neighbors=self.model.spatial.neighbors(key)
                   
                    p_neighbors=list(set(all_p_neighbors)-set(neighbors_chosen))
                    #get pheromone of each edge to neighbor
                    edge_pheromone_list=[]
                    
                    for potential_node in p_neighbors:
                        edge_heuristic=1.0#/((potential_node in s_sol)+1.0)
                        e=(key,potential_node)
                        e_0=list(p[sys][e][col][0]['e'])
                        e_1=list(p[sys][e][col][1]['e'])
                        e_ph=self.make_composite_ph(e_0,e_1,self.w)
                                                            
                        if sinks_included==0: #edge heuristic
                            paths=nx.shortest_path_length(self.model.spatial,
                                                      source=potential_node)
                            p_sink=[]
                            
                            #sinks not in walk
                            un_sink_list=list(set(s.keys())-set(s_sol.nodes()))
                            #print un_sink_list
                            
                            if un_sink_list:
                                for un_sink in un_sink_list:
                                    p_sink.append(paths[un_sink])
                                #print p_sink
                                # get dist to closest sink in terms of fraction of diameter
                                #d=nx.diameter(self.model.g)
                                #min_frac_dist=float(min(p_sink))/d
                                #if min_frac_dist==0.0: #on sink
                                    #min_frac_dist=0.1/d #1/th the min fractional distance (1/d)
                                #edge_heuristic*=1.0
#                                r = numpy.exp(numpy.log(2)/(d))-1 #rate to double over d*k
#                                T=(1+r)**consecutive_no_additions #temperature at step x
                                
#                                edge_heuristic*=(T)/\
#                                    (min_frac_dist+1.0)
                                
#                                edge_heuristic*=(consecutive_no_additions+1.0)/\
#                                    (min_frac_dist+1.0) # was .1
                                #edge_heuristic*=(steps)/(min_frac_dist) # was .1
                                
                            #Explore
                            if potential_node in s_sol:
                                edge_heuristic*=factor
                        else: #Exploit 
                            if potential_node not in s_sol:
                                edge_heuristic*=factor
                        #combine pheromones
                        if cycling:
                            e_ph=edge_heuristic**self.model.beta
                        else:
                            e_ph=e_ph[0]**self.model.alpha*\
                                edge_heuristic**self.model.beta #edited out for testing
                            
                        edge_pheromone_list.append(e_ph)
                    
                    #Choose edge
                    edge_index=self.make_decision(edge_pheromone_list)
                    next_node=p_neighbors[edge_index]
                    neighbors_chosen.append(next_node) #record branch taken 
                    
                    p=self.pheromone_reduction(p,sys,e,col,'e',0)
                    
                    #is this node new? **could check edges too**
                    if next_node not in s_sol.nodes():
                        new=1 
                    
                    #add node and edge
                    s_sol.add_edge(key,next_node)

                    #record node and predecessor for next step
                    if next_node in last_nodes_next:
                        last_nodes_next[next_node].append(key)
                    else:
                        last_nodes_next[next_node]=[key]

                    
            #Increment cycling tracker
            if new==0:
                consecutive_no_additions+=1
            else:
                consecutive_no_additions=0
            
            sinks_included=1 #check if all sinks in the walk
            for sink_node in t:
                if sink_node not in s_sol.nodes():
                    sinks_included=0
                    steps=0
            
#                #If cycling, either end or force ants to explore
#                if consecutive_no_additions>10:
#                    cycling=1
#                    if sinks_included==0:
#                        for node in self.last_nodes_next:#if cycling encourage ants to explore
#                            #record all predecessors as successors
#                            self.last_nodes_next[node].append(self.g.predecessors(node))                
#                            
#                            #record all successors as predecessors
#                            self.last_nodes_next[node].append(self.g.neighbors(node))
#                    else: #if sinks included and cycling, end
#                        self.last_nodes_next={} #cut list triggering termination
            #cycling=0
            if consecutive_no_additions>1.0*d:
                #print 'cycle termination'
                # for each sink not included, add path from existing node
                # add
                #exit_code=1
                cycling=0 #removed cycling
                ever_cycled=1
                if sinks_included==1:
                    #self.model.cycle+=1
                    last_nodes_next={} #cut list triggering termination
            
            movement=0
            for v,u in last_nodes_next.iteritems():
                if [v]!=u:
                    movement+=1
            
            
            #Check Termination Criteria
            if (not last_nodes_next) or (movement==0 and sinks_included==1):
                terminate=1

            last_nodes=dict(last_nodes_next)
        
        self.model.cycle+=ever_cycled
        s_sol=self.prune_graph(s_sol,t,s)
                
        return s_sol,tot_steps, steps
        
    def sys_capacity(self,s_sol,sys):
        """
        Creates capacited routing for system solution, s_sol
        """
        col=self.ant_id[1]
        
        #initialize run pheromone dict
        p=copy.deepcopy(self.model.p)
        
        #Get source and sink information
        s,t=self.get_s_t(sys) 
        
        # make dictionary of max supporting and recieved flow
        st_dict={}
        for n in s_sol:
            st_dict[n]={}
            st_dict[n]['t']=0.0 #mag of sinks node supports
            st_dict[n]['s']=0.0 #mag of sources node recieves from 
            
        # determine which sinks depend on node v
        for n,mag in t.iteritems():
            if n in s_sol:
                #mag=self.sink_mag[self.sink.index(t)]
                for i in dfs_tree(s_sol.reverse(),n).nodes(): #get nodes that feed sink
                    st_dict[i]['t']+=mag
            
        # determine which sources feed node v
        for n,mag in s.iteritems():
            if n in s_sol:
                #mag=self.source_mag[self.source.index(s)]
                for i in dfs_tree(s_sol,n).nodes(): #get nodes that feed sink
                    st_dict[i]['s']+=mag
                    
        sys_caps=self.model.capacities[sys]            
        for (u,v) in s_sol.edges():
            c_0=list(p[sys][(u,v)][col][0]['c'])
            c_1=list(p[sys][(u,v)][col][1]['c'])
            c_ph=self.make_composite_ph(c_0,c_1,self.w)
            #print 'cap_pheromone',(key,self.next_node),capacity_pheromone_list                         
            
            heuristic_list=[]
            #cap_lim=min([st_dict[u]['s'],st_dict[v]['t']])
            
            for cap in sys_caps:
                heuristic_value=1.0
                #heuristic_value+=1.0/(1.0+abs(cap_lim-cap))
                heuristic_list.append(heuristic_value**self.model.beta)
            
            adj_ph=[]
            for j in xrange(len(c_ph)):
                adj_ph.append((c_ph[j]**self.model.alpha)*heuristic_list[j]) 
            
            cap_i=self.make_decision(adj_ph)    
            cap_to_add=sys_caps[cap_i]
            s_sol[u][v]['capacity']=cap_to_add
            
            p=self.pheromone_reduction(p,sys,(u,v),col,'c',cap_i)
            
        
        return s_sol
    
    def prune_graph(self,graph,sinks,sources):
        """
        Removes nodes that only go out and back or are dead ends
        """
        done=0
        while not done:
            done=1
            for n in graph: 
                if graph.out_degree(n)==0 and (n not in sinks): #dead end, non-sink nodes
                    done=0
                    break
                
                if graph.in_degree(n)==0 and (n not in sources): #no start, non-source nodes
                    done=0
                    break
                
                if (graph.out_degree(n)==1) and (graph.in_degree(n)==1):
                    if (n not in sinks) and (n not in sources): #there and back non-critical nodes
                        neighbor=graph.neighbors(n)
                        if n in graph.neighbors(neighbor[0]):
                            done=0
                            break
            if done==0:
                graph.remove_node(n)
        
        return graph    
    
    
    
    def get_s_t(self,sys):
        """
        Gets source and sink info for system.
        """
        s_dict={}
        t_dict={}
        
        for c,info in self.model.components[sys].iteritems():
            if info['d']<0:
                s_dict[c]=-info['d']
                
            if info['d']>0:
                t_dict[c]=info['d']

        return s_dict,t_dict

    def make_composite_ph(self,l_1,l_2,w):
        """
        Returns composite pheromone list based on weighting.
        """
        return [l_1[i]**w*l_2[i]**(1.0-w) for i in xrange(len(l_1))]


    def make_decision(self,pheromone_list):
        """
        Return decision index, based on pheromone list.
        """
        #convert pheromones to percentage
        percent_list = [float(i)/sum(pheromone_list) for i in pheromone_list]   
        cumulative_percent=numpy.cumsum(percent_list)

        #Choose decision index
        select_index=bisect(cumulative_percent,numpy.random.uniform(0,1,1))
  
        return select_index
        
    def test_s(self,s_sol,sys):        
        """
        Test a single system for flow satisfaction
        """
        g=copy.deepcopy(s_sol)
        
            
        #Test flow
        cost,outflow=capacity_scaling(g)
        
        #Create inflow dict
        inflow={}
        for n in g.nodes():
            inflow[n]=0
        for pred,out in outflow.iteritems():
            for dest,in_f in out.iteritems():
                inflow[dest]+=in_f
        
        #Check demand satisfaction
        sat=1
        sat_dict={}
        for c,val in self.model.components[sys].iteritems():
            if val['d']>inflow[c]:
                sat=0
                sat_dict[c]=0
            else:
                sat_dict[c]=1
        
        return sat, inflow #, sat_dict
        
    def test_sos(self,sos):
        """
        Test system of system for flow satisfaction
        """
        sos_eval=copy.deepcopy(sos)
        
        
        #Check component operating or not, failed if any demand not sat.
        comp_keys=[]
        for sys in self.model.sys_types:
            comp_keys.extend(self.model.components[sys].keys())
        comp_keys=list(set(comp_keys))
        
        operating={k: 1 for k in comp_keys}        
        
        #Check if existing
        for sys,g in sos_eval.iteritems(): 
            for c,val in self.model.components[sys].iteritems():
                if c not in g:
                    #removed nodes are failed.
                    operating[c]=0
                #g.add_node(c,demand=val['d'])        
        #Add demand
        for sys,g in sos_eval.iteritems():
            sos_eval[sys]=self.add_demand(g,sys)
            
        #Store SoS information        
        sos_info={k: {'sat':0,'inflow':{}} for k in self.model.sys_types}
        op_list=[copy.deepcopy(operating)]
        while True:
            #Test individual systems 
            
            #print sos_eval
            for sys,g in sos_eval.iteritems():
                sat,inflow=self.test_s(g,sys)
                sos_info[sys]['sat']=sat
                sos_info[sys]['inflow']=inflow
                #sys_sat[sys]=sat_dict
            #print 'eval results', sos_info
            #check component dependence
            #failure=0
            
            
            for sys in self.model.components:
                for c,val in self.model.components[sys].iteritems():
                    #what are links                    
                    for l in val['l']:
                        #dependent system
                        d_sys=l[0]
                        threshold=l[1]
                        
                        #check flow on dependent
                        dep_flow=sos_info[d_sys]['inflow'][c]
                        dep_req=self.model.components[d_sys][c]['d']
                        dep_ratio=float(dep_flow)/dep_req
                        #print 'link', c,l,'in=',dep_flow,'req=',dep_req
                        if dep_ratio<threshold:
                            #failure - remove demand from graph
                            sos_eval[sys].node[c]['demand']=0
                            operating[c]=0
                            #failure=1
            
            #store operating list
            op_list.append(copy.deepcopy(operating))
            #if no failure, converged. End while loop.
            if op_list[-1]==op_list[-2]: #failure==0
                #print 'operating list',op_list
                break
        
        return sat, operating
        
    def rep_cost(self,sos):
        """
        Get representative cost of sos.
        """
        
        installed_cap=0
        num_over=0
        
        #iterate over edges        
        for u,v in self.model.spatial.edges_iter():
            edge_cap=0
            
            #iterate over systems            
            for sys,g in sos.iteritems():
                if g.has_edge(u,v):
                    edge_cap+=g[u][v]['capacity']
                if g.has_edge(v,u):
                    edge_cap+=g[v][u]['capacity']
            
            if edge_cap>self.model.max_cap:
                num_over+=1
            
            installed_cap+=edge_cap

        return installed_cap, num_over        
        
    def score_sos(self,sos):
        """
        Get score of system of systems
        """
        
        #Survivability
        results=[]
        for i in xrange(self.model.tests):
            sos_damage=copy.deepcopy(sos)
            #remove nodes due to damage
            sos_damage=self.inflict_damage(sos_damage)
            sat,operating=self.test_sos(sos_damage)
            working=sum(operating.itervalues())
            
            #trial score is fraction of working components
            results.append(float(working)/len(operating.keys()))
        
        s_score=numpy.average(results)
        #print results,s_score

        #Cost
        cost_sos=copy.deepcopy(sos)
        installed_cap,num_over=self.rep_cost(cost_sos)
        
        #Penalty function for infeasible edges
        #percentage over capacity edges
        pen=float(num_over)/self.model.spatial.number_of_edges()
        
        sp_score=s_score/(1.0+pen)
        cp_score=installed_cap*(1.0+pen)
        
        return (sp_score,cp_score)
        
    def inflict_damage(self,sos):
        """
        Removes nodes from sos.
        """
        nodes_to_remove=sample(self.model.spatial.nodes(),
                               self.model.removals)
        #print 'removed', nodes_to_remove
        
        #Remove node in each system                       
        for sys,g in sos.iteritems():
            for n in nodes_to_remove:
                if n in g:
                    g.remove_node(n)
                    
        return sos
        
        
        
        
    def __repr__(self):
        '''
        Return string representation.
        '''
        skip_none = True
        repr_string = type(self).__name__ + " ["
        except_list = "model"

        elements = [e for e in dir(self) if str(e) not in except_list]
        for e in elements:
            # Make sure we only display "public" fields; skip anything private (_*), that is a method/function, or that is a module.
            if not e.startswith("_") and eval('type(self.{0}).__name__'.format(e)) not in ['DataFrame', 'function', 'method', 'builtin_function_or_method', 'module', 'instancemethod']:
                    value = eval("self." + e)
                    if value != None and skip_none == True:
                        repr_string += "{0}={1}, ".format(e, value)

        # Clean up trailing space and comma.
        return repr_string.strip(" ").strip(",") + "]"    
        

class Space(object):
    """
    Space class, which encapsulates the entire behavior of a single "run" ACO.
    """
    
    def __init__(self,n_col=10,n_ant=10,
                 spatial=nx.grid_graph(dim=[3,3]),
                 components={1:{(0,0):{'d':-5,'l':[]},
                                (1,1):{'d':5,'l':[(1,1.0)]}},
                                2:{(1,1):{'d':-5,'l':[(1,1.0)]},
                                   (2,0):{'d':5,'l':[(2,1.0)]}}},
                 capacities={1:[5,10],2:[5]},
                 max_cap=20,
                 tests=30,removals=4,
                 alpha=1.0,beta=1.0,
                 initial_pheromone=.5,dissipation=.2,
                 keep_best=0,
                 figure_size=7):
        """
        Class constructor.
        """
        
        #Initialize space
        #components=dict{sys{node{d,l}}}
        ##d<0=>source mag.
        ##d>=>sink mag.
        ##l=[(recieving,threshold)]
        
        self.components=components
        self.spatial=spatial
        #print spatial
        self.g=self.init_space(self.spatial,self.components)
        
        #Initialize ants
        self.n_col=n_col
        self.n_ant=n_ant

        #capacities={sys:[caps]}        
        self.capacities=capacities
        
        #System information
        self.n_sys=len(self.components)
        self.sys_types=self.components.keys()
        
        
        #Pheromones
        self.ini_ph=initial_pheromone
        self.p=self.init_pheromones(self.g,self.n_col,self.capacities,
                                    self.ini_ph)
        self.dissipation=dissipation
                                        
        #Testing parameters
        self.tests=tests
        self.removals=removals 
        #maximum installed capacity on a bi-directional edge        
        self.max_cap=max_cap 
        
        #System design parameters
        self.alpha=alpha
        self.beta=beta
        
        #Generation List
        self.gen=0
        self.keep_best=keep_best
        self.pareto_history=[]
        self.ants=[] #self.init_ants(self.n_col,self.n_ant,self.gen)
        self.all_ants=[]
        self.all_scores=[]
        self.pareto_scores=[]
        
        #Generation tracking        
        self.convergence=0
        self.failure=0
        self.cycle=0
        self.fail_list=[]
        self.cycle_list=[]
        self.t_step_list=[]
        self.e_step_list=[]
        self.s_step_list=[]
        
        #System plotting
        self.figure_size=figure_size
    
    
    def init_space(self,spatial,components):
        """
        Constructs space with sources and sinks.
        """
        space_g=copy.deepcopy(spatial)
        space_g=nx.DiGraph(space_g)
        #print space_g
        
        g={}
        
        for sys in components:
            #create s,t,l info for each system
            g[sys]=copy.deepcopy(space_g)
            
            for n in components[sys]:
                g[sys].node[n]=components[sys][n]
            
        return g
        
        
    def init_pheromones(self,s,n_col,caps,ini_ph):
        """
        Creates pheromone data structure.
        """
        ph_struct={}
        #ph_struct={sys{element{colony{objective{type:ph_list}}}}}        
        
        
        #Create nested dict for all elements.
        for sys in caps:
            #system level            
            ph_struct[sys]={}
            
            #node level
            for n in s[sys].nodes():
                #colony level
                ph_struct[sys][n]={}
                for c in xrange(n_col):
                    #objective level
                    ph_struct[sys][n][c]={}
                    for obj in xrange(2):
                        ph_struct[sys][n][c][obj]={}
                        #node pheromone type level
                        #branch - possible branch choices = k
                        ph_struct[sys][n][c][obj]['b']= \
                            [ini_ph]*(s[sys].out_degree(n)+1) #allows no movement
                    
                        #termination
                        ph_struct[sys][n][c][obj]['t']= \
                            [ini_ph]*2
                    
            #edge level
            for e in s[sys].edges_iter():
                #colony level
                ph_struct[sys][e]={}
                for c in xrange(n_col):
                    #objective level
                    ph_struct[sys][e][c]={}
                    for obj in xrange(2):
                        ph_struct[sys][e][c][obj]={}
                        #node pheromone type level
                        #route
                        ph_struct[sys][e][c][obj]['e']=[ini_ph]                   
                    
                        #capacity - possible caps=number for system
                        ph_struct[sys][e][c][obj]['c']=[ini_ph]*len(caps[sys])
                    
        
        
        return ph_struct
        
        
    def init_ants(self,n_col,n_ant,gen):
        """
        Creates ant objects.
        """
        ants=[]
        #perm=0
        perm=2 #min and max permutations
        for c in xrange(n_col):
            for i in xrange(n_ant):
                for p in xrange(perm+1): 
                    ants.append(Ant(model=self,
                                    ant_id=(i),
                                    colony_id=(c),
                                    perm=p,
                                    gen=gen))
        
        return ants
        
    def set_sos(self,sos,score,a_id):
        """
        Set sos and score for an ant
        """
        a=self.ant_by_id(a_id)
        #print a_id
        #print a
        a.set_sos(sos,score)
        
        
    def ACO_generation(self,generation):
        """
        Gets solutions for the ACO generation.
        """
        ants=self.init_ants(self.n_col,self.n_ant,generation)
        tot_steps=[]
        end_steps=[]
        start_steps=[]
        #print ants,len(ants)
        
        for a in ants:
            #print a.ant_id
            if a.ant_id[-2]==0:
                #print a.ant_id
                a.make_sos(ants)
                tot_steps.append(a.total_steps)
                end_steps.append(a.end_steps)
                start_steps.append(a.start_steps)
            
            self.all_scores.append(a.score)
                
    
        
        #self.all_ants.extend(ants)
        if self.keep_best==1:    
            self.ants.extend(ants)         
        else:
            self.ants=ants
            
        return numpy.average(tot_steps),numpy.average(end_steps),numpy.average(start_steps)
            
    def rank_generation(self):
        """
        Ranks scores of the ACO.
        """
        #Get scores
        s=[]
        for a in self.ants:
            s.append(a.score)
            
        equality_sequence=[1,0] #[>=,<=]
        p_front, dom_points=self.simple_cull_front(s,self.dominates,
                                                   equality_sequence)
        p_front=list(p_front)
        
        #get ant ID's for pareto front
        p_ids=[]
        
        for a in self.ants:
            if a.score in p_front:
                p_ids.append(a.ant_id)
        
        #Get boundary points for pareto front        
        temp_list_rank=map(list,zip(*p_front)) #two list, one for each score
        #print temp_list_rank
        slr=[] #score list for ranking
        best=[] #best score [surv,cost]
        
        #survivability          
        array=numpy.array(temp_list_rank[0])
        temp=array.argsort()[::-1] #for reverse order, bigger is lower rank
        ranks=numpy.empty(len(array),int)
        ranks[temp]=numpy.arange(len(array))
        slr.append(ranks)
        best.append(max(temp_list_rank[0]))        
        
        #complexity
        array=numpy.array(temp_list_rank[1])
        temp=array.argsort()
        ranks=numpy.empty(len(array),int)
        ranks[temp]=numpy.arange(len(array))
        slr.append(ranks)
        best.append(min(temp_list_rank[1]))
        
        
        return p_ids,best
        
    def dominates(self, point_1, point_2, equality_sequence):
        '''
        Calculates if a point is dominated by another point, used in simple_cull_front
        equality_sequence:= 1 is '>=',0 is '<='
        '''
        score=0
        for i in range(len(point_1)):
            if equality_sequence[i]==1 and point_1[i]>=point_2[i]:
                score+=1
            elif equality_sequence[i]==0 and point_1[i]<=point_2[i]:
                score+=1
        dom=score==len(point_1)    
        return dom  
        
        
    def simple_cull_front(self, inputPoints, dominates, equality_sequence):
        '''
        Basic algorithm to find the pareto front of a set of points
        min or max is determined based on equality_sequence:= 0 is min, 1 is max
        '''
        paretoPoints = set()
        candidateRowNr = 0
        dominatedPoints = set()
        while True:
            candidateRow = inputPoints[candidateRowNr]
            inputPoints.remove(candidateRow)
            rowNr = 0
            nonDominated = True
            while len(inputPoints) != 0 and rowNr < len(inputPoints):
                row = inputPoints[rowNr]
                if self.dominates(candidateRow, row,equality_sequence):
                    # If it is worse on all features remove the row from the array
                    inputPoints.remove(row)
                    dominatedPoints.add(tuple(row))
                elif self.dominates(row, candidateRow, equality_sequence):
                    nonDominated = False
                    dominatedPoints.add(tuple(candidateRow))
                    rowNr += 1
                else:
                    rowNr += 1

            if nonDominated:
                # add the non-dominated point to the Pareto frontier
                paretoPoints.add(tuple(candidateRow))

            if len(inputPoints) == 0:
                break
        return paretoPoints, dominatedPoints
        
    def ant_by_id(self,a_id):
        """
        Return ant object by id.
        """
        return next((x for x in self.ants if x.ant_id==a_id), None)
        #return next((x for x in self.all_ants if x.ant_id==a_id), None)
        
    def ph_update(self,p_id,p_bounds):
        """
        Updates pheromone structure
        """
        #Dissipate
        self.ph_dissipation()
        
        #Add
        #calculate total pareto fitness
        f=[0.0,0.0]
        for ant in p_id:
            s=self.ant_by_id(ant).score
            f[0]+=s[0]/p_bounds[0]
            f[1]+=p_bounds[1]/s[1]
            
        #print f
        for ant in p_id:
            s=self.ant_by_id(ant).score
            inc=[s[0]/p_bounds[0],p_bounds[1]/s[1]]
            inc=[inc[0]/f[0],inc[1]/f[1]]
            self.ph_addition(ant,inc)
        
        
    def ph_dissipation(self):
        """
        Dissipates all pheromones.
        """
        #ph_struct={sys{element{colony{objective{type:ph_list}}}}}         
        for sys in self.p:
            for e in self.p[sys]:
                for col in self.p[sys][e]:
                    for obj in self.p[sys][e][col]:
                        for y in self.p[sys][e][col][obj]:
                            for i in xrange(len(self.p[sys][e][col][obj][y])):
                                self.p[sys][e][col][obj][y][i]*=\
                                    (1.0-self.dissipation)
                                if self.p[sys][e][col][obj][y][i]<0.01:
                                    self.p[sys][e][col][obj][y][i]=0.01
                                
            
        
    def ph_addition(self,ant_id,inc):
        """
        Adds to pheromones based on solutions.
        """
        #inc=[theta_1,theta_2]
        add=[inc[0]*self.dissipation, inc[1]*self.dissipation]
        #add=list(inc)
        #print add
        
        #Get sos and score from ant
        sos=copy.deepcopy(self.ant_by_id(ant_id).sos)
        #s=self.ant_by_id(ant_id).score
        
        #ant_id[0]=ant, ant_id[1]=colony
        col=ant_id[1]
        
        #Perform update
        #ph_struct={sys{element{colony{objective{type:ph_list}}}}}
        
        for sys,g in sos.iteritems():
            
            #g.edges() - e,c
            for e in g.edges_iter():
                #use of edge
                self.p[sys][e][col][0]['e'][0]+=add[0]
                self.p[sys][e][col][1]['e'][0]+=add[1]
                
                #capacity
                c=g[e[0]][e[1]]['capacity']
                #print c
                if c not in self.capacities[sys]:
                    print ant_id,sys,c
                    self.vizualize_sos(ant_id)

                cap_index=self.capacities[sys].index(c)
                self.p[sys][e][col][0]['c'][cap_index]+=add[0]
                self.p[sys][e][col][1]['c'][cap_index]+=add[1]
            
            #g.nodes() - b,t
            #Add empty nodes for review
            for n in self.spatial:
                if n not in g:
                    #g.add_node(n)
                    #positive termination
                    self.p[sys][n][col][0]['t'][1]+=add[0]
                    self.p[sys][n][col][1]['t'][1]+=add[1]
                    
                    #branch=0
#                    self.p[sys][n][col][0]['b'][0]+=add[0]
#                    self.p[sys][n][col][1]['b'][0]+=add[1]                    
                    
                if n in g:
                    #negative termination
                    self.p[sys][n][col][0]['t'][0]+=add[0]
                    self.p[sys][n][col][1]['t'][0]+=add[1]
                    
            for n in g.nodes():
                k=g.out_degree(n)
                
                self.p[sys][n][col][0]['b'][k]+=add[0]
                self.p[sys][n][col][1]['b'][k]+=add[1]
                
                #positive termination
#                if k==0:
#                    self.p[sys][n][col][0]['t'][1]+=add[0]
#                    self.p[sys][n][col][1]['t'][1]+=add[1]
#       
#                else: 
#                    #negative termination
#                    self.p[sys][n][col][0]['t'][0]+=add[0]
#                    self.p[sys][n][col][1]['t'][0]+=add[1] 
                
#                    #number of branches
#                    b=k
#                    self.p[sys][n][col][0]['b'][b]+=add[0]
#                    self.p[sys][n][col][1]['b'][b]+=add[1]
        
        
    def step(self):
        """
        One ACO step
        """
        
        #Get generation
        self.cycle=0
        self.failure=0
        self.total_steps,self.end_steps,self.start_steps=self.ACO_generation(self.gen)
        self.gen+=1
        self.t_step_list.append(self.total_steps)
        self.e_step_list.append(self.end_steps)
        self.s_step_list.append(self.start_steps)
        self.cycle_list.append(self.cycle)
        self.fail_list.append(self.failure)
        
        #Rank generation
        pareto_id, pareto_bounds=self.rank_generation()
        self.pareto_history.append(pareto_id)
        front_scores=[]
        for ant_id in pareto_id:
            s=self.ant_by_id(ant_id).score
            front_scores.append(s)
            
        self.pareto_scores.append(front_scores)
        
        #Update pheromones
        self.ph_update(pareto_id,pareto_bounds)
        
        #Only keep pareto ants for next generation        
        #print len(self.ants)
        #self.ants=[]
        keep=[]
        for a_id in pareto_id:
            keep.append(self.ant_by_id(a_id))
            #self.ants.append(self.ant_by_id(a_id)) #original with self.all ants
        self.ants=keep
        
        
        #Convergence check
        if len(self.pareto_history)>2:
            if self.pareto_history[-1]==self.pareto_history[-2]:
                self.convergence+=1
            else:
                self.convergence=0
        
    def vizualize_systems(self):
        """
        Creates plots of current pareto front.
        """
        for a_id in self.pareto_history[-1]:
            self.vizualize_sos(a_id)
        
        
        
    def vizualize_sos(self,ant_id):
        """
        Displays System of systems.
        """
        # activate latex text rendering
        #rc('text', usetex=True)
        color_seq=['cyan','red','green','magenta','yellow']     
        #g_layout=nx.spectral_layout(self.spatial)
        g_layout = dict(zip(self.spatial,self.spatial))
        
        f,axs=plt.subplots(1,self.n_sys, 
                           figsize=(self.figure_size*self.n_sys,
                                    self.figure_size),
                            dpi=1000,sharey=True)
        
        #Set boundaries
        cut = 1.15
        xmax = cut * max(xx for xx, yy in g_layout.values())
        xmin = cut * min(xx for xx, yy in g_layout.values())
        ymax = cut * max(yy for xx, yy in g_layout.values())        
        ymin = cut * min(yy for xx, yy in g_layout.values())
        s=self.ant_by_id(ant_id).score
        
        plt.suptitle(r'Ant {}: Survivability$={:1.2f}$, Capacity$={:6.2f}$'.format(ant_id,s[0],s[1]),
                  fontsize=17)
        
          
        plot_count=0
        #print axs
        sos=self.ant_by_id(ant_id).sos
        for sys,g in sos.iteritems():
            
            #Get axis            
            ax_g=axs[plot_count]
            color=color_seq[plot_count]

            #Set axis title
            ax_g.set_title("System {}, Demand and Capacities".format(sys,color),size=15)
            
            #Set plot limits
            ax_g.set_ylim([ymin-.25,ymax])#min(y_p)*.1
            ax_g.set_xlim([xmin-.25,xmax])
            #ax_g.set_axis_off()
            
            #Plot system
            
            
            nx.draw_networkx_nodes(g, g_layout, 
                                nodelist=self.components[sys],node_size=1000,
                                alpha=1, node_color=color, ax=ax_g)
            #Black
            nx.draw_networkx_edges(self.spatial, g_layout,
                              edgelist=self.spatial.edges(),width=1,
                              alpha=.8, ax=ax_g)
            #Color
            nx.draw_networkx_edges(g, g_layout,
                              edgelist=g.edges(),width=4,
                              alpha=.5,edge_color=color, ax=ax_g)
        
            n_labels=nx.get_node_attributes(self.g[sys],'d')
            
            nx.draw_networkx_labels(g,g_layout,n_labels,font_size=10,ax=ax_g)
            
            e_labels=nx.get_edge_attributes(g,'capacity')
            
            nx.draw_networkx_edge_labels(g,g_layout,edge_labels=e_labels,
                                         label_pos=.27,font_color=color,
                                         font_size=15,ax=ax_g)
            plot_count+=1
        
    def vizualize_pf(self):
        """
        Displays pareto front and history.
        """
        x=[]
        y=[]
        gen=[]
        #utopia=[]
        #seen=[]
        
        #Get pareto evolution
        gen_num=0
#        for front in self.pareto_history:
#            for ant_id in front:
#                s=self.ant_by_id(ant_id).score
#                x.append(s[0])
#                y.append(s[1])
#                gen.append(gen_num)
#            
#            gen_num+=1
        
        for front in self.pareto_scores:
            for s in front:
                x.append(s[0])
                y.append(s[1])
                gen.append(gen_num)
            
            gen_num+=1
                
        f,(ax1,ax2)=plt.subplots(1,2, figsize=(12,6),sharey=True)
        im2=ax2.scatter(x,y,c=gen,cmap=cm.rainbow)
        div2=make_axes_locatable(ax2)
        cax2=div2.append_axes('right',size='7%',pad=.05)
        cbar=plt.colorbar(im2,cax=cax2) 
        cbar.set_label('Generation', size=13)
        ax2.set_xlabel('Survivability', size=13)
        #ax2.set_ylabel('Complexity',size=13)
        ax2.set_title("Pareto front by generation",size=15)   
        
        #Get full history
        x=[]
        y=[]
        
        #print self.ants
#        for ant in self.all_ants:
#            s=ant.score
#            x.append(s[0])
#            y.append(s[1])
        
        for s in self.all_scores:
            x.append(s[0])
            y.append(s[1])
                
        
#        x_p=[]
#        y_p=[]
#        for p in pareto_history[-1]:
#                x_p.append(p[0])
#                y_p.append(p[1])
        x_p=[]
        y_p=[]
        colony=[]
        for ant_id in self.pareto_history[-1]:
            s=self.ant_by_id(ant_id).score
            x_p.append(s[0])
            y_p.append(s[1])
            colony.append(ant_id[1])
            
        ax1.scatter(x,y,color='grey',label='Dominated points')    
        im1=ax1.scatter(x_p,y_p,c=colony,cmap=cm.rainbow)
        div1=make_axes_locatable(ax1)
        cax1=div1.append_axes('right',size='7%',pad=.05)
        cbar=plt.colorbar(im1,cax=cax1) 
        cbar.set_label('Non-Dominated Points Colony of Origin', size=13)
        ax1.set_xlabel('Survivability', size=13)
        #ax2.set_ylabel('Complexity',size=13)
        ax1.set_title("Final Pareto front by colony",size=15)
        ax1.legend(shadow=True,frameon=True)
            
            
        
#        ax1.scatter(x,y,color='blue',label='Dominated points')
#        ax1.scatter(x_p,y_p,color='red',label='Pareto points')
#        
#        ax1.set_xlabel('Survivability', size=13)
#        ax1.set_ylabel('Capacity',size=13)
#        ax1.set_ylim([0,max(y_p)*1.2])#min(y_p)*.1
#        ax1.set_title("Final Pareto front by colony",size=15)
#        ax1.legend(shadow=True,frameon=True)
        
        plt.tight_layout(pad=1.5)
        
    def vizualize_c_f(self):
        """
        Plot cycles and failures from generations.
        """
        t=list(xrange(len(self.cycle_list)))
        
        f,(ax1,ax2)=plt.subplots(1,2, figsize=(12,6),sharex=True)
        ax1.set_title('Cycling and Failure Behavior',size=17)
        ax1.plot(t,self.cycle_list,'-b',label='Cycles')
        ax1.plot(t,self.fail_list,'-r',label='Failures')
        ax1.set_xlabel('Generation',size=15)
        ax1.legend()
        ax1.set_ylim(0,max([max(self.cycle_list),max(self.fail_list)])+5)
        ax2.set_title('Step Behavior', size=17)
        ax2.plot(t,self.t_step_list,'-g',label='Total Steps')
        ax2.plot(t,self.s_step_list,'-y',label='Start Steps')
        ax2.plot(t,self.e_step_list,'-c',label='End Steps')
        ax2.set_xlabel('Generation',size=15)
        
        ax2.legend()
        
if __name__=='__main__':
    space=Space(n_col=2,n_ant=2,keep_best=1,tests=10)
    
    count=0
    while (space.convergence<10) and (count<10): #
        print count,
        count+=1
        space.step()
    print ''
    space.vizualize_c_f()
    space.vizualize_pf()
#    for a in space.all_ants:
#        space.vizualize_sos(a.ant_id)
    #print space.p
    space.vizualize_systems()
        
    
        