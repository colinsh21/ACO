
# coding: utf-8

# In[1]:

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
from collections import Iterable
from bisect import bisect
from networkx.algorithms.flow import edmonds_karp
from networkx.algorithms.traversal.depth_first_search import dfs_tree
from random import choice

# Visualization imports
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
get_ipython().magic(u'matplotlib inline')

import seaborn; seaborn.set()


# In[4]:

class Ant(object):
    """
    Ant class, which encapsulates the entire behavior of an ant.
    """
    def __init__(self,model,ant_id,colony_id):
    
        # Set model link and ID
        self.model = model
        self.ant_id = (ant_id,colony_id)
        
    def random_walk(self,source,sink,s_mag,t_mag,alpha=1.0,beta=1.0,system=0):
        #Random Walk
        #initialize method
        self.alpha=alpha
        self.beta=beta
        self.source=source
        self.source_mag=s_mag
        self.sink_mag=t_mag
        self.sink=sink
        self.system=system
        self.g=nx.DiGraph()
        
        #initialize needed global data for heuristic
        self.capacity_list=self.get_capacity_list(self.system)
        
        #initialize start of random walk
        self.last_nodes={}
        for s in self.source:
            self.last_nodes[s]=[s]
        
        for key in self.last_nodes:
            self.g.add_node(key)
        
        #initialize walk trackers
        terminate=0
        steps=0
        consecutive_no_additions=0
        cycling=0
        sinks_included=0
        exit_code=0
        
        #start walk
        while not terminate:
            new=0 #did this step add new nodes?
                                  
            for key in self.last_nodes:
                '''
                Ant step is encompassed below
                '''
                steps+=1

                #track next nodes
                self.last_nodes_next={}
                  
                
                #Get number of branches     
                self.branch_pheromone_list=self.get_pheromone_branch(key,self.system)
#                if self.g.out_degree(key): #determine if this node has already been visited
#                    existing=1 
#                    if cycling==0: #if not cycling try to follow previous path
#                        self.branches=self.g.out_degree(key)
#                    else:
#                        self.branches=1 #if cycling try and create new branch
#                else:
#                    existing=0
#                    self.branches=self.make_decision(self.branch_pheromone_list)+1 #index of decision +1 is the number of branches chosen
                self.branches=self.make_decision(self.branch_pheromone_list)+1                
                self.neighbors_chosen=[] #list other branches taken out of that node
                
                if sinks_included==1:  #if all sinks included can terminate ants
                    terminate_ant=self.make_decision(self.get_pheromone_termination(key,self.system))
                    if terminate_ant==1: #0 is do not terminate
                        self.branches=0 #this ant does not branch
                
                #continue walk for each chosen branch       
                for branch in xrange(self.branches):
                    #get neighbors
                    self.all_p_neighbors=self.get_neighbors(key)
                   
                    self.p_neighbors=list(set(self.all_p_neighbors)-set(self.neighbors_chosen))
                    #get pheromone of each edge to neighbor
                    self.edge_pheromone_list=[]
                    edge_heuristic=1.0
                    for self.potential_node in self.p_neighbors:
                        edge_pheromone=self.get_pheromone_edge((key,self.potential_node),self.system)
                        
                        # Encourage movement to unfilled sink
                        # Add 12/11
#                        dist_to_unfilled=nx.diameter(self.g)
#                        for un_sink in (self.sink not in self.g.nodes()):
#                            d=nx.shortest_path_length(self.g,self.potential_node,un_sink)
#                            if d<dist_to_unfilled:
#                                dist_to_unfilled=d
                        
                        if sinks_included==0 and cycling==1: #
                            p=nx.shortest_path_length(self.model.g,source=self.potential_node)
                            p_sink=[]
                            un_sink_list=list(set(self.sink)-set(self.g.nodes()))
                            #print un_sink_list
                            if un_sink_list:
                                for un_sink in un_sink_list:
                                    p_sink.append(p[un_sink])
                                #print p_sink
                                # get dist to closest sink in terms of fraction of diameter
                                d=nx.diameter(self.model.g)
                                min_frac_dist=float(min(p_sink))/d
                                if min_frac_dist==0.0: #on sink
                                    min_frac_dist=0.1/d #1/th the min fractional distance (1/d) 
                                edge_heuristic*=1.0/(min_frac_dist) #h is 
                        
                        
                        if sinks_included==1: #and 
                            if self.potential_node in self.g.neighbors(key): #follow established path
                                edge_heuristic*=1.0 #3
                        
                        
#                        if self.potential_node in self.last_nodes[key] and cycling==1: #do not go backwards if possible
#                            edge_heuristic*=0.001 #.001
                            
                        edge_pheromone=edge_pheromone**self.alpha*edge_heuristic**self.beta
                        self.edge_pheromone_list.append(edge_pheromone)

                    #get next node
                    #print 'edge_pheromone',key,self.edge_pheromone_list
                    self.next_node=self.p_neighbors[self.make_decision(self.edge_pheromone_list)]
                    self.neighbors_chosen.append(self.next_node) #record branch taken
                    

                    #is this node new? **could check edges too**
                    if self.next_node not in self.g.nodes():
                        new=1 
                    
                    #add node and edge
                    self.g.add_edge(key,self.next_node)

                    #record node and predecessor for next step
                    if self.next_node in self.last_nodes_next:
                        self.last_nodes_next[self.next_node].append(key)
                    else:
                        self.last_nodes_next[self.next_node]=[key]

            
            #Increment cycling tracker
            if new==0:
                consecutive_no_additions+=1
            else:
                consecutive_no_additions=0
            
            sinks_included=1 #check if all sinks in the walk
            for sink_node in self.sink:
                if sink_node not in self.g.nodes():
                    sinks_included=0
            
            #If cycling, either end or force ants to explore
            if consecutive_no_additions>10:
                cycling=1
                if sinks_included==0:
                    for node in self.last_nodes_next:#if cycling encourage ants to explore
                        #record all predecessors as successors
                        self.last_nodes_next[node].append(self.g.predecessors(node))                
                        
                        #record all successors as predecessors
                        self.last_nodes_next[node].append(self.g.neighbors(node))
                else: #if sinks included and cycling, end
                    self.last_nodes_next={} #cut list triggering termination
            
            if consecutive_no_additions>100:
                #print 'cycle termination'
                # for each sink not included, add path from existing node
                # add
            
            
                exit_code=1
                self.last_nodes_next={} #cut list triggering termination
            
            #Check Termination Criteria
            if not self.last_nodes_next:
                terminate=1

            self.last_nodes=dict(self.last_nodes_next)
        
        #Prune walk
        out_graph=self.model.prune_graph(self.g,self.sink,self.source)
        #out_graph=self.g
        
        # Get capacities for graph
        #get capacity pheromones
        
        # make dictionary of max supporting and recieved flow
        st_dict={}
        for n in out_graph:
            st_dict[n]={}
            st_dict[n]['t']=0 #mag of sinks node supports
            st_dict[n]['s']=0 #mag of sources node recieves from 
            
        # determine which sinks depend on node v
        for t in self.sink:
            if t in out_graph:
                mag=self.sink_mag[self.sink.index(t)]
                for n in dfs_tree(out_graph.reverse(),t).nodes(): #get nodes that feed sink
                    st_dict[n]['t']+=mag
            
        # determine which sources feed node v
        for s in self.source:
            if s in out_graph:
                mag=self.source_mag[self.source.index(s)]
                for n in dfs_tree(out_graph,s).nodes(): #get nodes that feed sink
                    st_dict[n]['s']+=mag        
        
        for (u,v) in out_graph.edges():
            capacity_pheromone_list=self.get_pheromone_capacity((u,v),self.system)
            #print 'cap_pheromone',(key,self.next_node),capacity_pheromone_list                         
            
            heuristic_list=[]
            for cap in self.capacity_list:
                heuristic_value=1.0

                # souce support heuristic                
                if cap<=st_dict[u]['s']: #installing <= total recieved
                    heuristic_value+=0.5
                
                # sink support
                if cap>=st_dict[u]['t']: #installing >= total required
                    heuristic_value+=0.5/(1.0+abs(st_dict[u]['t']-cap))
                else:
                    heuristic_value+=0.0 #1.0/(1.0+abs(cap_supported-cap))
                
                #heuristic_value=1.0
                heuristic_list.append(heuristic_value**self.beta)
            
            
            #print 'heuristic',heuristic_list
            #apply heuristic to capacity pheromones
            self.adjusted_pheromone_list=[]
            for j in xrange(len(capacity_pheromone_list)):
                self.adjusted_pheromone_list.append((capacity_pheromone_list[j]**self.alpha)*heuristic_list[j])
            
            #decide capacity
            #print self.model.capacities[self.system]
            #print self.adjusted_pheromone_list
            dec=self.make_decision(self.adjusted_pheromone_list)                    
            #print dec
            #self.capacity_to_add=self.model.capacities[self.system][self.make_decision(self.adjusted_pheromone_list)]
            self.capacity_to_add=self.model.capacities[self.system][dec]

            #add capacity
            out_graph[u][v]['capacity']=self.capacity_to_add

        #out_graph=self.model.prune_graph(self.g,self.sink,self.source)        
        
        return  out_graph, exit_code
    
    
    def get_neighbors(self,node):
        """
        Return neighbors, calling through model.
        """
        return self.model.get_ant_neighbors(node)
    
    def make_decision(self,pheromone_list):
        """
        Return decision index, based on pheromone list.
        """
        #convert pheromones to percentage
        self.percent_list = [float(i)/sum(pheromone_list) for i in pheromone_list]   
        self.cumulative_percent=numpy.cumsum(self.percent_list)

        #Choose decision index
        self.select_index=bisect(self.cumulative_percent,numpy.random.uniform(0,1,1))
  
        return self.select_index
    
    def get_capacity_list(self,system):
        """
        Returns the capacity list from the space
        """
        return self.model.capacities[system]
    
    def get_pheromone_branch(self,node,system):
        """
        Return node pheromone, calling through model.
        """    
        return self.model.get_branch_pheromone(node,self.ant_id,system)
    
    def get_pheromone_edge(self,edge,system):
        """
        Return edge pheromone, calling through model.
        """
        return self.model.get_edge_pheromone(edge,self.ant_id,system)

    def get_pheromone_capacity(self,edge,system):
        """
        Return edge pheromone, calling through model.
        """
        return self.model.get_capacity_pheromone(edge,self.ant_id,system)
    
    def get_pheromone_termination(self,node,system):
        """
        Return node pheromone, calling through model.
        """    
        return self.model.get_termination_pheromone(node,self.ant_id,system)        
    

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
    def __init__(self,graph=nx.grid_graph(dim=[3,3]) ,num_ants=2,num_colonies=1,
                 source=[[(0,0)],[(1,1)]],source_magnitude=[[5],[5]],
                 sink=[[(1,1)],[(2,0)]],sink_magnitude=[[5],[5]], sink_threshold=[[1.0],[1.0]],
                 links=[(1,0,(1,1))],capacities=[[5,10],[5]],edge_capacity=20, 
                 fraction_removals=4.0/9,removal_trials=10,local_search_params=(2,1),
                 dissipation=.4,initial_pheromone=1.0,initial_termination=1.0,
                 alpha=1.0,beta=1.0,random=0):
                     
                     #size=[3,3]
        """
        Class constructor.
        """
        # Set our model parameters
        #self.size = size
        self.g=graph
        self.g_base=graph #clean graph with no attributes
        self.sink = sink
        self.sink_magnitude = sink_magnitude
        self.sink_threshold= sink_threshold #at what point is the sink not functioning
        self.source=source
        self.source_magnitude = source_magnitude
        self.links=links # format: (recieving_system,sending_system,node)
        #(number of mutations, edges per mutation)        
        self.local_search_params=local_search_params 
        self.random=random #have random ants
        
        #Ants
        self.num_ants=num_ants
        self.num_colonies=num_colonies
        self.dissipation=dissipation
        self.initial_pheromone=initial_pheromone
        self.capacities=capacities
        self.alpha=alpha
        self.beta=beta
        self.fraction_removals=fraction_removals
        self.removal_trials=removal_trials
        self.initial_termination=initial_termination
        self.edge_capacity=edge_capacity
          
        # Set our state variables
        self.t = 0
        self.ants = []
        self.ant_graphs=[]
        self.keep_ants=[]
        
        #entire history
        self.all_ants=[]

        # Call our setup methods to initialize space, people, and institution.
        self.setup_space()
        self.setup_ants()
        

    def setup_space(self):
        """
        Method to setup our space.
        """
        # Initialize a space with a grid network
        #self.g = nx.grid_graph(dim=self.size)
        self.g=self.g.to_directed()
        
        
        # Set Pheromones
        print 'Setting up network'
        capacity_pheromone_list=[self.initial_pheromone]*len(self.capacities[0])*2
        capacity_pheromone_list.extend([self.initial_pheromone]*len(self.capacities[1])*2)
        for e in self.g.edges_iter():
            self.g.add_edge(e[0],e[1],max_capacity=self.edge_capacity)
            self.g.add_edge(e[0],e[1],capacity=0) #initial capacity 
            self.g.add_edge(e[0],e[1],edge_pheromone=[self.initial_pheromone]*2*2) #pheromone per edge
            self.g.add_edge(e[0],e[1],capacity_pheromone=capacity_pheromone_list) #pheromone per capacity
            
        for n in self.g.nodes_iter():
            neighbors_n=self.g.neighbors(n)
            
            branch_factor=1 #was .5
            branch_pheromone_list=[]
            branch_pheromone_list=[self.initial_pheromone]
            branch_pheromone_list.extend([self.initial_pheromone*branch_factor]*(len(neighbors_n)-1))
            self.g.add_node(n,branch_pheromone=branch_pheromone_list*2*2)
            
            #termination=[live,die]
            live_factor=1.0 #was .25
            termination_pheromone_list=[self.initial_termination*live_factor,self.initial_termination]*2*2
            self.g.add_node(n,termination_pheromone=termination_pheromone_list)

        # Set layout    
        self.g_layout = nx.spectral_layout(self.g)
 
        
    def setup_ants(self):
        """
        Method to setup our space.
        """       
        # First, begin by creating all ants.
        int_id=0 #set up list ID of ants
        self.ant_id_dict={} #set up dict converting ant id to list id
        for c in xrange(self.num_colonies+1): #add last for random colony
            for i in xrange(self.num_ants):
                self.ants.append(Ant(model=self,ant_id=(i+1),colony_id=(c+1)))
                self.ant_id_dict[(i,c)]=int_id                
                int_id+=1
    
    def split_lists(self,input_list):
        """
        List manipulation for pheromone lists
        """       
        half = len(input_list)/2
        return input_list[:half], input_list[half:]
    
    def flatten(self,lis):
        """
        Reconstruct pheromone lists after split
        """       
        for item in lis:
            if isinstance(item, Iterable) and not isinstance(item, basestring):
                for x in self.flatten(item):
                    yield x
            else:        
                yield item
  
    def get_branch_pheromone(self,node,ant_id,system):     
        """
        Get branch decision pheromone for ant call
        """          
        ant_weight=float(ant_id[0]-1)/(self.num_ants-1)
        pheromone_full=list(self.g.node[node]['branch_pheromone'])
        system_pheromones=[[],[]]
        system_pheromones[0],system_pheromones[1]=self.split_lists(pheromone_full)
        pheromone_1, pheromone_2=self.split_lists(system_pheromones[system])
    
        weighted_pheromone_1=[i**ant_weight for i in pheromone_1]
        weighted_pheromone_2=[i**(1-ant_weight) for i in pheromone_2]
        composite_pheromone=[weighted_pheromone_1[i]*weighted_pheromone_2[i] for i in xrange(len(weighted_pheromone_1))]

        return composite_pheromone
            
    def get_edge_pheromone(self,edge,ant_id,system):
        """
        Get edge decision pheromone for ant call
        """        
        ant_weight=float(ant_id[0]-1)/(self.num_ants-1)
        if system==1:
            pheromone_1=float(self.g[edge[0]][edge[1]]['edge_pheromone'][0])
            pheromone_2=float(self.g[edge[0]][edge[1]]['edge_pheromone'][1])
        else:
            pheromone_1=float(self.g[edge[0]][edge[1]]['edge_pheromone'][2])
            pheromone_2=float(self.g[edge[0]][edge[1]]['edge_pheromone'][3])
                
        return (pheromone_1**ant_weight)*(pheromone_2**(1-ant_weight))
    
    def get_capacity_pheromone(self,edge,ant_id,system):
        """
        Get capacity decision pheromone for ant call
        """  
        ant_weight=float(ant_id[0]-1)/(self.num_ants-1)
        pheromone_full=list(self.g[edge[0]][edge[1]]['capacity_pheromone'])
        system_pheromones=[[],[]]
        system_pheromones[0]=pheromone_full[:len(self.capacities[0])*2]
        system_pheromones[1]=pheromone_full[len(self.capacities[0])*2:]
        pheromone_1, pheromone_2=self.split_lists(system_pheromones[system])
        
        weighted_pheromone_1=[i**ant_weight for i in pheromone_1]
        weighted_pheromone_2=[i**(1.0-ant_weight) for i in pheromone_2]
        composite_pheromone=[weighted_pheromone_1[i]*weighted_pheromone_2[i] for i in xrange(len(weighted_pheromone_1))]
        
        return composite_pheromone
    
    def get_termination_pheromone(self,node,ant_id,system):
        """
        Get termination decision pheromone for ant call
        """        
        ant_weight=float(ant_id[0]-1)/(self.num_ants-1)
        pheromone_full=list(self.g.node[node]['termination_pheromone'])
        system_pheromones=[[],[]]
        system_pheromones[0],system_pheromones[1]=self.split_lists(pheromone_full)
        pheromone_1, pheromone_2=self.split_lists(system_pheromones[system])
    
        weighted_pheromone_1=[i**ant_weight for i in pheromone_1]
        weighted_pheromone_2=[i**(1-ant_weight) for i in pheromone_2]
        composite_pheromone=[weighted_pheromone_1[i]*weighted_pheromone_2[i] for i in xrange(len(weighted_pheromone_1))]
        
        return composite_pheromone 
        
    def get_ant_neighbors(self,node):
        """
        Get node neighbors for ant call
        """
        return self.g.neighbors(node)
            
    def get_path(self, ant_id,system):
        """
        Get the path of an ant based on their ID.
        """
        ant_list_id=self.ant_id_dict[ant_id]
        
        if ant_id[1]==self.num_colonies and self.random==1:
            return self.ants[ant_list_id].random_walk(self.source[system],self.sink[system],
                                                    self.source_magnitude[system],
                                                    self.sink_magnitude[system],
                                                    0.0,self.beta,system) #don't follow pheromone
        else:
            return self.ants[ant_list_id].random_walk(self.source[system],self.sink[system],
                                                    self.source_magnitude[system],
                                                    self.sink_magnitude[system],
                                                    self.alpha,self.beta,system)
    
    def flow_test(self,graph):
        """
        Calculates the flow from sources to sinks in a graph.
        
        """
        flow_graph=copy.deepcopy(graph)
        
        # Create flow graphs for testing
        max_flow=[] #what is the maximum flow to sinks
        for sys in range(0,2):
            #add sinks       
            for i in xrange(len(self.sink[sys])):
                mag=self.sink_magnitude[sys][i]
                flow_graph[sys].add_node(self.sink[sys][i],demand=mag)
            max_flow.append(sum(self.sink_magnitude[sys]))
                
            #add sources
            #flow_graph[sys].add_node('place_holder') #fixes indexing error in maximum_flow
            for i in xrange(len(self.source[sys])):
                mag=self.source_magnitude[sys][i]
                flow_graph[sys].add_node(self.source[sys][i],demand=-mag)
                #sources have negative demand
                
        flow=[] #list of flow tuples by state
        
        # Test for cascading failures
        while True:
            #get flow values
            outflow=[] #list of dict of outflows to nodes
            inflow=[] #list of dicts of inflows to nodes
            flow_frac=[] #records the fraction of flow to sinks
            for sys in range(0,2):                    
                cost,flow_dict=capacity_scaling(flow_graph[sys])
                outflow.append(flow_dict) #store outflow dictionary
                
                #create inflow dictionary                
                inflow_dict={}
                for n in flow_graph[sys].nodes():
                    inflow_dict[n]=0
                for pred,out in flow_dict.iteritems():
                    for dest,in_f in out.iteritems():
                        inflow_dict[dest]+=in_f
                inflow.append(inflow_dict) #store inflow dictionary
                
                #get fractional flow to sinks  
                current_flow=0
                for t in self.sink[sys]:
                    current_flow+=inflow_dict[t]
                sys_frac=float(current_flow/max_flow[sys])
                flow_frac.append(sys_frac)
            flow.append(flow_frac)
                    
            failure=0
            for link in self.links: #if sink in linked nodes fails, corresponding source fails
                receiving_sys=link[0] #system recieving flow
                sending_sys=link[1] #system sending flow
                node=link[2]
                sending_node_index=self.sink[sending_sys].index(node)
                sending_flow=inflow[sending_sys][node]
                sink_mag=self.sink_magnitude[sending_sys][sending_node_index]
                sink_threshold=self.sink_threshold[sending_sys][sending_node_index]
                sending_threshold=float(sink_mag*sink_threshold)
                
                #failure=0
                if sending_flow < sending_threshold: #sink fails and linked source fails
                    failure=1
                    
                    #emulate failure by removing negative demand from souce
                    #print node,receiving_sys,flow_graph[receiving_sys].node[node]
                    flow_graph[receiving_sys].node[node]['demand']=0 
            
            if flow[-1]==[0.0,0.0]: #system off
                break
            if failure==0 or (len(flow)>=2 and (flow[-1]==flow[-2])): #continue until steady-state
                break
        #print flow[-1],inflow
        #if len(flow)>=2:
            #print 'failure'
            #print flow
        return flow[-1], inflow #last flow fraction is the steady state to sinks
    
    def graph_energy(self,graph):
        """
        Calculates the energy of a system graph. (not currently used)
        """
        e_val, e_vec=numpy.linalg.eig(nx.adjacency_matrix(graph.to_undirected()).todense())
    
        return sum(abs(e_val))
    
    def cap_score(self,graph):
        """
        Calculates the capacity of two system graph and returns overfull edges.
        """
        total_cap=0.0
        cap_dict={}
        for u,v in self.g.edges():
            cap_dict[(u,v)]=0.0
            
        for sys in xrange(2):
            for u,v in graph[sys].edges():
                e_cap=graph[sys][u][v]['capacity']
                cap_dict[(u,v)]+=e_cap
                total_cap+=e_cap
        
        infeasible_e=0
        for key,value in cap_dict.iteritems():
            if value>self.edge_capacity:
                infeasible_e+=1
        
        return total_cap, infeasible_e
        
        
    def complexity_score(self,graph):
        """
        Calculates the complexity of two system graph.
        """
        sys_complexity_edge=0
        for u,v in self.g.edges():
            temp_graph=copy.deepcopy(graph)
            
            #get max_cap from space 
            edge_max=self.g[u][v]['max_capacity']
   
            edge_cap=0
            num_sys_edge=0
            for g_index in xrange(2):
                #get capacity in edge
                sys_cap=0
                if temp_graph[g_index].has_edge(u,v):
                    sys_cap+=graph[g_index][u][v]['capacity']
                    temp_graph[g_index].remove_edge(u,v)
                    num_sys_edge+=1
                edge_cap+=sys_cap
                
            cap_difference=float(edge_cap)/edge_max
            if num_sys_edge>0:
                #complexity follows functional complexity
                #edge_complexity=edge_cap*2**(cap_difference*num_sys_edge)
                edge_complexity=num_sys_edge*2**(cap_difference*num_sys_edge)
            else:
                edge_complexity=0
                
            sys_complexity_edge+=edge_complexity
            
        sys_complexity_node=0
        for n in self.g:
            num_sys_node=0
            edge_in=0
            edge_out=0
            for g_index in xrange(2):
                #track how many edges the node supports
                if n in graph[g_index]:
                    num_sys_node+=1
                    edge_in+=graph[g_index].in_degree(n)
                    edge_out+=graph[g_index].out_degree(n)
            if num_sys_node>0:
                #complexity follows functional complexity
                #node_complexity=(edge_in+edge_out)*2**num_sys_node
                node_complexity=num_sys_node*2**(edge_in+edge_out)
            else:
                node_complexity=0
            sys_complexity_node+=node_complexity
            
        sys_complexity=sys_complexity_node+sys_complexity_edge
        
        return sys_complexity
        
    def flow_check(self,ant_graph):
        """
        check what initial flow is
        """
      
        flow_score_ini,flow_dict_ini=self.flow_test(ant_graph)
 
        #check if system provide initial stable flow
        for g_index in xrange(2):
            all_sink=1
            for i in xrange(len(self.sink[g_index])):
                sink=self.sink[g_index][i]
                mag=self.sink_magnitude[g_index][i]
                sink_flow=flow_dict_ini[g_index][sink]
                if sink_flow<mag: #sink_flow==0
                    all_sink=0
            flow_score_ini[g_index]*=all_sink

#        if (ant_graph[0].number_of_edges()==0) or (ant_graph[0].number_of_edges()==0):
#            print 'found empty', flow_score_ini, flow_dict_ini            
#            flow_score_ini=[0.0,0.0]
        
        return flow_score_ini
    
    def evaluate_graph(self, ant_graph):
        """
        Score graphs based on criteria.
        """
        #print 'pre copy', self.g[(0,0)][(0,1)]        
        #Score 1 - Flow        
        #get initial flow values
        flow_score_ini=self.flow_check(ant_graph)
        #print flow_score_ini
    

        #Percolate and test flow
        temp=copy.deepcopy(ant_graph)
        space_graph = copy.deepcopy(self.g_base)
        space_graph=space_graph.to_directed()
        base_percolation_graph=[]
        #create full space graph for percolation
        for g in xrange(2):
            base_percolation_graph.append(nx.compose(space_graph,temp[g]))
        #print base_percolation_graph[0].number_of_nodes()
        #print self.fraction_removals
        #get number of removal    
        num_removals=int(numpy.rint(base_percolation_graph[0].number_of_nodes()*self.fraction_removals))
        #print num_removals
        ##print base_percolation_graph[0].number_of_nodes()        
        #print num_removals
        #survivability_function=0
        num_trials=self.removal_trials #testing range
        trial_results=[]
        for i in xrange(num_trials):
            percolation_graph=copy.deepcopy(base_percolation_graph)
     
            for j in xrange(num_removals):
                #print n_list
#                n_list.remove(node_removed)
                n_list=percolation_graph[0].nodes() #get remaining nodes
                node_removed=choice(n_list)
                for g_index in xrange(2): #remove chosen node from both               
                    #print percolation_graph[g_index].edges()
                    
#                    for e in percolation_graph[g_index].edges(node_removed):
#                        percolation_graph[g_index].remove_edge(e[0],e[1])
#                        percolation_graph[g_index].remove_edge(e[1],e[0])
                    percolation_graph[g_index].remove_node(node_removed)
            #print 'post copy',self.g[(0,0)][(0,1)]        
            #Check flow value        
            #Check if there is a cascade and iterate.    
            flow_increment,flow_dict=self.flow_test(percolation_graph)
            function_increment=0
            for g_index in xrange(2):
                for sink in self.sink[g_index]: #count working sinks
                    sink_flow=flow_dict[g_index][sink]
                    #print sink_flow, flow_dict[g_index][self.sink[g_index][sink_index]]
                    sink_index=self.sink[g_index].index(sink)
                    sink_mag=self.sink_magnitude[g_index][sink_index]
                    sink_threshold=self.sink_threshold[g_index][sink_index]
                    frac_threshold=float(sink_mag*sink_threshold)                         
                    function_increment+=(sink_flow>=frac_threshold)
                
            trial_results.append(function_increment)
            
            #add statistical test here
            #std = numpy.std(trials,ddof=1)
            #smallest signifance=u_star=.01
            #z_alpha=1.96, for alpha=.05
            #norm_inv(1-beta)=.8416, for beta=.2
            #min_n=((z_alpha+norm_inv(1-beta)/(u_star/std))**2
            
            
            #OR
            #split trial lists, get average of first half
            #use scipy.stats.ttest_1sample, 
            #if p-value>.05 accept (95% confidence level)
        
        #system survivability is average fraction of functioning sinks after each removal trial
        num_sinks=len(self.sink_magnitude[0])+len(self.sink_magnitude[1])
        #print trial_results        
        if num_removals==0:
            survivability_score_function=1
        else:
            survivability_score_function=numpy.average(trial_results)/num_sinks

        #penalty on survivability function for less than full initial flow    
        #survivability_score_function*=(flow_score_ini[0]*flow_score_ini[1])**2
        
        #Score 2 - Complexity
#        complexity_score=self.complexity_score(ant_graph)
#        #penalty on complexity function for less than full initial flow
#        if flow_score_ini[0]*flow_score_ini[1]==0:
#            complexity_score*=1000
#        else:
#            complexity_score*=1/(flow_score_ini[0]*flow_score_ini[1])**3
        
        #Score 3 - Capacity
        cap_score, infeasible_e_count=self.cap_score(ant_graph)
        p_e=float(infeasible_e_count)/self.g.number_of_edges()
        p_s=flow_score_ini[0]*flow_score_ini[1]
        penalty=1.0 + p_e + (1.0-p_s)
        p_s_score=survivability_score_function/penalty
        p_c_score=cap_score*penalty
        
        #print flow_score_ini
        return (p_s_score,p_c_score),flow_score_ini
    
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

    
    def ant_ranking(self, ant_graphs):
        """
        Find best Ants based on scores from evaluate_graph.
        """
        score_list=[]
        #num_ants_generation=self.num_ants*self.num_colonies
        [score_list.append(graph[1]) for graph in ant_graphs] #Global Pareto
        cull_scores=score_list[:]
        equality_sequence=[1,0] #[>=,<=]
        self.paretoPoints, self.dominatedPoints=self.simple_cull_front(cull_scores,self.dominates,equality_sequence)
        self.paretoPoints=list(self.paretoPoints)

        best_ants_index=[]
        
        #match the score_list indices to the pareto front values
        for i in xrange(len(score_list)):
            if score_list[i] in self.paretoPoints:
                best_ants_index.append(i)
                
        return best_ants_index, score_list
    
    def pheromone_dissipation(self,pheromone_list,dissipation):
        """
        Updates a pheromone list with the proper increment based on system number.
        """
        return_list=[]
        for i in xrange(len(pheromone_list)):
            return_list.append(pheromone_list[i]*(1.0-dissipation))
        return return_list
        
    def cap_dissipation(self,pheromone_list,dissipation,system):
        """
        Updates a pheromone list with the proper increment based on system number.
        """
        # ph structure = [caps_sys_1*2, caps_sys_2*2]    
        # cap_sys = [cap_obj_1,cap_obj_2]
        sys_list=[[],[]]
        sys_list[0]=pheromone_list[:len(self.capacities[0])*2]
        sys_list[1]=pheromone_list[len(self.capacities[0])*2:]
        #print 'sys',sys_list
        pheromone_1, pheromone_2=self.split_lists(sys_list[system])

        for i in xrange(len(sys_list[system])):
            sys_list[system][i]*=(1.0-dissipation)
            
        return_list=list(self.flatten(sys_list))
                    
        return return_list
        
        
        
    def pheromone_update(self,pheromone_list,increment,system,index):
        """
        Updates a pheromone list with the proper increment based on system number.
        """
        sys_list=[[],[]]
        sys_list[0],sys_list[1]=self.split_lists(pheromone_list)
        objective_sys_list=[[[],[]],[[],[]]]
        for sys in xrange(2):
            objective_sys_list[sys][0],objective_sys_list[sys][1]=self.split_lists(sys_list[sys])
        for obj in xrange(2):
            objective_sys_list[system][obj][index]+=increment[obj]
            #MMAS            
            if objective_sys_list[system][obj][index]>1:
                objective_sys_list[system][obj][index]=1
        return_list=list(self.flatten(objective_sys_list))
        return return_list
    
    def pheromone_update_capacity(self,pheromone_list,increment,system,index):
        """
        Updates a pheromone list with the proper increment based on system number.
        """
        #print 'inc',increment
        sys_list=[[],[]]
        sys_list[0]=pheromone_list[:len(self.capacities[0])*2]
        sys_list[1]=pheromone_list[len(self.capacities[0])*2:]
        #print 'sys',sys_list
        pheromone_1, pheromone_2=self.split_lists(sys_list[system])
        #print 'ph',pheromone_1, pheromone_2
        objective_sys_list=[[[],[]],[[],[]]]
        for sys in xrange(2):
            objective_sys_list[sys][0],objective_sys_list[sys][1]=self.split_lists(sys_list[sys])
        for obj in xrange(2):
            objective_sys_list[system][obj][index]+=increment[obj]
            #MMAS            
            if objective_sys_list[system][obj][index]>1:
                objective_sys_list[system][obj][index]=1
        return_list=list(self.flatten(objective_sys_list))
        #print return_list
        
        return return_list
        
    def ant_path(self,ant_id):
        """
        Returns a feasible ant path.
        """
        full_flow=0
        #ant_failures=0
        #iterate until initial satisfactory flow is created
        while full_flow==0:
            ant_graph=[[],[]]
            for g in xrange(2):
                ant_graph[g],exit_code=self.get_path(ant_id,g)
                self.cycle_count+=exit_code
                #Prune Graph
                #ant_graph[g]=self.prune_graph(ant_graph[g],self.sink[g],self.source[g])
            
            #sat_flow=percentage of flow getting to sinks in each system
            full_flow=1
            #check if initial flow is satisfied
#            if (not ant_graph[0].edges()) or (not ant_graph[1].edges()):
#                full_flow=0
#                self.ant_failures+=1
#                continue
            sat_flow=self.flow_check(ant_graph)
            if sat_flow[0]!=1.0 or sat_flow[1]!=1.0:
                self.ant_failures+=1
                full_flow=0
#                self.visualize_system_single(ant_graph,
#                                      self.source,self.source_magnitude,
#                                      self.sink,self.sink_magnitude,
#                                      self.g,sat_flow,7)
        
        #Prune Graph
                   
        #Evaluate Paths:
        #print 'good ant, evaluate'
        
        graph_score,sat_flow=self.evaluate_graph(ant_graph)
        #print 'good ant, ini',ant_graph[0].number_of_edges(),ant_graph[1].number_of_edges(),graph_score
        return (ant_graph, graph_score)
    
    def add_ant(self, ant_id):#, ant_history
        """
        Adds a feasible ant path to the history
        """
        #return ant_history.append(self.ant_path(ant_id))
        return self.ant_path(ant_id)

    def update_institutions(self, ant_graphs, opt_ants, score_list, capacities,updating,dissipate):
        """
        Update institutions based on optimal ant solutions
        """
        #Get pheromones
        edge_pheromone_dict=dict(nx.get_edge_attributes(self.g,'edge_pheromone'))
        capacity_pheromone_dict=dict(nx.get_edge_attributes(self.g,'capacity_pheromone'))
        branch_pheromone_dict=dict(nx.get_node_attributes(self.g,'branch_pheromone'))
        termination_pheromone_dict=dict(nx.get_node_attributes(self.g,'termination_pheromone'))
        
        #Reduce edge pheromones by input amount
        
        #print capacity_pheromone_dict
        if dissipate==1:
            for key in edge_pheromone_dict:
                edge_pheromone_dict[key]=self.pheromone_dissipation(edge_pheromone_dict[key], self.dissipation)
            for key in capacity_pheromone_dict:
                capacity_pheromone_dict[key]=self.pheromone_dissipation(capacity_pheromone_dict[key], self.dissipation)
            #print capacity_pheromone_dict
            #Reduce node pheromones by input amount
            for key in branch_pheromone_dict:
                branch_pheromone_dict[key]=self.pheromone_dissipation(branch_pheromone_dict[key], self.dissipation)
            for key in termination_pheromone_dict:
                termination_pheromone_dict[key]=self.pheromone_dissipation(termination_pheromone_dict[key], self.dissipation)
        
        #Create rankings of ant scores
        
        temp_list_rank=map(list,zip(*self.paretoPoints)) #two list, one for each score
        #print temp_list_rank
        slr=[] #score list for ranking
        best=[] #best score
        
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
        
        ratios=[]
        ratios.append([s/best[0] for s in temp_list_rank[0]]) #survivability ratios
        ratios.append([best[1]/c for c in temp_list_rank[1]]) #cost ratios
        
        norm_tot=[]        
        for l in ratios:
            norm_tot.append(sum(l)) #normalize to ratio scores to 1
        
        #print opt_ants
        #print self.paretoPoints
        #print temp_list_rank
        #print slr
        #Update the optimum ant paths
        #print 'pre update',self.g[(0,0)][(0,1)]
        #dissipated_cap=[[],[]] #list of edges that have had their capacity dissipated by system
        #inc=1.0/updating 
        inc=self.dissipation*updating  #invariant     
        for ant in opt_ants:
            #score_rank=[]
            #print score_list[ant]
            #print slr[0][temp_list_rank[0].index(score_list[ant][0])]
            
            #rank
#            score_rank.append(slr[0][temp_list_rank[0].index(score_list[ant][0])]+1)
#            score_rank.append(slr[1][temp_list_rank[1].index(score_list[ant][1])]+1)
            #pheromone_increment=[1.0/score_rank[0],1.0/score_rank[1]] 
            
            #normalized and invariant
            norm_s=(score_list[ant][0]/best[0])/norm_tot[0]
            norm_c=(best[1]/score_list[ant][1])/norm_tot[1]
            
#            print 's',norm_s, score_list[ant][0]
#            print 'c',norm_c, score_list[ant][1]
            
            pheromone_increment=[inc*norm_s,inc*norm_c]
            if best[0]==0.0:
                pheromone_increment[0]=1.0
            if score_list[ant][1]==0.0:
                pheromone_increment[1]=1.0
                
            #pheromone_increment=[0.0,0.0]
            #print score_rank
            #score_rank.append(slr[1][numpy.where(array==score_list[ant][1])][0])
                           
            #pheromone_increment=[score_list[ant][0],2.0/len(opt_ants)]
            #print 'ph-increment',pheromone_increment  
            
            for sys in xrange(2):
                #pheromone update for edges
                for e in ant_graphs[ant][0][sys].edges():
                    
                    edge_pheromone_dict[e]=self.pheromone_update(edge_pheromone_dict[e], pheromone_increment,sys,0)
                    
                    #dissipate capacity pheromone - subproblem
#                    if disspate==1:
#                        if e not in dissipated_cap[sys]:
#                            capacity_pheromone_dict[e]=self.cap_dissipation(capacity_pheromone_dict[e],self.dissipation,sys)
#                            dissipated_cap[sys].append(e)
                        
                    
                    #get capacity chosen and update pheromones
                    capacity_chosen=ant_graphs[ant][0][sys][e[0]][e[1]]['capacity']
                    capacity_index=capacities[sys].index(capacity_chosen)
                    #print e,sys,capacity_pheromone_dict[e],capacity_chosen,capacity_index
                    capacity_pheromone_dict[e]=self.pheromone_update_capacity(capacity_pheromone_dict[e],
                                                                                   pheromone_increment,sys,
                                                                                   capacity_index)
                    #print e,sys,capacity_pheromone_dict[e]
                n_list=[]
                #update node pheromones
                #print len(ant_graphs)
                for n in ant_graphs[ant][0][sys].nodes():
                    n_list.append(n)
                    #branches
                    #print ant,sys,n
                    branches_chosen=len(ant_graphs[ant][0][sys].neighbors(n))
                    branches_index=branches_chosen-1
                    branch_pheromone_dict[n]=self.pheromone_update(branch_pheromone_dict[n],
                                                                        pheromone_increment,sys,
                                                                        branches_index)
                    #termination for sinks
                    if n in self.sink[sys]: 
                        if ant_graphs[ant][0][sys].out_degree(n)==0: #if terminated
                            termination_pheromone_dict[n]=self.pheromone_update(termination_pheromone_dict[n],
                                                                                     pheromone_increment,sys,1)
                        else:
                            termination_pheromone_dict[n]=self.pheromone_update(termination_pheromone_dict[n],
                                                                                pheromone_increment,sys,0)
                #nodes not in graph are considered terminated
                for n in self.g.nodes():
                    if n not in n_list:
                        termination_pheromone_dict[n]=self.pheromone_update(termination_pheromone_dict[n],
                                                                            pheromone_increment,sys,1)
        
                
        #Set new edge pheromones
        nx.set_edge_attributes(self.g, 'edge_pheromone', edge_pheromone_dict)

        #Set new capacity pheromones
        #print 'new cap dict', capacity_pheromone_dict[((0,0),(0,1))]
        nx.set_edge_attributes(self.g, 'capacity_pheromone', capacity_pheromone_dict)
        #print nx.get_edge_attributes(self.g,'capacity_pheromone')

        #Set new branch pheromones
        nx.set_node_attributes(self.g, 'branch_pheromone', branch_pheromone_dict)

        #Set new termination pheromones
        nx.set_node_attributes(self.g, 'termination_pheromone', termination_pheromone_dict)     
    
    def visualize_system_single(self,system,s,s_mag,t,t_mag,area,score,fig_size=7):
        """
        Plots graphs of the system
        """
#        num_e=[system[0].number_of_edges(),system[1].number_of_edges()]
#        print num_e
        # Remove the nonpath from path nodes
        path_nodes=[[],[]]
        s_t_list=s[0]+s[1]+t[0]+t[1]
        for i in xrange(2):
            path_nodes[i] = [node for node in area.nodes() if node in system[i] and node not in s_t_list]

        #nonpath_nodes = [node for node in area.nodes() if node not in system[0] or node not in system[1]]

        # Removed shared nodes
        shared_nodes=[]
        for n in area.nodes():
            if n in path_nodes[0] and n in path_nodes[1]:
                shared_nodes.append(n)
                path_nodes[0].remove(n)
                path_nodes[1].remove(n)
                
        #retrieve sources
        source_nodes=[s[0],s[1]]
        source_cap=[dict(zip(s[0],s_mag[0])),dict(zip(s[1],s_mag[1]))]

        for i in xrange(0,2):
            for s_n,m in source_cap[i].iteritems():
                source_cap[i][s_n]=r'+${}$'.format(m)
          
        #retrieve sinks
        sink_nodes=[t[0],t[1]]
        #sink_cap=[dict(zip(t[0],[-1*i for i in t_mag[0]])),
                  #dict(zip(t[1],[-1*i for i in t_mag[1]]))]
        sink_cap=[dict(zip(t[0],t_mag[0])),
                  dict(zip(t[1],t_mag[1]))]
        for i in xrange(0,2):
            for t_n,m in sink_cap[i].iteritems():
                sink_cap[i][t_n]=r'-${}$'.format(m)
        #print t_mag
        #print source_nodes,sink_nodes
        
        ss_nodes=[]
        ss_cap={}
        for s_n in source_nodes[0]:
            if s_n in sink_nodes[1]:
                ss_cap[s_n]=(source_cap[0][s_n],sink_cap[1][s_n])
                #source_nodes[0].remove(s_n)
                #sink_nodes[1].remove(s_n)
                del source_cap[0][s_n]
                del sink_cap[1][s_n]
                ss_nodes.append(s_n)
                
        for s_n in source_nodes[1]:
            if s_n in sink_nodes[0]:
                ss_cap[s_n]=(sink_cap[0][s_n],source_cap[1][s_n])
                #source_nodes[1].remove(s_n)
                #sink_nodes[0].remove(s_n)
                del source_cap[1][s_n]
                del sink_cap[0][s_n]
                ss_nodes.append(s_n)

        for st_n,m in ss_cap.iteritems():
            ss_cap[st_n]=r'{},{}'.format(m[0],m[1])
        
        temp_dict=[{},{},{}]
        for i in range(0,2):
            temp_dict[i]=source_cap[i].copy()
            temp_dict[i].update(sink_cap[i])
            #st_dict[i].update(ss_cap[i])
        temp_dict[2].update(ss_cap)
        
        st_dict={}
        for d in temp_dict:
            st_dict.update(d)
            
        #print st_dict
        #print ss_nodes, ss_cap

        # Remove edges
        path_edges=[[],[]]
        cap=[{},{},{}]
        for i in xrange(2):
            path_edges[i]=system[i].edges()
            for e in system[i].edges():
                cap[i][e]=r'${}$'.format(system[i][e[0]][e[1]]['capacity'])
        #print cap

        # Remove shared edges
        shared_edges=[]
        for e in area.edges():
            if e in path_edges[0] and e in path_edges[1]:
                shared_edges.append(e)
                path_edges[0].remove(e)
                path_edges[1].remove(e)
                cap[2][e]=r'{},{}'.format(cap[0][e],cap[1][e])
                del cap[1][e]
                del cap[0][e]
                
            

        area_layout=nx.spectral_layout(area)
        
        #Set boundaries
        cut = 1.15
        xmax = cut * max(xx for xx, yy in area_layout.values())
        ymax = cut * max(yy for xx, yy in area_layout.values())
        
        
        # Now we can visualize the infected node's position
        f = plt.figure(num=None, figsize=(fig_size,fig_size), dpi=1000)
        plt.axis('off')
        plt.xlim(-xmax, xmax)
        plt.ylim(-ymax, ymax)
        

        #nx.draw_networkx_nodes(area, area_layout,
                               #nodelist=nonpath_nodes,
                               #node_color='#dddddd')

        #nx.draw_networkx_nodes(system[0], area_layout, 
                                #nodelist=path_nodes[0],
                                #alpha=.1,
                                #node_color='red')
                                
        nx.draw_networkx_nodes(system[0], area_layout, 
                                nodelist=source_nodes[0],
                                node_size=1500,
                                alpha=1,
                                node_color='red')
        
        nx.draw_networkx_nodes(system[0], area_layout, 
                                nodelist=sink_nodes[0],
                                node_size=1500,
                                alpha=1,
                                node_color='red')

        #nx.draw_networkx_nodes(system[1], area_layout, 
                                #nodelist=path_nodes[1],
                                #alpha=.1,
                                #node_color='blue')
                                
        nx.draw_networkx_nodes(system[1], area_layout, 
                                nodelist=source_nodes[1],
                                node_size=1500,
                                alpha=1,
                                node_color='cyan')
        
        nx.draw_networkx_nodes(system[1], area_layout, 
                                nodelist=sink_nodes[1],
                                node_size=1500,
                                alpha=1,
                                node_color='cyan')

        #nx.draw_networkx_nodes(system[0], area_layout, 
                               #nodelist=shared_nodes,
                               #alpha=.1,
                               #node_size=400,
                               #node_color='orange')
                               
        nx.draw_networkx_nodes(system[1], area_layout, 
                                nodelist=ss_nodes,
                                node_size=1750,
                                alpha=1,
                                node_color='orange')

#        nx.draw_networkx_edges(area, area_layout, 
#                               width=1.0, 
#                               alpha=.5,
#                               edge_color='#111111')

        nx.draw_networkx_edges(system[0], area_layout,
                              edgelist=path_edges[0],
                              width=1.5, 
                              alpha=.8,
                              edge_color='red')
                              
        nx.draw_networkx_edges(system[1], area_layout,
                              edgelist=path_edges[1],
                              width=1.5, 
                              alpha=.8,
                              edge_color='cyan')

        nx.draw_networkx_edges(system[0], area_layout,
                              edgelist=shared_edges,
                              width=1.5, 
                              alpha=0.8,
                              edge_color='orange')

        _ = nx.draw_networkx_labels(area, area_layout,
                                st_dict,
                                font_size=12)

                                
        _ = nx.draw_networkx_edge_labels(area,area_layout,edge_labels=cap[0],
                                         label_pos=.27,font_color='red',
                                         font_size=12)
                                         
        _ = nx.draw_networkx_edge_labels(area,area_layout,edge_labels=cap[1],
                                         label_pos=.27,font_color='cyan',
                                         font_size=12)
                                         
        _ = nx.draw_networkx_edge_labels(area,area_layout,edge_labels=cap[2],
                                         label_pos=.35,font_color='orange',
                                         font_size=12)
        #_ = nx.draw_networkx_labels(area, area_layout,
                                #dict(zip(area.nodes(), area.nodes())),
                                #font_size=10)
        #num_e=[system[0].number_of_edges(),system[1].number_of_edges()]
        #sat_flow=self.flow_check(system)
#        plt.title(r'Survivability$={:1.2f}$, Capacity$={:6.2f}$, sat_flow={}'.format(score[0],score[1],sat_flow),
#                  fontsize=15)
        plt.title(r'Survivability$={:1.2f}$, Capacity$={:6.2f}$'.format(score[0],score[1]),
                  fontsize=15)
        
        
    def visualize_pareto(self, gen_history, pareto_history, final_pareto):
        """
        Plots movement of pareto front
        """
        x=[]
        y=[]
        gen=[]
        #utopia=[]
        #seen=[]
        
        #Get pareto evolution
        for i in xrange(len(pareto_history)):
            for p in pareto_history[i]:
                #if p in seen:
                    #continue
                x.append(p[0])
                y.append(p[1])
                gen.append(i)
                #seen.append(p)
                
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
        
        for i in xrange(len(gen_history)):
            for p in gen_history[i]:
                x.append(p[0])
                y.append(p[1])
                
        
#        x_p=[]
#        y_p=[]
#        for p in pareto_history[-1]:
#                x_p.append(p[0])
#                y_p.append(p[1])
        x_p=[]
        y_p=[]
        for p in final_pareto:
                x_p.append(p[0])
                y_p.append(p[1])
        
        ax1.scatter(x,y,color='blue',label='Dominated points')
        ax1.scatter(x_p,y_p,color='red',label='Pareto points')
        
        ax1.set_xlabel('Survivability', size=13)
        ax1.set_ylabel('Capacity',size=13)
        ax1.set_ylim([0,max(y_p)*1.2])#min(y_p)*.1
        ax1.set_title("Final Pareto front",size=15)
        ax1.legend(shadow=True,frameon=True)
        
        plt.tight_layout(pad=1.5)
        
    def local_search(self, ants,add):
        #add says if returning full list or just new ants        
        if add==1:        
            new_ants=copy.deepcopy(ants)
        else:
            new_ants=[]
        #new_ants=list(ants)
        #For each ant solution, take a random edge and change its capacity         
        for ant in ants:
            new_ant=copy.deepcopy(ant[0])
            #print new_ant
            s=0
            #mutates edges and adds them as ants
            while s<self.local_search_params[0]:
                #print 'mutation', s
                #test_ant=[[],[]]
                s+=1
                e=0
                added=1
                while (e<self.local_search_params[1]) and added :#added:
                    #mutate edges
                    e+=1
                    for i in range(0,2):
                     #systems
                        e_list=new_ant[i].edges()
                        #print 'sys', i, e_list
                        if not e_list:
                            added=0
                            #print 'no edges left'
                            break
                        target=choice(e_list)
                        new_cap=choice(self.capacities[i])
                        old_cap=new_ant[i][target[0]][target[1]]['capacity']
                        if new_cap!=old_cap:
                            new_ant[i][target[0]][target[1]]['capacity']=new_cap
                        else:
                            new_ant[i].remove_edge(target[0],target[1])
                            new_ant[i]=self.prune_graph(new_ant[i],self.sink[i],self.source[i])
                    
                        sat_flow=self.flow_check(new_ant)
                        if sat_flow[0]==1.0 and sat_flow[1]==1.0:
                            graph_score,sat_flow=self.evaluate_graph(new_ant)
                            #print 'good ant, prune',test_ant[0].number_of_edges(),test_ant[1].number_of_edges(),graph_score
                            added_ant=copy.deepcopy(new_ant)                        
                            new_ants.append((added_ant,graph_score))
    #                        num_e=[new_ants[-1][0][0].number_of_edges(),new_ants[-1][0][1].number_of_edges()]
    #                        print 'good ant, prune check',num_e,graph_score,len(new_ants)
                        else:
                            added=0
                            break
                        
                        
                        
#        for ant in new_ants:
#            num_e=[ant[0][0].number_of_edges(),ant[0][1].number_of_edges()]
#            print 'local search ants', num_e, ant[1]
        
        return new_ants
                
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
                            
        
        
    def get_generation(self,n_colonies,m_ants,last_gen_opt):
        #Step for each ant
        ants=copy.deepcopy(last_gen_opt)
        
#        ants=[]
#        #revaluate best ants
#        reval_ants=copy.deepcopy(last_gen_opt)
#        for ant in reval_ants:
#            graph_score,sat_flow=self.evaluate_graph(ant[0])
#            composite_score=(ant[1],graph_score)
#            
#            ants.append((ant[0],tuple(numpy.mean(composite_score,axis=0))))
        if self.random==1:    
            for c in xrange(n_colonies+1): #add random colony
                #print 'colony:',c
                for i in xrange(m_ants):
                    ant_sol=self.add_ant((i,c))
                    ants.append(ant_sol)
        else:
            for c in xrange(n_colonies): #add random colony
                #print 'colony:',c
                for i in xrange(m_ants):
                    ant_sol=self.add_ant((i,c))
                    ants.append(ant_sol)
        
        new_ants=self.local_search(ants,1)
        
#        for ant in new_ants:
#            sat_flow=self.flow_check(ant[0])
#            #num_e=[ant[0][0].number_of_edges(),ant[0][1].number_of_edges()]
#            print 'gen ants', sat_flow, ant[1]
        
        return new_ants
    
    
    def step(self):
        """
        Model step function.
        """
        #Step for each ant
        self.ant_failures=0
        self.cycle_count=0
        
        #Keep best
#        self.ant_graphs=self.get_generation(self.num_colonies,self.num_ants,
#                                            self.keep_ants)
        
        #Don't keep best
        self.ant_graphs=self.get_generation(self.num_colonies,self.num_ants,
                                            [])
         #Trying elitist
        #global_search=[]                                    
        global_search=self.local_search(self.keep_ants,0)
##        for ant in self.ant_graphs:
##            num_e=[ant[0][0].number_of_edges(),ant[0][1].number_of_edges()]
##            print 'step ants', num_e, ant[1] 
#        
##        for c in xrange(self.num_colonies):
##            #print 'colony:',c
##            for i in xrange(self.num_ants):
##                self.add_ant((i,c),self.ant_graphs)
#        
        self.all_ants_local=list(self.ant_graphs)
        if global_search:
#            print len(self.all_ants_global),len(global_search)
            self.all_ants_local=list(self.all_ants_local+global_search)
#        print 'lenlocal',len(self.all_ants_local)    
        if self.keep_ants:
            self.all_ants_global=list(self.all_ants_local+self.keep_ants)
        else:
            self.all_ants_global=list(self.all_ants_local)
        self.all_ants=list(self.all_ants_global)
#                
#        
#        #Get best score
#        #history
        self.opt_ant_global,self.score_list_global=self.ant_ranking(self.all_ants_global) 
#        print 'g', self.opt_ant_global
#        
#        self.opt_ants,self.score_list=self.ant_ranking(self.ant_graphs)
#        # remove global best from local
#        l_local=len(self.all_ants_local) #local ends at n-1
#        print 'lenlocal',l_local
#        for ant_index in list(reversed(self.opt_ant_global)):
#            print ant_index
#            if ant_index<l_local:
#                del self.all_ants_local[ant_index]
#                
#        #generation
        self.opt_ant_local,self.score_list_local=self.ant_ranking(self.all_ants_local)
#        print 'l', self.opt_ant_local
#        
#        total_update=len(self.opt_ant_global)+len(self.opt_ant_local)
#        print self.opt_ant_global, len(self.opt_ant_global)
#        print self.opt_ant_local,len(self.opt_ant_local)
#        print total_update
        opt_all=list(self.opt_ant_global+self.opt_ant_local)
        #print 'combo',opt_all
        #print 'set combo',set(opt_all)
        self.opt_ants=list(set(opt_all)) #indices of local and global pareto
        opt_ants_l=list(set(opt_all)-set(self.opt_ant_global)) #get only local pareto
        
        
#        print 'opt_ants',self.opt_ants,len(self.all_ants)
#        print 'opt_ants_l',opt_ants_l
        
        
        #Update Institutions - Edges and capacities for this walk - best
#        self.update_institutions(self.ant_graphs, self.opt_ants,
#                                 self.score_list,self.capacities,
#                                 len(self.opt_ants),1)
        # Elitest
#        self.update_institutions(self.all_ants_global, self.opt_ant_global,
#                                 self.score_list_global,self.capacities,
#                                 total_update,1)
#        #generation best
#        self.update_institutions(self.all_ants_local, self.opt_ant_local,
#                                 self.score_list_local,self.capacities,
#                                 total_update,0)
        #update global pareto 
        #separate
#        g_perc_update=float(len(self.opt_ant_global))/len(opt_all) #keep invariance
#        self.update_institutions(self.all_ants, self.opt_ant_global,
#                                 self.score_list_global,self.capacities,
#                                 g_perc_update,1)
#        if opt_ants_l: #update local pareto
#            l_perc_update=1.0-g_perc_update #ddpe invariance
#            self.update_institutions(self.all_ants, opt_ants_l,
#                                     self.score_list_global,self.capacities,
#                                     l_perc_update,0) #only dissipate once

        #together
        self.update_institutions(self.all_ants, opt_all,
                                 self.score_list_global,self.capacities,
                                 1.0,1)
                                 
        #Remove all except for pareto front ants
        self.keep_ants=[]
        #print 'opt_ant_index', self.opt_ant
#        for i in range(len(self.all_ants_global)):
#            if i in self.opt_ant_global:
#                self.keep_ants.append(self.all_ants_global[i])
#                print self.ant_graphs[i]
#                num_e=[self.ant_graphs[i][0][0].number_of_edges(),self.ant_graphs[i][0][1].number_of_edges()]
#                print 'ant',i,num_e,self.ant_graphs[i][1]
#                
#        for i in range(len(self.ant_graphs)):
#            if i in self.opt_ants:
#                self.keep_ants.append(self.ant_graphs[i])
                
        for i in range(len(self.all_ants)):
            if i in self.opt_ant_global:
                self.keep_ants.append(self.all_ants[i])        
        #self.ant_graphs=[]
#        print 'size keep',len(self.keep_ants)
        return self.keep_ants,self.all_ants
        #return keep_ants
               
if __name__=='__main__':
    space=Space()
    converged=0
    i=1
    pareto_history=[]
    gen_history=[]
    ant_hist=[]
    criteria=0
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
        if i>5 and pareto_history[-1]==pareto_history[-2]:
            criteria+=1
        else:
            criteria=0
            
        
        print (i,space.cycle_count,space.ant_failures),',',
        #print 'Pareto front of generation',i,':',paretoPoints, criteria #'\r',
        if criteria>10:
            converged=1

        i+=1
        if i>10:
            converged=1
            print '\n'
    
    

    # get overall pareto - for not keeping pareto ants in next gen
    #print ant_hist
    #opt_ants,score_list=space.ant_ranking(ant_hist)
    opt_ants,score_list=space.ant_ranking(all_ants)
    final_pareto=[]
    for i in opt_ants:
        final_pareto.append(score_list[i])
    
    print 'Ant Paths'
    #visualization
    f = plt.figure()
    space.visualize_pareto(gen_history, pareto_history, final_pareto)
 
#    for ant in opt_ants:
#        score=score_list[ant]
#        space.visualize_system_single(ant_hist[ant][0],
#                                      space.source,space.source_magnitude,
#                                      space.sink,space.sink_magnitude,
#                                      space.g,score,7)
 
    for ant in opt_ants:
        score=score_list[ant]
        space.visualize_system_single(all_ants[ant][0],
                                      space.source,space.source_magnitude,
                                      space.sink,space.sink_magnitude,
                                      space.g,score,7)
    
    


    


