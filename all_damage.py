#Imports
# Standard imports
import copy
import itertools
import numpy
import matplotlib.pyplot as plt
from matplotlib import colors
import networkx as nx
import pandas as pd
from datetime import datetime
from capacity_scaling import capacity_scaling
from networkx.algorithms.traversal.depth_first_search import dfs_tree
from random import sample


# Damage Test

def add_demand(g, sys, components):
    """
    Adds component demand for system.
    """
    g = copy.deepcopy(g)
    for i in g:
        g.add_node(i, demand=0)

    for c, val in components[sys].iteritems():
        g.add_node(c, demand=val['d'])
    return g


def test_s(components, s_sol, sys):
    """
    Test a single system for flow satisfaction
    """
    g = copy.deepcopy(s_sol)

    # Test flow
    cost, outflow = capacity_scaling(g)

    # Create inflow dict
    inflow = {}
    for n in g.nodes():
        inflow[n] = 0
    for pred, out in outflow.iteritems():
        for dest, in_f in out.iteritems():
            inflow[dest] += in_f

    # Check demand satisfaction
    sat = 1
    sat_dict = {}
    for c, val in components[sys].iteritems():
        if val['d'] > inflow[c]:
            sat = 0
            sat_dict[c] = 0
        else:
            sat_dict[c] = 1

    return sat, inflow  # , sat_dict


def test_s_connectivity(components, s_sol, sys):
    """
    Test a single system for flow satisfaction
    """
    g = copy.deepcopy(s_sol)

    # Get source and sink information
    # s,t=get_s_t(sys,components)
    # print sys,s,t

    # Find nodes that are connected to working sinks
    supported = set()  # list of all supported nodes
    for i in g:
        # print i, 'demand',g.node[i]['demand']
        if g.node[i]['demand'] < 0:
            supported.update([i])
            # print 'source',i,dfs_tree(s_sol,i).nodes()
            supported.update(dfs_tree(s_sol, i).nodes())
    # print supported

    # Check demand satisfaction
    sat = 1
    sat_dict = {}
    for c, val in components[sys].iteritems():
        if c in supported:
            sat = 1
            sat_dict[c] = 1
        else:
            sat = 0
            sat_dict[c] = 0

    return sat, sat_dict


def test_sos(base, components, sos):
    """
    Test system of system for flow satisfaction
    """
    # test_type =
    ## 'cc' - connectivity cross (interdependent)
    ## 'fc' - flow cross (interdependent)
    ## 'ci' - connectivity independent
    ## 'fi' - flow independent


    sos_eval = copy.deepcopy(sos)

    # Check component operating or not, failed if any demand not sat.
    comp_keys = []
    for sys in components:
        comp_keys.extend(components[sys].keys())
    comp_keys = list(set(comp_keys))

    operating = {k: 1 for k in comp_keys}

    # Check if existing
    for sys, g in sos_eval.iteritems():
        for c, val in components[sys].iteritems():
            # print c, g.nodes(),val
            if (c not in g) and (val['d'] != 0):
                # removed nodes are failed.
                operating[c] = 0
                # g.add_node(c,demand=val['d'])
    ini_op = copy.deepcopy(operating)

    # Add demand
    for sys, g in sos_eval.iteritems():
        sos_eval[sys] = add_demand(g, sys, components)

    # Test flow
    # Store SoS information
    sos_info = {k: {'sat': 0, 'inflow': {}} for k in components.keys()}
    op_list = [copy.deepcopy(operating)]
    while True:
        # Test individual systems

        # print sos_eval
        for sys, g in sos_eval.iteritems():
            sat, inflow = test_s(components, g, sys)
            sos_info[sys]['sat'] = sat
            sos_info[sys]['inflow'] = inflow
            # sys_sat[sys]=sat_dict
        # print 'eval results', sos_info
        # check component dependence
        # failure=0


        for sys in components:
            for c, val in components[sys].iteritems():
                # what are links
                for l in val['l']:
                    # dependent system
                    d_sys = l[0]
                    threshold = l[1]

                    # check flow on dependent
                    dep_flow = sos_info[d_sys]['inflow'][c]
                    dep_req = components[d_sys][c]['d']
                    dep_ratio = float(dep_flow) / dep_req
                    # print 'link', c,l,'in=',dep_flow,'req=',dep_req
                    if dep_ratio < threshold:
                        # failure - remove demand from graph
                        sos_eval[sys].node[c]['demand'] = 0
                        operating[c] = 0
                        # failure=1

        # store operating list
        op_list.append(copy.deepcopy(operating))
        # if no failure, converged. End while loop.
        if op_list[-1] == op_list[-2]:  # failure==0
            # print 'operating list',op_list
            break
    fi = copy.deepcopy(op_list[1])
    fc = copy.deepcopy(operating)

    # Test connectivity
    # Add demand
    for sys, g in sos_eval.iteritems():
        sos_eval[sys] = add_demand(g, sys, components)

    # Store SoS information
    sos_info = {k: {'sat': 0, 'inflow': {}} for k in components.keys()}
    op_list = [copy.deepcopy(ini_op)]
    operating = copy.deepcopy(ini_op)
    while True:
        # Test individual systems

        # print sos_eval
        for sys, g in sos_eval.iteritems():
            # print sys
            sat, inflow = test_s_connectivity(components, g, sys)
            # print inflow

            sos_info[sys]['sat'] = sat
            sos_info[sys]['inflow'] = inflow
            # sys_sat[sys]=sat_dict
        # print 'eval results', sos_info
        # check component dependence
        # failure=0


        for sys in components:
            for c, val in components[sys].iteritems():
                # what are links
                for l in val['l']:
                    # dependent system
                    d_sys = l[0]
                    threshold = l[1]

                    # check flow on dependent
                    dep_flow = sos_info[d_sys]['inflow'][c]
                    dep_req = components[d_sys][c]['d']
                    # dep_ratio=float(dep_flow)/dep_req
                    # print 'link', c,l,'in=',dep_flow,'req=',dep_req
                    if dep_req > 0 and dep_flow == 0:
                        # failure - remove demand from graph
                        sos_eval[sys].node[c]['demand'] = 0
                        operating[c] = 0
                        # failure=1

        # store operating list
        op_list.append(copy.deepcopy(operating))
        # if no failure, converged. End while loop.
        if op_list[-1] == op_list[-2]:  # failure==0
            # print 'operating list',op_list
            break
    # print op_list

    ci = copy.deepcopy(op_list[1])
    cc = copy.deepcopy(operating)

    # Test connectivity - undirected
    # Add demand
    for sys, g in sos_eval.iteritems():
        undir_g = nx.Graph()
        undir_g.add_edges_from(g.edges())
        sos_eval[sys] = add_demand(undir_g, sys, components)

    # Store SoS information
    sos_info = {k: {'sat': 0, 'inflow': {}} for k in components.keys()}
    op_list = [copy.deepcopy(ini_op)]
    operating = copy.deepcopy(ini_op)
    while True:
        # Test individual systems

        # print sos_eval
        for sys, g in sos_eval.iteritems():
            # print sys

            sat, inflow = test_s_connectivity(components, g, sys)
            # print inflow

            sos_info[sys]['sat'] = sat
            sos_info[sys]['inflow'] = inflow
            # sys_sat[sys]=sat_dict
        # print 'eval results', sos_info
        # check component dependence
        # failure=0


        for sys in components:
            for c, val in components[sys].iteritems():
                # what are links
                for l in val['l']:
                    # dependent system
                    d_sys = l[0]
                    threshold = l[1]

                    # check flow on dependent
                    dep_flow = sos_info[d_sys]['inflow'][c]
                    dep_req = components[d_sys][c]['d']
                    # dep_ratio=float(dep_flow)/dep_req
                    # print 'link', c,l,'in=',dep_flow,'req=',dep_req
                    if dep_req > 0 and dep_flow == 0:
                        # failure - remove demand from graph
                        sos_eval[sys].node[c]['demand'] = 0
                        operating[c] = 0
                        # failure=1

        # store operating list
        op_list.append(copy.deepcopy(operating))
        # if no failure, converged. End while loop.
        if op_list[-1] == op_list[-2]:  # failure==0
            # print 'operating list',op_list
            break
    # print op_list

    uci = copy.deepcopy(op_list[1])
    ucc = copy.deepcopy(operating)

    return fi, fc, ci, cc, uci, ucc


def get_s_t(sys, components):
    """
    Gets source and sink info for system.
    """
    s_dict = {}
    t_dict = {}

    for c, info in components[sys].iteritems():
        if info['d'] < 0:
            s_dict[c] = -info['d']

        if info['d'] > 0:
            t_dict[c] = info['d']

    return s_dict, t_dict


def score_sos(v, c, sos, tests, n_rem=1, blast=1):
    """
    Get score of system of systems
    """

    # Survivability
    results = {}
    results['fi'] = []
    results['fc'] = []
    results['ci'] = []
    results['cc'] = []
    results['uci'] = []
    results['ucc'] = []

    op = {}
    op['fi'] = []
    op['fc'] = []
    op['ci'] = []
    op['cc'] = []
    op['uci'] = []
    op['ucc'] = []

    for i in xrange(tests):
        sos_damage = copy.deepcopy(sos)
        # remove nodes due to damage
        sos_damage = inflict_damage(v, sos_damage, n_rem, blast)

        # for sys in sos_damage:
        # print sys,sos_damage[sys].nodes(data=True),sos_damage[sys].edges(data=True)

        # Test flow
        fi, fc, ci, cc, uci, ucc = test_sos(v, c, sos_damage)

        # flow
        # no-cross
        working = sum(fi.itervalues())
        results['fi'].append(float(working) / len(fi.keys()))
        op['fi'].append(fi)

        # cross
        working = sum(fc.itervalues())
        results['fc'].append(float(working) / len(fc.keys()))
        op['fc'].append(fc)

        # connectivity
        # no-cross
        working = sum(ci.itervalues())
        results['ci'].append(float(working) / len(ci.keys()))
        op['ci'].append(ci)

        # cross
        working = sum(cc.itervalues())
        results['cc'].append(float(working) / len(cc.keys()))
        op['cc'].append(cc)

        # connectivity-undirected
        # no-cross
        working = sum(uci.itervalues())
        results['uci'].append(float(working) / len(uci.keys()))
        op['uci'].append(uci)

        # cross
        working = sum(ucc.itervalues())
        results['ucc'].append(float(working) / len(ucc.keys()))
        op['ucc'].append(ucc)

    score = {}
    for t in results:
        score[t] = numpy.average(results[t])
    # print results,s_score


    return score, results, op


# Damage inflict, blast radius=1


def inflict_damage(v, sos, n_rem, blast):
    """
    Removes nodes from sos.
    """
    targets = sample(v.nodes(),
                     n_rem)
    # print 'removed', nodes_to_remove
    # Get radius
    nodes_to_remove = set()
    # print 'num', n_rem

    # targets=[(0,0)]

    nodes_to_remove.update(targets)
    # nodes_to_remove.update([(0,0)])
    #print nodes_to_remove

    for n in targets:
        path = nx.single_source_shortest_path(v,
                                              n,
                                              cutoff=blast)
        for t in path:
            nodes_to_remove.update(path[t])

    # print 'final',nodes_to_remove


    # Remove node in each system
    # print nodes_to_remove
    for sys, g in sos.iteritems():
        for n in nodes_to_remove:

            if n in g:
                g.remove_node(n)

    return sos


def testing(area, c, sos, tests, number_of_removals, radius):
    s, r, o = score_sos(area, c, sos, tests, n_rem=number_of_removals, blast=radius)

    component_names = []
    for sys, comps in c.iteritems():
        for comp in comps:
            component_names.append(comp)

    component_names = list(set(component_names))

    data = []

    for a_type in r:
        for trial in zip(r[a_type], o[a_type]):
            operating = []
            for comp in component_names:
                operating.append(trial[1][comp])
            # print operating
            # print trial[0]
            trial_data = list(itertools.chain([number_of_removals, radius], [a_type], [trial[0]], operating))
            data.append(trial_data)

    cs = list(itertools.chain(['hits', 'blast', 'eval', 'score'], component_names))

    d = pd.DataFrame(data, columns=cs)

    return d

def test_fvc():
    # Define vessel
    area = nx.grid_graph(dim=[2, 2])

    # Define components
    c = {1: {(0, 1): {'d': -1, 'l': [(2, 1.0)]},
             (0, 0): {'d': -1, 'l': [(2, 1.0)]},
             (1, 0): {'d': 1, 'l': [(1, 1.0)]}},
         2: {(0, 1): {'d': 1, 'l': [(2, 1.0)]},
             (0, 0): {'d': 1, 'l': [(2, 1.0)]},
             (1, 0): {'d': -2, 'l': [(1, 1.0)]}}}

    # Define SoS
    sos = {}

    g = nx.DiGraph()
    g.add_edges_from([((0, 1), (1, 1), {'capacity': 1}),
                      ((0, 0), (1, 0), {'capacity': 1}),
                      ((1, 1), (1, 0), {'capacity': 1})])

    sos[1] = copy.deepcopy(g)

    g = nx.DiGraph()
    g.add_edges_from([((1, 0), (0, 0), {'capacity': 2}),
                      ((0, 0), (0, 1), {'capacity': 1})])

    sos[2] = copy.deepcopy(g)

    data = testing(area, c, sos, tests=1, number_of_removals=1, radius=0)
    return data

print test_fvc()