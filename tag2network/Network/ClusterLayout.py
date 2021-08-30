"""
Modified tSNE layout to pull clusters together into visually coherent groups.

1) tSNE on whole network
2) Kamada-Kawai on each cluster
3) Pull 'distant nodes in to limit total radius of each cluster
4) (Optional) GTree to move clsuters to eliminate overlap
"""
from Network.tSNELayout import runTSNELayout
import networkx as nx
import numpy as np
import scipy as sp


def remove_overlap(nodes):
    """Implement GTree algorithm https://arxiv.org/pdf/1608.02653.pdf."""
    nodes = [n.copy() for n in nodes]

    def dist(idx1, idx2, pos, nodes):
        d = pos[idx1] - pos[idx2]
        center_to_center = np.sqrt((d * d).sum())
        return center_to_center - nodes[idx1]['radius'] - nodes[idx2]['radius']

    def get_next(mst, previous, current):
        edges = list(mst.edges(current))
        next_nodes = []
        for e in edges:
            if previous is None or previous not in e:
                next_nodes.append(e[1] if e[0] == current else e[0])
        return next_nodes

    def shift_nodes(nodes, mst, source, target, delta_x, delta_y):
        # shift the target
        trg_node = nodes[target]
        trg_node['x'] += delta_x
        trg_node['y'] += delta_y
        # shift nodes recursively
        next_nodes = get_next(mst, source, target)
        for next_n in next_nodes:
            shift_nodes(nodes, mst, target, next_n, delta_x, delta_y)

    def process_tree(nodes, mst, previous, current):
        # process mst recursively
        next_nodes = get_next(mst, previous, current)
        for next_n in next_nodes:
            wt = mst.edges[(current, next_n)]['weight']
            if wt < 0:
                # compute the shift x, y
                src_node = nodes[current]
                trg_node = nodes[next_n]
                dx = trg_node['x'] - src_node['x']
                dy = trg_node['y'] - src_node['y']
                dist = np.sqrt(dx ** 2 + dy ** 2)
                frac_x = dx / dist
                frac_y = dy / dist
                wt = mst.edges[(current, next_n)]['weight']
                delta_x = -wt * frac_x
                delta_y = -wt * frac_y
                # shift target and its children
                shift_nodes(nodes, mst, current, next_n, delta_x, delta_y)
            process_tree(nodes, mst, current, next_n)

    max_steps = 10
    for step in range(max_steps):
        # extract position data to numpy
        pos = np.array([[n['x'], n['y']] for n in nodes])
        # build delauney triangulation
        tri = sp.spatial.Delaunay(pos)
        # build weighted networkx graph. Weight is distance between node edges
        raw_edges = set()
        for sim in tri.simplices:
            raw_edges.add((sim[0], sim[1]))
            raw_edges.add((sim[1], sim[2]))
            raw_edges.add((sim[2], sim[0]))
        nw = nx.Graph()
        n_overlap = 0
        for e in raw_edges:
            d = dist(e[0], e[1], pos, nodes)
            nw.add_edge(e[0], e[1], weight=d)
            if d < 0:
                n_overlap += 1
        print(f'Step {step} n_overlap = {n_overlap}')
        # quit looping if all weights are positive
        if n_overlap == 0:
            break
        # get minimal spanning tree of weighted graph
        mst = nx.minimum_spanning_tree(nw)
        # roots have degree == 1
        root = [n for n, d in mst.degree if d == 1][0]
        # recursively process mst from root
        process_tree(nodes, mst, None, root)
    return {n['name']: np.array([n['x'], n['y']]) for n in nodes}


def compress_groups(nw, nodes_df, layout_dict, cluster, no_overlap,
                    max_expansion=1.5, scale_factor=1.0):
    clus_nodes = nodes_df.groupby(cluster)
    subgraphs = {clus: cdf.id.to_list() for clus, cdf in clus_nodes}
    new_positions = {}
    clusters = []
    clus_id = 0
    for clus, subg in subgraphs.items():
        # get starting positions (from tSNE layout)
        pos = {k: layout_dict[k] for k in subg}
        clus_pos = np.array(list(pos.values()))
        # get cluster centroid, final scale and distance from center
        center = np.median(clus_pos, axis=0)
        scale = np.sqrt(len(subg)) * scale_factor
        # add cluster node
        clusters.append({'id': clus_id,
                         'name': clus,
                         'x': center[0],
                         'y': center[1],
                         'radius': scale})
        clus_id += 1
    # move cluster centers to remove overlap
    print("Repositioning cluster centers")
    centers = {cl['name']: (cl['x'], cl['y']) for cl in clusters}
    if no_overlap:
        new_centers = remove_overlap(clusters)
    else:
        new_centers = centers
    print("Compressing layout of nodes in clusters")
    for clus, subg in subgraphs.items():
        # get starting positions (from tSNE layout)
        pos = {k: layout_dict[k] for k in subg}
        clus_pos = np.array(list(pos.values()))
        # get cluster centroid, final scale and distance from center
        center = np.median(clus_pos, axis=0)
        scale = np.sqrt(len(subg)) * scale_factor
        dists = np.sqrt(((clus_pos - center) ** 2).sum(axis=1))
        # use a truncated, normalized Mechelis-Menten function
        # to rescale distance from center
        om = max(dists)
        nm = scale
        k = om * nm / (om - nm / 2)
        rescale = np.clip(k / (k / 2 + dists), None, max_expansion)
        center = centers[clus]
        new_center = new_centers[clus]
        new_pos = new_center + ((clus_pos - center) * rescale.reshape(-1, 1))
        new_positions.update(dict(zip(pos.keys(), new_pos)))
    return new_positions


def run_cluster_layout(nw, nodes_df, dists=None, maxdist=5, cluster='Cluster', no_overlap=True):
    layout_dict, layout = runTSNELayout(nw, nodes_df, dists, maxdist, cluster)
    clus_nodes = nodes_df.groupby(cluster)
    subgraphs = {clus: nw.subgraph(cdf.id.to_list()) for clus, cdf in clus_nodes}
    new_positions = {}
    for clus, subg in subgraphs.items():
        print(f"Laying out subgraph for {clus}")
        # get starting positions (from tSNE layout)
        pos = {k: layout_dict[k] for k in subg.nodes}
        clus_pos = np.array(list(pos.values()))
        # get cluster centroid
        center = np.median(clus_pos, axis=0)
        scale = np.sqrt(len(subg))
        new_pos = nx.kamada_kawai_layout(subg, pos=pos, weight=None,
                                         scale=scale,
                                         center=center)
        new_positions.update(new_pos)
    new_positions = compress_groups(nw, nodes_df, new_positions, cluster, no_overlap)
    layout = [new_positions[idx] for idx in layout_dict.keys()]
    return new_positions, layout


if __name__ == "__main__":
    # build test dataset
    init_nodes = [
        {'id': 1,
         'name': 'node1',
         'x': -0.5,
         'y': 0,
         'radius': 0.3
         },
        {'id': 2,
         'name': 'node2',
         'x': 0,
         'y': 0,
         'radius': 0.3
         },
        {'id': 3,
         'name': 'node3',
         'x': 0.5,
         'y': 0.1,
         'radius': 0.2
         },
        {'id': 4,
         'name': 'node4',
         'x': 1.0,
         'y': 0.1,
         'radius': 0.4
         }
        ]

    new_centers = remove_overlap(init_nodes)