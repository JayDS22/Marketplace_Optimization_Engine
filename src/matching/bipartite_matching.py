import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Any, Optional
import logging

class BipartiteMatchingEngine:
    """
    Advanced bipartite matching engine for marketplace optimization
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def maximum_bipartite_matching(self, edges: List[Tuple[int, int]], 
                                 left_nodes: List[int], 
                                 right_nodes: List[int]) -> Dict[str, Any]:
        """
        Find maximum cardinality bipartite matching
        """
        # Create bipartite graph
        G = nx.Graph()
        G.add_nodes_from(left_nodes, bipartite=0)
        G.add_nodes_from(right_nodes, bipartite=1)
        G.add_edges_from(edges)
        
        # Find maximum matching
        matching = nx.bipartite.maximum_matching(G, top_nodes=left_nodes)
        
        # Calculate statistics
        matching_size = len(matching)
        max_possible = min(len(left_nodes), len(right_nodes))
        matching_ratio = matching_size / max_possible if max_possible > 0 else 0
        
        # Analyze unmatched nodes
        matched_left = set(node for node in matching if node in left_nodes)
        matched_right = set(node for node in matching if node in right_nodes)
        
        unmatched_left = set(left_nodes) - matched_left
        unmatched_right = set(right_nodes) - matched_right
        
        return {
            'matching': dict(matching),
            'matching_size': matching_size,
            'matching_ratio': matching_ratio,
            'unmatched_left': list(unmatched_left),
            'unmatched_right': list(unmatched_right),
            'efficiency': matching_ratio * 100
        }
    
    def weighted_bipartite_matching(self, weighted_edges: List[Tuple[int, int, float]],
                                  left_nodes: List[int], 
                                  right_nodes: List[int]) -> Dict[str, Any]:
        """
        Find maximum weight bipartite matching
        """
        # Create weighted bipartite graph
        G = nx.Graph()
        G.add_nodes_from(left_nodes, bipartite=0)
        G.add_nodes_from(right_nodes, bipartite=1)
        
        for u, v, weight in weighted_edges:
            G.add_edge(u, v, weight=weight)
        
        # Find maximum weight matching
        matching = nx.bipartite.minimum_weight_full_matching(G, top_nodes=left_nodes)
        
        # Calculate total weight
        total_weight = sum(G[u][v]['weight'] for u, v in matching.items() if u in left_nodes)
        
        # Calculate efficiency metrics
        all_weights = [data['weight'] for u, v, data in G.edges(data=True)]
        max_possible_weight = sum(sorted(all_weights, reverse=True)[:min(len(left_nodes), len(right_nodes))])
        
        weight_efficiency = (total_weight / max_possible_weight * 100) if max_possible_weight > 0 else 0
        
        return {
            'matching': matching,
            'total_weight': total_weight,
            'weight_efficiency': weight_efficiency,
            'matching_size': len([k for k in matching.keys() if k in left_nodes]),
            'average_weight': total_weight / len(matching) if matching else 0
        }
