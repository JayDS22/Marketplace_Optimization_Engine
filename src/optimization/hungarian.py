import numpy as np
from scipy.optimize import linear_sum_assignment
import networkx as nx
from typing import Tuple, List, Dict, Any, Optional
import logging
import time

class HungarianMatchingOptimizer:
    """
    Hungarian algorithm implementation for optimal bipartite matching
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.matching_history = []
        
    def solve_assignment_problem(self, cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Solve the assignment problem using Hungarian algorithm
        
        Args:
            cost_matrix: n×m cost matrix where cost_matrix[i,j] is cost of assigning i to j
            
        Returns:
            row_indices: Array of row indices for optimal assignment
            col_indices: Array of column indices for optimal assignment  
            total_cost: Total cost of optimal assignment
        """
        start_time = time.time()
        
        # Validate input
        if cost_matrix.size == 0:
            raise ValueError("Cost matrix cannot be empty")
        
        if np.any(np.isnan(cost_matrix)) or np.any(np.isinf(cost_matrix)):
            raise ValueError("Cost matrix contains NaN or infinite values")
        
        # Apply Hungarian algorithm
        row_indices, col_indices = linear_sum_assignment(cost_matrix)
        total_cost = cost_matrix[row_indices, col_indices].sum()
        
        execution_time = time.time() - start_time
        
        # Calculate efficiency metrics
        efficiency_metrics = self._calculate_efficiency_metrics(
            cost_matrix, row_indices, col_indices, total_cost
        )
        
        # Store matching history
        self.matching_history.append({
            'timestamp': time.time(),
            'cost_matrix_shape': cost_matrix.shape,
            'total_cost': total_cost,
            'execution_time': execution_time,
            'efficiency': efficiency_metrics['efficiency'],
            'utilization': efficiency_metrics['utilization']
        })
        
        self.logger.info(
            f"Hungarian algorithm completed: {efficiency_metrics['efficiency']:.1f}% efficiency, "
            f"{execution_time*1000:.1f}ms runtime"
        )
        
        return row_indices, col_indices, total_cost
    
    def solve_max_weight_matching(self, cost_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Solve maximum weight bipartite matching (convert to min cost)
        """
        # Convert to minimization problem
        max_cost = np.max(cost_matrix)
        min_cost_matrix = max_cost - cost_matrix
        
        row_indices, col_indices, min_total_cost = self.solve_assignment_problem(min_cost_matrix)
        
        # Convert back to maximum weight
        max_total_weight = cost_matrix[row_indices, col_indices].sum()
        
        return row_indices, col_indices, max_total_weight
    
    def solve_with_constraints(self, cost_matrix: np.ndarray, 
                             forbidden_pairs: List[Tuple[int, int]] = None,
                             required_pairs: List[Tuple[int, int]] = None) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Solve assignment with additional constraints
        
        Args:
            cost_matrix: Original cost matrix
            forbidden_pairs: List of (row, col) pairs that cannot be matched
            required_pairs: List of (row, col) pairs that must be matched
        """
        modified_cost_matrix = cost_matrix.copy()
        
        # Handle forbidden pairs
        if forbidden_pairs:
            for row, col in forbidden_pairs:
                if 0 <= row < cost_matrix.shape[0] and 0 <= col < cost_matrix.shape[1]:
                    modified_cost_matrix[row, col] = np.inf
        
        # Handle required pairs (set cost to very low value)
        if required_pairs:
            min_cost = np.min(cost_matrix)
            for row, col in required_pairs:
                if 0 <= row < cost_matrix.shape[0] and 0 <= col < cost_matrix.shape[1]:
                    modified_cost_matrix[row, col] = min_cost - 1000
        
        return self.solve_assignment_problem(modified_cost_matrix)
    
    def solve_multiple_objectives(self, cost_matrices: List[np.ndarray], 
                                weights: List[float]) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Solve multi-objective assignment problem using weighted sum approach
        
        Args:
            cost_matrices: List of cost matrices for different objectives
            weights: Importance weights for each objective
        """
        if len(cost_matrices) != len(weights):
            raise ValueError("Number of cost matrices must match number of weights")
        
        if not np.allclose(sum(weights), 1.0):
            raise ValueError("Weights must sum to 1.0")
        
        # Normalize all cost matrices to [0, 1] range
        normalized_matrices = []
        for matrix in cost_matrices:
            min_val, max_val = matrix.min(), matrix.max()
            if max_val > min_val:
                normalized = (matrix - min_val) / (max_val - min_val)
            else:
                normalized = np.zeros_like(matrix)
            normalized_matrices.append(normalized)
        
        # Weighted combination
        combined_matrix = np.zeros_like(normalized_matrices[0])
        for matrix, weight in zip(normalized_matrices, weights):
            combined_matrix += weight * matrix
        
        return self.solve_assignment_problem(combined_matrix)
    
    def _calculate_efficiency_metrics(self, cost_matrix: np.ndarray, 
                                    row_indices: np.ndarray, 
                                    col_indices: np.ndarray, 
                                    total_cost: float) -> Dict[str, float]:
        """Calculate matching efficiency metrics"""
        
        # Matching efficiency (percentage of optimal)
        theoretical_min = np.min(cost_matrix, axis=1).sum()
        if theoretical_min > 0:
            efficiency = (theoretical_min / total_cost) * 100
        else:
            efficiency = 100.0
        
        # Utilization rate
        num_matched = len(row_indices)
        max_possible_matches = min(cost_matrix.shape)
        utilization = (num_matched / max_possible_matches) * 100
        
        # Average cost per match
        avg_cost_per_match = total_cost / num_matched if num_matched > 0 else 0
        
        return {
            'efficiency': efficiency,
            'utilization': utilization,
            'avg_cost_per_match': avg_cost_per_match,
            'total_matches': num_matched
        }
    
    def analyze_matching_stability(self, cost_matrix: np.ndarray, 
                                 row_indices: np.ndarray, 
                                 col_indices: np.ndarray) -> Dict[str, Any]:
        """
        Analyze stability of the matching solution
        """
        n_rows, n_cols = cost_matrix.shape
        
        # Calculate regret for each agent
        row_regrets = []
        col_regrets = []
        
        for i, j in zip(row_indices, col_indices):
            # Row agent regret (could have gotten better match?)
            row_alternatives = cost_matrix[i, :]
            current_cost = cost_matrix[i, j]
            best_alternative = np.min(row_alternatives)
            row_regrets.append(current_cost - best_alternative)
            
            # Column agent regret
            col_alternatives = cost_matrix[:, j]
            best_col_alternative = np.min(col_alternatives)
            col_regrets.append(current_cost - best_col_alternative)
        
        stability_metrics = {
            'row_regrets': row_regrets,
            'col_regrets': col_regrets,
            'max_row_regret': max(row_regrets) if row_regrets else 0,
            'max_col_regret': max(col_regrets) if col_regrets else 0,
            'avg_row_regret': np.mean(row_regrets) if row_regrets else 0,
            'avg_col_regret': np.mean(col_regrets) if col_regrets else 0,
            'is_stable': max(row_regrets + col_regrets) < self.config.get('stability_threshold', 0.1)
        }
        
        return stability_metrics
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary from matching history"""
        if not self.matching_history:
            return {}
        
        efficiencies = [entry['efficiency'] for entry in self.matching_history]
        execution_times = [entry['execution_time'] for entry in self.matching_history]
        
        return {
            'total_matchings': len(self.matching_history),
            'avg_efficiency': np.mean(efficiencies),
            'min_efficiency': np.min(efficiencies),
            'max_efficiency': np.max(efficiencies),
            'avg_execution_time': np.mean(execution_times),
            'max_execution_time': np.max(execution_times),
            'efficiency_std': np.std(efficiencies)
        }
