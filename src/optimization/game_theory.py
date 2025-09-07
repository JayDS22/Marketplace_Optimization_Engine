import pulp
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
import logging
import time

class LinearProgrammingOptimizer:
    """
    Linear Programming optimizer for marketplace resource allocation
    """
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.solver_history = []
        
    def solve_resource_allocation(self, supply: np.ndarray, demand: np.ndarray, 
                                costs: np.ndarray) -> Dict[str, Any]:
        """
        Solve transportation/resource allocation problem
        
        Args:
            supply: Supply capacities for each source
            demand: Demand requirements for each destination  
            costs: Cost matrix for transportation/allocation
            
        Returns:
            Dictionary with solution details
        """
        start_time = time.time()
        
        # Validate inputs
        if len(supply) != costs.shape[0] or len(demand) != costs.shape[1]:
            raise ValueError("Dimension mismatch between supply/demand and cost matrix")
        
        if np.sum(supply) < np.sum(demand):
            raise ValueError("Total supply must be >= total demand")
        
        # Create the LP problem
        prob = pulp.LpProblem("Resource_Allocation", pulp.LpMinimize)
        
        # Decision variables
        sources = range(len(supply))
        destinations = range(len(demand))
        
        # x[i][j] = amount transported from source i to destination j
        x = {}
        for i in sources:
            for j in destinations:
                x[i, j] = pulp.LpVariable(f"x_{i}_{j}", lowBound=0, cat='Continuous')
        
        # Objective function: minimize total cost
        prob += pulp.lpSum([costs[i][j] * x[i, j] for i in sources for j in destinations])
        
        # Supply constraints
        for i in sources:
            prob += pulp.lpSum([x[i, j] for j in destinations]) <= supply[i], f"Supply_{i}"
        
        # Demand constraints  
        for j in destinations:
            prob += pulp.lpSum([x[i, j] for i in sources]) >= demand[j], f"Demand_{j}"
        
        # Solve the problem
        solver = pulp.PULP_CBC_CMD(msg=0)
        prob.solve(solver)
        
        execution_time = time.time() - start_time
        
        # Extract solution
        solution = self._extract_solution(prob, x, sources, destinations, costs, execution_time)
        
        # Store in history
        self.solver_history.append({
            'timestamp': time.time(),
            'problem_size': (len(supply), len(demand)),
            'status': solution['status'],
            'objective_value': solution['objective_value'],
            'execution_time': execution_time
        })
        
        return solution
    
    def solve_capacity_planning(self, demand_forecast: np.ndarray, 
                              capacity_costs: np.ndarray,
                              operational_costs: np.ndarray,
                              service_levels: np.ndarray) -> Dict[str, Any]:
        """
        Solve capacity planning optimization with service level constraints
        """
        start_time = time.time()
        
        time_periods = len(demand_forecast)
        capacity_options = len(capacity_costs)
        
        # Create LP problem
        prob = pulp.LpProblem("Capacity_Planning", pulp.LpMinimize)
        
        # Decision variables
        # y[t][k] = capacity of type k in period t
        y = {}
        # x[t] = unmet demand in period t
        x = {}
        
        for t in range(time_periods):
            x[t] = pulp.LpVariable(f"unmet_demand_{t}", lowBound=0)
            for k in range(capacity_options):
                y[t, k] = pulp.LpVariable(f"capacity_{t}_{k}", lowBound=0, cat='Integer')
        
        # Objective: minimize total cost (capacity + operational + penalty for unmet demand)
        penalty_cost = self.config.get('unmet_demand_penalty', 1000)
        
        total_cost = (
            pulp.lpSum([capacity_costs[k] * y[t, k] 
                       for t in range(time_periods) for k in range(capacity_options)]) +
            pulp.lpSum([operational_costs[k] * y[t, k] 
                       for t in range(time_periods) for k in range(capacity_options)]) +
            pulp.lpSum([penalty_cost * x[t] for t in range(time_periods)])
        )
        
        prob += total_cost
        
        # Capacity constraints: capacity must meet demand
        for t in range(time_periods):
            total_capacity = pulp.lpSum([y[t, k] for k in range(capacity_options)])
            prob += total_capacity + x[t] >= demand_forecast[t], f"Demand_Period_{t}"
        
        # Service level constraints
        for t in range(time_periods):
            required_capacity = demand_forecast[t] * service_levels[t]
            prob += pulp.lpSum([y[t, k] for k in range(capacity_options)]) >= required_capacity
        
        # Solve
        solver = pulp.PULP_CBC_CMD(msg=0)
        prob.solve(solver)
        
        execution_time = time.time() - start_time
        
        # Extract and return solution
        return self._extract_capacity_solution(prob, y, x, time_periods, capacity_options, execution_time)
    
    def solve_multi_objective_optimization(self, objectives: List[np.ndarray],
                                         weights: List[float],
                                         constraints: List[Dict]) -> Dict[str, Any]:
        """
        Solve multi-objective optimization using weighted sum method
        """
        if len(objectives) != len(weights):
            raise ValueError("Number of objectives must match number of weights")
        
        if not np.allclose(sum(weights), 1.0):
            weights = np.array(weights) / sum(weights)  # Normalize
        
        start_time = time.time()
        
        # Create problem
        prob = pulp.LpProblem("Multi_Objective", pulp.LpMinimize)
        
        # Normalize objectives to [0, 1] range
        normalized_objectives = []
        for obj in objectives:
            obj_min, obj_max = np.min(obj), np.max(obj)
            if obj_max > obj_min:
                normalized = (obj - obj_min) / (obj_max - obj_min)
            else:
                normalized = np.zeros_like(obj)
            normalized_objectives.append(normalized)
        
        # Decision variables (assuming same structure for all objectives)
        n_vars = len(objectives[0])
        x_vars = [pulp.LpVariable(f"x_{i}", lowBound=0) for i in range(n_vars)]
        
        # Combined objective
        combined_objective = pulp.lpSum([
            weights[obj_idx] * pulp.lpSum([normalized_objectives[obj_idx][i] * x_vars[i] 
                                         for i in range(n_vars)])
            for obj_idx in range(len(objectives))
        ])
        
        prob += combined_objective
        
        # Add constraints
        for constraint in constraints:
            self._add_constraint(prob, constraint, x_vars)
        
        # Solve
        solver = pulp.PULP_CBC_CMD(msg=0)
        prob.solve(solver)
        
        execution_time = time.time() - start_time
        
        return self._extract_multi_objective_solution(prob, x_vars, objectives, weights, execution_time)
    
    def solve_network_flow(self, nodes: List[str], edges: List[Tuple], 
                          capacities: Dict, costs: Dict, supplies: Dict) -> Dict[str, Any]:
        """
        Solve minimum cost network flow problem
        """
        start_time = time.time()
        
        # Create problem
        prob = pulp.LpProblem("Network_Flow", pulp.LpMinimize)
        
        # Decision variables for flow on each edge
        flow_vars = {}
        for edge in edges:
            flow_vars[edge] = pulp.LpVariable(f"flow_{edge[0]}_{edge[1]}", lowBound=0)
        
        # Objective: minimize total cost
        prob += pulp.lpSum([costs.get(edge, 0) * flow_vars[edge] for edge in edges])
        
        # Flow conservation constraints
        for node in nodes:
            # Incoming flow - outgoing flow = supply (negative for demand)
            incoming = pulp.lpSum([flow_vars[(i, node)] for (i, j) in edges if j == node])
            outgoing = pulp.lpSum([flow_vars[(node, j)] for (i, j) in edges if i == node])
            
            prob += incoming - outgoing == supplies.get(node, 0), f"Flow_Conservation_{node}"
        
        # Capacity constraints
        for edge in edges:
            if edge in capacities:
                prob += flow_vars[edge] <= capacities[edge], f"Capacity_{edge[0]}_{edge[1]}"
        
        # Solve
        solver = pulp.PULP_CBC_CMD(msg=0)
        prob.solve(solver)
        
        execution_time = time.time() - start_time
        
        return self._extract_network_flow_solution(prob, flow_vars, edges, execution_time)
    
    def _extract_solution(self, prob, x_vars, sources, destinations, costs, execution_time):
        """Extract solution from solved LP problem"""
        
        status = pulp.LpStatus[prob.status]
        
        if prob.status == pulp.LpStatusOptimal:
            # Extract allocation matrix
            allocation = np.zeros((len(sources), len(destinations)))
            for i in sources:
                for j in destinations:
                    allocation[i, j] = x_vars[i, j].varValue or 0
            
            objective_value = pulp.value(prob.objective)
            
            # Calculate utilization metrics
            total_capacity = np.sum([x_vars[i, j].varValue or 0 
                                   for i in sources for j in destinations])
            
            solution = {
                'status': status,
                'allocation_matrix': allocation,
                'objective_value': objective_value,
                'total_allocation': total_capacity,
                'execution_time': execution_time,
                'solver_efficiency': self._calculate_solver_efficiency(allocation, costs)
            }
        else:
            solution = {
                'status': status,
                'allocation_matrix': None,
                'objective_value': None,
                'execution_time': execution_time,
                'error': f"Solver failed with status: {status}"
            }
        
        return solution
    
    def _extract_capacity_solution(self, prob, y_vars, x_vars, time_periods, capacity_options, execution_time):
        """Extract capacity planning solution"""
        
        status = pulp.LpStatus[prob.status]
        
        if prob.status == pulp.LpStatusOptimal:
            # Extract capacity allocation
            capacity_plan = np.zeros((time_periods, capacity_options))
            unmet_demand = np.zeros(time_periods)
            
            for t in range(time_periods):
                unmet_demand[t] = x_vars[t].varValue or 0
                for k in range(capacity_options):
                    capacity_plan[t, k] = y_vars[t, k].varValue or 0
            
            solution = {
                'status': status,
                'capacity_plan': capacity_plan,
                'unmet_demand': unmet_demand,
                'total_cost': pulp.value(prob.objective),
                'execution_time': execution_time,
                'service_level_achieved': np.sum(unmet_demand) == 0
            }
        else:
            solution = {
                'status': status,
                'capacity_plan': None,
                'total_cost': None,
                'execution_time': execution_time,
                'error': f"Solver failed with status: {status}"
            }
        
        return solution
    
    def _extract_multi_objective_solution(self, prob, x_vars, objectives, weights, execution_time):
        """Extract multi-objective solution"""
        
        status = pulp.LpStatus[prob.status]
        
        if prob.status == pulp.LpStatusOptimal:
            solution_values = [var.varValue or 0 for var in x_vars]
            
            # Calculate individual objective values
            individual_objectives = []
            for obj in objectives:
                obj_value = sum(obj[i] * solution_values[i] for i in range(len(solution_values)))
                individual_objectives.append(obj_value)
            
            solution = {
                'status': status,
                'solution': solution_values,
                'combined_objective': pulp.value(prob.objective),
                'individual_objectives': individual_objectives,
                'weights_used': weights,
                'execution_time': execution_time
            }
        else:
            solution = {
                'status': status,
                'solution': None,
                'execution_time': execution_time,
                'error': f"Solver failed with status: {status}"
            }
        
        return solution
    
    def _extract_network_flow_solution(self, prob, flow_vars, edges, execution_time):
        """Extract network flow solution"""
        
        status = pulp.LpStatus[prob.status]
        
        if prob.status == pulp.LpStatusOptimal:
            flows = {}
            for edge in edges:
                flows[edge] = flow_vars[edge].varValue or 0
            
            solution = {
                'status': status,
                'flows': flows,
                'total_cost': pulp.value(prob.objective),
                'execution_time': execution_time,
                'total_flow': sum(flows.values())
            }
        else:
            solution = {
                'status': status,
                'flows': None,
                'execution_time': execution_time,
                'error': f"Solver failed with status: {status}"
            }
        
        return solution
    
    def _add_constraint(self, prob, constraint, x_vars):
        """Add constraint to LP problem"""
        constraint_type = constraint.get('type')
        
        if constraint_type == 'linear_inequality':
            coefficients = constraint['coefficients']
            rhs = constraint['rhs']
            operator = constraint.get('operator', '<=')
            
            lhs = pulp.lpSum([coefficients[i] * x_vars[i] for i in range(len(coefficients))])
            
            if operator == '<=':
                prob += lhs <= rhs
            elif operator == '>=':
                prob += lhs >= rhs
            elif operator == '==':
                prob += lhs == rhs
        
        elif constraint_type == 'bound':
            var_index = constraint['variable']
            lower_bound = constraint.get('lower_bound')
            upper_bound = constraint.get('upper_bound')
            
            if lower_bound is not None:
                prob += x_vars[var_index] >= lower_bound
            if upper_bound is not None:
                prob += x_vars[var_index] <= upper_bound
    
    def _calculate_solver_efficiency(self, allocation, costs):
        """Calculate solver efficiency metrics"""
        if allocation is None:
            return 0.0
        
        total_cost = np.sum(allocation * costs)
        theoretical_min = np.sum(np.min(costs, axis=1))
        
        if theoretical_min > 0:
            efficiency = (theoretical_min / total_cost) * 100
        else:
            efficiency = 100.0
        
        return min(efficiency, 100.0)
