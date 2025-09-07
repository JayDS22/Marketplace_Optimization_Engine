from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

# Import optimization modules
from ..optimization.hungarian import HungarianMatchingOptimizer
from ..optimization.linear_programming import LinearProgrammingOptimizer  
from ..optimization.game_theory import GameTheoryOptimizer
from ..matching.bipartite_matching import BipartiteMatchingEngine

app = FastAPI(
    title="Marketplace Optimization API",
    description="API for marketplace optimization using OR and game theory",
    version="1.0.0"
)

# Pydantic models
class HungarianRequest(BaseModel):
    cost_matrix: List[List[float]]
    
class BipartiteMatchingRequest(BaseModel):
    edges: List[List[int]]  # List of [u, v] pairs
    left_nodes: List[int]
    right_nodes: List[int]
    
class GameTheoryRequest(BaseModel):
    payoff_matrix_1: List[List[float]]
    payoff_matrix_2: List[List[float]]

# Global optimizers
hungarian_optimizer = None
lp_optimizer = None
game_optimizer = None
matching_engine = None

@app.on_event("startup")
async def initialize_optimizers():
    global hungarian_optimizer, lp_optimizer, game_optimizer, matching_engine
    
    config = {
        'stability_threshold': 0.1,
        'nash_tolerance': 1e-6,
        'first_price_bid_factor': 0.8
    }
    
    hungarian_optimizer = HungarianMatchingOptimizer(config)
    lp_optimizer = LinearProgrammingOptimizer(config)
    game_optimizer = GameTheoryOptimizer(config)
    matching_engine = BipartiteMatchingEngine(config)
    
    logging.info("Optimization engines initialized")

@app.get("/")
async def root():
    return {"message": "Marketplace Optimization API", "version": "1.0.0"}

@app.post("/optimize/hungarian")
async def solve_hungarian(request: HungarianRequest):
    """Solve assignment problem using Hungarian algorithm"""
    try:
        cost_matrix = np.array(request.cost_matrix)
        
        row_indices, col_indices, total_cost = hungarian_optimizer.solve_assignment_problem(cost_matrix)
        
        # Create assignment pairs
        assignments = [(int(i), int(j)) for i, j in zip(row_indices, col_indices)]
        
        return {
            "assignments": assignments,
            "total_cost": float(total_cost),
            "efficiency": hungarian_optimizer._calculate_efficiency_metrics(
                cost_matrix, row_indices, col_indices, total_cost
            )
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/optimize/bipartite_matching")  
async def solve_bipartite_matching(request: BipartiteMatchingRequest):
    """Solve maximum bipartite matching"""
    try:
        edges = [(pair[0], pair[1]) for pair in request.edges]
        
        result = matching_engine.maximum_bipartite_matching(
            edges, request.left_nodes, request.right_nodes
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/optimize/nash_equilibrium")
async def find_nash_equilibrium(request: GameTheoryRequest):
    """Find Nash equilibrium for 2-player game"""
    try:
        payoff_1 = np.array(request.payoff_matrix_1)
        payoff_2 = np.array(request.payoff_matrix_2)
        
        result = game_optimizer.find_nash_equilibrium_2player(payoff_1, payoff_2)
        
        # Convert numpy arrays to lists for JSON serialization
        if result['status'] == 'success':
            result['player_1_strategy'] = result['player_1_strategy'].tolist()
            result['player_2_strategy'] = result['player_2_strategy'].tolist()
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
