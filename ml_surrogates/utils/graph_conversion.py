"""
Graph Conversion Utilities for IRH ML Surrogate

THEORETICAL FOUNDATION: IRH v21.1 §1.2-1.3
Format conversion and compatibility utilities

Provides conversions between:
1. HolographicState ↔ NetworkX graphs
2. HolographicState ↔ dictionary representations
3. RG trajectories ↔ pandas DataFrames
4. Export/import for external tools

Enables integration with graph analysis libraries and data formats.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False
    print("Warning: NetworkX not available. Install with: pip install networkx")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not available. Install with: pip install pandas")

try:
    from ..engines import CouplingState, HolographicState
except (ImportError, ValueError):
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))
    from engines import CouplingState, HolographicState


def holographic_state_to_networkx(state: HolographicState) -> Any:
    """
    Convert HolographicState to NetworkX graph.
    
    Args:
        state: HolographicState to convert
        
    Returns:
        NetworkX DiGraph
        
    Raises:
        ImportError: If NetworkX not available
    """
    if not NETWORKX_AVAILABLE:
        raise ImportError("NetworkX required. Install with: pip install networkx")
    
    G = nx.DiGraph()
    
    # Add nodes
    for i, node in enumerate(state.graph_repr['nodes']):
        G.add_node(i, **node)
    
    # Add edges
    for edge in state.graph_repr['edges']:
        source = edge['source']
        target = edge['target']
        G.add_edge(source, target,
                  beta_lambda=edge['beta_lambda'],
                  beta_gamma=edge['beta_gamma'],
                  beta_mu=edge['beta_mu'])
    
    return G


def networkx_to_holographic_state(G: Any) -> HolographicState:
    """
    Convert NetworkX graph to HolographicState.
    
    Args:
        G: NetworkX DiGraph
        
    Returns:
        HolographicState
        
    Raises:
        ImportError: If NetworkX not available
    """
    if not NETWORKX_AVAILABLE:
        raise ImportError("NetworkX required. Install with: pip install networkx")
    
    # Get nodes in order
    nodes = sorted(G.nodes())
    
    # Create initial state from first node
    first_node = G.nodes[nodes[0]]
    initial_state = CouplingState(
        lambda_tilde=first_node['lambda_tilde'],
        gamma_tilde=first_node['gamma_tilde'],
        mu_tilde=first_node['mu_tilde'],
        k=first_node['k']
    )
    
    holo_state = HolographicState(initial_state)
    
    # Add subsequent states
    for i in range(1, len(nodes)):
        node = G.nodes[nodes[i]]
        new_state = CouplingState(
            lambda_tilde=node['lambda_tilde'],
            gamma_tilde=node['gamma_tilde'],
            mu_tilde=node['mu_tilde'],
            k=node['k']
        )
        
        # Get beta functions from edge if available
        if i > 0 and G.has_edge(nodes[i-1], nodes[i]):
            edge = G.edges[nodes[i-1], nodes[i]]
            beta_functions = (
                edge.get('beta_lambda', 0.0),
                edge.get('beta_gamma', 0.0),
                edge.get('beta_mu', 0.0)
            )
        else:
            beta_functions = None
        
        holo_state.add_rg_step(new_state, beta_functions)
    
    return holo_state


def trajectory_to_dataframe(state: HolographicState) -> Any:
    """
    Convert RG trajectory to pandas DataFrame.
    
    Args:
        state: HolographicState with trajectory
        
    Returns:
        pandas DataFrame with columns for each coupling and RG scale
        
    Raises:
        ImportError: If pandas not available
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas required. Install with: pip install pandas")
    
    data = []
    for i, node in enumerate(state.graph_repr['nodes']):
        row = {
            'step': i,
            'lambda_tilde': node['lambda_tilde'],
            'gamma_tilde': node['gamma_tilde'],
            'mu_tilde': node['mu_tilde'],
            'k': node['k']
        }
        data.append(row)
    
    # Add beta functions from edges if available
    for i, edge in enumerate(state.graph_repr['edges']):
        if i < len(data) - 1:
            data[i+1]['beta_lambda'] = edge['beta_lambda']
            data[i+1]['beta_gamma'] = edge['beta_gamma']
            data[i+1]['beta_mu'] = edge['beta_mu']
    
    return pd.DataFrame(data)


def dataframe_to_trajectory(df: Any) -> HolographicState:
    """
    Convert pandas DataFrame to RG trajectory.
    
    Args:
        df: DataFrame with coupling and RG scale columns
        
    Returns:
        HolographicState
        
    Raises:
        ImportError: If pandas not available
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas required. Install with: pip install pandas")
    
    # Create initial state from first row
    first_row = df.iloc[0]
    initial_state = CouplingState(
        lambda_tilde=first_row['lambda_tilde'],
        gamma_tilde=first_row['gamma_tilde'],
        mu_tilde=first_row['mu_tilde'],
        k=first_row['k']
    )
    
    holo_state = HolographicState(initial_state)
    
    # Add subsequent states
    for i in range(1, len(df)):
        row = df.iloc[i]
        new_state = CouplingState(
            lambda_tilde=row['lambda_tilde'],
            gamma_tilde=row['gamma_tilde'],
            mu_tilde=row['mu_tilde'],
            k=row['k']
        )
        
        # Get beta functions if available
        if 'beta_lambda' in row and not pd.isna(row['beta_lambda']):
            beta_functions = (
                row['beta_lambda'],
                row['beta_gamma'],
                row['beta_mu']
            )
        else:
            beta_functions = None
        
        holo_state.add_rg_step(new_state, beta_functions)
    
    return holo_state


def export_trajectory_csv(state: HolographicState, path: str) -> None:
    """
    Export RG trajectory to CSV file.
    
    Args:
        state: HolographicState to export
        path: Path to CSV file
    """
    df = trajectory_to_dataframe(state)
    df.to_csv(path, index=False)


def import_trajectory_csv(path: str) -> HolographicState:
    """
    Import RG trajectory from CSV file.
    
    Args:
        path: Path to CSV file
        
    Returns:
        HolographicState
    """
    if not PANDAS_AVAILABLE:
        raise ImportError("pandas required. Install with: pip install pandas")
    
    df = pd.read_csv(path)
    return dataframe_to_trajectory(df)


def export_trajectory_json(state: HolographicState, path: str) -> None:
    """
    Export RG trajectory to JSON file.
    
    Args:
        state: HolographicState to export
        path: Path to JSON file
    """
    import json
    
    data = {
        'nodes': state.graph_repr['nodes'],
        'edges': state.graph_repr['edges'],
        'metadata': {
            'num_nodes': len(state.graph_repr['nodes']),
            'num_edges': len(state.graph_repr['edges']),
            'fixed_point': state.fixed_point
        }
    }
    
    with open(path, 'w') as f:
        json.dump(data, f, indent=2)


def import_trajectory_json(path: str) -> HolographicState:
    """
    Import RG trajectory from JSON file.
    
    Args:
        path: Path to JSON file
        
    Returns:
        HolographicState
    """
    import json
    
    with open(path, 'r') as f:
        data = json.load(f)
    
    # Reconstruct from nodes
    nodes = data['nodes']
    first_node = nodes[0]
    
    initial_state = CouplingState(
        lambda_tilde=first_node['lambda_tilde'],
        gamma_tilde=first_node['gamma_tilde'],
        mu_tilde=first_node['mu_tilde'],
        k=first_node['k']
    )
    
    holo_state = HolographicState(initial_state)
    
    # Add subsequent states
    for i in range(1, len(nodes)):
        node = nodes[i]
        new_state = CouplingState(
            lambda_tilde=node['lambda_tilde'],
            gamma_tilde=node['gamma_tilde'],
            mu_tilde=node['mu_tilde'],
            k=node['k']
        )
        
        # Get beta functions from edges
        if i-1 < len(data['edges']):
            edge = data['edges'][i-1]
            beta_functions = (
                edge['beta_lambda'],
                edge['beta_gamma'],
                edge['beta_mu']
            )
        else:
            beta_functions = None
        
        holo_state.add_rg_step(new_state, beta_functions)
    
    return holo_state


def to_dict(state: HolographicState) -> Dict[str, Any]:
    """
    Convert HolographicState to plain dictionary.
    
    Args:
        state: HolographicState to convert
        
    Returns:
        Dictionary representation
    """
    return {
        'nodes': state.graph_repr['nodes'],
        'edges': state.graph_repr['edges'],
        'fixed_point': state.fixed_point,
        'trajectory_length': state.get_trajectory_length()
    }


def from_dict(data: Dict[str, Any]) -> HolographicState:
    """
    Create HolographicState from dictionary.
    
    Args:
        data: Dictionary representation
        
    Returns:
        HolographicState
    """
    nodes = data['nodes']
    first_node = nodes[0]
    
    initial_state = CouplingState(
        lambda_tilde=first_node['lambda_tilde'],
        gamma_tilde=first_node['gamma_tilde'],
        mu_tilde=first_node['mu_tilde'],
        k=first_node['k']
    )
    
    holo_state = HolographicState(initial_state)
    
    for i in range(1, len(nodes)):
        node = nodes[i]
        new_state = CouplingState(
            lambda_tilde=node['lambda_tilde'],
            gamma_tilde=node['gamma_tilde'],
            mu_tilde=node['mu_tilde'],
            k=node['k']
        )
        
        if i-1 < len(data['edges']):
            edge = data['edges'][i-1]
            beta_functions = (
                edge['beta_lambda'],
                edge['beta_gamma'],
                edge['beta_mu']
            )
        else:
            beta_functions = None
        
        holo_state.add_rg_step(new_state, beta_functions)
    
    return holo_state


# Example usage
if __name__ == "__main__":
    print("Testing Graph Conversion Utilities...")
    
    # Create test trajectory
    from engines import CouplingState, HolographicState
    
    initial = CouplingState(10.0, 10.0, 10.0, 1.0)
    trajectory = HolographicState(initial)
    
    for i in range(5):
        new_state = CouplingState(
            10.0 + i,
            10.0 + i * 0.5,
            10.0 + i * 0.3,
            1.0 - i * 0.1
        )
        trajectory.add_rg_step(new_state, beta_functions=(-0.5, -0.3, -0.2))
    
    # Test 1: NetworkX conversion
    if NETWORKX_AVAILABLE:
        print("\n1. Testing NetworkX conversion...")
        G = holographic_state_to_networkx(trajectory)
        print(f"  ✓ Converted to NetworkX: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        trajectory_back = networkx_to_holographic_state(G)
        print(f"  ✓ Converted back to HolographicState: {trajectory_back.get_trajectory_length()} steps")
    else:
        print("\n1. NetworkX not available (skipped)")
    
    # Test 2: pandas DataFrame conversion
    if PANDAS_AVAILABLE:
        print("\n2. Testing pandas DataFrame conversion...")
        df = trajectory_to_dataframe(trajectory)
        print(f"  ✓ Converted to DataFrame: {len(df)} rows, {len(df.columns)} columns")
        print(f"    Columns: {list(df.columns)}")
        
        trajectory_back = dataframe_to_trajectory(df)
        print(f"  ✓ Converted back to HolographicState: {trajectory_back.get_trajectory_length()} steps")
    else:
        print("\n2. pandas not available (skipped)")
    
    # Test 3: Dictionary conversion
    print("\n3. Testing dictionary conversion...")
    data_dict = to_dict(trajectory)
    print(f"  ✓ Converted to dict: {data_dict['trajectory_length']} steps")
    
    trajectory_back = from_dict(data_dict)
    print(f"  ✓ Converted back to HolographicState: {trajectory_back.get_trajectory_length()} steps")
    
    # Test 4: JSON export/import
    print("\n4. Testing JSON export/import...")
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        json_path = os.path.join(tmpdir, "trajectory.json")
        
        export_trajectory_json(trajectory, json_path)
        print(f"  ✓ Exported to JSON: {json_path}")
        
        trajectory_back = import_trajectory_json(json_path)
        print(f"  ✓ Imported from JSON: {trajectory_back.get_trajectory_length()} steps")
    
    # Test 5: CSV export/import
    if PANDAS_AVAILABLE:
        print("\n5. Testing CSV export/import...")
        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = os.path.join(tmpdir, "trajectory.csv")
            
            export_trajectory_csv(trajectory, csv_path)
            print(f"  ✓ Exported to CSV: {csv_path}")
            
            trajectory_back = import_trajectory_csv(csv_path)
            print(f"  ✓ Imported from CSV: {trajectory_back.get_trajectory_length()} steps")
    else:
        print("\n5. CSV export/import requires pandas (skipped)")
    
    print("\n✅ All graph conversion utilities tested successfully!")
