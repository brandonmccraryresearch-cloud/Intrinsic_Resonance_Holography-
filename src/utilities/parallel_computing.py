"""
Parallel Computing Utilities for Intrinsic Resonance Holography v21.0

This module provides parallel computing infrastructure for
IRH computations.

Key Features:
    - Parallel map function
    - Distributed summation
    - Thread pool management

Authors: IRH Computational Framework Team
Last Updated: December 2025
"""

from typing import Callable, List, Any, Optional, Iterator
import numpy as np
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing


__version__ = "21.0.0"


# Theoretical Reference: IRH v21.4



def parallel_map(
    func: Callable,
    items: List[Any],
    n_workers: Optional[int] = None,
    use_processes: bool = False,
) -> List[Any]:
    """
    Apply function to items in parallel.
    
    Parameters
    ----------
    func : Callable
        Function to apply
    items : List
        Items to process
    n_workers : int, optional
        Number of workers (default: CPU count)
    use_processes : bool
        Use processes instead of threads
        
    Returns
    -------
    List
        Results
    """
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    
    # For small inputs, don't parallelize
    if len(items) < 2 * n_workers:
        return [func(item) for item in items]
    
    Executor = ProcessPoolExecutor if use_processes else ThreadPoolExecutor
    
    with Executor(max_workers=n_workers) as executor:
        results = list(executor.map(func, items))
    
    return results


# Theoretical Reference: IRH v21.4



def distributed_sum(
    func: Callable[[Any], float],
    items: Iterator[Any],
    n_workers: Optional[int] = None,
    chunk_size: int = 1000,
) -> float:
    """
    Compute distributed sum over items.
    
    Parameters
    ----------
    func : Callable
        Function returning float for each item
    items : Iterator
        Items to sum over
    n_workers : int, optional
        Number of workers
    chunk_size : int
        Items per chunk
        
    Returns
    -------
    float
        Total sum
    """
    if n_workers is None:
        n_workers = multiprocessing.cpu_count()
    
    items_list = list(items)
    n_items = len(items_list)
    
    # Create chunks
    chunks = [
        items_list[i:i + chunk_size]
        for i in range(0, n_items, chunk_size)
    ]
    
    def sum_chunk(chunk):
        """
        # Theoretical Reference: IRH v21.4
        """
        return sum(func(item) for item in chunk)
    
    partial_sums = parallel_map(sum_chunk, chunks, n_workers)
    
    return sum(partial_sums)


# Theoretical Reference: IRH v21.4



def batch_compute(
    func: Callable[[np.ndarray], np.ndarray],
    data: np.ndarray,
    batch_size: int = 1000,
    n_workers: Optional[int] = None,
) -> np.ndarray:
    """
    Apply function to batches of data in parallel.
    
    Parameters
    ----------
    func : Callable
        Function operating on arrays
    data : np.ndarray
        Input data
    batch_size : int
        Size of each batch
    n_workers : int, optional
        Number of workers
        
    Returns
    -------
    np.ndarray
        Results
    """
    n_samples = len(data)
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    batches = [
        data[i * batch_size:(i + 1) * batch_size]
        for i in range(n_batches)
    ]
    
    results = parallel_map(func, batches, n_workers)
    
    return np.concatenate(results)


# Theoretical Reference: IRH v21.4



def get_optimal_workers(data_size: int) -> int:
    """
    Get optimal number of workers for given data size.
    
    Parameters
    ----------
    data_size : int
        Size of data to process
        
    Returns
    -------
    int
        Optimal number of workers
    """
    cpu_count = multiprocessing.cpu_count()
    
    # Heuristic: use more workers for larger data
    if data_size < 100:
        return 1
    elif data_size < 1000:
        return min(4, cpu_count)
    elif data_size < 10000:
        return min(8, cpu_count)
    else:
        return cpu_count


__all__ = [
    'parallel_map',
    'distributed_sum',
    'batch_compute',
    'get_optimal_workers',
]
