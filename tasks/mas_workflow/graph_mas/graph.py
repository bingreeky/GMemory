import random
from typing import Literal
from dataclasses import dataclass

@dataclass
class GraphMaskInfo:

    fixed_spatial_masks: tuple[tuple[int]]
    fixed_temporal_masks: tuple[tuple[int]]  

    def __post_init__(self):

        self._validate_matrix(self.fixed_spatial_masks, "fixed_spatial_masks")
        self._validate_matrix(self.fixed_temporal_masks, "fixed_temporal_masks")

    def _validate_matrix(self, matrix: tuple[tuple[int]], name: str):
        size = len(matrix)

        for row in matrix:
            if len(row) != size:
                raise ValueError(f"{name} must be a square matrix")

            if not all(value in (0, 1) for value in row):
                raise ValueError(f"{name} must contain only 0 and 1")
            
def gen_graph_mask_info(mode: Literal['DirectAnswer', 'FullConnected', 'Random', 'Chain', 'Debate', 'Layered', 'Star'], N: int) -> GraphMaskInfo:
    
    def generate_layered_graph(N: int, layer_num: int = 2) -> tuple[tuple[int]]:

        adj_matrix = [[0] * N for _ in range(N)]
        base_size = N // layer_num
        remainder = N % layer_num
        layers = []
        for i in range(layer_num):
            size = base_size + (1 if i < remainder else 0)
            layers.extend([i] * size)
        random.shuffle(layers)
        for i in range(N):
            for j in range(N):
                if layers[j] == layers[i] + 1:
                    adj_matrix[i][j] = 1
        return tuple(tuple(row) for row in adj_matrix)
    
    def generate_star_graph(N: int) -> tuple[tuple[int]]:
        return tuple(tuple(1 if i == 0 and j > 0 else 0 for j in range(N)) for i in range(N))
    
    fixed_spatial_masks: tuple[tuple[int]] = tuple()
    fixed_temporal_masks: tuple[tuple[int]] = tuple()

    if mode == 'DirectAnswer':
        fixed_spatial_masks = ((0,),)
        fixed_temporal_masks = ((0,),)
    elif mode == 'FullConnected':
        fixed_spatial_masks = tuple(tuple(1 if i != j else 0 for i in range(N)) for j in range(N))
        fixed_temporal_masks = tuple(tuple(1 for _ in range(N)) for _ in range(N))
    elif mode == 'Random':
        fixed_spatial_masks = tuple(tuple(random.randint(0, 1) if i != j else 0 for i in range(N)) for j in range(N))
        fixed_temporal_masks = tuple(tuple(random.randint(0, 1) for _ in range(N)) for _ in range(N))
    elif mode == 'Chain':
        fixed_spatial_masks = tuple(tuple(1 if i == j + 1 else 0 for i in range(N)) for j in range(N))
        fixed_temporal_masks = tuple(tuple(1 if i == 0 and j == N - 1 else 0 for i in range(N)) for j in range(N))
    elif mode == 'Debate':
        fixed_spatial_masks = tuple(tuple(0 for _ in range(N)) for _ in range(N))
        fixed_temporal_masks = tuple(tuple(1 for _ in range(N)) for _ in range(N))
    elif mode == 'Layered':
        fixed_spatial_masks = generate_layered_graph(N)
        fixed_temporal_masks = tuple(tuple(1 for _ in range(N)) for _ in range(N))
    elif mode == 'Star':
        fixed_spatial_masks = generate_star_graph(N)
        fixed_temporal_masks = tuple(tuple(1 for _ in range(N)) for _ in range(N))
    else:
        raise ValueError('Unsupported Graph Connection Type.')
    
    return GraphMaskInfo(
        fixed_spatial_masks=fixed_spatial_masks,
        fixed_temporal_masks=fixed_temporal_masks
    )  