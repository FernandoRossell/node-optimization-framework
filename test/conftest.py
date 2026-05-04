"""
conftest.py — fixtures compartidos para todo ``tests/``.

Asegura que ``ejecutable.py`` (un nivel arriba) sea importable y construye
**una sola vez** los datasets sintéticos reutilizados por las pruebas.
"""

import os
import sys
import pytest

# importamos ejecutable.py desde la carpeta del proyecto

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from ejecutable import (  # noqa: E402  (import after sys.path tweak)
    SyntheticDatasetGenerator,
    CostCalculator,
    GraphBuilder,
    GraphContext,
    RouteSolver,
)

# ----------------------------------------------------------------- fixtures
COST_PARAMS = {
    "weights": {"tiempo": 1.0, "riesgo": 6.0, "fuel_cost": 2.0},
    "normalize": "none",
}

# Replicamos lo que se haría en prod
def _build_ctx(nodos_df, routes_df):
    routes_costed = CostCalculator(COST_PARAMS).compute(routes_df)
    de = GraphBuilder.build_directed_edges(routes_costed)
    return nodos_df, de, GraphContext.from_dataframes(nodos_df, de)


@pytest.fixture(scope="session")
def main_dataset():
    """
    Dataset 30×30, sin border, seed=123.

    Pinned facts (verificados deterministamente):
      * BANK_001 está open in_grid.
      * ATM_010 está open in_grid.
      * BANK_002, BANK_003, BANK_014 están **closed** in_grid.
      * La arista canónica ((5, 5), (5, 6)) existe en ambos sentidos.
      * La arista one-way (7, 12) → (7, 13) **NO** existe (sólo la inversa).
    """
    nodos_df, routes_df, _ = SyntheticDatasetGenerator(
        grid_max_x=30, grid_max_y=30, seed=123
    ).generate()
    return _build_ctx(nodos_df, routes_df)


@pytest.fixture(scope="session")
def border_dataset():
    """Dataset 30×30 con caso borde aislado en (4, 4) <-> (5, 4)."""
    nodos_df, routes_df, _ = SyntheticDatasetGenerator(
        grid_max_x=30, grid_max_y=30, seed=123,
        border_cases=True, border_nodes=((4, 4), (5, 4)),
    ).generate()
    return _build_ctx(nodos_df, routes_df)


@pytest.fixture
def solver():
    """Solver canónico — el del primer ``solve_route`` del notebook."""
    return RouteSolver()
