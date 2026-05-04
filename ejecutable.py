"""
ejecutable.py

Módulo para el Examen C11 — Optimización de rutas en grafos dirigidos ponderados
(caso de negocio: red de cajeros automáticos / bancos).

Clases principales
------------------
- 'SyntheticDatasetGenerator'  : construye 'nodos_df' y 'routes_df' reproducibles (incluye casos borde y OOB).

- 'CostCalculator'             : calcula el costo total escalar de cada arista.

- 'GraphBuilder'               : helpers estáticos para expandir aristas dirigidas, adjacency list y mapeos de nodos.

- 'GraphContext'               : contenedor reutilizable para múltiples consultas (cacheo).

- 'RouteResult'                : resultado tipado de cada consulta.

- 'BaseRouteSolver'            : base abstracta con el Dijkstra sobre el grafo de estados '(coord, k_waypoint, used_mask)'.
                                   Soporta **N aristas obligatorias** vía bitmask.

    - 'RouteSolver'            : solver canónico . Acepta 'nodes_df + directed_edges_df' o un 'ctx' pre-construido (modo cached implícito, usado por 'CaseManager').

- 'CaseManager'                : maneja el DataFrame de casos, casos de estrés y ejecución batch.

- 'BaseMapPlotter'             : plantilla con la lógica común de grilla, aristas y nodos.

    - 'SyntheticMapPlotter'    : mapa base (equivalente a 'plot_synthetic_map').
    - 'CaseRoutePlotter'       : mapa base + overlay de ruta para un caso (equivalente a 'plot_case_route').

- 'RiskHeatmapPlotter'         : heatmap de riesgo incidente por nodo.
"""

from __future__ import annotations

# Imports
import heapq
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch

# Aliases de tipo
Coord = Tuple[int, int]
Edge = Tuple[Coord, Coord]

# Nombres de columnas por defecto (centralizados — single source of truth)
NODE_ID_COL: str = "node_id"
NODE_TYPE_COL: str = "status"          # ATM / BANCO / NODO
NODE_OPEN_COL: str = "operativo"       # open / closed
NODE_X_COL: str = "x"
NODE_Y_COL: str = "y"
NODE_INGRID_COL: str = "in_grid"

ROUTE_ID_COL: str = "route_id"
FX, FY = "from_x", "from_y"
TX, TY = "to_x", "to_y"
DIR_COL: str = "direction"
RISK_COL: str = "riesgo"
TIME_COL: str = "tiempo"
FUEL_COL: str = "fuel_cost"

# Paleta formal (tonos sobrios, contraste claro, líneas finas)
PALETTE: Dict[str, str] = {
    # aristas base
    "two_way":      "#3b5b85",   # azul pizarra
    "one_way":      "#2e8b75",   # verde mar suave
    "blocked":      "#c8c8c8",   # gris claro
    # nodos
    "atm_open":     "#d97e2b",   # ámbar
    "bank_open":    "#3d4f7d",   # azul corporativo profundo
    "closed_edge":  "#a02828",   # rojo carmín apagado
    "border_edge":  "#bf9b30",   # dorado oliva
    # overlays de ruta / caso
    "route":        "#a83232",   # rojo terracota
    "mandatory":    "#c79a1c",   # dorado elegante
    "stop":         "#3f7e8c",   # verde-azulado profundo
    "origin":       "#111111",
    "destination":  "#111111",
    # grid
    "grid":         "#9c9c9c",
}


# 1) Generador de dataset sintético
class SyntheticDatasetGenerator:
    """
    Genera de manera reproducible un dataset sintético de:
      - 'nodos_df'  : ATMs + BANCOs (in-grid) + nodos OOB (fuera de la grilla).
      - 'routes_df' : aristas adyacentes (recta + diagonal) con dirección y costos.

    Mantiene la **misma lógica** de generación que la función 'generate_synthetic_datasets' del notebook original.

    Parámetros (constructor)
    -----------------------
    grid_max_x, grid_max_y : int
        Tamaño de la grilla cuadrada (incluye borde superior).

    n_atms, n_banks : int
        Número de cajeros y bancos a colocar.

    atm_open_prob, bank_open_prob : float
        Probabilidad de que un ATM/BANCO esté 'open'.

    pct_two_way, pct_one_way, pct_blocked : float
        Distribución de tipos de aristas. Deben sumar 1.0.

    seed : int
        Semilla para reproducibilidad.

    avoid_overlap_between_types : bool
        Si True, ATMs y BANCOs nunca comparten coordenadas.

    border_cases : bool
        Si True, fuerza un ATM y un BANCO en 'border_nodes' y
        bloquea todas las aristas incidentes (caso sin solución).

    border_nodes : tuple
        Pareja de coordenadas adyacentes para los nodos borde.

    border_operativo : tuple
        Estado operativo del ATM y BANCO de borde.
    """

    def __init__(
        self,
        grid_max_x: int = 10,
        grid_max_y: int = 10,
        n_atms: int = 20,
        n_banks: int = 20,
        atm_open_prob: float = 0.90,
        bank_open_prob: float = 0.90,
        pct_two_way: float = 0.70,
        pct_one_way: float = 0.25,
        pct_blocked: float = 0.05,
        seed: int = 42,
        avoid_overlap_between_types: bool = True,
        border_cases: bool = False,
        border_nodes: Tuple[Coord, Coord] = ((0, 0), (0, 1)),
        border_operativo: Tuple[str, str] = ("open", "open"),
    ) -> None:
        self.grid_max_x     = grid_max_x
        self.grid_max_y     = grid_max_y
        self.n_atms         = n_atms
        self.n_banks        = n_banks
        self.atm_open_prob  = atm_open_prob
        self.bank_open_prob = bank_open_prob
        self.pct_two_way    = pct_two_way
        self.pct_one_way    = pct_one_way
        self.pct_blocked    = pct_blocked
        self.seed           = seed
        self.border_cases   = border_cases
        self.border_nodes   = border_nodes
        self.border_operativo = border_operativo
        self.avoid_overlap_between_types = avoid_overlap_between_types

    # ------------------------------------------------------------------ utils
    def _in_bounds(self, x: int, y: int) -> bool:
        """True si (x, y) está dentro de la grilla."""
        return (0 <= x <= self.grid_max_x) and (0 <= y <= self.grid_max_y)

    @staticmethod
    def _sample_oob_coords(
        n: int, grid_max_x: int, grid_max_y: int, margin: int, rng: np.random.Generator
    ) -> np.ndarray:
        """Muestrea 'n' coordenadas enteras fuera de la grilla (OOB)."""
        coords: List[Coord] = []
        while len(coords) < n:
            x = int(rng.integers(-margin, grid_max_x + margin + 1))
            y = int(rng.integers(-margin, grid_max_y + margin + 1))
            if (x < 0) or (x > grid_max_x) or (y < 0) or (y > grid_max_y):
                coords.append((x, y))
        return np.array(coords, dtype=int)

    # ---------------------------------------------------------------- public
    def generate(self) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]:
        """
        Genera el dataset sintético.

        Returns
        -------
        nodos_df : pd.DataFrame
            Columnas: 'node_id, status, operativo, x, y, in_grid, case'.
        routes_df : pd.DataFrame
            Columnas: 'route_id, from_x, from_y, to_x, to_y, direction,
            riesgo, tiempo, fuel_cost'.
        summary : dict
            Estadísticas de la generación.
        """
        rng = np.random.default_rng(self.seed)

        # 1) Grilla
        xs = np.arange(0, self.grid_max_x + 1)
        ys = np.arange(0, self.grid_max_y + 1)
        nodes = [(x, y) for x in xs for y in ys]
        n_nodes = len(nodes)

        # 2) Aristas (8 vecinos, orden canónico)
        edges_set: Set[Edge] = set()
        for (x, y) in nodes:
            for dx in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    if dx == 0 and dy == 0:
                        continue
                    nx, ny = x + dx, y + dy
                    if self._in_bounds(nx, ny):
                        a, b = (x, y), (nx, ny)
                        u, v = (a, b) if a < b else (b, a)
                        edges_set.add((u, v))
        edges = list(edges_set)
        n_edges = len(edges)

        # 3) Direcciones
        if not np.isclose(self.pct_two_way + self.pct_one_way + self.pct_blocked, 1.0):
            raise ValueError(
                "pct_two_way + pct_one_way + pct_blocked debe sumar 1.0"
            )
        n_two_way = int(round(n_edges * self.pct_two_way))
        n_one_way = int(round(n_edges * self.pct_one_way))
        n_blocked = n_edges - n_two_way - n_one_way

        direction_base = np.array(
            [2] * n_two_way + [1] * n_one_way + [0] * n_blocked, dtype=int
        )
        rng.shuffle(direction_base)
        direction = direction_base.copy()
        one_way_mask = direction_base == 1
        direction[one_way_mask] = rng.choice([-1, +1], size=one_way_mask.sum())

        # 4) Costos aleatorios
        riesgo = rng.integers(0, 6, size=n_edges)
        tiempo = rng.integers(10, 61, size=n_edges)
        fuel_cost = rng.uniform(0.01, 1.0, size=n_edges)

        # 5) routes_df
        routes_df = pd.DataFrame({
            "from_x": [u[0] for (u, v) in edges],
            "from_y": [u[1] for (u, v) in edges],
            "to_x":   [v[0] for (u, v) in edges],
            "to_y":   [v[1] for (u, v) in edges],
            "direction": direction.astype(int),
            "riesgo":    riesgo.astype(int),
            "tiempo":    tiempo.astype(int),
            "fuel_cost": fuel_cost.astype(float),
        })

        # 6) Border case
        border_info: Optional[Dict[str, Any]] = None
        reserved_coords: Set[Coord] = set()
        if self.border_cases:
            (ax, ay), (bx, by) = self.border_nodes
            if not self._in_bounds(ax, ay) or not self._in_bounds(bx, by):
                raise ValueError("border_nodes está fuera del rango de la grilla.")
            if max(abs(ax - bx), abs(ay - by)) != 1:
                raise ValueError(
                    "border_nodes debe contener dos nodos adyacentes (recta o diagonal)."
                )
            reserved_coords = {(ax, ay), (bx, by)}
            incident_mask = (
                  ((routes_df["from_x"] == ax) & (routes_df["from_y"] == ay))
                | ((routes_df["to_x"]   == ax) & (routes_df["to_y"]   == ay))
                | ((routes_df["from_x"] == bx) & (routes_df["from_y"] == by))
                | ((routes_df["to_x"]   == bx) & (routes_df["to_y"]   == by))
            )
            routes_df.loc[incident_mask, "direction"] = 0

            u, v = (ax, ay), (bx, by)
            u_can, v_can = (u, v) if u < v else (v, u)
            border_info = {
                "atm_border_coord": (ax, ay),
                "bank_border_coord": (bx, by),
                "border_edge_canonical": (u_can, v_can),
                "blocked_incident_edges_count": int(incident_mask.sum()),
            }

        # 7) Muestreo de coords ATM/BANCO (in-grid)
        if self.border_cases:
            if self.n_atms < 1 or self.n_banks < 1:
                raise ValueError("Si border_cases=True, n_atms y n_banks deben ser >= 1.")
            n_atms_random = self.n_atms - 1
            n_banks_random = self.n_banks - 1

        else:
            n_atms_random = self.n_atms
            n_banks_random = self.n_banks

        available_nodes = [n for n in nodes if n not in reserved_coords]
        available_arr = np.array(available_nodes, dtype=int)

        if (
            n_atms_random + n_banks_random > len(available_nodes)
            and self.avoid_overlap_between_types
        ):
            raise ValueError(
                "No hay suficientes nodos para ubicar ATMs/Bancos sin traslape "
                "(considerando nodos reservados). Reduce n_atms/n_banks o "
                "permite traslape."
            )

        # ATMs in-grid
        if n_atms_random > 0:
            atm_idx = rng.choice(len(available_nodes), size=n_atms_random, replace=False)
            atm_coords = available_arr[atm_idx]
            atm_operativo = rng.choice(
                ["open", "closed"], size=n_atms_random,
                p=[self.atm_open_prob, 1 - self.atm_open_prob],
            )
            used_coords = set(map(tuple, atm_coords.tolist()))
        else:
            atm_coords = np.empty((0, 2), dtype=int)
            atm_operativo = np.array([], dtype=object)
            used_coords = set()

        # BANCOs in-grid
        if self.avoid_overlap_between_types:
            remaining_for_banks = [n for n in available_nodes if n not in used_coords]
            remaining_arr = np.array(remaining_for_banks, dtype=int)
            if n_banks_random > len(remaining_for_banks):
                raise ValueError("No hay suficientes nodos restantes para bancos sin traslape.")
            if n_banks_random > 0:
                bank_idx = rng.choice(len(remaining_for_banks), size=n_banks_random, replace=False)
                bank_coords = remaining_arr[bank_idx]
            else:
                bank_coords = np.empty((0, 2), dtype=int)
        else:
            if n_banks_random > 0:
                bank_idx = rng.choice(len(available_nodes), size=n_banks_random, replace=False)
                bank_coords = available_arr[bank_idx]
            else:
                bank_coords = np.empty((0, 2), dtype=int)

        if n_banks_random > 0:
            bank_operativo = rng.choice(
                ["open", "closed"], size=n_banks_random,
                p=[self.bank_open_prob, 1 - self.bank_open_prob],
            )
        else:
            bank_operativo = np.array([], dtype=object)

        # 8) Construcción del nodos_df base
        atms_rows: List[Dict[str, Any]] = []
        if self.border_cases:
            (ax, ay) = self.border_nodes[0]
            atms_rows.append({
                "node_id": "ATM_BORDER",
                "status": "ATM",
                "operativo": self.border_operativo[0],
                "x": ax, "y": ay,
            })
        for i in range(n_atms_random):
            atms_rows.append({
                "node_id": f"ATM_{i+1:03d}",
                "status": "ATM",
                "operativo": atm_operativo[i],
                "x": int(atm_coords[i, 0]),
                "y": int(atm_coords[i, 1]),
            })
        atms_df = pd.DataFrame(atms_rows)

        banks_rows: List[Dict[str, Any]] = []
        if self.border_cases:
            (bx, by) = self.border_nodes[1]
            banks_rows.append({
                "node_id": "BANK_BORDER",
                "status": "BANCO",
                "operativo": self.border_operativo[1],
                "x": bx, "y": by,
            })
        for i in range(n_banks_random):
            banks_rows.append({
                "node_id": f"BANK_{i+1:03d}",
                "status": "BANCO",
                "operativo": bank_operativo[i],
                "x": int(bank_coords[i, 0]),
                "y": int(bank_coords[i, 1]),
            })
        banks_df = pd.DataFrame(banks_rows)

        nodos_df = pd.concat([atms_df, banks_df], ignore_index=True)

        # 9) Resumen
        counts = routes_df["direction"].value_counts().to_dict()
        two_way_cnt = counts.get(2, 0)
        one_way_cnt = counts.get(-1, 0) + counts.get(1, 0)
        blocked_cnt = counts.get(0, 0)
        summary: Dict[str, Any] = {
            "grid_nodes": n_nodes,
            "routes_edges": n_edges,
            "nodos_total": len(nodos_df),
            "n_atms": int((nodos_df["status"] == "ATM").sum()),
            "n_bancos": int((nodos_df["status"] == "BANCO").sum()),
            "direction_counts": {
                "two_way(2)": two_way_cnt,
                "one_way(+1/-1)": one_way_cnt,
                "blocked(0)": blocked_cnt,
            },
            "direction_shares": {
                "two_way(2)": two_way_cnt / n_edges,
                "one_way(+1/-1)": one_way_cnt / n_edges,
                "blocked(0)": blocked_cnt / n_edges,
            },
            "border_info": border_info,
        }

        # 10) ID de ruta
        routes_df.insert(
            0, "route_id", [f"RTE_{i:06d}" for i in range(1, len(routes_df) + 1)]
        )

        # 11) Casos OOB explícitos
        n_atms_oob, n_banks_oob, n_generic_oob, oob_margin = 3, 3, 5, 5

        if "in_grid" not in nodos_df.columns:
            nodos_df["in_grid"] = True
        if "case" not in nodos_df.columns:
            nodos_df["case"] = "NORMAL"

        atm_oob_coords = self._sample_oob_coords(n_atms_oob,  self.grid_max_x, self.grid_max_y, oob_margin, rng)
        bank_oob_coords = self._sample_oob_coords(n_banks_oob, self.grid_max_x, self.grid_max_y, oob_margin, rng)
        gen_oob_coords = self._sample_oob_coords(n_generic_oob, self.grid_max_x, self.grid_max_y, oob_margin, rng)

        atm_oob_oper = rng.choice(["open", "closed"], size=n_atms_oob,  p=[self.atm_open_prob,  1 - self.atm_open_prob])
        bank_oob_oper = rng.choice(["open", "closed"], size=n_banks_oob, p=[self.bank_open_prob, 1 - self.bank_open_prob])
        gen_oob_oper = rng.choice(["open", "closed"], size=n_generic_oob, p=[0.90, 0.10])

        atms_oob_df = pd.DataFrame({
            "node_id":   [f"ATM_OOB_{i:03d}" for i in range(1, n_atms_oob + 1)],
            "status":    "ATM",
            "operativo": atm_oob_oper,
            "x": atm_oob_coords[:, 0], "y": atm_oob_coords[:, 1],
            "in_grid": False, "case": "OOB",
        })
        banks_oob_df = pd.DataFrame({
            "node_id":   [f"BANK_OOB_{i:03d}" for i in range(1, n_banks_oob + 1)],
            "status":    "BANCO",
            "operativo": bank_oob_oper,
            "x": bank_oob_coords[:, 0], "y": bank_oob_coords[:, 1],
            "in_grid": False, "case": "OOB",
        })
        generic_oob_df = pd.DataFrame({
            "node_id":   [f"NODE_OOB_{i:03d}" for i in range(1, n_generic_oob + 1)],
            "status":    "NODO",
            "operativo": gen_oob_oper,
            "x": gen_oob_coords[:, 0], "y": gen_oob_coords[:, 1],
            "in_grid": False, "case": "OOB",
        })

        nodos_df = pd.concat(
            [nodos_df, atms_oob_df, banks_oob_df, generic_oob_df], ignore_index=True
        )
        return nodos_df, routes_df, summary


# 2) Calculadora de costo
class CostCalculator:
    """
    Calcula el costo total escalar por arista a partir de pesos lineales y
    términos de interacción (product / min / max).

    Parámetros
    ----------
    cost_params : dict
        Dict con claves opcionales:
          - 'weights' (dict[str, float]): peso por variable.
          - 'interactions' (list[dict]): cada dict con
            'vars=(a,b)', 'coef', 'type' ∈ {product, min, max}.
          - 'normalize' ∈ {none, minmax, zscore}.
    """

    SUPPORTED_INTERACTIONS = ("product", "min", "max")

    def __init__(self, cost_params: Dict[str, Any]) -> None:
        self.cost_params = cost_params or {}

    @staticmethod
    def _apply_interaction(a: float, b: float, kind: str) -> float:
        """Aplica una interacción puntual a dos floats. (Helper escalar.)"""
        if kind == "product":
            return a * b
        if kind == "min":
            return min(a, b)
        if kind == "max":
            return max(a, b)
        raise ValueError(f"Tipo de interacción no soportado: {kind}")

    def compute(self, routes_df: pd.DataFrame, out_col: str = "total_cost") -> pd.DataFrame:
        """
        Devuelve una **copia** de 'routes_df' con la columna 'out_col' que
        contiene el costo total escalar (no negativo).

        Parameters
        ----------
        routes_df : pd.DataFrame
        out_col : str

        Returns
        -------
        pd.DataFrame
        """
        df = routes_df.copy()
        weights: Dict[str, float] = self.cost_params.get("weights", {})
        interactions: List[Dict[str, Any]] = self.cost_params.get("interactions", [])
        normalize: str = self.cost_params.get("normalize", "none")

        # Validaciones
        for var in weights.keys():
            if var not in df.columns:
                raise KeyError(f"Falta columna '{var}' para calcular costo total.")
        for inter in interactions:
            v1, v2 = inter["vars"]
            if v1 not in df.columns or v2 not in df.columns:
                raise KeyError(f"Faltan columnas para interacción: {v1}, {v2}")

        X = df[list(weights.keys())].astype(float)

        if normalize == "minmax":
            mins, maxs = X.min(), X.max()
            denom = (maxs - mins).replace(0, 1.0)
            Xn = (X - mins) / denom
        elif normalize == "zscore":
            mu = X.mean()
            sigma = X.std(ddof=0).replace(0, 1.0)
            Xn = (X - mu) / sigma
        else:
            Xn = X

        total = np.zeros(len(df), dtype=float)
        for var, w in weights.items():
            total += w * Xn[var].values

        for inter in interactions:
            v1, v2 = inter["vars"]
            coef = float(inter.get("coef", 0.0))
            kind = inter.get("type", "product")
            a = df[v1].astype(float).values
            b = df[v2].astype(float).values
            if kind == "product":
                total += coef * (a * b)
            elif kind == "min":
                total += coef * np.minimum(a, b)
            elif kind == "max":
                total += coef * np.maximum(a, b)
            else:
                raise ValueError(f"Tipo de interacción no soportado: {kind}")

        df[out_col] = np.clip(total, 0.0, None)
        return df


# 3) Helpers de grafo
class GraphBuilder:
    """
    Conjunto de helpers estáticos para transformar las tablas crudas
    ('routes_df' no dirigido + 'nodes_df') en estructuras eficientes
    para el solver:

      - 'build_directed_edges' : expande el 'direction' a aristas dirigidas.
      - 'build_adjacency'      : adjacency list 'coord -> [(coord, cost, meta)]'.
      - 'build_node_maps'      : 'id2coord' y 'coord2info'.
    """

    @staticmethod
    def build_directed_edges(routes_df: pd.DataFrame) -> pd.DataFrame:
        """
        Expande según 'direction':
          2  -> A->B y B->A (ruta inversa con 'route_id' sufijo '_REV')
          1  -> A->B
         -1  -> B->A
          0  -> bloqueada (no genera aristas)

        Returns
        -------
        pd.DataFrame
            Mismas columnas que 'routes_df' más 'src_x, src_y, dst_x, dst_y, dir_expanded'.
        """
        df = routes_df.copy()
        required = [ROUTE_ID_COL, FX, FY, TX, TY, DIR_COL]
        for c in required:
            if c not in df.columns:
                raise KeyError(f"Falta columna '{c}' en routes_df.")

        rows: List[Dict[str, Any]] = []
        for _, r in df.iterrows():
            rid = r[ROUTE_ID_COL]
            a = (int(r[FX]), int(r[FY]))
            b = (int(r[TX]), int(r[TY]))
            d = int(r[DIR_COL])
            base = r.to_dict()

            if d == 0:
                continue
            elif d == 2:
                out1 = base.copy()
                out1.update({"src_x": a[0], "src_y": a[1], "dst_x": b[0], "dst_y": b[1], "dir_expanded": "A2B"})
                rows.append(out1)
                out2 = base.copy()
                out2.update({"src_x": b[0], "src_y": b[1], "dst_x": a[0], "dst_y": a[1], "dir_expanded": "B2A", "route_id": f"{rid}_REV"})
                rows.append(out2)
            elif d == 1:
                out = base.copy()
                out.update({"src_x": a[0], "src_y": a[1], "dst_x": b[0], "dst_y": b[1], "dir_expanded": "A2B"})
                rows.append(out)
            elif d == -1:
                out = base.copy()
                out.update({"src_x": b[0], "src_y": b[1], "dst_x": a[0], "dst_y": a[1], "dir_expanded": "B2A"})
                rows.append(out)
            else:
                raise ValueError(f"direction inválido: {d} en route_id={rid}")
        return pd.DataFrame(rows)

    @staticmethod
    def build_adjacency(
        directed_edges_df: pd.DataFrame, cost_col: str = "total_cost"
    ) -> Dict[Coord, List[Tuple[Coord, float, Dict[str, Any]]]]:
        """
        Adjacency list: 'adj[(x, y)] = [((nx, ny), cost, meta_dict), ...]'.
        """
        needed = ["src_x", "src_y", "dst_x", "dst_y", cost_col, ROUTE_ID_COL]
        for c in needed:
            if c not in directed_edges_df.columns:
                raise KeyError(f"Falta columna '{c}' en directed_edges_df.")

        adj: Dict[Coord, List[Tuple[Coord, float, Dict[str, Any]]]] = {}
        for _, r in directed_edges_df.iterrows():
            src = (int(r["src_x"]), int(r["src_y"]))
            dst = (int(r["dst_x"]), int(r["dst_y"]))
            cost = float(r[cost_col])
            meta = {
                "route_id": r[ROUTE_ID_COL],
                "riesgo":    float(r.get(RISK_COL, np.nan)),
                "tiempo":    float(r.get(TIME_COL, np.nan)),
                "fuel_cost": float(r.get(FUEL_COL, np.nan)),
            }
            adj.setdefault(src, []).append((dst, cost, meta))
        return adj

    @staticmethod
    def build_node_maps(
        nodes_df: pd.DataFrame,
    ) -> Tuple[Dict[str, Coord], Dict[Coord, Dict[str, Any]]]:
        """
        Construye:
          - 'id2coord'    : 'node_id -> (x, y)'.
          - 'coord2info'  : '(x, y) -> {in_grid, open, types, node_ids}'.
        """
        required = [NODE_ID_COL, NODE_X_COL, NODE_Y_COL, NODE_OPEN_COL, NODE_INGRID_COL]
        for c in required:
            if c not in nodes_df.columns:
                raise KeyError(f"Falta columna '{c}' en nodes_df.")

        id2coord: Dict[str, Coord] = {}
        coord2info: Dict[Coord, Dict[str, Any]] = {}

        for _, r in nodes_df.iterrows():
            nid = str(r[NODE_ID_COL])
            coord = (int(r[NODE_X_COL]), int(r[NODE_Y_COL]))
            id2coord[nid] = coord
            if coord not in coord2info:
                coord2info[coord] = {
                    "in_grid": bool(r[NODE_INGRID_COL]),
                    "open": str(r[NODE_OPEN_COL]).lower() == "open",
                    "types": {str(r.get(NODE_TYPE_COL, ""))},
                    "node_ids": [nid],
                }
            else:
                coord2info[coord]["types"].add(str(r.get(NODE_TYPE_COL, "")))
                coord2info[coord]["node_ids"].append(nid)
                coord2info[coord]["open"] = (
                    coord2info[coord]["open"]
                    or (str(r[NODE_OPEN_COL]).lower() == "open")
                )
                coord2info[coord]["in_grid"] = (
                    coord2info[coord]["in_grid"] and bool(r[NODE_INGRID_COL])
                )
        return id2coord, coord2info


# 4) Helpers operacionales
def normalize_operational_status(
    value: Any,
    open_values: Optional[Iterable[str]] = None,
    closed_values: Optional[Iterable[str]] = None,
    default_open: bool = False,
) -> bool:
    """
    Convierte un valor (string libre) de 'operativo' a booleano (utilizable / no).

    Parameters
    ----------
    value : Any
    open_values, closed_values : Iterable[str], optional
    default_open : bool
        Política para valores desconocidos. Por defecto, 'False' (fail-safe).

    Returns
    -------
    bool
    """
    if open_values is None:
        open_values = {"open", "abierto", "activo", "active", "operational", "ok", "enabled"}
    if closed_values is None:
        closed_values = {
            "closed", "cerrado", "inactivo", "inactive", "maintenance", "mantenimiento",
            "down", "blocked", "fuera_de_servicio", "out_of_service",
        }
    if value is None:
        return default_open
    v = str(value).strip().lower()
    if v in open_values:
        return True
    if v in closed_values:
        return False
    return default_open


# 5) Contexto de grafo (cacheable y reutilizable)
@dataclass
class GraphContext:
    """
    Contexto reutilizable construido **una sola vez** sobre el grafo y
    consultable por cualquier solver.

    Attributes
    ----------
    id2coord :       Dict[str, Coord]
    coord_flags :    Dict[Coord, Dict[str, Any]]
    adj :            Dict[Coord, List[Tuple[Coord, float, Dict[str, Any]]]]
    edge_set :       Set[Edge]
    edges_pool :     List[Edge]
    atms_open_ids :  List[str]
    banks_open_ids : List[str]
    cost_col :       str
    """

    id2coord:       Dict[str, Coord]
    coord_flags:    Dict[Coord, Dict[str, Any]]
    adj:            Dict[Coord, List[Tuple[Coord, float, Dict[str, Any]]]]
    edge_set:       Set[Edge]
    edges_pool:     List[Edge]
    atms_open_ids:  List[str]
    banks_open_ids: List[str]
    cost_col:       str = "total_cost"

    # ---- helpers ------------------------------------------------------
    def _coord_is_usable(self, c: Coord) -> bool:
        """True si 'c' está abierta y dentro de la grilla (o si no figura
        en el mapa de nodos, en cuyo caso se considera transitable)."""
        info = self.coord_flags.get(c)
        if info is None:
            return True
        return bool(info["in_grid"]) and bool(info["open"])

    # ---- factory ------------------------------------------------------
    @classmethod
    def from_dataframes(
        cls,
        nodes_df: pd.DataFrame,
        directed_edges_df: pd.DataFrame,
        *,
        node_id_col: str = NODE_ID_COL,
        node_type_col: str = NODE_TYPE_COL,
        operativo_col: str = NODE_OPEN_COL,
        in_grid_col: str = NODE_INGRID_COL,
        x_col: str = NODE_X_COL,
        y_col: str = NODE_Y_COL,
        cost_col: str = "total_cost",
        open_values: Optional[Iterable[str]] = None,
        closed_values: Optional[Iterable[str]] = None,
        default_open_unknown: bool = False,
    ) -> "GraphContext":
        """
        Construye el contexto a partir del 'nodes_df' y el
        'directed_edges_df' ya expandido por 'GraphBuilder.build_directed_edges'
        y costeado por 'CostCalculator'.

        Returns
        -------
        GraphContext
        """
        # validaciones
        required_nodes = {node_id_col, node_type_col, operativo_col, in_grid_col, x_col, y_col}
        missing_nodes = required_nodes - set(nodes_df.columns)
        if missing_nodes:
            raise KeyError(f"nodes_df: faltan columnas {missing_nodes}")

        required_edges = {"src_x", "src_y", "dst_x", "dst_y", cost_col}
        missing_edges = required_edges - set(directed_edges_df.columns)
        if missing_edges:
            raise KeyError(
                f"directed_edges_df: faltan columnas {missing_edges}. "
                f"¿Ya expandiste direction a src/dst con GraphBuilder.build_directed_edges?"
            )

        id2coord: Dict[str, Coord] = {}
        coord_flags: Dict[Coord, Dict[str, Any]] = {}

        for _, r in nodes_df.iterrows():
            nid = str(r[node_id_col])
            coord = (int(r[x_col]), int(r[y_col]))
            id2coord[nid] = coord
            open_flag = normalize_operational_status(
                r[operativo_col],
                open_values=open_values,
                closed_values=closed_values,
                default_open=default_open_unknown,
            )
            in_grid = bool(r[in_grid_col])
            ntype = str(r[node_type_col]).strip().upper()

            if coord not in coord_flags:
                coord_flags[coord] = {
                    "in_grid": in_grid,
                    "open": open_flag,
                    "types": {ntype},
                    "node_ids": [nid],
                }
            else:
                coord_flags[coord]["in_grid"] = coord_flags[coord]["in_grid"] and in_grid
                coord_flags[coord]["open"] = coord_flags[coord]["open"] or open_flag
                coord_flags[coord]["types"].add(ntype)
                coord_flags[coord]["node_ids"].append(nid)

        # adjacency + edge_set
        adj: Dict[Coord, List[Tuple[Coord, float, Dict[str, Any]]]] = {}
        edge_set: Set[Edge] = set()

        route_id_col    = "route_id"  if "route_id"  in directed_edges_df.columns else None
        risk_col        = "riesgo"    if "riesgo"    in directed_edges_df.columns else None
        time_col        = "tiempo"    if "tiempo"    in directed_edges_df.columns else None
        fuel_col        = "fuel_cost" if "fuel_cost" in directed_edges_df.columns else None

        for _, r in directed_edges_df.iterrows():
            src = (int(r["src_x"]), int(r["src_y"]))
            dst = (int(r["dst_x"]), int(r["dst_y"]))
            w = float(r[cost_col])
            edge_set.add((src, dst))
            meta = {
                "route_id": r[route_id_col] if route_id_col else None,
                "riesgo":    float(r[risk_col]) if risk_col else None,
                "tiempo":    float(r[time_col]) if time_col else None,
                "fuel_cost": float(r[fuel_col]) if fuel_col else None,
            }
            adj.setdefault(src, []).append((dst, w, meta))

        # nodes_open por tipo (para muestreo en stress)
        nodes_in_grid = nodes_df[nodes_df[in_grid_col] == True].copy()
        open_mask = nodes_in_grid[operativo_col].apply(
            lambda x: normalize_operational_status(
                x, open_values=open_values, closed_values=closed_values,
                default_open=default_open_unknown,
            )
        )
        nodes_open = nodes_in_grid[open_mask].copy()
        type_upper = nodes_open[node_type_col].astype(str).str.upper()
        atms_open_ids = nodes_open[type_upper.isin(["ATM"])][node_id_col].astype(str).tolist()
        banks_open_ids = nodes_open[type_upper.isin(["BANCO", "BANK"])][node_id_col].astype(str).tolist()

        return cls(
            id2coord=id2coord,
            coord_flags=coord_flags,
            adj=adj,
            edge_set=edge_set,
            edges_pool=list(edge_set),
            atms_open_ids=atms_open_ids,
            banks_open_ids=banks_open_ids,
            cost_col=cost_col,
        )


# 6) Resultado tipado
@dataclass
class RouteResult:
    """
    Resultado tipado de una consulta.

    Attributes
    ----------
    feasible : bool
    total_cost : Optional[float]
    path_coords : List[Coord]
    path_edges : List[Edge]
    reason : Optional[str]
        Código de error si infeasible (p.ej. 'MANDATORY_EDGE_MISSING').
    mandatory_edge_in_path : bool
        True si **todas** las aristas obligatorias fueron transitadas.
        (Si sólo se pasó una arista, equivalente al booleano clásico.)
    mandatory_edges_in_path : List[bool]
        Bandera por arista obligatoria (orden = orden de entrada).
    metrics : Dict[str, Any]
        'runtime_ms', 'expanded_states', 'hops', ...
    audit : Optional[Dict[str, Any]]
    """

    feasible: bool
    total_cost: Optional[float]
    path_coords: List[Coord]
    path_edges: List[Edge]
    reason: Optional[str]
    mandatory_edge_in_path: bool
    metrics: Dict[str, Any]
    mandatory_edges_in_path: List[bool] = field(default_factory=list)
    audit: Optional[Dict[str, Any]] = None


# 7) Solver base + variantes (HERENCIA)
class BaseRouteSolver:
    """
    Núcleo abstracto del algoritmo.

    Implementa **Dijkstra** sobre un grafo de estados
    '(coord, k, used_mask)' donde:

      - 'coord'     : posición actual,
      - 'k'         : índice del último waypoint alcanzado en orden,
      - 'used_mask' : bitmask de 'N' bits con el i-ésimo bit en 1 si la
                        arista obligatoria 'i' ya fue transitada.

    Generaliza el original (que sólo soportaba 1 arista obligatoria) a un
    número arbitrario 'N' de aristas mediante el bitmask. La condición de
    meta es 'coord == dest and k == target_k and used_mask == (1<<N) - 1'.

    Las subclases concretas (p. ej. :class:`RouteSolver`) deciden cómo se
    construye o se recibe el 'GraphContext'; la lógica algorítmica vive
    una sola vez aquí.
    """

    def __init__(self) -> None:
        pass

    # ------------------------------------------------------------------
    # Núcleo del algoritmo (protected): asume mandatory_edges_list ya
    # validada y un GraphContext listo. CaseManager y RouteSolver lo llaman.
    # ------------------------------------------------------------------
    def _solve(
        self,
        ctx: GraphContext,
        origin_node_id: str,
        destination_node_id: str,
        mandatory_edges_list: List[Edge],
        mandatory_stops: Optional[List[str]] = None,
        forbidden_nodes: Optional[Set[Coord]] = None,
        forbidden_edges: Optional[Set[Edge]] = None,
        tie_breaker: Optional[str] = None,
        audit: bool = False,
    ) -> RouteResult:
        """
        Dijkstra sobre el grafo de estados '(coord, k, used_mask)'.

        Parameters
        ----------
        ctx : GraphContext
        origin_node_id, destination_node_id : str
        mandatory_edges_list : List[Edge]
            Lista de 1..N aristas obligatorias; el orden define la posición
            de cada bit en 'used_mask'.
        mandatory_stops : Optional[List[str]]
            'node_id' a visitar **en orden**.
        forbidden_nodes : Optional[Set[Coord]]
        forbidden_edges : Optional[Set[Edge]]
        tie_breaker : Optional[str]
            'None' (solo costo) o '"min_hops"'.
        audit : bool

        Returns
        -------
        RouteResult
        """
        t0 = time.perf_counter()

        mandatory_stops = mandatory_stops or []
        forbidden_nodes = forbidden_nodes or set()
        forbidden_edges = forbidden_edges or set()

        id2coord = ctx.id2coord
        adj = ctx.adj
        edge_set = ctx.edge_set
        _coord_is_usable = ctx._coord_is_usable

        N = len(mandatory_edges_list)
        full_mask = (1 << N) - 1 if N > 0 else 0
        # Mapa edge -> bit index (para máscara rápida en cada relax)
        edge2bit: Dict[Edge, int] = {e: i for i, e in enumerate(mandatory_edges_list)}

        # ---------- Validaciones de entrada ----------
        def _fail(reason: str, **extra) -> RouteResult:
            metrics = {"runtime_ms": (time.perf_counter() - t0) * 1000}
            metrics.update(extra)
            return RouteResult(
                feasible=False, total_cost=None, path_coords=[], path_edges=[],
                reason=reason, mandatory_edge_in_path=False,
                metrics=metrics, mandatory_edges_in_path=[False] * N, audit=None,
            )

        if origin_node_id not in id2coord:
            return _fail("ORIGIN_NOT_FOUND")
        if destination_node_id not in id2coord:
            return _fail("DEST_NOT_FOUND")

        origin = id2coord[origin_node_id]
        dest = id2coord[destination_node_id]

        if not _coord_is_usable(origin):
            return _fail("ORIGIN_CLOSED_OR_OOB")
        if not _coord_is_usable(dest):
            return _fail("DEST_CLOSED_OR_OOB")

        # stops -> coords
        stop_coords: List[Coord] = []
        for s in mandatory_stops:
            if s not in id2coord:
                return _fail("STOP_NOT_FOUND", bad_stop=s)
            c = id2coord[s]
            if not _coord_is_usable(c):
                return _fail("STOP_CLOSED_OR_OOB", bad_stop=s)
            stop_coords.append(c)

        # Validar cada arista obligatoria
        for i, (u, v) in enumerate(mandatory_edges_list):
            if (u, v) in forbidden_edges:
                return _fail("MANDATORY_EDGE_FORBIDDEN", bad_edge_index=i, bad_edge=(u, v))
            if (u, v) not in edge_set:
                return _fail("MANDATORY_EDGE_MISSING", bad_edge_index=i, bad_edge=(u, v))
            if not _coord_is_usable(u) or not _coord_is_usable(v):
                return _fail("MANDATORY_EDGE_ENDPOINT_OOB_OR_CLOSED",
                             bad_edge_index=i, bad_edge=(u, v))

        # ---------- Dijkstra sobre (coord, k, used_mask) ----------
        waypoints = [origin] + stop_coords + [dest]     # Secuencia obligatoria de coordenadas
        target_k = len(waypoints) - 1

        def advance_k(coord: Coord, k: int) -> int:     # Si llegamos al waypóint esperado avanzamos, si no nos quedamos en k
            if k < target_k and coord == waypoints[k + 1]:
                return k + 1
            return k

        # Inicializamos el algoritmo sobre el grafo de estados
        start_state: Tuple[Coord, int, int] = (origin, 0, 0)
        dist: Dict[Tuple[Coord, int, int], Tuple[float, int]] = {start_state: (0.0, 0)}
        prev: Dict[Tuple[Coord, int, int], Tuple[Tuple[Coord, int, int], Edge]] = {}
        pq: List[Tuple[float, int, Coord, int, int]] = [(0.0, 0, origin, 0, 0)]
        expanded = 0

        # Loop principal: Dijkstra sobre el grafo de estados
        while pq:
            cur_cost, cur_hops, coord, k, used_mask = heapq.heappop(pq) # Ebtenemos losa elementos mas pequeños de pq
            state = (coord, k, used_mask)                               # creamos un nuevo estado

            best = dist.get(state)
            if best is None:
                continue

            # Chequeo de dominancia;  poda de caminos subóptimos
            best_cost, best_hops = best     
            if cur_cost > best_cost or (cur_cost == best_cost and cur_hops > best_hops):
                continue
                
            # Contador de cuantos estados realmente exploró el algoritmo    
            expanded += 1

            # Meta: estar en destino, último waypoint y todas las mandatorias usadas
            if coord == dest and k == target_k and used_mask == full_mask:
                break

            # Filtros de estados inválidos    
            if coord in forbidden_nodes:        # Estados que hay que evitar
                continue
            if not _coord_is_usable(coord):     # Estados usables
                continue

            # Expansión de vecinos para saltarnos el estado   
            for (nbr, w, _meta) in adj.get(coord, []):
                edge = (coord, nbr)
                
                # Evita movimientos inválidos.
                if edge in forbidden_edges:
                    continue

                if nbr in forbidden_nodes:
                    continue

                if not _coord_is_usable(nbr):
                    continue

                # Si la arista es obligatoria -> se prende su bit
                # Si no -> la máscara queda igual

                # Esto es el núcleo del soporte multi‑mandatoria.
                bit = edge2bit.get(edge)
                used_mask2 = used_mask | (1 << bit) if bit is not None else used_mask

                # Verifica si con este movimiento se cumple el siguiente waypoint.
                k2 = advance_k(nbr, k)

                # Cálculo de costo y hops
                new_cost = cur_cost + float(w)
                new_hops = cur_hops + 1
                st2 = (nbr, k2, used_mask2)

                # Minimiza costo
                
                old = dist.get(st2)
                better = False
                
                if old is None:
                    better = True

                else:
                    old_cost, old_hops = old
                    if new_cost < old_cost:
                        better = True
                    
                    # En empate, usamos la ruta con menos aristas
                    elif new_cost == old_cost and tie_breaker == "min_hops" and new_hops < old_hops:
                        better = True

                # Actualización de estructuras
                # Guarda el mejor camino al nuevo estado
                # Registra cómo llegaste (para reconstrucción)
                
                if better:
                    dist[st2] = (new_cost, new_hops)
                    prev[st2] = (state, edge)
                    heapq.heappush(pq, (new_cost, new_hops, nbr, k2, used_mask2))

        # ---------- Final ----------
        final_state: Tuple[Coord, int, int] = (dest, target_k, full_mask)

        # Caso: no se encontró solución válida
        if final_state not in dist:
            reason = "NO_FEASIBLE_PATH"
            # ¿Llegamos al dest pero sin alguna mandatoria?
            for partial in range(full_mask):
                if (dest, target_k, partial) in dist:
                    reason = "REACHED_DEST_BUT_MANDATORY_EDGE_NOT_USED"
                    break
            return RouteResult(
                feasible=False, total_cost=None, path_coords=[], path_edges=[],
                reason=reason, mandatory_edge_in_path=False,
                metrics={"expanded_states": expanded,
                         "runtime_ms": (time.perf_counter() - t0) * 1000},
                mandatory_edges_in_path=[False] * N,
                audit={"waypoints": waypoints,
                       "mandatory_edges": mandatory_edges_list} if audit else None,
            )

        total_cost, total_hops = dist[final_state]

        # reconstrucción de aristas
        path_edges: List[Edge] = []
        st: Tuple[Coord, int, int] = final_state
        while st != start_state:
            p, e = prev[st]
            path_edges.append(e)
            st = p
        path_edges.reverse()

        # Reconstrucción de coordenadas 
        path_coords = [origin]
        for a, b in path_edges:
            path_coords.append(b)

        # Marcar cada mandatoria por separado y la agregada
        mandatory_edges_in_path = [(e in path_edges) for e in mandatory_edges_list]
        mandatory_in_path = all(mandatory_edges_in_path) if N > 0 else True

        #  Métricas y auditoría
        metrics = {
            "expanded_states": expanded,
            "hops": total_hops,
            "runtime_ms": (time.perf_counter() - t0) * 1000,
            "n_mandatory_edges": N,
        }

        audit_info: Optional[Dict[str, Any]] = None
        if audit:
            audit_info = {
                "origin_node_id": origin_node_id,
                "destination_node_id": destination_node_id,
                "origin_coord": origin,
                "destination_coord": dest,
                "mandatory_stops": mandatory_stops,
                "waypoints_coords": waypoints,
                "mandatory_edges": [{"index": i, "from": u, "to": v,
                                     "in_path": mandatory_edges_in_path[i]}
                                    for i, (u, v) in enumerate(mandatory_edges_list)],
                "mandatory_edge_in_path": mandatory_in_path,
                "tie_breaker": tie_breaker,
                "forbidden_nodes_count": len(forbidden_nodes),
                "forbidden_edges_count": len(forbidden_edges),
                "metrics": metrics,
            }

        return RouteResult(
            feasible=True,
            total_cost=total_cost,
            path_coords=path_coords,
            path_edges=path_edges,
            reason=None,
            mandatory_edge_in_path=mandatory_in_path,
            metrics=metrics,
            mandatory_edges_in_path=mandatory_edges_in_path,
            audit=audit_info,
        )


class RouteSolver(BaseRouteSolver):
    """
    Solver canónico

    API:
        'solver = RouteSolver()'
        'res = solver.solve(nodes_df, directed_edges_df, origin_node_id=..., ...)'

    Acepta uno o varios 'mandatory_edges' (extensión punto 6: vía bitmask).
    Para cargas masivas (stress) puede recibir un 'ctx' ya construido y
    saltarse la construcción del 'GraphContext' por "llamada", esto es lo que
    usa internamente :class:`CaseManager`.
    """

    # ------------------------------------------------------------------
    @staticmethod
    def _normalize_mandatory_edges(
        mandatory_edge: Optional[Edge] = None,
        mandatory_edges: Optional[List[Edge]] = None,
    ) -> List[Edge]:
        """
        Acepta cualquiera de las dos formas y devuelve 'List[Edge]'.

        - 'mandatory_edge' (singular)  -> lista de 1.
        - 'mandatory_edges' (lista)    -> tal cual.
        - ambos None  -> ValueError.
        - ambos definidos -> ValueError.
        """

        if mandatory_edge is not None and mandatory_edges is not None:
            raise ValueError("Pasa solo uno: 'mandatory_edge' o 'mandatory_edges'.")
        
        if mandatory_edge is not None:
            return [mandatory_edge]
        
        if mandatory_edges is not None:
            if not isinstance(mandatory_edges, list) or not mandatory_edges:
                raise ValueError("'mandatory_edges' debe ser una lista no vacía de aristas.")
            return list(mandatory_edges)
        raise ValueError("Debes especificar 'mandatory_edge' o 'mandatory_edges'.")

    # ------------------------------------------------------------------
    def solve(
        self,
        nodes_df: Optional[pd.DataFrame] = None,
        directed_edges_df: Optional[pd.DataFrame] = None,
        *,
        origin_node_id: str,
        destination_node_id: str,
        mandatory_edge: Optional[Edge] = None,
        mandatory_edges: Optional[List[Edge]] = None,
        mandatory_stops: Optional[List[str]] = None,
        forbidden_nodes: Optional[Set[Coord]] = None,
        forbidden_edges: Optional[Set[Edge]] = None,
        tie_breaker: Optional[str] = None,
        audit: bool = False,
        cost_col: str = "total_cost",
        ctx: Optional[GraphContext] = None,
    ) -> RouteResult:
        """
        Resuelve la ruta de costo mínimo desde 'origin_node_id' a
        'destination_node_id' que (a) pase por todos los 'mandatory_stops'
        en orden y (b) utilice **todas** las aristas obligatorias.

        Parameters
        ----------
        nodes_df, directed_edges_df : pd.DataFrame
            optional Requeridos si 'ctx is None'

        ctx : Optional[GraphContext] 
            Si se provee, se reutiliza (modo cached, para cargas masivas).

        origin_node_id, destination_node_id : str

        mandatory_edge : Optional[Edge] 
            Una sola arista obligatoria (compat hacia atrás).

        mandatory_edges : Optional[List[Edge]] 
            Lista de N>=1 aristas obligatorias. Mutuamente excluyente con 'mandatory_edge'.

        mandatory_stops : Optional[List[str]]

        forbidden_nodes : Optional[Set[Coord]]

        forbidden_edges : Optional[Set[Edge]]

        tie_breaker : Optional[str]

        audit : bool

        cost_col : str Columna con el costo escalar en 'directed_edges_df'.

        Returns
        -------
        RouteResult
        """
        edges_list = self._normalize_mandatory_edges(mandatory_edge, mandatory_edges)

        if ctx is None:
            if nodes_df is None or directed_edges_df is None:
                raise ValueError(
                    "Modo fresh: nodes_df y directed_edges_df son requeridos "
                    "cuando no se pasa un 'ctx' pre-construido."
                )
            ctx = GraphContext.from_dataframes(
                nodes_df, directed_edges_df, cost_col=cost_col
            )

        return self._solve(
            ctx=ctx,
            origin_node_id=origin_node_id,
            destination_node_id=destination_node_id,
            mandatory_edges_list=edges_list,
            mandatory_stops=mandatory_stops,
            forbidden_nodes=forbidden_nodes,
            forbidden_edges=forbidden_edges,
            tie_breaker=tie_breaker,
            audit=audit,
        )


# 8) CaseManager — batch & stress
class CaseManager:
    """
    Gestor de casos sobre un 'GraphContext' reusable.

    Permite:
      - Crear un DataFrame vacío con el esquema esperado.
      - Generar casos de estrés sintéticos.
      - Ejecutar un caso por índice y volcar resultados al DataFrame.
      - Ejecutar todo el batch con timing.
    """

    EMPTY_COLUMNS = [
        # ---- Inputs
        "case_id",
        "origin_node_id",
        "destination_node_id",
        "mandatory_from_x", "mandatory_from_y",
        "mandatory_to_x",   "mandatory_to_y",
        "mandatory_stops",
        "forbidden_nodes",
        "forbidden_edges",
        "tie_breaker",
        "audit",
        # ---- Outputs
        "feasible",
        "total_cost",
        "mandatory_edge_in_path",
        "reason",
        "metrics",
        "path_edges",
    ]

    def __init__(
        self,
        ctx: GraphContext,
        solver: Optional[BaseRouteSolver] = None,
    ) -> None:
        
        """
        Parameters
        ----------
        ctx : GraphContext        
        solver : Optional[BaseRouteSolver]
            Si se omite, se usa :class:'RouteSolver'. 
            Para batch reutiliza el 'ctx' sin reconstruirlo por-llamada.
        """

        self.ctx    = ctx
        self.solver = solver if solver is not None else RouteSolver()

    # ------------------------------------------------------------------
    @staticmethod
    def make_empty_cases_df() -> pd.DataFrame:
        """DataFrame vacío con el esquema de casos."""
        return pd.DataFrame(columns=CaseManager.EMPTY_COLUMNS)

    # ------------------------------------------------------------------
    def generate_stress_cases(
        self,
        n_queries: int = 5000,
        seed: int = 42,
        p_add_stops: float = 0.40,
        max_stops: int = 3,
        p_forbidden: float = 0.10,
        p_infeasible: float = 0.05,
        tie_breaker: Optional[str] = "min_hops",
        audit: bool = False,
    ) -> pd.DataFrame:
        """
        Genera 'n_queries' casos de estrés muestreando del 'GraphContext'.

        Returns
        -------
        pd.DataFrame
        """
        rng = random.Random(seed)
        banks = self.ctx.banks_open_ids
        atms = self.ctx.atms_open_ids
        edges_pool: List[Edge] = self.ctx.edges_pool
        id2coord = self.ctx.id2coord

        if not banks or not atms:
            raise ValueError("ctx no tiene suficientes BANKs o ATMs abiertos para generar casos.")
        if not edges_pool:
            raise ValueError("ctx no tiene edges_pool (¿directed_edges_df vacío?)")

        cases = self.make_empty_cases_df()
        open_node_ids = list(set(banks + atms))
        open_coords = [id2coord[nid] for nid in open_node_ids if nid in id2coord]

        for i in range(n_queries):
            case_id = f"STRESS_{i:06d}"
            origin_id = rng.choice(banks)
            dest_id = rng.choice(atms)

            stops: List[str] = []
            if rng.random() < p_add_stops:
                k = rng.randint(1, max_stops)
                stops = rng.sample(atms, k=min(k, len(atms)))
                stops = [s for s in stops if s != dest_id]

            forb_nodes: List[Coord] = []
            forb_edges: List[Edge] = []
            if rng.random() < p_forbidden and open_coords:
                forb_nodes = rng.sample(open_coords, k=min(2, len(open_coords)))
            if rng.random() < p_forbidden and edges_pool:
                forb_edges = rng.sample(edges_pool, k=min(2, len(edges_pool)))

            if rng.random() < p_infeasible:
                mf, mt = (9999, 9999), (9999, 10000)
            else:
                (mf, mt) = rng.choice(edges_pool)

            cases.loc[len(cases)] = {
                "case_id": case_id,
                "origin_node_id": origin_id,
                "destination_node_id": dest_id,
                "mandatory_from_x": int(mf[0]), "mandatory_from_y": int(mf[1]),
                "mandatory_to_x":   int(mt[0]), "mandatory_to_y":   int(mt[1]),
                "mandatory_stops": stops,
                "forbidden_nodes": forb_nodes,
                "forbidden_edges": forb_edges,
                "tie_breaker": tie_breaker,
                "audit": audit,
                "feasible": None, "total_cost": None, "mandatory_edge_in_path": None,
                "reason": None,   "metrics": None,    "path_edges": None,
            }
        return cases

    # ------------------------------------------------------------------
    def run_case_by_index(self, cases_df: pd.DataFrame, idx: Any) -> RouteResult:
        """
        Ejecuta el caso 'cases_df.loc[idx]', vuelca outputs y devuelve
        el 'RouteResult'.

        Acepta dos modos para las aristas obligatorias:
          * Columnas 'mandatory_from_x/y' y 'mandatory_to_x/y'
            (modo legacy de 1 arista).
          * Columna opcional 'mandatory_edges' (lista de tuplas) que tiene
            **prioridad** sobre las anteriores si está presente y no vacía.

        Returns
        -------
        RouteResult
        """
        row = cases_df.loc[idx]

        # --- Resolver la lista de aristas obligatorias ---
        edges_list: List[Edge] = []
        if "mandatory_edges" in cases_df.columns:
            cell = row["mandatory_edges"]
            if isinstance(cell, list) and len(cell) > 0:
                edges_list = [tuple(map(tuple, e)) for e in cell]  # type: ignore[arg-type]
        if not edges_list:
            edges_list = [(
                (int(row["mandatory_from_x"]), int(row["mandatory_from_y"])),
                (int(row["mandatory_to_x"]),   int(row["mandatory_to_y"])),
            )]

        stops = row["mandatory_stops"] if isinstance(row["mandatory_stops"], list) else []
        forb_nodes: Set[Coord] = (
            set(row["forbidden_nodes"]) if isinstance(row["forbidden_nodes"], (list, set)) else set()
        )
        forb_edges: Set[Edge] = (
            set(row["forbidden_edges"]) if isinstance(row["forbidden_edges"], (list, set)) else set()
        )
        tie_breaker = row["tie_breaker"] if pd.notna(row["tie_breaker"]) else None
        audit = bool(row["audit"]) if pd.notna(row["audit"]) else False

        # Llamada al solver pasando ctx pre-construido (modo cached implícito)
        res = self.solver.solve(
            ctx=self.ctx,
            origin_node_id=row["origin_node_id"],
            destination_node_id=row["destination_node_id"],
            mandatory_edges=edges_list,
            mandatory_stops=stops,
            forbidden_nodes=forb_nodes,
            forbidden_edges=forb_edges,
            tie_breaker=tie_breaker,
            audit=audit,
        )

        cases_df.at[idx, "feasible"] = res.feasible
        cases_df.at[idx, "total_cost"] = res.total_cost
        cases_df.at[idx, "mandatory_edge_in_path"] = res.mandatory_edge_in_path
        cases_df.at[idx, "reason"] = res.reason
        cases_df.at[idx, "metrics"] = res.metrics
        cases_df.at[idx, "path_edges"] = res.path_edges
        if "mandatory_edges_in_path" in cases_df.columns:
            cases_df.at[idx, "mandatory_edges_in_path"] = res.mandatory_edges_in_path
        return res

    # ------------------------------------------------------------------
    def run_stress(
        self,
        cases_df: pd.DataFrame,
        limit: Optional[int] = None,
        print_every: int = 200,
    ) -> pd.DataFrame:
        """
        Ejecuta los casos del DataFrame con barra de progreso simple.

        Returns
        -------
        pd.DataFrame
            'cases_df' mutado in-place y devuelto por conveniencia.
        """
        n = len(cases_df) if limit is None else min(limit, len(cases_df))
        t0 = time.perf_counter()
        for j in range(n):
            idx = cases_df.index[j]
            self.run_case_by_index(cases_df, idx)
            if print_every and (j + 1) % print_every == 0:
                elapsed = time.perf_counter() - t0
                feasible_count = int(cases_df["feasible"].iloc[: j + 1].fillna(False).sum())
                print(f"[{j + 1}/{n}] elapsed={elapsed:.2f}s feasible={feasible_count}/{j + 1}")
        total = time.perf_counter() - t0
        if n > 0:
            print(f"[DONE] {n} casos en {total:.2f}s  => {1000 * total / n:.2f} ms/caso aprox.")
        return cases_df


# 9) Plotters — paleta formal y líneas finas
class BaseMapPlotter:
    """
    Plantilla con la lógica común de los mapas:
      - Configuración de la grilla (ejes, ticks, aspect).
      - Dibujo de aristas según 'direction'.
      - Dibujo de nodos ATM/BANCO con state open/closed.
      - Resaltado de nodos borde si existen.
      - Colocación de leyendas externas al eje.

    Las subclases sólo añaden overlays propios sobre el lienzo ya construido.
    """

    def __init__(
        self,
        palette: Optional[Dict[str, str]] = None,
        figsize: Tuple[float, float] = (10, 10),
        node_size: float = 60,
        # parámetros visuales (líneas finas + paleta sobria)
        edge_lw_two_way: float = 0.7,
        edge_lw_one_way: float = 0.7,
        edge_lw_blocked: float = 0.5,
        alpha_edges: float = 0.55,
        arrow_mutation_scale: float = 7,
    ) -> None:
        self.palette = palette if palette is not None else PALETTE
        self.figsize = figsize
        self.node_size = node_size
        self.edge_lw_two_way = edge_lw_two_way
        self.edge_lw_one_way = edge_lw_one_way
        self.edge_lw_blocked = edge_lw_blocked
        self.alpha_edges = alpha_edges
        self.arrow_mutation_scale = arrow_mutation_scale

    # ----- helpers --------------------------------------------------------
    def _setup_grid(self, ax, grid_max_x: int, grid_max_y: int) -> None:
        """Configura ejes, ticks y grilla."""
        ax.set_xlim(-0.5, grid_max_x + 0.5)
        ax.set_ylim(-0.5, grid_max_y + 0.5)
        ax.set_xticks(range(0, grid_max_x + 1))
        ax.set_yticks(range(0, grid_max_y + 1))
        ax.grid(True, which="both", linewidth=0.4, alpha=0.30, color=self.palette["grid"])
        ax.set_aspect("equal", adjustable="box")
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
            spine.set_color("#666666")

    def _add_arrow(
        self,
        ax,
        x1: float, y1: float, x2: float, y2: float,
        color: str = "#2e8b75",
        lw: float = 0.7,
        alpha: float = 0.75,
        ms: Optional[float] = None,
        z: int = 5,
        style: str = "-|>",
    ) -> None:
        """Añade una flecha fina y elegante."""
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            arrowstyle=style,
            mutation_scale=ms if ms is not None else self.arrow_mutation_scale,
            linewidth=lw,
            color=color,
            alpha=alpha,
            zorder=z,
        )
        ax.add_patch(arrow)

    def _draw_edges(
        self,
        ax,
        routes_df: pd.DataFrame,
        show_blocked: bool,
        arrows: bool,
        legend_handles: List,
        legend_labels: List[str],
    ) -> None:
        """Dibuja aristas según 'direction' y rellena la leyenda manual."""
        # Bloqueadas
        if show_blocked:
            blocked = routes_df[routes_df["direction"] == 0]
            for _, r in blocked.iterrows():
                ax.plot(
                    [r["from_x"], r["to_x"]], [r["from_y"], r["to_y"]],
                    color=self.palette["blocked"], alpha=self.alpha_edges,
                    linewidth=self.edge_lw_blocked, zorder=1, linestyle=(0, (3, 2)),
                )
            if len(blocked) > 0:
                legend_labels.append("Ruta bloqueada (0)")
                legend_handles.append(plt.Line2D([0], [0], color=self.palette["blocked"],
                                                 lw=1.0, linestyle=(0, (3, 2))))

        # Doble sentido
        two_way = routes_df[routes_df["direction"] == 2]
        for _, r in two_way.iterrows():
            ax.plot(
                [r["from_x"], r["to_x"]], [r["from_y"], r["to_y"]],
                color=self.palette["two_way"], alpha=self.alpha_edges,
                linewidth=self.edge_lw_two_way, zorder=2,
            )
        if len(two_way) > 0:
            legend_labels.append("Ruta doble sentido (2)")
            legend_handles.append(plt.Line2D([0], [0], color=self.palette["two_way"], lw=1.2))

        # Un sentido
        one_way = routes_df[routes_df["direction"].isin([1, -1])]
        if arrows:
            for _, r in one_way.iterrows():
                x1, y1 = r["from_x"], r["from_y"]
                x2, y2 = r["to_x"],   r["to_y"]
                if r["direction"] == -1:
                    x1, y1, x2, y2 = x2, y2, x1, y1
                self._add_arrow(
                    ax, x1, y1, x2, y2,
                    color=self.palette["one_way"],
                    lw=self.edge_lw_one_way, alpha=0.80, z=3,
                )
            if len(one_way) > 0:
                legend_labels.append("Ruta un sentido (+1/-1)")
                legend_handles.append(plt.Line2D([0], [0], color=self.palette["one_way"], lw=1.2))
        else:
            for _, r in one_way.iterrows():
                ax.plot(
                    [r["from_x"], r["to_x"]], [r["from_y"], r["to_y"]],
                    color=self.palette["one_way"], alpha=self.alpha_edges,
                    linewidth=self.edge_lw_one_way, zorder=3,
                )

    def _draw_nodes(self, ax, nodos_df: pd.DataFrame) -> None:
        """Dibuja ATMs y BANCOs (open/closed) con estilo formal."""
        atms = nodos_df[nodos_df["status"] == "ATM"].copy()
        banks = nodos_df[nodos_df["status"] == "BANCO"].copy()

        def scatter_nodes(df, marker, label_open, label_closed, color_main):
            if df.empty:
                return
            open_df = df[df["operativo"] == "open"]
            closed_df = df[df["operativo"] == "closed"]
            if not open_df.empty:
                ax.scatter(
                    open_df["x"], open_df["y"], s=self.node_size, marker=marker,
                    c=color_main, edgecolors="#1a1a1a", linewidths=0.5,
                    zorder=10, label=label_open,
                )
            if not closed_df.empty:
                ax.scatter(
                    closed_df["x"], closed_df["y"], s=self.node_size, marker=marker,
                    facecolors="none", edgecolors=self.palette["closed_edge"],
                    linewidths=1.0, zorder=11, label=label_closed,
                )

        scatter_nodes(atms,  marker="o", label_open="ATM open",   label_closed="ATM closed",
                      color_main=self.palette["atm_open"])
        scatter_nodes(banks, marker="s", label_open="BANCO open", label_closed="BANCO closed",
                      color_main=self.palette["bank_open"])

    def _highlight_border_nodes(self, ax, nodos_df: pd.DataFrame) -> None:
        """Resalta los nodos borde si existen."""
        if "ATM_BORDER" in nodos_df["node_id"].values:
            atm_b = nodos_df[nodos_df["node_id"] == "ATM_BORDER"].iloc[0]
            ax.scatter(
                [atm_b["x"]], [atm_b["y"]], s=self.node_size * 2.2, marker="o",
                facecolors="none", edgecolors=self.palette["border_edge"],
                linewidths=1.8, zorder=20, label="ATM_BORDER",
            )
        if "BANK_BORDER" in nodos_df["node_id"].values:
            bank_b = nodos_df[nodos_df["node_id"] == "BANK_BORDER"].iloc[0]
            ax.scatter(
                [bank_b["x"]], [bank_b["y"]], s=self.node_size * 2.2, marker="s",
                facecolors="none", edgecolors=self.palette["border_edge"],
                linewidths=1.8, zorder=20, label="BANK_BORDER",
            )

    def _add_legends(
        self,
        ax,
        legend_handles: List,
        legend_labels: List[str],
    ) -> None:
        """Coloca dos leyendas (rutas / nodos) fuera del eje."""
        if legend_handles:
            leg1 = ax.legend(
                legend_handles, legend_labels,
                loc="upper left", bbox_to_anchor=(1.02, 1.00),
                borderaxespad=0.0, frameon=True, title="Rutas",
                fontsize=9, title_fontsize=10,
            )
            leg1.get_frame().set_linewidth(0.5)
            ax.add_artist(leg1)

        handles2, labels2 = ax.get_legend_handles_labels()
        seen: Set[str] = set()
        uniq = [(h, l) for h, l in zip(handles2, labels2) if not (l in seen or seen.add(l))]
        if uniq:
            handles2_u, labels2_u = zip(*uniq)
            leg2 = ax.legend(
                handles2_u, labels2_u,
                loc="upper left", bbox_to_anchor=(1.02, 0.55),
                borderaxespad=0.0, frameon=True, title="Nodos / Puntos",
                fontsize=9, title_fontsize=10,
            )
            leg2.get_frame().set_linewidth(0.5)

    # API a sobreescribir por subclases
    def plot(self, *args, **kwargs):  # pragma: no cover
        raise NotImplementedError


class SyntheticMapPlotter(BaseMapPlotter):
    """
    Plotea el mapa sintético (nodos + rutas), sin overlays de ruta.
    Equivalente a 'plot_synthetic_map' del notebook.
    """

    def plot(
        self,
        nodos_df: pd.DataFrame,
        routes_df: pd.DataFrame,
        grid_max_x: int = 10,
        grid_max_y: int = 10,
        show_blocked: bool = True,
        arrows: bool = True,
        title: str = "Mapa sintético (nodos + rutas)",
        figsize: Optional[Tuple[float, float]] = None,
    ) -> None:
        """
        Renderiza el mapa.

        Parameters
        ----------
        nodos_df, routes_df : pd.DataFrame
        grid_max_x, grid_max_y : int
        show_blocked : bool
        arrows : bool
        title : str
        figsize : Optional[Tuple[float, float]]
        """
        fig, ax = plt.subplots(figsize=figsize or self.figsize)

        self._setup_grid(ax, grid_max_x, grid_max_y)

        legend_handles: List = []
        legend_labels: List[str] = []
        self._draw_edges(ax, routes_df, show_blocked, arrows, legend_handles, legend_labels)
        self._draw_nodes(ax, nodos_df)
        self._highlight_border_nodes(ax, nodos_df)

        ax.set_title(title, fontsize=12, color="#1a1a1a", pad=10)
        self._add_legends(ax, legend_handles, legend_labels)
        plt.tight_layout(rect=[0, 0, 0.78, 1])
        plt.show()


class CaseRoutePlotter(BaseMapPlotter):
    """
    Plotea el mapa base **+ overlay** de la ruta solución de un caso.
    Equivalente a 'plot_case_route' del notebook.

    Reutiliza 'BaseMapPlotter' para todo el lienzo base; sólo agrega:
      - Origen y destino resaltados.
      - Stops obligatorios numerados.
      - Arista obligatoria en dorado.
      - Path solución como flechas en la paleta de overlay.
    """

    def __init__(
        self,
        palette: Optional[Dict[str, str]] = None,
        figsize: Tuple[float, float] = (10, 10),
        node_size: float = 60,
        # overlays más finos (formales)
        route_lw: float = 1.6,
        mandatory_lw: float = 2.0,
        stop_lw: float = 1.2,
        **kwargs: Any,
    ) -> None:
        super().__init__(palette=palette, figsize=figsize, node_size=node_size, **kwargs)
        self.route_lw = route_lw
        self.mandatory_lw = mandatory_lw
        self.stop_lw = stop_lw

    def plot(
        self,
        cases_df: pd.DataFrame,
        case_idx: Optional[Any] = None,
        case_id: Optional[str] = None,
        nodos_df: Optional[pd.DataFrame] = None,
        routes_df: Optional[pd.DataFrame] = None,
        grid_max_x: int = 10,
        grid_max_y: int = 10,
        show_blocked: bool = True,
        arrows: bool = True,
        title_prefix: str = "Ruta óptima (caso)",
        figsize: Optional[Tuple[float, float]] = None,
    ) -> None:
        """
        Renderiza mapa base + overlay del caso.

        Parameters
        ----------
        cases_df : pd.DataFrame
        case_idx : Optional[Any]
            Índice del caso. Alternativamente usar 'case_id'.
        case_id : Optional[str]
        nodos_df, routes_df : pd.DataFrame
            Tablas para dibujar el mapa base.
        grid_max_x, grid_max_y : int
        show_blocked, arrows : bool
        title_prefix : str
        figsize : Optional[Tuple[float, float]]
        """
        if nodos_df is None or routes_df is None:
            raise ValueError("Debes pasar nodos_df y routes_df.")

        # Selección del caso
        if case_id is not None:
            if "case_id" not in cases_df.columns:
                raise KeyError("cases_df no tiene columna 'case_id'.")
            matches = cases_df.index[cases_df["case_id"] == case_id].tolist()
            if not matches:
                raise ValueError(f"No encontré case_id='{case_id}' en cases_df.")
            idx = matches[0]
        else:
            if case_idx is None:
                raise ValueError("Pasa case_idx o case_id.")
            idx = case_idx

        row = cases_df.loc[idx]
        if not bool(row["feasible"]):
            raise ValueError("Ruta invalida")

        # Lienzo base
        fig, ax = plt.subplots(figsize=figsize or self.figsize)
        self._setup_grid(ax, grid_max_x, grid_max_y)

        legend_handles: List = []
        legend_labels: List[str] = []
        self._draw_edges(ax, routes_df, show_blocked, arrows, legend_handles, legend_labels)
        self._draw_nodes(ax, nodos_df)
        self._highlight_border_nodes(ax, nodos_df)

        # Overlays
        id2xy = dict(zip(nodos_df["node_id"], list(zip(nodos_df["x"], nodos_df["y"]))))

        origin_id = row.get("origin_node_id", None)
        dest_id = row.get("destination_node_id", None)
        origin_xy = id2xy.get(origin_id) if origin_id is not None else None
        dest_xy = id2xy.get(dest_id) if dest_id is not None else None

        stops = row.get("mandatory_stops", [])
        if not isinstance(stops, list):
            stops = []
        stops_xy = [id2xy[s] for s in stops if s in id2xy]

        mf = (int(row["mandatory_from_x"]), int(row["mandatory_from_y"])) if pd.notna(row.get("mandatory_from_x", np.nan)) else None
        mt = (int(row["mandatory_to_x"]),   int(row["mandatory_to_y"]))   if pd.notna(row.get("mandatory_to_x",   np.nan)) else None

        path_edges = row.get("path_edges", [])
        if not isinstance(path_edges, list):
            path_edges = []

        feasible = row.get("feasible", None)
        total_cost = row.get("total_cost", None)
        mandatory_in_path = row.get("mandatory_edge_in_path", None)
        reason = row.get("reason", None)

        # Origen / Destino
        if origin_xy is not None:
            ax.scatter(
                [origin_xy[0]], [origin_xy[1]],
                s=self.node_size * 2.2, marker="*",
                c=self.palette["origin"], edgecolors="white",
                linewidths=0.9, zorder=30, label="ORIGEN",
            )
            ax.text(
                origin_xy[0] + 0.15, origin_xy[1] + 0.15, f"O:{origin_id}",
                fontsize=8, color="#1a1a1a", zorder=31,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )
        if dest_xy is not None:
            ax.scatter(
                [dest_xy[0]], [dest_xy[1]],
                s=self.node_size * 2.2, marker="X",
                c=self.palette["destination"], edgecolors="white",
                linewidths=0.9, zorder=30, label="DESTINO",
            )
            ax.text(
                dest_xy[0] + 0.15, dest_xy[1] + 0.15, f"D:{dest_id}",
                fontsize=8, color="#1a1a1a", zorder=31,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )

        # Stops obligatorios (numerados)
        for j, (sx, sy) in enumerate(stops_xy, start=1):
            ax.scatter(
                [sx], [sy], s=self.node_size * 1.5, marker="o",
                facecolors="none", edgecolors=self.palette["stop"],
                linewidths=self.stop_lw, zorder=25, label="STOP" if j == 1 else None,
            )
            ax.text(
                sx + 0.12, sy - 0.18, f"{j}", fontsize=9,
                color=self.palette["stop"], zorder=26,
                bbox=dict(facecolor="white", alpha=0.7, edgecolor="none"),
            )

        # Arista obligatoria
        if mf is not None and mt is not None:
            self._add_arrow(
                ax, mf[0], mf[1], mt[0], mt[1],
                color=self.palette["mandatory"], lw=self.mandatory_lw,
                alpha=0.95, ms=10, z=40,
            )
            ax.text(
                (mf[0] + mt[0]) / 2 + 0.1, (mf[1] + mt[1]) / 2 + 0.1, "MANDATORY",
                fontsize=8, color="#7a5e0a", zorder=41,
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
            )

        # Ruta solución
        if path_edges and bool(feasible):
            for (a, b) in path_edges:
                self._add_arrow(
                    ax, a[0], a[1], b[0], b[1],
                    color=self.palette["route"], lw=self.route_lw,
                    alpha=0.95, ms=9, z=35,
                )
            legend_handles.append(plt.Line2D([0], [0], color=self.palette["route"], lw=1.8))
            legend_labels.append("Ruta óptima (overlay)")
        else:
            if origin_xy is not None and dest_xy is not None:
                ax.plot(
                    [origin_xy[0], dest_xy[0]], [origin_xy[1], dest_xy[1]],
                    color=self.palette["route"], linestyle="--", linewidth=1.0,
                    alpha=0.6, zorder=20,
                )
                legend_handles.append(plt.Line2D([0], [0], color=self.palette["route"],
                                                 lw=1.0, linestyle="--"))
                legend_labels.append("Sin ruta válida (referencia)")

        # Título informativo
        title_parts = [f"{title_prefix}: {row.get('case_id', idx)}"]
        if feasible is not None:
            title_parts.append(f"feasible={feasible}")
        if total_cost is not None and pd.notna(total_cost):
            title_parts.append(f"cost={float(total_cost):.2f}")
        if mandatory_in_path is not None:
            title_parts.append(f"mandatory_in_path={mandatory_in_path}")
        if (feasible is False) and reason:
            title_parts.append(f"reason={reason}")
        ax.set_title(" | ".join(title_parts), fontsize=11, color="#1a1a1a", pad=10)

        self._add_legends(ax, legend_handles, legend_labels)
        plt.tight_layout(rect=[0, 0, 0.78, 1])
        plt.show()


class RiskHeatmapPlotter:
    """
    Heatmap de riesgo incidente acumulado por nodo.
    Equivalente a 'plot_node_risk_heatmap'.
    """

    def __init__(self, cmap: str = "Reds", figsize: Tuple[float, float] = (8.5, 6.5)) -> None:
        self.cmap = cmap
        self.figsize = figsize

    def plot(
        self,
        routes_df: pd.DataFrame,
        grid_max_x: int = 10,
        grid_max_y: int = 10,
        title: str = "Heatmap de riesgo (incidente por nodo)",
    ) -> None:
        """
        Renderiza el heatmap.

        Parameters
        ----------
        routes_df : pd.DataFrame
        grid_max_x, grid_max_y : int
        title : str
        """
        risk_map = np.zeros((grid_max_y + 1, grid_max_x + 1), dtype=float)
        for _, r in routes_df.iterrows():
            x1, y1 = int(r["from_x"]), int(r["from_y"])
            x2, y2 = int(r["to_x"]),   int(r["to_y"])
            risk = float(r["riesgo"])
            risk_map[y1, x1] += risk
            risk_map[y2, x2] += risk

        plt.figure(figsize=self.figsize)
        sns.heatmap(
            risk_map, cmap=self.cmap, cbar=True,
            linewidths=0.3, linecolor="#ffffff", square=True,
            cbar_kws={"shrink": 0.75, "label": "Riesgo acumulado"},
        )
        plt.title(title, fontsize=12, color="#1a1a1a")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()


# 10) Compatibilidad con el flujo del notebook original (wrappers opcionales)
# Estas funciones existen sólo para que el código del notebook viejo siga
# corriendo si lo importas en bruto. La interfaz **recomendada** es la OOP.
def generate_synthetic_datasets(*args, **kwargs):
    """Wrapper que delega en :class:`SyntheticDatasetGenerator`."""
    return SyntheticDatasetGenerator(*args, **kwargs).generate()


def compute_total_cost(routes_df, cost_params, out_col="total_cost"):
    """Wrapper que delega en :class:`CostCalculator`."""
    return CostCalculator(cost_params).compute(routes_df, out_col=out_col)


def build_directed_edges(routes_df):
    """Wrapper que delega en :meth:`GraphBuilder.build_directed_edges`."""
    return GraphBuilder.build_directed_edges(routes_df)


def prepare_route_context(nodes_df, directed_edges_df, **kwargs):
    """Wrapper que delega en :meth:`GraphContext.from_dataframes`."""
    return GraphContext.from_dataframes(nodes_df, directed_edges_df, **kwargs)


__all__ = [
    # constantes / paleta
    "PALETTE", "Coord", "Edge",
    # generación / costo / grafo
    "SyntheticDatasetGenerator", "CostCalculator", "GraphBuilder",
    "GraphContext", "normalize_operational_status",
    # solver (unificado: RouteSolver acepta DataFrames o ctx pre-construido)
    "RouteResult", "BaseRouteSolver", "RouteSolver",
    # casos
    "CaseManager",
    # plots
    "BaseMapPlotter", "SyntheticMapPlotter", "CaseRoutePlotter", "RiskHeatmapPlotter",
    # wrappers compatibilidad
    "generate_synthetic_datasets", "compute_total_cost",
    "build_directed_edges", "prepare_route_context",
]
