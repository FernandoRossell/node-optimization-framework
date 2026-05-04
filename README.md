# Optimización de rutas en grafos

**Caso de negocio:** Red de cajeros automáticos / bancos

**Stack:** Python 3.10+, NumPy, pandas, matplotlib, seaborn

**Autor:** Abel Soto

---

## 1. Descripción del problema

Dada una red dirigida ponderada de relaciones entre entidades, encontrar la
**ruta de costo mínimo** entre un nodo origen `o` y un nodo destino `d` que
**incluya obligatoriamente** una arista `(u, v)`. Si no existe, reportarlo
con un código de error trazable.

El problema se materializa en un caso de negocio bancario: una unidad de
abastecimiento de ATMs (camión de valores o equipo técnico) debe trasladarse
entre ubicaciones de la ciudad y, por compliance, debe transitar por al menos
un tramo específico (corredor seguro, vialidad monitoreada, etc.).

### Solución elegida

* **Modelo:** grafo dirigido ponderado en una grilla 2D donde cada celda es
  una intersección y cada arista una conexión recta o diagonal con costo no
  negativo.
* **Algoritmo:** Dijkstra sobre un **grafo de estados** ampliado
  `(coord, k_waypoint, used_mandatory)`:
  * `coord` = posición actual,
  * `k_waypoint` = índice del último waypoint alcanzado en orden,
  * `used_mandatory ∈ {0, 1}` = bandera de si ya se transitó la arista obligatoria.
  Esto resuelve simultáneamente el problema base y las extensiones de stops
  ordenados, nodos/aristas prohibidos y desempates.
* **Costos:** suma ponderada (tiempo, riesgo, fuel) más interacciones
  opcionales (product, min, max). Se garantiza no negatividad para que
  Dijkstra sea aplicable.

---

## 2. Estructura del proyecto

```
proyecto_rutas/
├── ejecutable.py            # Módulo principal (clases POO con herencia)
├── Problema_nodos_OOP.ipynb # Notebook de ejecución end-to-end
├── README.md                # Este archivo
├── MEMO_TECNICO.md          # Memo de 1–2 páginas (trade-offs y evolución)
└── tests/
    ├── conftest.py          # Fixtures compartidos (datasets seed=123)
    └── test_solver.py       # 25 tests funcionales (pytest)
```

### Mapa de clases

| Categoría        | Clase                              | Responsabilidad                                                   |
| ---------------- | ---------------------------------- | ----------------------------------------------------------------- |
| Datos            | `SyntheticDatasetGenerator`        | Genera `nodos_df`, `routes_df` reproducibles (incluye OOB y borde)|
| Costo            | `CostCalculator`                   | Calcula `total_cost` con pesos + interacciones                    |
| Grafo            | `GraphBuilder`                     | Helpers estáticos: edges dirigidas, adyacencia, mapeos            |
| Grafo            | `GraphContext`                     | Contexto cacheable para múltiples consultas                       |
| Solver           | `BaseRouteSolver` (abstract)       | **Dijkstra** sobre `(coord, k, used_mask)` - soporta N aristas obligatorias |
| Solver           | `RouteSolver(BaseRouteSolver)`     | Solver canónico (réplica del 1º `solve_route` del notebook)       |
| Casos            | `CaseManager`                      | DataFrame de casos, batch y stress                                |
| Plot             | `BaseMapPlotter`                   | Plantilla con grilla, aristas y nodos                             |
| Plot             | `SyntheticMapPlotter(BaseMapPlotter)` | Mapa base                                                      |
| Plot             | `CaseRoutePlotter(BaseMapPlotter)` | Mapa base + overlay de ruta                                       |
| Plot             | `RiskHeatmapPlotter`               | Heatmap de riesgo                                                 |


---

## 3. Cómo ejecutar

### 3.1 Requisitos

Python 3.10 o superior. Paquetes:

```bash
pip install numpy pandas matplotlib seaborn jupyter
```

> El notebook trae una **celda inicial** que instala automáticamente los
> paquetes faltantes (idempotente).

### 3.2 Vía notebook (recomendado)

```bash
cd proyecto_rutas/
jupyter notebook Problema_nodos_OOP.ipynb
```

Y correr todas las celdas en orden. Genera datos, resuelve casos manuales,
plotea mapas y corre un stress de 10.000 consultas.

### 3.3 Vía Python puro

```python
from ejecutable import (
    SyntheticDatasetGenerator, CostCalculator, GraphBuilder,
    GraphContext, RouteSolver, CaseManager,
    SyntheticMapPlotter, CaseRoutePlotter, RiskHeatmapPlotter,
)

# 1. Datos
nodos_df, routes_df, summary = SyntheticDatasetGenerator(
    grid_max_x=30, grid_max_y=30, seed=123,
    border_cases=True, border_nodes=((4, 4), (5, 4)),
).generate()

# 2. Costo + grafo dirigido
COST = {"weights": {"tiempo": 1.0, "riesgo": 6.0, "fuel_cost": 2.0}}
routes_costed = CostCalculator(COST).compute(routes_df)
directed_edges = GraphBuilder.build_directed_edges(routes_costed)

# 3. Contexto reutilizable + manager (CaseManager usa RouteSolver por default)
ctx = GraphContext.from_dataframes(nodos_df, directed_edges)
mgr = CaseManager(ctx)

# 4.a Consulta puntual
solver = RouteSolver()
res = solver.solve(
    nodes_df=nodos_df,
    directed_edges_df=directed_edges,
    origin_node_id="BANK_001", destination_node_id="ATM_010",
    mandatory_edge=((5, 5), (5, 6)),
    audit=True,
)

# 4.b Consulta cacheada - mismo método con ctx pre-construido
res2 = solver.solve(
    ctx=ctx,
    origin_node_id="BANK_001", destination_node_id="ATM_010",
    mandatory_edges=[((5, 5), (5, 6)), ((10, 10), (9, 11))],
)
print(res2.feasible, res2.total_cost, res2.mandatory_edges_in_path)
```

---

## 4. Generación de datos

### 4.1 Datos base

`SyntheticDatasetGenerator(...).generate()` produce:

* **`nodos_df`**: ATMs y BANCOs in-grid + nodos OOB explícitos.
  Columnas: `node_id, status, operativo, x, y, in_grid, case`.
* **`routes_df`**: aristas adyacentes (recta + diagonal) en orden canónico.
  Columnas: `route_id, from_x, from_y, to_x, to_y, direction, riesgo, tiempo, fuel_cost`.
* **`summary`**: estadísticas (conteos por dirección, n_atms, n_bancos, info de borde).

Parámetros clave reproducibles vía `seed`:

| Parámetro                 | Default | Descripción                                          |
| ------------------------- | ------- | ---------------------------------------------------- |
| `grid_max_x`, `grid_max_y`| 30      | Tamaño de la grilla                                  |
| `n_atms`, `n_banks`       | 20      | Densidad de nodos                                    |
| `pct_two_way / one_way / blocked` | 0.70/0.25/0.05 | Distribución de direcciones                |
| `atm_open_prob`, `bank_open_prob` | 0.90 | Probabilidad de estar `open`                       |
| `border_cases`            | False   | Aísla un ATM y un BANCO bloqueando aristas incidentes|

### 4.2 Datos de extensión

* **Stress masivo**: `CaseManager(ctx).generate_stress_cases(n_queries=10_000, seed=42)`.
  Mezcla casos válidos, infeasibles y restricciones (forbidden_*).
* **Casos manuales** (válido / nodo cerrado / arista inexistente) ver Sección 7
  del notebook.

### 4.3 Justificación del dataset

* **Reproducible** (semilla fija) y **paramétrico** (escala con `grid_max_*`).
* **Cubre escenarios**:
  * solución válida (mayoría de stress),
  * sin solución (border_cases, mandatory_edge inexistente, destino cerrado),
  * casos borde (origen ≡ destino, OOB, stops cerrados),
  * escala (10 000 queries sobre el mismo grafo).
* **Costos heterogéneos** (riesgo 0–5, tiempo 10–60 min, fuel 0.01–1.0)
  permiten ejercitar el desempate y la sensibilidad a los pesos.

---

## 5. Formato de entradas y salidas

### 5.1 Entrada del solver

| Campo                | Tipo                            | Notas                                              |
| -------------------- | ------------------------------- | -------------------------------------------------- |
| `nodes_df`           | `pd.DataFrame`                  | Requerido si `ctx is None`                         |
| `directed_edges_df`  | `pd.DataFrame`                  | Requerido si `ctx is None`                         |
| `ctx`                | `Optional[GraphContext]`        | Si se provee, se reusa (modo cached implícito)     |
| `origin_node_id`     | `str`                           | Debe existir en `nodos_df`                         |
| `destination_node_id`| `str`                           | Debe existir en `nodos_df`                         |
| `mandatory_edge`     | `Optional[Edge]`                | Una arista obligatoria (compat hacia atrás)        |
| `mandatory_edges`    | `Optional[List[Edge]]`          | **N** aristas obligatorias (extensión punto 6)     |
| `mandatory_stops`    | `Optional[List[str]]`           | Lista **ordenada** de node_id intermedios          |
| `forbidden_nodes`    | `Optional[Set[Tuple[int,int]]]` | Coordenadas prohibidas                             |
| `forbidden_edges`    | `Optional[Set[Edge]]`           | Aristas dirigidas prohibidas                       |
| `tie_breaker`        | `Optional[str]`                 | `None` o `"min_hops"`                              |
| `audit`              | `bool`                          | Devuelve trazabilidad detallada                    |
| `cost_col`           | `str`                           | Default `"total_cost"`                             |

`mandatory_edge` y `mandatory_edges` son mutuamente excluyentes (uno o el otro).

### 5.2 Salida (`RouteResult`)

```python
@dataclass
class RouteResult:
    feasible: bool
    total_cost: Optional[float]
    path_coords: List[Tuple[int, int]]
    path_edges: List[Tuple[Tuple[int,int], Tuple[int,int]]]
    reason: Optional[str]                       # ej. MANDATORY_EDGE_MISSING
    mandatory_edge_in_path: bool                # AND lógico de todas las mandatorias
    metrics: Dict[str, Any]                     # runtime_ms, expanded_states, hops, n_mandatory_edges
    mandatory_edges_in_path: List[bool]         # bandera por arista (orden = entrada)
    audit: Optional[Dict[str, Any]]             # contexto + waypoints + flags
```

Códigos de `reason` posibles cuando `feasible=False`:

`ORIGIN_NOT_FOUND`, `DEST_NOT_FOUND`,
`ORIGIN_CLOSED_OR_OOB`, `DEST_CLOSED_OR_OOB`,
`STOP_NOT_FOUND`, `STOP_CLOSED_OR_OOB`,
`MANDATORY_EDGE_FORBIDDEN`, `MANDATORY_EDGE_MISSING`,
`MANDATORY_EDGE_ENDPOINT_OOB_OR_CLOSED`,
`REACHED_DEST_BUT_MANDATORY_EDGE_NOT_USED`,
`NO_FEASIBLE_PATH`.

---

## 6. Supuestos, limitaciones y riesgos

| # | Supuesto                                           | Riesgo si se rompe                                |
| - | -------------------------------------------------- | ------------------------------------------------- |
| 1 | Costos no negativos                                | Dijkstra dejaría de ser óptimo (usar Bellman-Ford)|
| 2 | El grafo cabe en memoria                           | OOM en escenarios extremos (>10⁷ aristas)         |
| 3 | `nodos_df` tiene `in_grid` y `operativo` confiables| Falsos positivos/negativos en alcanzabilidad      |
| 4 | La lista de stops es **ordenada**                  | Confundir con un TSP no resuelto                  |


**Limitación principal**: el solver expande estados `(coord, k, used)` y, si
`mandatory_stops` crece, la complejidad se multiplica por el número de
permutaciones implícitas (con stops *ordenados* es lineal en `k`, pero si se
quiere TSP de stops sería NP-hard).

---

## 7. Estrategia de pruebas

### 7.1 Suite automatizada (pytest)

```bash
# desde proyecto_rutas/
pip install pytest
python -m pytest -v
```

Cobertura actual: **25 tests / 9 grupos**.

| Grupo                            | Cubre                                                    |
| -------------------------------- | -------------------------------------------------------- |
| `TestHappyPath`                  | Camino feasible, evidencia explícita de mandatory en path, consistencia path_coords <-> path_edges |
| `TestInvalidIDs`                 | `ORIGIN_NOT_FOUND`, `DEST_NOT_FOUND`                     |
| `TestOperationalStatus`          | `ORIGIN_CLOSED_OR_OOB`, `DEST_CLOSED_OR_OOB`, `STOP_NOT_FOUND`, `STOP_CLOSED_OR_OOB` |
| `TestMandatoryEdgeValidation`    | `MANDATORY_EDGE_MISSING`, `MANDATORY_EDGE_FORBIDDEN`, `*_ENDPOINT_OOB_OR_CLOSED`, **edge inverso** (sólo la dirección contraria existe) |
| `TestBorderIsolated`             | `border_cases=True` -> ruta a `ATM_BORDER` infeasible     |
| `TestMultipleMandatoryEdges`     | N=2 happy, N=3 happy, índice de la arista que falla, mutex de argumentos |
| `TestTieBreaker`                 | `min_hops` no degrada el costo óptimo                    |
| `TestAudit`                      | Payload de auditoría no nulo cuando `audit=True`; nulo por default |
| `TestFreshMode`                  | Modo "fresh" (sin ctx) y consistencia fresh <-> cached     |

Los datos de prueba se generan **una sola vez por sesión** con semilla fija
(seed=123) en `conftest.py`, y los hechos pinned (BANK_002 closed, edge
(7,12)->(7,13) inexistente) están documentados como invariantes del fixture.

### 7.2 Pruebas manuales adicionales en notebook

| Tipo                | Cómo se evidencia                                                          |
| ------------------- | -------------------------------------------------------------------------- |
| **Funcional happy** | `MANUAL_001` -> `feasible=True`, `mandatory_edge_in_path=True`              |
| **Multi-arista**    | `MANUAL_004_MULTI` -> 2 aristas, ambas en `path_edges`                      |
| **Edge inexistente**| `MANUAL_003` -> `feasible=False`, `reason=MANDATORY_EDGE_MISSING`           |
| **Border (aislado)**| `border_cases=True` -> toda ruta a borde devuelve `NO_FEASIBLE_PATH`        |
| **Stress**          | 10 000 queries con `p_infeasible=0.05` y `p_forbidden=0.10`. Tasas y timing|
| **Reproducibilidad**| Semillas fijas en generación y stress. Re-ejecutar produce los mismos costos|

---

## 8. Complejidad y rendimiento

* **Dijkstra con heap binario**: O((V + E) log V) sobre el grafo de estados.
* **Estados**: `V_state = V_grafo · (k+1) · 2` con `k = len(stops)+1`.
  Para grilla 30×30 con 1 stop ⇒ ~3 700 estados, despreciable.
* **Throughput medido**: ~10 ms / consulta en M1 / cloud-CPU estándar para
  grilla 30×30. Stress de 10 000 consultas: ~20 s.
* **Cuello de botella**: `pd.DataFrame.iterrows` durante `build_directed_edges`
  y `from_dataframes`. Pasar a vectorización o a `to_numpy()` reduce esto en ~10×.

---

## 9. Decision log

| Decisión                                                  | Alternativa descartada                | Justificación                                                                |
| --------------------------------------------------------- | ------------------------------------- | ---------------------------------------------------------------------------- |
| Dijkstra sobre `(coord, k, used)`                         | Dijkstra simple `o->u` + Dijkstra `v->d`| El método elegido respeta restricciones de stops y forbidden simultáneamente |
| Escalarizar costos en una sola dimensión                  | Multi-objetivo (Pareto)               | Suficiente para el problema base; multi-objetivo añade complejidad no requerida|
| Tie-break `min_hops`                                      | Lexicográfico por riesgo/tiempo       | `min_hops` es objetivo y barato; los lexicográficos son fáciles de añadir    |

---

## 10. Evolución a producción

1. **Empaquetado**: convertir el módulo en paquete (`pyproject.toml`,
   `route_optimizer/`) con submódulos `data/`, `graph/`, `solvers/`, `plot/`.
2. **Logging**: cambiar `print` por `logging` con niveles. Trazas estructuradas
   (`structlog`) con `case_id`, `runtime_ms`, `reason`.
3. **Observabilidad**: métricas Prometheus (`route_query_seconds`, `feasible_total`).
   Alertas si `p99` > 50 ms o tasa de infeasibles > 20 %.
4. **API**: FastAPI con `/route` (single) y `/route/batch` (stream con SSE).
   Validación con Pydantic.
5. **Persistencia del grafo**: serializar `GraphContext` a `pickle/parquet` y
   cargarlo en arranque para evitar reconstruirlo. En grafos > 10⁵ edges,
   migrar a `igraph` o `networkit` (C++ backend).
6. **Auditoría**: persistir cada `RouteResult.audit` en una BD (Postgres) con
   `case_id`, `caller`, `timestamp` para reportes de cumplimiento.
77. **CI/CD**: GitHub Actions con lint (ruff), tipos (mypy), tests, build de
    contenedor y despliegue blue/green.

---

## 11. Preguntas que esta entrega responde (del documento C11)

| Pregunta                                          | Respuesta corta                                                          |
| ------------------------------------------------- | ------------------------------------------------------------------------ |
| Mapeo a caso de negocio                           | Sección 1 (red ATM/BANCO + corredor seguro).                             |
| ¿Por qué este enfoque?                            | Sección 9 (decision log).                                                |
| Supuestos y dónde se rompen                       | Sección 6.                                                               |
| Comportamiento a 10× / 100×                       | Sección 8 + memo técnico.                                                |
| Parte más frágil y refuerzo                       | `iterrows` en hot path -> vectorizar (memo).                              |
| Qué cambiaría en prod                             | Sección 10.                                                              |
| ¿Las extensiones agregan valor?                   | Stress + auditoría + forbidden están instrumentadas y medidas.           |
