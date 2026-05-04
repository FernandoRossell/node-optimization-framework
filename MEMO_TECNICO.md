# Memo técnico
**Tema:** Optimización de rutas en grafos dirigidos ponderados con arista
obligatoria. Caso de negocio: red ATM / banco.

**Autor:** Abel Soto

---

## Decisión central

Dijkstra sobre un grafo de **estados ampliado** `(coord, k_waypoint, used_mandatory)`.
Esto permite resolver en **una sola pasada** el problema base más las
extensiones de stops ordenados, nodos / aristas prohibidos y desempate por
saltos. La alternativa de descomponer el problema en `o → u` + `(u, v)` + `v → d`
funciona para una arista obligatoria sola, pero no compone bien con
prohibidos y stops, y tiende a producir rutas sub-óptimas cuando los caminos
parciales se reusan.

Costos escalarizados (suma ponderada) para mantener no-negatividad y no
salirnos del régimen donde Dijkstra es óptimo.

## Trade-offs

| Eje                | Decisión                 | Costo asumido                                            |
| ------------------ | ------------------------ | -------------------------------------------------------- |
| Algoritmo          | Dijkstra ampliado        | Si introducen costos negativos, hay que migrar a SPFA.   |
| Modelo de costo    | Suma ponderada           | Pierde Pareto; requiere re-tunear pesos por dominio.     |
| Estructuras        | `dict` + `heapq` puros   | Throughput aceptable hasta ~10⁴ edges; arriba toca C++.  |
| Datos              | `pandas.iterrows`        | Hot path lento; aceptable en la fase de construcción por comodidad.    |
| Plots              | matplotlib + seaborn     | No interactivo; para demos largas conviene plotly/bokeh. |


## Riesgos

1. **Operacional**: si `nodos_df.in_grid` o `operativo` traen ruido, las
   queries fallan silenciosamente con `*_CLOSED_OR_OOB`. Mitigar con
   validación al inicio + alertas si la tasa de errores crece.
2. **Escalabilidad**: en grafos densos (>10⁵ edges), la construcción del
   `GraphContext` con `iterrows` se vuelve dominante (varios segundos).
   Vectorizar con `to_numpy()` y `groupby` reduce ~10×.
3. **Multi-objetivo encubierto**: empacar tiempo + riesgo + fuel en un
   escalar oculta el trade-off real. Si negocio pide "minimo riesgo a
   costo de tiempo razonable", hay que exponer ambos.
4. **Determinismo en empates**: con `tie_breaker=None`, el orden del heap
   y el orden de inserción en `adj` deciden el resultado. Implícitamente
   reproducible mientras la generación use la misma semilla, pero frágil
   si se cambia la pipeline.

## Limitaciones

* Stops son **ordenados** (no TSP). Si negocio pide TSP-like sobre stops,
  el problema es NP-hard y hay que cambiar de algoritmo (Held-Karp para
  tamaños chicos; meta-heurísticas para grandes).
* `forbidden_edges` se compara como tuplas exactas; si el negocio quiere
  prohibir bidireccional, hay que pasarle ambas direcciones.
* Coords no listadas en `nodes_df` se asumen transitables. En operación
  esto debería ser parametrizable (estricto vs. permisivo).

## Plan de evolución (3 horizontes)

**Sprint 1 — semana corta**
* Vectorizar `build_directed_edges` y `from_dataframes` (de `iterrows` a
  numpy arrays).
* Suite de pytest con casos del README (~85 % cobertura).
* `logging` estructurado en lugar de `print`.

**Sprint 2 — un mes**
* API FastAPI con `/route` y `/route/batch`.
* Persistencia del `GraphContext` (pickle/parquet).
* Métricas Prometheus + dashboard Grafana.
* Tests de propiedad con `hypothesis` para invariantes
  (mandatory_edge_in_path ⇔ feasible).

**Sprint 3 — un trimestre**
* Migrar a `igraph` o `networkit` para grafos > 10⁵ edges.
* Soporte multi-arista obligatoria (bitmask de hasta 16 bits).
* Multi-objetivo Pareto opcional.
* Auditoría persistente en Postgres por cumplimiento regulatorio.

## ¿Cómo crece a 10× / 100×?

* **10× en queries (mismo grafo)**: ~210 s en lugar de 21 s. Aceptable
  con paralelismo simple (multiprocessing por shards de queries) o
  caché LRU sobre `(o, d, edge)` si hay repetidos.
* **10× en grafo (V·10)**: la construcción de adyacencia pasa de ~0.3 s
  a ~3 s. El solver sigue lineal-logarítmico, pero la huella de memoria
  sube. Solución: cargar el grafo una vez por proceso y compartirlo.
* **100× en grafo**: requiere C++ backend (`networkit`) y sharding
  geográfico (descomposición del grafo por regiones).

## Validación de extensiones

Las extensiones implementadas (stress, forbidden, audit) tienen evidencia
medible en el notebook:

* **Stress de 10 000 queries**: tiempo total y % de feasibles imprimibles
  permiten ver que el cached evita reconstruir el contexto N veces.
* **Forbidden**: el % esperado (10 %) se valida por la columna `reason`.
* **Audit**: cada `RouteResult.audit` es un dict serializable, listo
  para bitácora.

Lo que **falta** validar formalmente: tests unitarios automatizados
(pytest), benchmarks comparativos (Dijkstra simple vs. ampliado en grafos
sin restricciones, para confirmar mismo costo) y stress con grafos de
50×50 y 100×100.
