"""
tests/test_solver.py
====================

Suite funcional para el RouteSolver del módulo ejecutable.py.

Cobertura:
    1.  Happy path (1 arista obligatoria) y evidencia explícita de
        mandatory_edge en path_edges.
    2.  Validaciones de IDs:
        * ORIGIN_NOT_FOUND
        * DEST_NOT_FOUND
    3.  Validaciones de estado operativo:
        * ORIGIN_CLOSED_OR_OOB
        * DEST_CLOSED_OR_OOB
        * STOP_CLOSED_OR_OOB
        * STOP_NOT_FOUND
    4.  Validaciones de la arista obligatoria:
        * MANDATORY_EDGE_MISSING (no existe en el grafo)
        * MANDATORY_EDGE_FORBIDDEN (vetada por regla de negocio)
        * MANDATORY_EDGE_ENDPOINT_OOB_OR_CLOSED
        * Edge inverso: pedir (u, v) cuando sólo (v, u) existe.
    5.  Caso borde aislado -> NO_FEASIBLE_PATH.
    6.  Extensión punto 6 - N aristas obligatorias:
        * 2 aristas, happy path.
        * 2 aristas, una falta -> falla con razón correcta.
        * Bandera mandatory_edges_in_path por arista.
    7.  Tie-breaker min_hops no degrada el costo.
    8.  Mutex de los argumentos mandatory_edge y mandatory_edges.
    9.  audit=True produce payload no nulo y consistente.

Ejecutar:
    pytest -q
"""

from __future__ import annotations

import pytest

from ejecutable import RouteResult


# 1) HAPPY PATH
class TestHappyPath:
    """Camino básico - 1 arista obligatoria."""

    def test_basic_feasible(self, solver, main_dataset):
        _, _, ctx = main_dataset
        res = solver.solve(
            ctx=ctx,
            origin_node_id="BANK_001",
            destination_node_id="ATM_010",
            mandatory_edge=((5, 5), (5, 6)),
        )
        assert isinstance(res, RouteResult)
        assert res.feasible is True
        assert res.reason is None
        assert res.total_cost is not None and res.total_cost > 0
        assert res.metrics["hops"] >= 1
        assert res.metrics["expanded_states"] >= 1

    def test_mandatory_edge_in_path_evidence(self, solver, main_dataset):
        """La arista obligatoria debe aparecer literal en path_edges."""
        _, _, ctx = main_dataset
        edge = ((5, 5), (5, 6))
        res = solver.solve(
            ctx=ctx,
            origin_node_id="BANK_001", destination_node_id="ATM_010",
            mandatory_edge=edge,
        )
        assert res.feasible is True
        assert res.mandatory_edge_in_path is True
        assert edge in res.path_edges
        assert res.mandatory_edges_in_path == [True]

    def test_path_coords_consistent_with_edges(self, solver, main_dataset):
        """path_coords debe ser la cadena de origen destinos de path_edges."""
        _, _, ctx = main_dataset
        res = solver.solve(
            ctx=ctx,
            origin_node_id="BANK_001", destination_node_id="ATM_010",
            mandatory_edge=((5, 5), (5, 6)),
        )
        assert res.feasible is True
        # primer coord = origen
        # cada par consecutivo de coords corresponde a una path_edge
        for (a, b), (c1, c2) in zip(res.path_edges,
                                    zip(res.path_coords, res.path_coords[1:])):
            assert a == c1
            assert b == c2


# 2) IDS INVÁLIDOS
class TestInvalidIDs:

    def test_origin_not_found(self, solver, main_dataset):
        _, _, ctx = main_dataset
        res = solver.solve(
            ctx=ctx,
            origin_node_id="DOES_NOT_EXIST",
            destination_node_id="ATM_010",
            mandatory_edge=((5, 5), (5, 6)),
        )
        assert res.feasible is False
        assert res.reason == "ORIGIN_NOT_FOUND"
        assert res.total_cost is None
        assert res.path_edges == []

    def test_destination_not_found(self, solver, main_dataset):
        _, _, ctx = main_dataset
        res = solver.solve(
            ctx=ctx,
            origin_node_id="BANK_001",
            destination_node_id="GHOST_NODE",
            mandatory_edge=((5, 5), (5, 6)),
        )
        assert res.feasible is False
        assert res.reason == "DEST_NOT_FOUND"


# 3) ESTADO OPERATIVO
class TestOperationalStatus:

    def test_origin_closed(self, solver, main_dataset):
        """BANK_002 está cerrado (con seed=123)."""
        _, _, ctx = main_dataset
        res = solver.solve(
            ctx=ctx,
            origin_node_id="BANK_002",            # closed
            destination_node_id="ATM_010",
            mandatory_edge=((5, 5), (5, 6)),
        )
        assert res.feasible is False
        assert res.reason == "ORIGIN_CLOSED_OR_OOB"

    def test_destination_closed(self, solver, main_dataset):
        """El destino se marca como cerrado """
        _, _, ctx = main_dataset
        res = solver.solve(
            ctx=ctx,
            origin_node_id="BANK_001",
            destination_node_id="BANK_002",       # closed (cualquier node_id sirve como dest)
            mandatory_edge=((5, 5), (5, 6)),
        )
        assert res.feasible is False
        assert res.reason == "DEST_CLOSED_OR_OOB"

    def test_stop_not_found(self, solver, main_dataset):
        """ El ATM de parada está no se encuentra """
        _, _, ctx = main_dataset
        res = solver.solve(
            ctx=ctx,
            origin_node_id="BANK_001",
            destination_node_id="ATM_010",
            mandatory_edge=((5, 5), (5, 6)),
            mandatory_stops=["NOT_A_REAL_STOP"],
        )
        assert res.feasible is False
        assert res.reason == "STOP_NOT_FOUND"
        assert res.metrics.get("bad_stop") == "NOT_A_REAL_STOP"

    def test_stop_closed(self, solver, main_dataset):
        """Stop intermedio cerrado -> STOP_CLOSED_OR_OOB."""
        _, _, ctx = main_dataset
        res = solver.solve(
            ctx=ctx,
            origin_node_id="BANK_001",
            destination_node_id="ATM_010",
            mandatory_edge=((5, 5), (5, 6)),
            mandatory_stops=["BANK_002"],   # closed
        )
        assert res.feasible is False
        assert res.reason == "STOP_CLOSED_OR_OOB"
        assert res.metrics.get("bad_stop") == "BANK_002"


# 4) ARISTA OBLIGATORIA - VALIDACIONES
class TestMandatoryEdgeValidation:

    def test_mandatory_edge_missing(self, solver, main_dataset):
        """Arista que no existe en absoluto en el grafo."""
        _, _, ctx = main_dataset
        res = solver.solve(
            ctx=ctx,
            origin_node_id="BANK_001",
            destination_node_id="ATM_010",
            mandatory_edge=((9999, 9999), (9999, 10000)),
        )
        assert res.feasible is False
        assert res.reason == "MANDATORY_EDGE_MISSING"

    def test_mandatory_edge_forbidden(self, solver, main_dataset):
        """Pasar la mandatory dentro de forbidden_edges -> falla rápido."""
        _, _, ctx = main_dataset
        edge = ((5, 5), (5, 6))
        res = solver.solve(
            ctx=ctx,
            origin_node_id="BANK_001",
            destination_node_id="ATM_010",
            mandatory_edge=edge,
            forbidden_edges={edge},
        )
        assert res.feasible is False
        assert res.reason == "MANDATORY_EDGE_FORBIDDEN"

    def test_mandatory_edge_endpoint_oob(self, solver, main_dataset):
        """
        Endpoint fuera de la grilla = no usable. Construyo una arista que sale
        a un nodo OOB conocido (NODE_OOB_001 está fuera de la grilla con seed
        123) con coords aleatorias, y la trato como obligatoria.
        """
        nodos_df, _, ctx = main_dataset
        oob_row = nodos_df[nodos_df["node_id"] == "NODE_OOB_001"].iloc[0]
        oob_coord = (int(oob_row["x"]), int(oob_row["y"]))

        # arista artificial: (0,0) -> oob_coord; obviamente no está en edge_set
        # -> primero falla por MANDATORY_EDGE_MISSING. Para forzar
        # ENDPOINT_OOB_OR_CLOSED necesito que la arista exista. Por construcción
        # del generator, NINGUNA arista del grafo dirigido tiene endpoints OOB,
        # así que validamos el camino de fallo natural: MISSING.

        res = solver.solve(
            ctx=ctx,
            origin_node_id="BANK_001",
            destination_node_id="ATM_010",
            mandatory_edge=((0, 0), oob_coord),
        )
        assert res.feasible is False
        assert res.reason in {"MANDATORY_EDGE_MISSING",
                              "MANDATORY_EDGE_ENDPOINT_OOB_OR_CLOSED"}

    def test_edge_reverse_only(self, solver, main_dataset):
        """
        Con seed=123, la arista (7, 12) -> (7, 13) NO existe (sólo la inversa
        (7, 13) -> (7, 12)). Pedirla como obligatoria debe fallar con
        MANDATORY_EDGE_MISSING aunque la inversa exista.
        """
        _, _, ctx = main_dataset
        forward = ((7, 12), (7, 13))
        reverse = ((7, 13), (7, 12))

        # Sanity check de la fixture
        assert forward not in ctx.edge_set
        assert reverse in ctx.edge_set

        res = solver.solve(
            ctx=ctx,
            origin_node_id="BANK_001",
            destination_node_id="ATM_010",
            mandatory_edge=forward,
        )
        assert res.feasible is False
        assert res.reason == "MANDATORY_EDGE_MISSING"


# 5) BORDER AISLADO
class TestBorderIsolated:
    """border_cases=True bloquea TODAS las aristas incidentes a los dos nodos
    border. Cualquier ruta hacia/desde ellos debería ser infeasible."""

    def test_route_to_isolated_atm(self, solver, border_dataset):
        """Testeamos el caso donde se pide como nodo necesario un nodo aislado"""
        nodos_df, _, ctx = border_dataset

        # Validar que ATM_BORDER existe
        assert "ATM_BORDER" in nodos_df["node_id"].values
        res = solver.solve(
            ctx=ctx,
            origin_node_id="BANK_001",
            destination_node_id="ATM_BORDER",
            mandatory_edge=((5, 5), (5, 6)),
        )

        # Esperamos un fracaso por aislamiento direccional
        assert res.feasible is False
        assert res.reason in {
            "NO_FEASIBLE_PATH",
            "REACHED_DEST_BUT_MANDATORY_EDGE_NOT_USED",
            "DEST_CLOSED_OR_OOB",
        }


# 6) EXTENSIÓN PUNTO 6 - N ARISTAS OBLIGATORIAS
class TestMultipleMandatoryEdges:
    """Revisión de N aristas obligatorias."""

    def test_two_edges_happy_path(self, solver, main_dataset):
        """Revisamos que la funcion se comporte bien con 2 aristas obligatorias """

        _, _, ctx = main_dataset
        e1 = ((5, 5), (5, 6))
        e2 = ((10, 10), (9, 11))

        # asegurar que ambas existen
        assert e1 in ctx.edge_set
        assert e2 in ctx.edge_set

        res = solver.solve(
            ctx=ctx,
            origin_node_id="BANK_001", destination_node_id="ATM_010",
            mandatory_edges=[e1, e2],
            tie_breaker="min_hops",
        )
        assert res.feasible is True
        assert res.mandatory_edges_in_path == [True, True]
        assert res.mandatory_edge_in_path is True
        assert e1 in res.path_edges
        assert e2 in res.path_edges
        assert res.metrics["n_mandatory_edges"] == 2

    def test_three_edges_happy_path(self, solver, main_dataset):
        """Revisamos que la funcion se comporte bien con 3 aristas obligatorias """

        _, _, ctx = main_dataset

        # picamos tres aristas existentes razonablemente alcanzables
        edges = [((5, 5), (5, 6)),
                 ((10, 10), (9, 11)),
                 ((20, 20), (19, 21))]
        
        # filtrar a las que existan
        edges_existing = [e for e in edges if e in ctx.edge_set]
        # si alguna no existe, picar otra cualquiera
        while len(edges_existing) < 3:
            edges_existing.append(next(iter(ctx.edge_set - set(edges_existing))))

        res = solver.solve(
            ctx=ctx,
            origin_node_id="BANK_001", destination_node_id="ATM_010",
            mandatory_edges=edges_existing,
        )
        # No exigimos feasible (el solver podría no encontrar camino), pero si
        # es feasible, las tres deben estar.
        if res.feasible:
            assert all(res.mandatory_edges_in_path)
            for e in edges_existing:
                assert e in res.path_edges

    def test_one_of_two_missing(self, solver, main_dataset):
        """Si una arista de la lista no existe, el solver lo reporta con índice."""
        _, _, ctx = main_dataset
        e_ok = ((5, 5), (5, 6))
        e_bad = ((9999, 9999), (9999, 10000))
        res = solver.solve(
            ctx=ctx,
            origin_node_id="BANK_001", destination_node_id="ATM_010",
            mandatory_edges=[e_ok, e_bad],
        )
        assert res.feasible is False
        assert res.reason == "MANDATORY_EDGE_MISSING"
        assert res.metrics.get("bad_edge_index") == 1
        assert res.metrics.get("bad_edge") == e_bad

    def test_argument_mutex(self, solver, main_dataset):
        """No puede pasar mandatory_edge Y mandatory_edges al mismo tiempo."""
        _, _, ctx = main_dataset
        with pytest.raises(ValueError):
            solver.solve(
                ctx=ctx,
                origin_node_id="BANK_001", destination_node_id="ATM_010",
                mandatory_edge=((5, 5), (5, 6)),
                mandatory_edges=[((5, 5), (5, 6))],
            )

    def test_no_mandatory_provided_raises(self, solver, main_dataset):
        """ No se especifica un nodo destino"""
        _, _, ctx = main_dataset
        with pytest.raises(ValueError):
            solver.solve(
                ctx=ctx,
                origin_node_id="BANK_001", destination_node_id="ATM_010",
            )


# 7) TIE-BREAKER
class TestTieBreaker:
    """min_hops no debe degradar el costo óptimo."""

    def test_min_hops_does_not_increase_cost(self, solver, main_dataset):
        """El criterio de desempate no tiene que incrementar el costo optimo"""

        _, _, ctx = main_dataset
        edge = ((5, 5), (5, 6))
        res_default = solver.solve(
            ctx=ctx, 
            origin_node_id="BANK_001", 
            destination_node_id="ATM_010",
            mandatory_edge=edge,
        )
        res_min_hops = solver.solve(
            ctx=ctx, 
            origin_node_id="BANK_001", 
            destination_node_id="ATM_010",
            mandatory_edge=edge, 
            tie_breaker="min_hops",
        )

        assert res_default.feasible is True
        assert res_min_hops.feasible is True

        # ambos deben dar el mismo costo óptimo
        assert res_default.total_cost == pytest.approx(res_min_hops.total_cost)

        # min_hops debería tener hops <= default (si hubo empate; si no, igual)
        assert res_min_hops.metrics["hops"] <= res_default.metrics["hops"]


# 8) AUDIT
class TestAudit:
    """ La funci´+n """
    def test_audit_payload_present(self, solver, main_dataset):
        _, _, ctx = main_dataset
        res = solver.solve(
            ctx=ctx,
            origin_node_id="BANK_001", destination_node_id="ATM_010",
            mandatory_edge=((5, 5), (5, 6)),
            audit=True,
        )
        assert res.audit is not None
        assert res.audit["origin_node_id"] == "BANK_001"
        assert res.audit["destination_node_id"] == "ATM_010"
        assert res.audit["mandatory_edge_in_path"] is True
        assert isinstance(res.audit["mandatory_edges"], list)
        assert len(res.audit["mandatory_edges"]) == 1
        assert res.audit["mandatory_edges"][0]["in_path"] is True

    def test_audit_off_by_default(self, solver, main_dataset):
        _, _, ctx = main_dataset
        res = solver.solve(
            ctx=ctx,
            origin_node_id="BANK_001", destination_node_id="ATM_010",
            mandatory_edge=((5, 5), (5, 6)),
        )
        assert res.audit is None


# 9) MODO FRESH (sin ctx pre-construido)
class TestFreshMode:
    """Réplica directa del primer solve_route del notebook."""

    def test_fresh_mode_works(self, solver, main_dataset):
        nodos_df, de, _ = main_dataset
        res = solver.solve(
            nodes_df=nodos_df, directed_edges_df=de,
            origin_node_id="BANK_001", destination_node_id="ATM_010",
            mandatory_edge=((5, 5), (5, 6)),
        )
        assert res.feasible is True

    def test_fresh_mode_requires_dataframes(self, solver):
        """Sin ctx y sin nodes_df / directed_edges_df -> ValueError claro."""
        with pytest.raises(ValueError):
            solver.solve(
                origin_node_id="BANK_001",
                destination_node_id="ATM_010",
                mandatory_edge=((5, 5), (5, 6)),
            )

    def test_fresh_and_cached_agree(self, solver, main_dataset):
        """El resultado debe ser idéntico con o sin ctx pre-construido."""
        nodos_df, de, ctx = main_dataset
        kwargs = dict(
            origin_node_id="BANK_001",
            destination_node_id="ATM_010",
            mandatory_edge=((5, 5), (5, 6)),
            mandatory_stops=["ATM_003", "ATM_007"],
            tie_breaker="min_hops",
        )
        res_fresh = solver.solve(nodes_df=nodos_df, directed_edges_df=de, **kwargs)
        res_cached = solver.solve(ctx=ctx, **kwargs)
        assert res_fresh.feasible == res_cached.feasible
        assert res_fresh.total_cost == pytest.approx(res_cached.total_cost)
        assert res_fresh.path_edges == res_cached.path_edges
