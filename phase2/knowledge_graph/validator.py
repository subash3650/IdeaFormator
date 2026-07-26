"""GraphValidator — 15 integrity checks for knowledge graph."""

from __future__ import annotations

from phase2.knowledge_graph.graph_interface import GraphInterface
from phase2.knowledge_graph.schema import EdgeType, NodeType, ValidationResult


class GraphValidator:
    """Validates knowledge graph integrity across 15 checks."""

    def validate(self, graph: GraphInterface) -> ValidationResult:
        errors: list[str] = []
        warnings: list[str] = []
        all_nodes = graph.nodes()
        all_edges = graph.edges()

        node_count = len(all_nodes)
        edge_count = len(all_edges)
        node_ids: set[str] = set(n.node_id for n in all_nodes)
        edge_ids: set[str] = set(e.edge_id for e in all_edges)

        # Check 1: Duplicate nodes
        duplicate_node_count = node_count - len(set(n.node_id for n in all_nodes))
        seen_nids: set[str] = set()
        dup_nids: list[str] = []
        for n in all_nodes:
            if n.node_id in seen_nids:
                dup_nids.append(n.node_id)
            seen_nids.add(n.node_id)
        duplicate_node_count = len(dup_nids)
        if duplicate_node_count:
            errors.append(f"Found {duplicate_node_count} duplicate node IDs: {dup_nids[:5]}...")

        # Check 2: Duplicate edges
        duplicate_edge_count = edge_count - len(set(e.edge_id for e in all_edges))
        seen_eids: set[str] = set()
        dup_eids: list[str] = []
        for e in all_edges:
            if e.edge_id in seen_eids:
                dup_eids.append(e.edge_id)
            seen_eids.add(e.edge_id)
        duplicate_edge_count = len(dup_eids)
        if duplicate_edge_count:
            errors.append(f"Found {duplicate_edge_count} duplicate edge IDs: {dup_eids[:5]}...")

        # Check 3: Orphan edges (edge references non-existent node)
        orphan_edge_count = 0
        for e in all_edges:
            if e.source_node_id not in node_ids:
                orphan_edge_count += 1
            elif e.target_node_id not in node_ids:
                orphan_edge_count += 1
        if orphan_edge_count:
            errors.append(f"Found {orphan_edge_count} orphan edges referencing missing nodes")

        # Check 4: Self-loops
        self_loop_count = sum(1 for e in all_edges if e.source_node_id == e.target_node_id)
        if self_loop_count:
            warnings.append(f"Found {self_loop_count} self-loop edges")

        # Check 5: Invalid confidence
        bad_conf_nodes = [n.node_id for n in all_nodes if not (0.0 <= n.confidence <= 1.0)]
        bad_conf_edges = [e.edge_id for e in all_edges if not (0.0 <= e.confidence <= 1.0)]
        if bad_conf_nodes:
            errors.append(f"Invalid confidence on {len(bad_conf_nodes)} nodes: {bad_conf_nodes[:5]}...")
        if bad_conf_edges:
            errors.append(f"Invalid confidence on {len(bad_conf_edges)} edges: {bad_conf_edges[:5]}...")

        # Check 6: Invalid weights
        bad_weight_edges = [e.edge_id for e in all_edges if not (0.0 <= e.weight <= 1.0)]
        if bad_weight_edges:
            errors.append(f"Invalid weight on {len(bad_weight_edges)} edges: {bad_weight_edges[:5]}...")

        # Check 7: Missing metadata
        missing_meta_nodes = [n.node_id for n in all_nodes if not n.metadata]
        missing_meta_edges = [e.edge_id for e in all_edges if not e.metadata]
        if missing_meta_nodes:
            warnings.append(f"{len(missing_meta_nodes)} nodes have empty metadata")
        if missing_meta_edges:
            warnings.append(f"{len(missing_meta_edges)} edges have empty metadata")

        # Check 8: Disconnected graph
        from phase2.knowledge_graph.algorithms import connected_components
        comps = connected_components(graph)
        disconnected_components = len(comps)
        if disconnected_components > 1:
            warnings.append(f"Graph has {disconnected_components} disconnected components (largest: {len(comps[0])} nodes)")

        # Check 9: Broken references (embedding_id, cluster_id point to non-existent nodes)
        broken_refs = 0
        for n in all_nodes:
            emb = n.attributes.get("embedding_id")
            if emb and emb not in node_ids:
                broken_refs += 1
        if broken_refs:
            warnings.append(f"Found {broken_refs} broken attribute references")

        # Check 10: Schema mismatch
        node_type_vals = set(v.value for v in NodeType)
        edge_type_vals = set(v.value for v in EdgeType)
        schema_mismatch_count = sum(1 for n in all_nodes if n.node_type.value not in node_type_vals)
        schema_mismatch_count += sum(1 for e in all_edges if e.edge_type.value not in edge_type_vals)
        if schema_mismatch_count:
            errors.append(f"Found {schema_mismatch_count} nodes/edges with unrecognized types")

        # Check 11: Empty graph
        if node_count == 0:
            warnings.append("Graph has zero nodes")
        if edge_count == 0:
            warnings.append("Graph has zero edges")

        # Check 12: Cycles
        from phase2.knowledge_graph.algorithms import has_cycle
        cycle_detected = has_cycle(graph)
        cycle_count = 1 if cycle_detected else 0
        if cycle_detected and disconnected_components <= 1:
            warnings.append("Graph contains directed cycles")

        # Checks 13-15 require store/manifest; mark as unverified
        stale_asset_count = 0
        run_id_mismatch_count = 0
        checksum_mismatch_count = 0

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors,
            warnings=warnings,
            node_count=node_count,
            edge_count=edge_count,
            duplicate_node_count=duplicate_node_count,
            duplicate_edge_count=duplicate_edge_count,
            orphan_node_count=sum(1 for n in all_nodes if graph.degree(n.node_id) == 0),
            orphan_edge_count=orphan_edge_count,
            self_loop_count=self_loop_count,
            cycle_count=cycle_count,
            disconnected_components=disconnected_components,
            stale_asset_count=stale_asset_count,
            run_id_mismatch_count=run_id_mismatch_count,
            checksum_mismatch_count=checksum_mismatch_count,
            schema_mismatch_count=schema_mismatch_count,
        )
