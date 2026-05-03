"""
Build the SDN ownership graph from OFAC Advanced XML ProfileRelationships.

The OFAC Advanced XML ProfileRelationships section contains 8,558 directed
relationships between explicitly sanctioned entities. Relationship types used
to build the ownership graph:

    15003  Owned or Controlled By    From=asset, To=owner  → edge: owner→asset
    92019  Owns, controls, or operates  From=owner, To=asset → edge: owner→asset

Other relationship types (family, associate, support) are parsed but not added
to the ownership graph; they are returned separately for potential use in
community detection or additional analysis.

Important limitation:
    All nodes in this graph are explicitly listed SDN entities. The 50% rule is
    most operationally significant for identifying entities that are NOT on the
    list but should be blocked due to SDN ownership. Extending this graph with
    off-list entities requires additional data sources (OpenSanctions, commercial
    corporate registries, Sayari, Kharon, etc.).
"""

from __future__ import annotations

from pathlib import Path

import lxml.etree as ET
import networkx as nx
import pandas as pd

NS = "https://sanctionslistservice.ofac.treas.gov/api/PublicationPreview/exports/ADVANCED_XML"


def _q(tag: str) -> str:
    return f"{{{NS}}}{tag}"


# Relationship types that represent ownership/control
_OWNERSHIP_TYPES = {
    "15003",  # Owned or Controlled By  (From=asset, To=owner → reverse edge)
    "92019",  # Owns, controls, or operates (From=owner, To=asset → direct edge)
}

# Human-readable labels for all relationship types
REL_TYPE_NAMES = {
    "1555":  "Associate Of",
    "15001": "Providing support to",
    "15002": "Acting for or on behalf of",
    "15003": "Owned or Controlled By",
    "15004": "Family member of",
    "91422": "Playing a significant role in",
    "91725": "Leader or official of",
    "91900": "Principal Executive Officer",
    "92019": "Owns, controls, or operates",
    "92122": "Property in the interest of",
}

REL_QUALITY_NAMES = {
    "1":    "High",
    "1540": "High",
    "1541": "Medium",
    "1542": "Low",
    "1543": "Unknown",
}


def build_ownership_graph(xml_path: str | Path) -> tuple[nx.DiGraph, pd.DataFrame, dict]:
    """
    Parse the OFAC Advanced XML and return:
        G          — directed ownership graph (owner → asset edges)
        entities   — DataFrame of all DistinctParty nodes with name, type, programs
        rel_df     — DataFrame of all ProfileRelationships (all types)
    """
    tree = ET.parse(str(xml_path))
    root = tree.getroot()

    # ── Build program lookup (ProfileID → list of programs) ───────────────────
    programs_by_profile: dict[str, list[str]] = {}
    for entry in root.find(_q("SanctionsEntries")):
        pid = entry.get("ProfileID")
        if not pid:
            continue
        for measure in entry.findall(_q("SanctionsMeasure")):
            comment = (measure.findtext(_q("Comment")) or "").strip()
            if comment:
                programs_by_profile.setdefault(pid, []).append(comment)

    # ── Parse reference values ─────────────────────────────────────────────────
    party_subtype: dict[str, str] = {}
    party_type: dict[str, tuple] = {}  # subtype_id → (text, type_id)
    type_name: dict[str, str] = {}

    for child in root.find(_q("ReferenceValueSets")):
        tag = ET.QName(child).localname
        for item in child:
            iid = item.get("ID")
            val = (item.text or "").strip()
            if tag == "PartySubTypeValues":
                party_type[iid] = (val, item.get("PartyTypeID"))
            elif tag == "PartyTypeValues":
                type_name[iid] = val

    def resolve_type(subtype_id: str) -> str:
        subtype_text, type_id = party_type.get(subtype_id, (None, None))
        if subtype_text and subtype_text != "Unknown":
            return subtype_text
        return type_name.get(type_id, "Unknown")

    # ── Parse DistinctParties → entity nodes ──────────────────────────────────
    node_attrs: dict[str, dict] = {}
    for party in root.find(_q("DistinctParties")):
        fixed_ref = party.get("FixedRef")
        profile = party.find(_q("Profile"))
        if profile is None:
            continue
        profile_id = profile.get("ID", fixed_ref)
        subtype_id = profile.get("PartySubTypeID")

        # Primary name
        primary_name = None
        identity = profile.find(_q("Identity"))
        if identity is not None:
            for alias in identity.findall(_q("Alias")):
                if alias.get("Primary") == "true":
                    nv = alias.find(
                        f"{_q('DocumentedName')}/{_q('DocumentedNamePart')}/{_q('NamePartValue')}"
                    )
                    if nv is not None and nv.text:
                        primary_name = nv.text.strip()
                    break

        node_attrs[profile_id] = {
            "entity_id":   fixed_ref,
            "full_name":   primary_name or f"#{profile_id}",
            "entity_type": resolve_type(subtype_id or ""),
            "programs":    programs_by_profile.get(profile_id, []),
            "on_sdn":      True,
        }

    # ── Parse ProfileRelationships → edges ────────────────────────────────────
    rel_rows = []
    G = nx.DiGraph()

    # Add all nodes first
    for pid, attrs in node_attrs.items():
        G.add_node(pid, **attrs)

    for rel in root.find(_q("ProfileRelationships")):
        from_id  = rel.get("From-ProfileID")
        to_id    = rel.get("To-ProfileID")
        rel_type = rel.get("RelationTypeID")
        quality  = rel.get("RelationQualityID", "")
        former   = rel.get("Former", "false") == "true"

        rel_rows.append({
            "from_id":    from_id,
            "to_id":      to_id,
            "rel_type_id": rel_type,
            "rel_type":   REL_TYPE_NAMES.get(rel_type, rel_type),
            "quality":    REL_QUALITY_NAMES.get(quality, quality),
            "former":     former,
            "from_name":  node_attrs.get(from_id, {}).get("full_name", f"#{from_id}"),
            "to_name":    node_attrs.get(to_id, {}).get("full_name", f"#{to_id}"),
        })

        if rel_type not in _OWNERSHIP_TYPES or former:
            continue

        # Add ownership edge: owner → asset
        if rel_type == "15003":  # Owned or Controlled By: From=asset, To=owner
            owner, asset = to_id, from_id
        else:                    # Owns, controls, or operates: From=owner, To=asset
            owner, asset = from_id, to_id

        if owner in G and asset in G:
            G.add_edge(owner, asset, rel_type=rel_type, quality=quality)

    entities_df = pd.DataFrame(node_attrs.values())
    rel_df = pd.DataFrame(rel_rows)

    return G, entities_df, rel_df
