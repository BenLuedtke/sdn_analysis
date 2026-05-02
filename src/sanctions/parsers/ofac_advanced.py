"""
OFAC SDN / Consolidated Advanced XML parser.

Source:   https://sanctionslistservice.ofac.treas.gov/api/download/SDN_ADVANCED.XML
          https://sanctionslistservice.ofac.treas.gov/api/download/CONS_ADVANCED.XML
Format:   XML with namespace https://...ADVANCED_XML
Cadence:  Updated as designations change, typically multiple times per week.

Produces five normalized DataFrames, persisted as parquet to data/processed/:

    entities  — one row per SDN entry
                (entity_id, full_name, entity_type, programs, listed_date, list_id)

    akas      — one row per alias / documented name
                (entity_id, aka_name, alias_type, is_primary, is_weak, script)

    addresses — one row per location linked to an entity
                (entity_id, location_id, country, city, region, postal_code, address)

    documents — one row per ID document
                (entity_id, doc_type, doc_number, issuing_country, validity)

    features  — one row per feature (DOB, gender, nationality, vessel details, etc.)
                (entity_id, feature_type, value, location_id)

Usage:
    from sanctions.parsers.ofac_advanced import OFACAdvancedParser
    parser = OFACAdvancedParser("data/raw/SDN_ADVANCED.XML")
    tables = parser.parse_all("data/processed/")
"""

from __future__ import annotations

from pathlib import Path

import lxml.etree as ET
import pandas as pd

NS = "https://sanctionslistservice.ofac.treas.gov/api/PublicationPreview/exports/ADVANCED_XML"
_ns = {"s": NS}


def _q(tag: str) -> str:
    """Return Clark-notation qualified tag name."""
    return f"{{{NS}}}{tag}"


def _localname(element: ET.Element) -> str:
    return ET.QName(element).localname


class OFACAdvancedParser:
    def __init__(self, xml_path: str | Path) -> None:
        self._path = Path(xml_path)
        tree = ET.parse(str(self._path))
        self.root = tree.getroot()
        self.refs = self._build_refs()

    # ------------------------------------------------------------------
    # Reference value tables
    # ------------------------------------------------------------------

    def _build_refs(self) -> dict[str, dict]:
        """Build lookup dicts from ReferenceValueSets. Values are element text."""
        refs: dict[str, dict] = {
            "country": {},          # CountryID -> country name
            "area_code": {},        # AreaCodeID -> ISO-2 code  (stored in Description attr)
            "party_subtype": {},    # subtype_id -> (text, party_type_id)
            "party_type": {},       # type_id -> "Individual" / "Entity" / "Vessel" / ...
            "feature_type": {},     # feature_type_id -> feature name
            "doc_type": {},         # doc_type_id -> document type name
            "alias_type": {},       # alias_type_id -> "A.K.A." / "F.K.A." / ...
            "script": {},           # script_id -> script name
            "program": {},          # program_id -> program code
            "legal_basis": {},      # legal_basis_id -> SanctionsProgramID
            "loc_part_type": {},    # loc_part_type_id -> part name (City, Region, etc.)
            "validity": {},         # validity_id -> "Valid" / "Invalid"
        }

        ref_sets = self.root.find(_q("ReferenceValueSets"))
        if ref_sets is None:
            return refs

        for child in ref_sets:
            tag = _localname(child)
            for item in child:
                item_id = item.get("ID")
                value = (item.text or "").strip()
                if not item_id:
                    continue

                if tag == "CountryValues":
                    refs["country"][item_id] = value
                elif tag == "AreaCodeValues":
                    refs["area_code"][item_id] = item.get("Description", value)
                elif tag == "PartySubTypeValues":
                    refs["party_subtype"][item_id] = (value, item.get("PartyTypeID"))
                elif tag == "PartyTypeValues":
                    refs["party_type"][item_id] = value
                elif tag == "FeatureTypeValues":
                    refs["feature_type"][item_id] = value
                elif tag == "IDRegDocTypeValues":
                    refs["doc_type"][item_id] = value
                elif tag == "AliasTypeValues":
                    refs["alias_type"][item_id] = value
                elif tag == "ScriptValues":
                    refs["script"][item_id] = value
                elif tag == "SanctionsProgramValues":
                    refs["program"][item_id] = value
                elif tag == "LegalBasisValues":
                    refs["legal_basis"][item_id] = item.get("SanctionsProgramID")
                elif tag == "LocPartTypeValues":
                    refs["loc_part_type"][item_id] = value
                elif tag == "ValidityValues":
                    refs["validity"][item_id] = value

        return refs

    def _resolve_entity_type(self, subtype_id: str | None) -> str | None:
        """
        Two-step lookup: PartySubTypeID -> PartyTypeID -> PartyType name.
        Falls back to PartyType when SubType text is "Unknown" or blank.
        """
        if not subtype_id:
            return None
        subtype_text, type_id = self.refs["party_subtype"].get(subtype_id, (None, None))
        if subtype_text and subtype_text != "Unknown":
            return subtype_text
        return self.refs["party_type"].get(type_id)

    # ------------------------------------------------------------------
    # Section parsers (build lookup tables before the main entity loop)
    # ------------------------------------------------------------------

    def _parse_sanctions_entries(self) -> dict[str, dict]:
        """ProfileID -> {programs, listed_date, list_id}."""
        entries: dict[str, dict] = {}
        section = self.root.find(_q("SanctionsEntries"))
        if section is None:
            return entries

        for entry in section:
            profile_id = entry.get("ProfileID")
            if not profile_id:
                continue

            list_id = entry.get("ListID")

            date_el = entry.find(f"{_q('EntryEvent')}/{_q('Date')}")
            listed_date = None
            if date_el is not None:
                y = date_el.findtext(_q("Year"))
                mo = date_el.findtext(_q("Month"))
                d = date_el.findtext(_q("Day"))
                if y:
                    listed_date = f"{y}-{(mo or '01').zfill(2)}-{(d or '01').zfill(2)}"

            programs = []
            for measure in entry.findall(_q("SanctionsMeasure")):
                comment = (measure.findtext(_q("Comment")) or "").strip()
                if comment:
                    programs.append(comment)

            if profile_id not in entries:
                entries[profile_id] = {
                    "programs": programs,
                    "listed_date": listed_date,
                    "list_id": list_id,
                }

        return entries

    def _parse_locations(self) -> dict[str, dict]:
        """LocationID -> {country, area_code, city, region, postal_code, address}."""
        locations: dict[str, dict] = {}
        section = self.root.find(_q("Locations"))
        if section is None:
            return locations

        for loc in section:
            loc_id = loc.get("ID")
            if not loc_id:
                continue

            country_el = loc.find(_q("LocationCountry"))
            country = None
            if country_el is not None:
                country = self.refs["country"].get(country_el.get("CountryID", ""))

            area_el = loc.find(_q("LocationAreaCode"))
            area_code = None
            if area_el is not None:
                area_code = self.refs["area_code"].get(area_el.get("AreaCodeID", ""))

            # Structured address parts: City, Region, Postal Code, Address, etc.
            parts: dict[str, str] = {}
            for part in loc.findall(_q("LocationPart")):
                part_type = self.refs["loc_part_type"].get(
                    part.get("LocPartTypeID", ""), part.get("LocPartTypeID", "")
                )
                value_el = part.find(f"{_q('LocationPartValue')}/{_q('Value')}")
                if value_el is not None and value_el.text:
                    parts[part_type] = value_el.text.strip()

            # Combine up to three address lines; LocPartType names are uppercase in the XML
            address_parts = [parts.get(k) for k in ("ADDRESS1", "ADDRESS2", "ADDRESS3")]
            locations[loc_id] = {
                "country": country,
                "area_code": area_code,
                "city": parts.get("CITY"),
                "region": parts.get("STATE/PROVINCE") or parts.get("REGION"),
                "postal_code": parts.get("POSTAL CODE"),
                "address": ", ".join(p for p in address_parts if p) or None,
            }

        return locations

    def _parse_id_documents(self) -> dict[str, list[dict]]:
        """IdentityID -> list of document dicts."""
        docs: dict[str, list] = {}
        section = self.root.find(_q("IDRegDocuments"))
        if section is None:
            return docs

        for doc in section:
            identity_id = doc.get("IdentityID")
            if not identity_id:
                continue

            doc_type = self.refs["doc_type"].get(doc.get("IDRegDocTypeID", ""))
            doc_number = (doc.findtext(_q("IDRegistrationNo")) or "").strip() or None
            issuing_country = self.refs["country"].get(doc.get("IssuedBy-CountryID", ""))
            validity = self.refs["validity"].get(doc.get("ValidityID", ""))

            docs.setdefault(identity_id, []).append({
                "doc_type": doc_type,
                "doc_number": doc_number,
                "issuing_country": issuing_country,
                "validity": validity,
            })

        return docs

    # ------------------------------------------------------------------
    # Main parse
    # ------------------------------------------------------------------

    def parse_all(self, output_dir: str | Path) -> dict[str, pd.DataFrame]:
        """
        Parse all sections into five normalized DataFrames.
        Writes parquet files to output_dir and returns the dict.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        sanctions_entries = self._parse_sanctions_entries()
        locations = self._parse_locations()
        id_documents = self._parse_id_documents()

        entity_rows: list[dict] = []
        aka_rows: list[dict] = []
        address_rows: list[dict] = []
        document_rows: list[dict] = []
        feature_rows: list[dict] = []

        parties = self.root.find(_q("DistinctParties"))
        if parties is None:
            raise ValueError("No DistinctParties section found in XML.")

        for party in parties:
            fixed_ref = party.get("FixedRef")
            profile = party.find(_q("Profile"))
            if profile is None:
                continue

            profile_id = profile.get("ID", fixed_ref)
            subtype_id = profile.get("PartySubTypeID")
            entity_type = self._resolve_entity_type(subtype_id)
            entry = sanctions_entries.get(profile_id, {})

            identity = profile.find(_q("Identity"))
            identity_id = identity.get("ID") if identity is not None else None
            primary_name = None

            # --- Names / AKAs ---
            if identity is not None:
                for alias in identity.findall(_q("Alias")):
                    alias_type_id = alias.get("AliasTypeID")
                    is_primary = alias.get("Primary") == "true"
                    is_weak = alias.get("LowQuality") == "true"
                    alias_type = self.refs["alias_type"].get(alias_type_id, "A.K.A.")

                    doc_name = alias.find(_q("DocumentedName"))
                    if doc_name is None:
                        continue

                    name_parts = doc_name.findall(
                        f"{_q('DocumentedNamePart')}/{_q('NamePartValue')}"
                    )
                    if not name_parts:
                        continue

                    script_id = name_parts[0].get("ScriptID")
                    aka_name = " ".join(p.text for p in name_parts if p.text).strip()
                    script = self.refs["script"].get(script_id or "")

                    if is_primary:
                        primary_name = aka_name

                    aka_rows.append({
                        "entity_id": fixed_ref,
                        "aka_name": aka_name,
                        "alias_type": alias_type,
                        "is_primary": is_primary,
                        "is_weak": is_weak,
                        "script": script,
                    })

            entity_rows.append({
                "entity_id": fixed_ref,
                "full_name": primary_name,
                "entity_type": entity_type,
                "programs": entry.get("programs", []),
                "listed_date": entry.get("listed_date"),
                "list_id": entry.get("list_id"),
            })

            # --- Features (DOB, gender, nationality, vessel details, etc.) ---
            for feature in profile.findall(_q("Feature")):
                feature_type_id = feature.get("FeatureTypeID")
                feature_type = self.refs["feature_type"].get(feature_type_id, feature_type_id)

                for fv in feature.findall(_q("FeatureVersion")):
                    # Location-linked feature (country of registration, org location, etc.)
                    vl = fv.find(_q("VersionLocation"))
                    if vl is not None:
                        loc = locations.get(vl.get("LocationID", ""), {})
                        feature_rows.append({
                            "entity_id": fixed_ref,
                            "feature_type": feature_type,
                            "value": loc.get("country"),
                            "location_id": vl.get("LocationID"),
                        })
                        continue

                    # Date-period feature (Birthdate, etc.)
                    dp = fv.find(_q("DatePeriod"))
                    if dp is not None:
                        y = dp.findtext(f"{_q('Start')}/{_q('From')}/{_q('Year')}") or \
                            dp.findtext(f"{_q('Start')}/{_q('Year')}")
                        mo = dp.findtext(f"{_q('Start')}/{_q('From')}/{_q('Month')}") or \
                            dp.findtext(f"{_q('Start')}/{_q('Month')}")
                        d = dp.findtext(f"{_q('Start')}/{_q('From')}/{_q('Day')}") or \
                            dp.findtext(f"{_q('Start')}/{_q('Day')}")
                        if y:
                            date_val = y
                            if mo:
                                date_val += f"-{mo.zfill(2)}"
                                if d:
                                    date_val += f"-{d.zfill(2)}"
                            feature_rows.append({
                                "entity_id": fixed_ref,
                                "feature_type": feature_type,
                                "value": date_val,
                                "location_id": None,
                            })
                        continue

                    # Detail/text value feature (gender, vessel type, etc.)
                    for detail in fv.findall(_q("VersionDetail")):
                        detail_ref = detail.get("DetailReferenceID")
                        # DetailReference values are in refs["detail_reference"] if we added it
                        # For now store the raw text content
                        text = (detail.text or "").strip()
                        if text:
                            feature_rows.append({
                                "entity_id": fixed_ref,
                                "feature_type": feature_type,
                                "value": text,
                                "location_id": None,
                            })

            # --- Addresses (location-linked features of address type) ---
            # Also captured via feature_rows above; here we build the flat addresses table
            for feature in profile.findall(_q("Feature")):
                for fv in feature.findall(_q("FeatureVersion")):
                    vl = fv.find(_q("VersionLocation"))
                    if vl is None:
                        continue
                    loc_id = vl.get("LocationID")
                    loc = locations.get(loc_id, {})
                    if loc.get("country") or loc.get("city"):
                        address_rows.append({
                            "entity_id": fixed_ref,
                            "location_id": loc_id,
                            "country": loc.get("country"),
                            "area_code": loc.get("area_code"),
                            "city": loc.get("city"),
                            "region": loc.get("region"),
                            "postal_code": loc.get("postal_code"),
                            "address": loc.get("address"),
                        })

            # --- ID Documents ---
            if identity_id and identity_id in id_documents:
                for doc in id_documents[identity_id]:
                    document_rows.append({"entity_id": fixed_ref, **doc})

        # ------------------------------------------------------------------
        # Build DataFrames and write parquet
        # ------------------------------------------------------------------
        entities = pd.DataFrame(entity_rows)
        if not entities.empty:
            entities["listed_date"] = pd.to_datetime(entities["listed_date"], errors="coerce")

        tables = {
            "entities": entities,
            "akas": pd.DataFrame(aka_rows),
            "addresses": pd.DataFrame(address_rows).drop_duplicates(
                subset=["entity_id", "location_id"]
            ),
            "documents": pd.DataFrame(document_rows),
            "features": pd.DataFrame(feature_rows),
        }

        for name, df in tables.items():
            df.to_parquet(output_dir / f"{name}.parquet", index=False)
            print(f"  {name}: {len(df):,} rows → {name}.parquet")

        return tables
