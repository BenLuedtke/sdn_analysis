"""Quick validation script for the OFAC Advanced XML parser."""
from pathlib import Path
from sanctions.parsers.ofac_advanced import OFACAdvancedParser

parser = OFACAdvancedParser("data/raw/SDN_ADVANCED.XML")
tables = parser.parse_all("data/processed/")

addr = tables["addresses"]
print("Addresses with city:", addr["city"].notna().sum())
print(addr[addr["city"].notna()].head(4).to_string())
print()
print("Entity types:")
print(tables["entities"]["entity_type"].value_counts())
print()
print("Feature types (top 12):")
print(tables["features"]["feature_type"].value_counts().head(12))
print()
print("Documents sample:")
print(tables["documents"].head(4).to_string())
