#!/usr/bin/env python3
"""
Parse coverage.xml and print total coverage percentage as a float.
Works with coverage.py XML format.
"""
import sys
import xml.etree.ElementTree as ET

if len(sys.argv) < 2:
    print("0.0")
    sys.exit(0)

path = sys.argv[1]
tree = ET.parse(path)
root = tree.getroot()

# coverage.py stores line-rate on root as 'line-rate' (0..1)
rate = root.get('line-rate', None)
if rate is None:
    print("0.0")
    sys.exit(0)

pct = float(rate) * 100.0
print(f"{pct:.2f}")
