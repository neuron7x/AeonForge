#!/usr/bin/env python3
import sys, xml.etree.ElementTree as ET

def get_line_rate(xml_path: str) -> float:
    tree = ET.parse(xml_path)
    root = tree.getroot()
    # Cobertura XML: attributes 'line-rate' or lines-covered/lines-valid
    rate = root.attrib.get("line-rate")
    if rate is not None:
        return float(rate)
    # Fallback
    metrics = root.find(".//coverage")
    if metrics is not None and 'line-rate' in metrics.attrib:
        return float(metrics.attrib['line-rate'])
    # Try totals
    lines_valid = 0
    lines_covered = 0
    for pkg in root.findall(".//packages/package"):
        for cls in pkg.findall(".//classes/class"):
            lines = cls.find("lines")
            if lines is None:
                continue
            for line in lines.findall("line"):
                lines_valid += 1
                if int(line.attrib.get("hits", "0")) > 0:
                    lines_covered += 1
    if lines_valid == 0:
        return 0.0
    return lines_covered / lines_valid

def pct(frac: float) -> str:
    return f"{frac*100:.2f}%"

def main():
    if len(sys.argv) != 3:
        print("Usage: coverage_diff.py BASE.xml PR.xml", file=sys.stderr)
        sys.exit(2)
    base = get_line_rate(sys.argv[1])
    pr = get_line_rate(sys.argv[2])
    print(f"Base coverage: {pct(base)}")
    print(f"PR   coverage: {pct(pr)}")
    if pr + 1e-9 < base:
        print(f"::error::Coverage decreased: {pct(base)} -> {pct(pr)}")
        sys.exit(1)
    else:
        print("Coverage OK (no decrease).")
        sys.exit(0)

if __name__ == "__main__":
    main()
