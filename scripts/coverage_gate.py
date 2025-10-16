#!/usr/bin/env python3
import argparse, sys
from xml.etree import ElementTree as ET

def read_rate(path: str) -> float:
    root = ET.parse(path).getroot()
    rate = root.get('line-rate')
    if rate is None:
        valid = int(root.get('lines-valid', '0'))
        covered = int(root.get('lines-covered', '0'))
        return (covered / valid) if valid else 0.0
    return float(rate)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("min_percent", type=float, help="Minimal coverage (percent)")
    parser.add_argument("--path", default="coverage.xml")
    args = parser.parse_args()

    rate = read_rate(args.path) * 100.0
    print(f"Coverage: {rate:.2f}%")
    if rate + 1e-9 < args.min_percent:
        print(f"ERROR: Coverage below threshold {args.min_percent:.2f}%", file=sys.stderr)
        return 1
    print("OK: Coverage gate passed")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
