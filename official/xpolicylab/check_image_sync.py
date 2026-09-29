"""Check that the agent image and a submission wheelhouse hold the same Python packages.

Bundles are developed and rehearsed in the agent image but run on the evaluator from
the wheelhouse. A package the image has and the wheelhouse lacks (or a different
version) would import in rehearsal and fail officially.

Usage: check_image_sync.py <wheelhouse> [image=robodojo-official:dev]
"""
import json
from pathlib import Path
import re
import subprocess
import sys

# Base-interpreter tooling, and development-only asset inspection (see the Dockerfile).
IMAGE_ONLY = {"pip", "setuptools", "wheel", "usd-core"}


def canonical(name):
    return re.sub(r"[-_.]+", "-", name).lower()


def wheelhouse_packages(directory):
    packages = {}
    for wheel in Path(directory).glob("*.whl"):
        name, version = wheel.name.split("-")[:2]
        packages[canonical(name)] = version
    return packages


def image_packages(image):
    listing = subprocess.run(["docker", "run", "--rm", "--network=none", "--entrypoint", "python", image,
                              "-m", "pip", "list", "--format=json"], capture_output=True, text=True,
                             timeout=300, check=True).stdout
    return {canonical(p["name"]): p["version"] for p in json.loads(listing)}


def compare(wheels, image):
    problems = []
    for name, version in sorted(image.items()):
        if name in IMAGE_ONLY:
            continue
        if name not in wheels:
            problems.append(f"{name}=={version} is in the image but not in the wheelhouse")
        elif canonical(wheels[name]) != canonical(version):
            problems.append(f"{name}: image {version}, wheelhouse {wheels[name]}")
    problems += [f"{name}=={version} is in the wheelhouse but not installed in the image"
                 for name, version in sorted(wheels.items()) if name not in image]
    return problems


def main():
    wheelhouse = sys.argv[1]
    image = sys.argv[2] if len(sys.argv) > 2 else "robodojo-official:dev"
    problems = compare(wheelhouse_packages(wheelhouse), image_packages(image))
    if problems:
        print(f"Agent image {image} and wheelhouse {wheelhouse} differ:", *problems, sep="\n  ", file=sys.stderr)
        print("Rebuild the image (scripts/build_official_image.sh) from this wheelhouse.", file=sys.stderr)
        raise SystemExit(1)
    print(f"Agent image {image} matches the submission wheelhouse")


if __name__ == "__main__":
    main()
