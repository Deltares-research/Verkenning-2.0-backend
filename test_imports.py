"""Test if all required packages are installed"""

packages = [
    ('geopandas', 'gpd'),
    ('shapely', None),
    ('pyproj', None),
    ('fastapi', None),
    ('matplotlib', 'plt'),
    ('plotly', None),
    ('numpy', 'np'),
    ('pandas', 'pd'),
    ('scipy', None),
    ('requests', None),
]

print("Checking installed packages:\n")

for package, alias in packages:
    try:
        if alias:
            exec(f"import {package} as {alias}")
        else:
            exec(f"import {package}")
        print(f"✓ {package} installed")
    except ImportError:
        print(f"✗ {package} NOT installed - run: pip install {package}")

try:
    from volume_calc import DikeModel
    print("\n✓ DikeModel can be imported")
except ValueError as e:
    print(f"\n✗ DikeModel import failed with ValueError: {e}")
    print("\nTo fix: Edit AHN5.py and change:")
    print("  AHN_data: np.ndarray = np.array([])")
    print("to:")
    print("  from dataclasses import dataclass, field")
    print("  AHN_data: np.ndarray = field(default_factory=lambda: np.array([]))")
except ImportError as e:
    print(f"\n✗ DikeModel import failed: {e}")

print("\nAll checks complete!")
