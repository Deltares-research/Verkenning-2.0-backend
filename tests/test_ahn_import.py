"""Test AHN imports"""

try:
    from AHN5 import API_ahn
    print("✓ AHN5.API_ahn imported successfully")
except Exception as e:
    print(f"✗ AHN5.API_ahn import failed: {e}")

try:
    from AHN_raster_API import AHN4, AHN4_API
    print("✓ AHN_raster_API imported successfully")
except Exception as e:
    print(f"✗ AHN_raster_API import failed: {e}")

try:
    from volume_calc import DikeModel
    print("✓ volume_calc.DikeModel imported successfully")
except Exception as e:
    print(f"✗ volume_calc.DikeModel import failed: {e}")

print("\nAll import checks complete!")
