"""Test AHN imports"""


try:
    from source.AHN_raster_API import AHN4, AHN4_API
    print("✓ AHN_raster_API imported successfully")
except Exception as e:
    print(f"✗ AHN_raster_API import failed: {e}")

try:
    from source.volume_calc import DikeModel
    print("✓ volume_calc.DikeModel imported successfully")
except Exception as e:
    print(f"✗ volume_calc.DikeModel import failed: {e}")

print("\nAll import checks complete!")
