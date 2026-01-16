# 2D Ruimtebeslag API Documentation

## Overview

The 2D Ruimtebeslag endpoint calculates the footprint area where a 3D design surface is above ground level (AHN). It creates concave hull polygons around grid points where the design elevation exceeds the actual terrain elevation.

## Endpoint

**POST** `/api/calculate_ruimtebeslag`

### Authentication

Requires API key in header:
```
X-API-Key: your-api-key-here
```

### Request Parameters

#### Body (JSON)
- **geojson** (required): GeoJSON FeatureCollection with 3D polygon features
  - Must include Z coordinates (elevation) for all polygon vertices
  - Coordinates should be in WGS84 (EPSG:4326) format: `[longitude, latitude, elevation]`

#### Query Parameters
- **alpha** (optional, default: 5.0): Alpha parameter for concave hull algorithm
  - Lower values (1-3) = tighter fit around points
  - Higher values (5-10) = looser, more generalized hull
  - Use 0 for convex hull

### Response

Returns a GeoJSON FeatureCollection with the following structure:

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "geometry": {
        "type": "Polygon",
        "coordinates": [[[x1, y1], [x2, y2], ...]]
      },
      "properties": {
        "area_m2": 1234.56,
        "polygon_id": 0
      }
    }
  ],
  "crs": {
    "type": "name",
    "properties": {"name": "EPSG:28992"}
  },
  "total_area_m2": 1234.56,
  "num_polygons": 1,
  "points_above_ground": 156,
  "calculation_time": 2.345
}
```

#### Response Fields
- **features**: Array of polygon features representing above-ground areas
- **crs**: Coordinate reference system (RD New / EPSG:28992)
- **total_area_m2**: Total area in square meters across all polygons
- **num_polygons**: Number of distinct polygon features
- **points_above_ground**: Number of grid points where design > ground elevation
- **calculation_time**: Calculation time in seconds

## Algorithm Details

The calculation follows these steps:

1. **Grid Generation**: Creates a regular grid (default 0.5m spacing) over the design extent in RD New coordinates (EPSG:28992)

2. **Elevation Queries**:
   - Samples design elevation at each grid point using linear interpolation
   - Queries AHN ground elevation at each grid point via WCS

3. **Above-Ground Filtering**: Filters grid points where `design_elevation > ground_elevation`

4. **Alpha Shape Creation**: Creates a concave hull (alpha shape) around filtered points:
   - Uses Shapely's `concave_hull` method (Shapely >= 2.0)
   - Falls back to convex hull if concave hull fails
   - Can produce single or multiple polygons

5. **Area Calculation**: Computes area in m² for each resulting polygon

## Example Usage

### Python (using requests)

```python
import requests
import json

# Sample 3D GeoJSON
geojson_data = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [[
                    [4.3821, 51.9225, 15.0],  # lon, lat, elevation
                    [4.3825, 51.9225, 15.0],
                    [4.3825, 51.9228, 15.5],
                    [4.3821, 51.9228, 15.5],
                    [4.3821, 51.9225, 15.0]
                ]]
            },
            "properties": {"name": "Design Area"}
        }
    ]
}

# Make request
response = requests.post(
    "http://localhost:8000/api/calculate_ruimtebeslag",
    headers={
        "X-API-Key": "your-api-key",
        "Content-Type": "application/json"
    },
    json=geojson_data,
    params={"alpha": 5.0}
)

result = response.json()
print(f"Total area: {result['total_area_m2']:.2f} m²")
```

### JavaScript/TypeScript (fetch)

```typescript
const geojsonData = {
  type: "FeatureCollection",
  features: [/* your 3D features */]
};

const response = await fetch(
  "http://localhost:8000/api/calculate_ruimtebeslag?alpha=5.0",
  {
    method: "POST",
    headers: {
      "X-API-Key": "your-api-key",
      "Content-Type": "application/json",
    },
    body: JSON.stringify(geojsonData),
  }
);

const result = await response.json();
console.log(`Total area: ${result.total_area_m2.toFixed(2)} m²`);
```

### cURL

```bash
curl -X POST "http://localhost:8000/api/calculate_ruimtebeslag?alpha=5.0" \
  -H "X-API-Key: your-api-key" \
  -H "Content-Type: application/json" \
  -d @input.geojson
```

## Frontend Integration

To integrate with your frontend (similar to the provided code):

```typescript
export async function calculateRuimtebeslag2D(geojson: any): Promise<any> {
  const response = await fetch(
    `${API_BASE_URL}/api/calculate_ruimtebeslag?alpha=5.0`,
    {
      method: "POST",
      headers: {
        "X-API-Key": API_KEY,
        "Content-Type": "application/json",
      },
      body: JSON.stringify(geojson),
    }
  );

  if (!response.ok) {
    throw new Error(`API error: ${response.statusText}`);
  }

  return await response.json();
}

// Use it
const result = await calculateRuimtebeslag2D(designGeoJSON);

// Add polygons to map
result.features.forEach(feature => {
  // Convert from RD (EPSG:28992) to your map projection if needed
  const polygon = createPolygonFromGeoJSON(feature);
  map.add(polygon);
});
```

## Notes

- **Coordinate Systems**: Input is WGS84 (EPSG:4326), output polygons are in RD New (EPSG:28992)
- **Grid Resolution**: Default 0.5m grid provides accurate results but can be adjusted in `DikeModel`
- **Performance**: Calculation time increases with area and grid resolution
- **Alpha Parameter**: Experiment with values between 1-10 to find the best fit for your use case
- **Fallback**: System falls back to convex hull if concave hull algorithm fails

## Error Handling

The API returns HTTP status codes:
- **200**: Success
- **400**: Bad request (missing data, invalid GeoJSON, non-3D geometry)
- **403**: Invalid API key
- **500**: Server error (calculation failure, AHN data unavailable)


## Response

```
{
  "volume": {
    ...
  },
  "calculation_time": 0.123,
  "ruimtebeslag_2d_points": [[x, y], ...]
}
```

### Notes
- `ruimtebeslag_2d_points` is a list of [x, y] or [x, y, z] coordinates in EPSG:3857 (Web Mercator).
- The `ruimtebeslag_2d` GeoJSON is no longer returned. Use the points for frontend ruimtebeslag calculation.
| Use Case | Cut/fill quantities | Footprint visualization |
