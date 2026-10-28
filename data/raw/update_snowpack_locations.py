import re
import pandas as pd
from pathlib import Path

# Configuration
BASE_PATH = Path("/ssd/snowpack/output/2025")
OUTPUT_CSV = Path("SnowPackLocations_updated.csv")

def parse_pro_file_header(pro_file_path):
    """
    Reads a SNOWPACK .pro file and extracts metadata from the [STATION_PARAMETERS] header
    or the standard header lines.
    Returns: dict with lat, lon, elev
    """
    metadata = {'lat': None, 'lon': None, 'elev': None}
    
    try:
        # pathlib Path objects can be passed directly to open() in modern Python
        with pro_file_path.open('r', encoding='utf-8', errors='ignore') as f:
            # Read first 50 lines to find header info
            for _ in range(50):
                line = f.readline()
                if not line:
                    break
                
                # Clean line
                line = line.strip()
                
                # Check for standard SNOWPACK header keys
                # Example: Latitude = 40.76
                if line.startswith("Latitude"):
                    parts = line.split('=')
                    if len(parts) > 1:
                        metadata['lat'] = float(parts[1].strip())
                        
                elif line.startswith("Longitude"):
                    parts = line.split('=')
                    if len(parts) > 1:
                        metadata['lon'] = float(parts[1].strip())
                        
                elif line.startswith("Altitude"):
                    parts = line.split('=')
                    if len(parts) > 1:
                        metadata['elev'] = float(parts[1].strip())

                # Stop if we found everything
                if all(metadata.values()):
                    break
                    
    except Exception as e:
        print(f"Error reading {pro_file_path}: {e}")
        return None

    if all(metadata.values()):
        return metadata
    else:
        return None

def main():
    print(f"Scanning directory: {BASE_PATH}...")
    
    data_rows = []
    
    # 1. Find all Zone Directories (e.g. "zone002", "zone101")
    if not BASE_PATH.exists():
        print(f"Error: Base path {BASE_PATH} does not exist.")
        return

    # Iterate over items in the base path
    for zone_dir in BASE_PATH.iterdir():
        # We only care about directories matching 'zoneXYZ'
        if not zone_dir.is_dir():
            continue

        match = re.match(r'zone(\d+)', zone_dir.name)
        if match:
            zone_str = match.group(1)
            try:
                zone_id = int(zone_str) # Convert "002" -> 2
            except ValueError:
                continue
                
            # 2. Iterate through Station Number directories inside the zone
            for station_dir in zone_dir.iterdir():
                # Must be a directory and look like a number (the ID)
                if station_dir.is_dir() and station_dir.name.isdigit():
                    try:
                        station_id = int(station_dir.name)
                    except ValueError:
                        continue
                    
                    # 3. Find the .pro file inside
                    # pathlib's glob returns a generator, convert to list
                    pro_files = list(station_dir.glob("*.pro"))
                    
                    if not pro_files:
                        print(f"Warning: No .pro file found in {station_dir}")
                        continue
                        
                    # Parse the first .pro file found
                    meta = parse_pro_file_header(pro_files[0])
                    
                    if meta:
                        data_rows.append({
                            'id': station_id,
                            'zone': zone_id,
                            'lat': meta['lat'],
                            'lon': meta['lon'],
                            'elev': int(meta['elev'])
                        })
                        print(f"Found Station: {station_id} in Zone {zone_id}")
                    else:
                        print(f"Warning: Could not parse metadata for station {station_id}")

    # 4. Create DataFrame and Save
    if data_rows:
        df = pd.DataFrame(data_rows)
        
        # Sort for cleanliness
        df = df.sort_values(by=['zone', 'id'])
        
        # Save to CSV
        df.to_csv(OUTPUT_CSV, index=False)
        print("-" * 30)
        print(f"Success! Scanned {len(df)} stations.")
        print(f"Updated file saved to: {OUTPUT_CSV.resolve()}")
    else:
        print("No valid stations found.")

if __name__ == "__main__":
    main()