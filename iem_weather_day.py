"""
IEM Weather Data Download Script
Downloads daily weather data for Medium and Small hub airports

Based on IEM API format:
curl 'https://mesonet.agron.iastate.edu/cgi-bin/request/daily.py?network=TX_ASOS&stations=DAL&stations=HOU&year1=2023&month1=6&day1=1&year2=2023&month2=7&day2=31&var=max_temp_f&format=csv'
"""

import requests
import pandas as pd
import time
from datetime import datetime
import os
from io import StringIO

# Import network mapping
from airport_network_mapping import AIRPORT_NETWORK_MAP, group_by_network

# Study period
START_DATE = "2023-06-01"
END_DATE = "2023-07-31"

# Weather variables to download
VARIABLES = [
    'max_temp_f',
    'min_temp_f',
    'max_dewpoint_f',
    'min_dewpoint_f',
    'precip_in',
    'avg_wind_speed_kts',
    'snow_in',
    'avg_feel'
]

# IEM API endpoint
BASE_URL = "https://mesonet.agron.iastate.edu/cgi-bin/request/daily.py"

def download_network_weather(network_name, airports, start_date, end_date, variables):
    """
    Download weather data for multiple airports in the same network from IEM.
    
    Args:
        network_name: Network name (e.g., 'TX_ASOS')
        airports: List of dicts with 'iata' and 'icao' keys
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        variables: List of variable names to download
        
    Returns:
        pandas DataFrame with weather data for all airports
    """
    iata_codes = [apt['iata'] for apt in airports]
    print(f"\nDownloading {network_name}: {', '.join(iata_codes)}")
    
    # Parse dates
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    
    # Build URL with multiple stations and vars
    # Format: ?network=TX_ASOS&stations=DAL&stations=HOU&var=max_temp_f&var=min_temp_f
    params = {
        'network': network_name,
        'year1': start.year,
        'month1': start.month,
        'day1': start.day,
        'year2': end.year,
        'month2': end.month,
        'day2': end.day,
        'format': 'csv',
        'na': 'blank'
    }
    
    # Add multiple stations parameters
    params_list = []
    for key, value in params.items():
        params_list.append(f"{key}={value}")
    
    # Add stations HI_ASOS and PR_ASOS require ICAO codes, others use IATA
    use_icao_networks = {'HI_ASOS', 'PR_ASOS'}
    
    for apt in airports:
        if network_name in use_icao_networks:
            params_list.append(f"stations={apt['icao']}")
        else:
            params_list.append(f"stations={apt['iata']}")
    
    # Add variables
    for var in variables:
        params_list.append(f"var={var}")
    
    url = f"{BASE_URL}?{'&'.join(params_list)}"
    
    try:
        # Make request
        response = requests.get(url, timeout=60)
        response.raise_for_status()
        
        # Check for errors in response
        if response.text.startswith('ERROR'):
            print(f"  API Error: {response.text.strip()}")
            return None

        df = pd.read_csv(StringIO(response.text))

        if len(df) == 0:
            print(f"  No data returned")
            return None

        print(f"  Downloaded {len(df)} records, {df['station'].nunique()} airports")
        
        # Map station codes back to IATA for consistency
        # For HI/PR networks, station column has ICAO codes, need to map back
        if network_name in use_icao_networks:
            icao_to_iata = {apt['icao']: apt['iata'] for apt in airports}
            df['IATA'] = df['station'].map(icao_to_iata)
        else:
            station_to_iata = {apt['iata']: apt['iata'] for apt in airports}
            df['IATA'] = df['station'].map(station_to_iata)
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"  Network error: {e}")
        return None
    except pd.errors.EmptyDataError:
        print(f"  No data in response")
        return None
    except Exception as e:
        print(f"  Parse error: {e}")
        print(f"  Response preview: {response.text[:200]}")
        return None

def main():
    print(f"IEM weather download: {START_DATE} to {END_DATE}, {len(VARIABLES)} variables")

    output_dir = "weather_data"
    os.makedirs(output_dir, exist_ok=True)

    network_groups = group_by_network()
    print(f"{len(AIRPORT_NETWORK_MAP)} airports across {len(network_groups)} networks")
    
    all_data = []
    failed_networks = []
    
    # Download data for each network group
    for i, (network, airports) in enumerate(sorted(network_groups.items()), 1):
        print(f"[{i}/{len(network_groups)}] ", end="")
        
        df = download_network_weather(
            network_name=network,
            airports=airports,
            start_date=START_DATE,
            end_date=END_DATE,
            variables=VARIABLES
        )
        
        if df is not None and len(df) > 0:
            all_data.append(df)
            network_file = os.path.join(output_dir, f"{network}_weather.csv")
            df.to_csv(network_file, index=False)
            print(f"  Saved {network_file}")
        else:
            failed_networks.append(network)
        
        # Be polite to the server
        time.sleep(1)
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        combined_file = os.path.join(output_dir, "all_airports_weather_combined.csv")
        combined_df.to_csv(combined_file, index=False)
        print(f"Combined: {combined_file} ({len(combined_df):,} records, "
              f"{combined_df['IATA'].nunique()} airports, "
              f"{combined_df['day'].min()} to {combined_df['day'].max()})")

    print(f"Downloaded: {len(all_data)} networks, "
          f"{sum(df['IATA'].nunique() for df in all_data)} airports")
    if failed_networks:
        print(f"Failed: {', '.join(failed_networks)}")

    data_dict_file = os.path.join(output_dir, "README_DATA_DICTIONARY.txt")
    with open(data_dict_file, 'w') as f:
        f.write("WEATHER DATA DICTIONARY\n")
        f.write("="*70 + "\n\n")
        f.write("Source: Iowa Environmental Mesonet (IEM)\n")
        f.write("URL: https://mesonet.agron.iastate.edu/\n")
        f.write(f"Download Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Study Period: {START_DATE} to {END_DATE}\n\n")
        f.write("VARIABLES:\n")
        f.write("-"*70 + "\n")
        f.write("day                  - Date (YYYY-MM-DD)\n")
        f.write("station              - IATA station code (3-letter)\n")
        f.write("max_temp_f           - Maximum Air Temperature [F]\n")
        f.write("min_temp_f           - Minimum Air Temperature [F]\n")
        f.write("max_dewpoint_f       - Maximum Dew Point [F]\n")
        f.write("min_dewpoint_f       - Minimum Dew Point [F]\n")
        f.write("precip_in            - Daily Precipitation [inch]\n")
        f.write("avg_wind_speed_kts   - Average Wind Speed [knots]\n")
        f.write("snow_in              - Reported Snowfall [inch]\n")
        f.write("avg_feel             - Average 'Feels Like' Temperature [F]\n")
        f.write("IATA                 - 3-letter IATA airport code\n\n")
        f.write("NOTES:\n")
        f.write("-"*70 + "\n")
        f.write("- Missing values appear as blank\n")
        f.write("- Precipitation values of 0.0001 represent trace amounts\n")
        f.write("- Data is based on local calendar day for each station\n")
    print(f"Data dictionary saved: {data_dict_file}")

if __name__ == "__main__":
    main()