"""Airport-to-IEM-network mapping for weather data fetching."""

AIRPORT_NETWORK_MAP = {
    'DAL': {'name': 'Dallas Love Field', 'state': 'TX', 'network': 'TX_ASOS', 'icao': 'KDAL'},
    'HNL': {'name': 'Honolulu', 'state': 'HI', 'network': 'HI_ASOS', 'icao': 'PHNL'},
    'PDX': {'name': 'Portland', 'state': 'OR', 'network': 'OR_ASOS', 'icao': 'KPDX'},
    'HOU': {'name': 'Houston Hobby', 'state': 'TX', 'network': 'TX_ASOS', 'icao': 'KHOU'},
    'RSW': {'name': 'Fort Myers', 'state': 'FL', 'network': 'FL_ASOS', 'icao': 'KRSW'},
    'STL': {'name': 'St. Louis', 'state': 'MO', 'network': 'MO_ASOS', 'icao': 'KSTL'},
    'SMF': {'name': 'Sacramento', 'state': 'CA', 'network': 'CA_ASOS', 'icao': 'KSMF'},
    'SJU': {'name': 'San Juan', 'state': 'PR', 'network': 'PR_ASOS', 'icao': 'TJSJ'},
    'RDU': {'name': 'Raleigh-Durham', 'state': 'NC', 'network': 'NC_ASOS', 'icao': 'KRDU'},
    'MSY': {'name': 'New Orleans', 'state': 'LA', 'network': 'LA_ASOS', 'icao': 'KMSY'},
    'OAK': {'name': 'Oakland', 'state': 'CA', 'network': 'CA_ASOS', 'icao': 'KOAK'},
    'SNA': {'name': 'Santa Ana', 'state': 'CA', 'network': 'CA_ASOS', 'icao': 'KSNA'},
    'MCI': {'name': 'Kansas City', 'state': 'MO', 'network': 'MO_ASOS', 'icao': 'KMCI'},
    'SAT': {'name': 'San Antonio', 'state': 'TX', 'network': 'TX_ASOS', 'icao': 'KSAT'},
    'SJC': {'name': 'San Jose', 'state': 'CA', 'network': 'CA_ASOS', 'icao': 'KSJC'},
    'OKC': {'name': 'Oklahoma City', 'state': 'OK', 'network': 'OK_ASOS', 'icao': 'KOKC'},
    'RIC': {'name': 'Richmond', 'state': 'VA', 'network': 'VA_ASOS', 'icao': 'KRIC'},
    'GEG': {'name': 'Spokane', 'state': 'WA', 'network': 'WA_ASOS', 'icao': 'KGEG'},
    'MYR': {'name': 'Myrtle Beach', 'state': 'SC', 'network': 'SC_ASOS', 'icao': 'KMYR'},
    'SRQ': {'name': 'Sarasota-Bradenton', 'state': 'FL', 'network': 'FL_ASOS', 'icao': 'KSRQ'},
    'SDF': {'name': 'Louisville', 'state': 'KY', 'network': 'KY_ASOS', 'icao': 'KSDF'},
    'GRR': {'name': 'Grand Rapids', 'state': 'MI', 'network': 'MI_ASOS', 'icao': 'KGRR'},
    'ELP': {'name': 'El Paso', 'state': 'TX', 'network': 'TX_ASOS', 'icao': 'KELP'},
    'BUF': {'name': 'Buffalo', 'state': 'NY', 'network': 'NY_ASOS', 'icao': 'KBUF'},
    'KOA': {'name': 'Kona', 'state': 'HI', 'network': 'HI_ASOS', 'icao': 'PHKO'},
    'SAV': {'name': 'Savannah', 'state': 'GA', 'network': 'GA_ASOS', 'icao': 'KSAV'},
    'TUS': {'name': 'Tucson', 'state': 'AZ', 'network': 'AZ_ASOS', 'icao': 'KTUS'},
    'SFB': {'name': 'Orlando Sanford', 'state': 'FL', 'network': 'FL_ASOS', 'icao': 'KSFB'},
    'PNS': {'name': 'Pensacola', 'state': 'FL', 'network': 'FL_ASOS', 'icao': 'KPNS'},
    'PVD': {'name': 'Providence', 'state': 'RI', 'network': 'RI_ASOS', 'icao': 'KPVD'},
}

def group_by_network():
    network_groups = {}
    for airport, info in AIRPORT_NETWORK_MAP.items():
        network = info['network']
        if network not in network_groups:
            network_groups[network] = []
        network_groups[network].append({
            'iata': airport,
            'icao': info['icao'],
            'name': info['name'],
            'state': info['state']
        })
    return network_groups

def print_network_summary():
    groups = group_by_network()
    print(f"Total airports: {len(AIRPORT_NETWORK_MAP)}, networks: {len(groups)}")
    for network in sorted(groups.keys()):
        airports = groups[network]
        print(f"\n{network} ({len(airports)} airports)")
        for apt in airports:
            print(f"  {apt['iata']:4s} | {apt['icao']:4s} | {apt['name']}")

def get_network_for_airport(iata_code):
    if iata_code in AIRPORT_NETWORK_MAP:
        return AIRPORT_NETWORK_MAP[iata_code]['network']
    return None

if __name__ == '__main__':
    print_network_summary()
