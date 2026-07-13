import json

NEPSE_SECTORS = {
    "Commercial Banks": [
        "ADBL", "BOKL", "CBL", "CCBL", "CZBIL", "EBL", "GBIME", "HBL", "KBL", "LBL",
        "LSL", "MBL", "MEGA", "NABIL", "NBB", "NBL", "NCCB", "NIB", "NICA", "NIMB",
        "NMB", "PCBL", "PRVU", "SANIMA", "SBI", "SBL", "SCB"
    ],
    "Development Banks": [
        "CORBL", "EDBL", "GBBL", "GRDBL", "JBBL", "KRBL", "KSBBL", "LBBL",
        "MDB", "MERO", "MLBL", "MNBBL", "NABBC", "SADBL", "SAPDBL", "SHBL",
        "SINDU", "SRBL"
    ],
    "Finance Companies": [
        "BFC", "CFCL", "CFL", "GFCL", "GMFIL", "GUFL", "ICFC", "JFL",
        "MFIL", "MPFL", "NFS", "PFL", "PROFL", "RLFL", "SFCL", "SIFC"
    ],
    "Hydro Power": [
        "AHL", "AHPC", "AKJCL", "AKPL", "API", "BARUN", "BHGK", "BPCL", "BSPC",
        "CHCL", "CHDC", "CHL", "CWJC", "DHPL", "GHL", "GLH", "HDHPC",
        "HIDCL", "HPPL", "HURJA", "JOSHI", "KHL", "KKHC", "KPCL", "LEC", "MBJC",
        "MEN", "MHK", "MKJC", "MHNL", "MLJ", "NGPL", "NHDL", "NHPC", "NYADI",
        "PMHPL", "PPCL", "RADHI", "RBDC", "RHPC", "RHPL", "RRHP", "RURU",
        "SAHAS", "SGHL", "SHPC", "SJCL", "SJVCL", "SMJC", "SPC", "SPDL", "SSHL",
        "SWM", "TAKSAR", "TPC", "UMHL", "UMRH", "UNHPL", "UPCL", "UPPER", "URC"
    ],
    "Manufacturing And Processing": [
        "BIRL", "BOT", "CSBBL", "DDBL", "DLBS", "FHL", "GMF", "HBT", 
        "HDL", "HRL", "LSL", "SAB", "SCL", "SHINE", "SHEL", "UN"
    ],
    "Hotels And Tourism": [
        "CGH", "CITY", "KGL", "OHL", "SHL", "TRH"
    ],
    "Trading": [
        "BBC", "STC"
    ],
    "Mutual Fund": [
        "BFED", "CFFF", "CMF1", "CMF2", "GIBF1", "KBF", "KEF", "LBF", 
        "MEOF1", "NIBF1", "NIBF2", "NMB50", "NMBHF1", "PMF", "SAEF", "SIGS2"
    ],
    "Microfinance": [
        "ACLBSL", "ALBSL", "ANLB", "CBBL", "CLBSL", "DDBL", "FOWAD", "GGBSL", 
        "GMFBS", "ILBS", "JALPA", "KMCDB", "LLBS", "MERO", "MLBBL", "MLBSL", 
        "NESDO", "NICLBSL", "NMFBS", "NUBL", "SABSL", "SADBL", "SAMAJ", "SDLBSL", 
        "SKBBL", "SMB", "SMFBS", "SMHL", "SWBBL", "USLB"
    ],
    "Life Insurance": [
        "ALICL", "CLI", "GLICL", "ILI", "JLI", "LICN", "NLIC", "NLICL",
        "PLI", "PLIC", "RIL", "RLI", "SLI", "SLICL", "ULI"
    ],
    "Non Life Insurance": [
        "EIC", "GIC", "HGI", "IGI", "LGIL", "NIL", "NICL", "NLG", 
        "PRIN", "RBCLI", "SICL", "SIL", "SPIL", "UIC"
    ],
    "Investment": [
        "CIT", "ENL", "HATHY", "IID", "NIFRA", "NRN"
    ],
    "Others": [
        "NTC"
    ]
}

TICKER_TO_SECTOR = {}
for sector, tickers in NEPSE_SECTORS.items():
    for ticker in tickers:
        TICKER_TO_SECTOR[ticker.upper()] = sector

def get_sector(ticker):
    return TICKER_TO_SECTOR.get(ticker.upper(), "Unknown")

def get_all_sectors():
    return list(NEPSE_SECTORS.keys())

def count_all_tickers():
    return len(TICKER_TO_SECTOR)

def get_tickers_by_sector(sector_name):
    return NEPSE_SECTORS.get(sector_name, [])

if __name__ == '__main__':
    test_ticker = "SSHL"
    print(f"Ticker '{test_ticker}' belongs to: {get_sector(test_ticker)}")
    
    print("\n--- Available Sectors ---")
    print(json.dumps(get_all_sectors(), indent=2))
    
    print("\n--- Metrics Summary ---")
    print(f"Total Unique Mapped Tickers: {count_all_tickers()}")