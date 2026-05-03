NEPSE_SECTORS = {
    "Commercial Banks": [
        "NMB", "NBL", "BOKL", "SBL", "SCB", "EBL", "SBIB", "MBL", 
        "GBBL", "LBBL", "SADBL", "CBL", "CCBL", "GBIME", "BPCL", "NCCB",
        "NRB", "SANIMA", "SNPB", "NICBL", "KPCL", "NABIL", "HBL", "CIT",
        "NBB", "SBI", "NLICL", "PRVU", "MNBBL", "NABBC", "NIB", "KBL"
    ],
    "Development Banks": [
        "MDB", "CORBL", "KRBL", "GLH", "SIFC", "SHBL", "PROFL", "GFCL",
        "GRDBL", "JBBL", "LBL", "CZBIL", "EDBL", "KSBBL", "RLFL", "SAPDBL",
        "GYAN", "MERO", "ICFC", "BFC", "MFIL", "MPFL", "MLBL", "NHPC",
        "PMS", "PFL", "SLI", "ADBL", "BARUN", "CFCL", "CGH", "CHDC", "CHL",
        "GHL", "GLICL", "HDHPC", "HURJA", "JLI", "JOSHI", "KKHC", "LEC",
        "LICN", "MEN", "MHNL", "NHDL", "NICA", "PCBL", "PLI", "PLIC",
        "PMHPL", "SAHAS", "SRBL", "SSHL", "UNHPL", "UPCL", "ULI", "UMRH"
    ],
    "Finance Companies": [
        "SFCL", "MFIL", "GFCL", "GMFIL", "JFL", "SIC", "PROFL", "GUFL",
        "CFL", "KAFIN", "NIFRA", "NLIC", "MEGA"
    ],
    "Hydro Power": [
        "NHPC", "HIDCL", "KPCL", "DHPL", "CHCL", "PPCL", "AKJCL", "RHPC",
        "HPPL", "AHPC", "API", "SJVCL", "NGPL", "UMHL", "UPPER", "RRHP",
        "RHPL", "RURU", "TLBS", "NYADI", "MHK", "RADHI", "RBDC", "AKPL",
        "BSPC", "HPPL", "SPDL", "BHGK", "UPCL", "CWJC", "KHL", "LLBS",
        "MBJC", "MKJC", "MLJ", "SMJC", "SWM", "TPC", "TRH", "UNHPL", "URC",
        "SPC"
    ],
    "Manufacturing And Processing": [
        "BIRL", "BOT", "CEBRO", "CFL", "CSBBL", "DDBL", "DLBS", "FHL",
        "GMF", "GROTE", "GURU", "HAT", "HAWA", "HBT", "HERO", "HLC",
        "HMHL", "HMM", "HRL", "IC", "JAR", "JTS", "KCL", "KMCL", "KTI",
        "LG", "LIC", "LSL", "MBL", "MCD", "MH", "MHL", "MIDI", "ML",
        "MMF", "MOJ", "MP", "MSM", "MTB", "NBB", "NBM", "NBS", "NCC",
        "NLIC", "NRN", "NSM", "NTW", "ODC", "OK", "PBT", "PC", "PF",
        "PGC", "PGR", "PH", "PIC", "PL", "PM", "PR", "PRIN", "PRO",
        "RAD", "RBM", "RC", "RDB", "RH", "RII", "RLG", "SAB", "SAF",
        "SB", "SC", "SCL", "SG", "SH", "SHL", "SI", "SIC", "SK", "SL",
        "SMC", "SMS", "SO", "SP", "SR", "STC", "SW", "TAW", "TCC", "TF",
        "TG", "TH", "TI", "TK", "TL", "TM", "TN", "TR", "TS", "TT", "TV",
        "UL", "UN", "US", "UT", "VTL", "WCF", "WEL", "WH", "WIC", "WMF",
        "SHEL", "SHINE", "SHPC"
    ],
    "Hotels And Tourism": [
        "SHL", "TRH", "OHL", "KGL", "VL", "CM", "DGM", "GM", "HDL", "HRB",
        "LEI", "LI", "ME", "NATH", "NI", "OZ", "PT", "RI", "RR", "SH",
        "SI", "SRS", "TH", "TR", "TTH", "TTI", "UM", "UR", "VO"
    ],
    "Trading": [
        "STC", "SG", "SW", "WMF", "WEL", "SINDU"
    ],
    "Mutual Fund": [
        "NMB", "LBS", "KMLB", "SNMF", "KSMBF", "SMB", "CFM", "GBF", "LGF",
        "MKF", "NSF", "OMF", "PRF", "PTF", "TMF", "TRF"
    ],
    "Microfinance": [
        "MMF", "KMF", "SMF", "KMFB", "KMF", "NMF", "CMF", "MGF", "SGMF",
        "GMF", "FOWAD", "SAMAJ", "MAW", "MCC", "SCB", "NAGRI", "PRAG", "SJCL"
    ],
    "Insurance": [
        "NIC", "NLIC", "SBI", "LIC", "ALICL", "BIMAL", "GURU", "PRABHAT",
        "Sagar", "Swastik", "Union", "Vijaya", "SLICL"
    ],
    "Investment": [
        "CIT", "ICFC", "IID", "MKHC", "NIFRA", "PBT", "PF", "PIC", "PYE",
        "RTL", "SAI", "SIC", "SK", "SMC", "SW", "TDV", "TEA", "TF", "RLI"
    ],
    "Others": [
        "NFS", "NWD", "ENM", "KBL", "NIB", "OS", "PST",
        "SAND", "SIL", "SIN", "SIT", "SM", "SN", "SP", "SR", "ST"
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