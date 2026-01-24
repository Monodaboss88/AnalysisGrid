"""
Watchlist Management System
===========================
Comprehensive watchlist with major indices, sectors, and custom lists.
Includes search, filtering, and persistence.

Author: Rob's Trading Systems
Version: 1.0.0
"""

import json
import os
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set
from datetime import datetime
from enum import Enum
import re


# =============================================================================
# ENUMS AND DATA STRUCTURES
# =============================================================================

class WatchlistCategory(Enum):
    """Predefined watchlist categories"""
    INDEX = "Index"
    INDEX_COMPONENT = "Index Component"
    SECTOR_ETF = "Sector ETF"
    INDUSTRY_ETF = "Industry ETF"
    MEGA_CAP = "Mega Cap"
    LARGE_CAP = "Large Cap"
    MID_CAP = "Mid Cap"
    SMALL_CAP = "Small Cap"
    GROWTH = "Growth"
    VALUE = "Value"
    DIVIDEND = "Dividend"
    TECH = "Technology"
    CRYPTO = "Crypto"
    COMMODITY = "Commodity"
    FOREX = "Forex"
    FUTURES = "Futures"
    CUSTOM = "Custom"
    FAVORITES = "Favorites"


@dataclass
class WatchlistSymbol:
    """Individual symbol in a watchlist"""
    symbol: str
    name: str = ""
    category: str = "Custom"
    sector: str = ""
    industry: str = ""
    market_cap: str = ""  # "mega", "large", "mid", "small", "micro"
    
    # Metadata
    added_date: str = ""
    notes: str = ""
    tags: List[str] = field(default_factory=list)
    
    # State
    enabled: bool = True
    is_favorite: bool = False
    
    # Index membership
    in_sp500: bool = False
    in_nasdaq100: bool = False
    in_dow30: bool = False
    in_russell2000: bool = False
    
    def __post_init__(self):
        if not self.added_date:
            self.added_date = datetime.now().isoformat()
        self.symbol = self.symbol.upper()


@dataclass
class Watchlist:
    """A named collection of symbols"""
    name: str
    description: str = ""
    symbols: List[WatchlistSymbol] = field(default_factory=list)
    created_date: str = ""
    modified_date: str = ""
    is_builtin: bool = False
    
    def __post_init__(self):
        if not self.created_date:
            self.created_date = datetime.now().isoformat()
        self.modified_date = self.created_date
    
    @property
    def symbol_count(self) -> int:
        return len(self.symbols)
    
    @property
    def enabled_count(self) -> int:
        return sum(1 for s in self.symbols if s.enabled)


# =============================================================================
# BUILT-IN INDEX DATA
# =============================================================================

class IndexData:
    """
    Built-in data for major indices and their components
    
    For Brokers:
    -----------
    Pre-loaded with:
    - Major Index ETFs (SPY, QQQ, DIA, IWM)
    - S&P 500 Top 50 components
    - Nasdaq 100 Top 50 components
    - Dow 30 components
    - All 11 Sector ETFs
    - Popular Industry ETFs
    - Mega Cap stocks
    
    For Programmers:
    ---------------
    Static data that can be refreshed from external source.
    """
    
    # Major Index ETFs
    INDEX_ETFS = {
        "SPY": ("SPDR S&P 500 ETF", "Index"),
        "QQQ": ("Invesco Nasdaq 100 ETF", "Index"),
        "DIA": ("SPDR Dow Jones ETF", "Index"),
        "IWM": ("iShares Russell 2000 ETF", "Index"),
        "IWB": ("iShares Russell 1000 ETF", "Index"),
        "IWF": ("iShares Russell 1000 Growth", "Index"),
        "IWD": ("iShares Russell 1000 Value", "Index"),
        "MDY": ("SPDR S&P MidCap 400 ETF", "Index"),
        "IJH": ("iShares Core S&P MidCap", "Index"),
        "VTI": ("Vanguard Total Stock Market", "Index"),
        "VOO": ("Vanguard S&P 500 ETF", "Index"),
    }
    
    # Sector ETFs (Select Sector SPDRs)
    SECTOR_ETFS = {
        "XLK": ("Technology Select Sector", "Technology"),
        "XLF": ("Financial Select Sector", "Financials"),
        "XLV": ("Health Care Select Sector", "Healthcare"),
        "XLE": ("Energy Select Sector", "Energy"),
        "XLI": ("Industrial Select Sector", "Industrials"),
        "XLY": ("Consumer Discretionary Select", "Consumer Discretionary"),
        "XLP": ("Consumer Staples Select", "Consumer Staples"),
        "XLU": ("Utilities Select Sector", "Utilities"),
        "XLB": ("Materials Select Sector", "Materials"),
        "XLRE": ("Real Estate Select Sector", "Real Estate"),
        "XLC": ("Communication Services Select", "Communication Services"),
    }
    
    # Industry/Thematic ETFs
    INDUSTRY_ETFS = {
        "SMH": ("VanEck Semiconductor ETF", "Semiconductors"),
        "SOXX": ("iShares Semiconductor ETF", "Semiconductors"),
        "XBI": ("SPDR S&P Biotech ETF", "Biotech"),
        "IBB": ("iShares Biotech ETF", "Biotech"),
        "XHB": ("SPDR S&P Homebuilders ETF", "Homebuilders"),
        "ITB": ("iShares Home Construction ETF", "Homebuilders"),
        "XRT": ("SPDR S&P Retail ETF", "Retail"),
        "KRE": ("SPDR S&P Regional Banking ETF", "Banks"),
        "XOP": ("SPDR S&P Oil & Gas Exploration", "Oil & Gas"),
        "GDX": ("VanEck Gold Miners ETF", "Gold Miners"),
        "ARKK": ("ARK Innovation ETF", "Innovation"),
        "ARKW": ("ARK Next Gen Internet ETF", "Internet"),
        "HACK": ("ETFMG Prime Cyber Security", "Cybersecurity"),
        "BOTZ": ("Global X Robotics & AI ETF", "Robotics/AI"),
        "TAN": ("Invesco Solar ETF", "Solar"),
        "LIT": ("Global X Lithium & Battery ETF", "Lithium/Battery"),
        "JETS": ("US Global Jets ETF", "Airlines"),
    }
    
    # DOW 30 Components
    DOW_30 = {
        "AAPL": ("Apple Inc", "Technology", True, True, True),
        "AMGN": ("Amgen Inc", "Healthcare", True, True, True),
        "AXP": ("American Express", "Financials", True, False, True),
        "BA": ("Boeing Co", "Industrials", True, False, True),
        "CAT": ("Caterpillar Inc", "Industrials", True, False, True),
        "CRM": ("Salesforce Inc", "Technology", True, False, True),
        "CSCO": ("Cisco Systems", "Technology", True, True, True),
        "CVX": ("Chevron Corp", "Energy", True, False, True),
        "DIS": ("Walt Disney Co", "Communication Services", True, False, True),
        "DOW": ("Dow Inc", "Materials", True, False, True),
        "GS": ("Goldman Sachs", "Financials", True, False, True),
        "HD": ("Home Depot", "Consumer Discretionary", True, False, True),
        "HON": ("Honeywell", "Industrials", True, False, True),
        "IBM": ("IBM Corp", "Technology", True, False, True),
        "INTC": ("Intel Corp", "Technology", True, True, True),
        "JNJ": ("Johnson & Johnson", "Healthcare", True, False, True),
        "JPM": ("JPMorgan Chase", "Financials", True, False, True),
        "KO": ("Coca-Cola Co", "Consumer Staples", True, False, True),
        "MCD": ("McDonald's Corp", "Consumer Discretionary", True, False, True),
        "MMM": ("3M Company", "Industrials", True, False, True),
        "MRK": ("Merck & Co", "Healthcare", True, False, True),
        "MSFT": ("Microsoft Corp", "Technology", True, True, True),
        "NKE": ("Nike Inc", "Consumer Discretionary", True, False, True),
        "PG": ("Procter & Gamble", "Consumer Staples", True, False, True),
        "TRV": ("Travelers Companies", "Financials", True, False, True),
        "UNH": ("UnitedHealth Group", "Healthcare", True, False, True),
        "V": ("Visa Inc", "Financials", True, False, True),
        "VZ": ("Verizon", "Communication Services", True, False, True),
        "WBA": ("Walgreens Boots Alliance", "Healthcare", True, True, True),
        "WMT": ("Walmart Inc", "Consumer Staples", True, False, True),
    }
    
    # NASDAQ 100 Top Components (Top 50 by weight)
    NASDAQ_100_TOP = {
        "AAPL": ("Apple Inc", "Technology"),
        "MSFT": ("Microsoft Corp", "Technology"),
        "NVDA": ("NVIDIA Corp", "Technology"),
        "AMZN": ("Amazon.com Inc", "Consumer Discretionary"),
        "META": ("Meta Platforms", "Communication Services"),
        "GOOGL": ("Alphabet Inc Class A", "Communication Services"),
        "GOOG": ("Alphabet Inc Class C", "Communication Services"),
        "AVGO": ("Broadcom Inc", "Technology"),
        "TSLA": ("Tesla Inc", "Consumer Discretionary"),
        "COST": ("Costco Wholesale", "Consumer Staples"),
        "NFLX": ("Netflix Inc", "Communication Services"),
        "AMD": ("Advanced Micro Devices", "Technology"),
        "ADBE": ("Adobe Inc", "Technology"),
        "PEP": ("PepsiCo Inc", "Consumer Staples"),
        "CSCO": ("Cisco Systems", "Technology"),
        "TMUS": ("T-Mobile US", "Communication Services"),
        "INTC": ("Intel Corp", "Technology"),
        "CMCSA": ("Comcast Corp", "Communication Services"),
        "INTU": ("Intuit Inc", "Technology"),
        "QCOM": ("Qualcomm Inc", "Technology"),
        "TXN": ("Texas Instruments", "Technology"),
        "AMGN": ("Amgen Inc", "Healthcare"),
        "AMAT": ("Applied Materials", "Technology"),
        "ISRG": ("Intuitive Surgical", "Healthcare"),
        "HON": ("Honeywell", "Industrials"),
        "BKNG": ("Booking Holdings", "Consumer Discretionary"),
        "VRTX": ("Vertex Pharmaceuticals", "Healthcare"),
        "MU": ("Micron Technology", "Technology"),
        "LRCX": ("Lam Research", "Technology"),
        "ADP": ("Automatic Data Processing", "Industrials"),
        "REGN": ("Regeneron Pharma", "Healthcare"),
        "ADI": ("Analog Devices", "Technology"),
        "MDLZ": ("Mondelez International", "Consumer Staples"),
        "PANW": ("Palo Alto Networks", "Technology"),
        "KLAC": ("KLA Corp", "Technology"),
        "SNPS": ("Synopsys Inc", "Technology"),
        "CDNS": ("Cadence Design", "Technology"),
        "SBUX": ("Starbucks Corp", "Consumer Discretionary"),
        "GILD": ("Gilead Sciences", "Healthcare"),
        "ASML": ("ASML Holding", "Technology"),
        "MAR": ("Marriott International", "Consumer Discretionary"),
        "MELI": ("MercadoLibre", "Consumer Discretionary"),
        "ORLY": ("O'Reilly Automotive", "Consumer Discretionary"),
        "PYPL": ("PayPal Holdings", "Financials"),
        "CTAS": ("Cintas Corp", "Industrials"),
        "CSX": ("CSX Corp", "Industrials"),
        "ABNB": ("Airbnb Inc", "Consumer Discretionary"),
        "MRVL": ("Marvell Technology", "Technology"),
        "PCAR": ("PACCAR Inc", "Industrials"),
        "FTNT": ("Fortinet Inc", "Technology"),
    }
    
    # S&P 500 Top 50 by Weight
    SP500_TOP = {
        "AAPL": ("Apple Inc", "Technology", "mega"),
        "MSFT": ("Microsoft Corp", "Technology", "mega"),
        "NVDA": ("NVIDIA Corp", "Technology", "mega"),
        "AMZN": ("Amazon.com Inc", "Consumer Discretionary", "mega"),
        "META": ("Meta Platforms", "Communication Services", "mega"),
        "GOOGL": ("Alphabet Inc Class A", "Communication Services", "mega"),
        "GOOG": ("Alphabet Inc Class C", "Communication Services", "mega"),
        "BRK.B": ("Berkshire Hathaway B", "Financials", "mega"),
        "LLY": ("Eli Lilly", "Healthcare", "mega"),
        "AVGO": ("Broadcom Inc", "Technology", "mega"),
        "JPM": ("JPMorgan Chase", "Financials", "mega"),
        "TSLA": ("Tesla Inc", "Consumer Discretionary", "mega"),
        "UNH": ("UnitedHealth Group", "Healthcare", "mega"),
        "XOM": ("Exxon Mobil", "Energy", "mega"),
        "V": ("Visa Inc", "Financials", "mega"),
        "MA": ("Mastercard Inc", "Financials", "mega"),
        "COST": ("Costco Wholesale", "Consumer Staples", "mega"),
        "PG": ("Procter & Gamble", "Consumer Staples", "mega"),
        "JNJ": ("Johnson & Johnson", "Healthcare", "mega"),
        "HD": ("Home Depot", "Consumer Discretionary", "mega"),
        "NFLX": ("Netflix Inc", "Communication Services", "large"),
        "ABBV": ("AbbVie Inc", "Healthcare", "large"),
        "WMT": ("Walmart Inc", "Consumer Staples", "mega"),
        "CRM": ("Salesforce Inc", "Technology", "large"),
        "BAC": ("Bank of America", "Financials", "large"),
        "MRK": ("Merck & Co", "Healthcare", "large"),
        "CVX": ("Chevron Corp", "Energy", "large"),
        "KO": ("Coca-Cola Co", "Consumer Staples", "large"),
        "ORCL": ("Oracle Corp", "Technology", "large"),
        "AMD": ("Advanced Micro Devices", "Technology", "large"),
        "ACN": ("Accenture plc", "Technology", "large"),
        "PEP": ("PepsiCo Inc", "Consumer Staples", "large"),
        "LIN": ("Linde plc", "Materials", "large"),
        "TMO": ("Thermo Fisher Scientific", "Healthcare", "large"),
        "ADBE": ("Adobe Inc", "Technology", "large"),
        "MCD": ("McDonald's Corp", "Consumer Discretionary", "large"),
        "CSCO": ("Cisco Systems", "Technology", "large"),
        "WFC": ("Wells Fargo", "Financials", "large"),
        "ABT": ("Abbott Laboratories", "Healthcare", "large"),
        "IBM": ("IBM Corp", "Technology", "large"),
        "GE": ("General Electric", "Industrials", "large"),
        "DHR": ("Danaher Corp", "Healthcare", "large"),
        "CAT": ("Caterpillar Inc", "Industrials", "large"),
        "INTU": ("Intuit Inc", "Technology", "large"),
        "QCOM": ("Qualcomm Inc", "Technology", "large"),
        "DIS": ("Walt Disney Co", "Communication Services", "large"),
        "VZ": ("Verizon", "Communication Services", "large"),
        "AMAT": ("Applied Materials", "Technology", "large"),
        "NOW": ("ServiceNow Inc", "Technology", "large"),
        "TXN": ("Texas Instruments", "Technology", "large"),
    }
    
    # Mega Cap Tech (Mag 7 + others)
    MEGA_CAP_TECH = {
        "AAPL": ("Apple Inc", "Consumer Electronics"),
        "MSFT": ("Microsoft Corp", "Software"),
        "NVDA": ("NVIDIA Corp", "Semiconductors"),
        "GOOGL": ("Alphabet Inc", "Internet"),
        "AMZN": ("Amazon.com Inc", "E-Commerce/Cloud"),
        "META": ("Meta Platforms", "Social Media"),
        "TSLA": ("Tesla Inc", "Electric Vehicles"),
        "AVGO": ("Broadcom Inc", "Semiconductors"),
        "ORCL": ("Oracle Corp", "Enterprise Software"),
        "CRM": ("Salesforce Inc", "Cloud Software"),
        "ADBE": ("Adobe Inc", "Software"),
        "AMD": ("Advanced Micro Devices", "Semiconductors"),
        "NFLX": ("Netflix Inc", "Streaming"),
        "INTC": ("Intel Corp", "Semiconductors"),
    }
    
    # Crypto-Related Stocks
    CRYPTO_RELATED = {
        "COIN": ("Coinbase Global", "Crypto Exchange"),
        "MSTR": ("MicroStrategy", "Bitcoin Holdings"),
        "MARA": ("Marathon Digital", "Bitcoin Mining"),
        "RIOT": ("Riot Platforms", "Bitcoin Mining"),
        "CLSK": ("CleanSpark Inc", "Bitcoin Mining"),
        "HUT": ("Hut 8 Mining", "Bitcoin Mining"),
        "BITF": ("Bitfarms Ltd", "Bitcoin Mining"),
        "SQ": ("Block Inc", "Crypto/Payments"),
        "HOOD": ("Robinhood Markets", "Crypto Trading"),
    }
    
    # Commodity ETFs
    COMMODITY_ETFS = {
        "GLD": ("SPDR Gold Shares", "Gold"),
        "SLV": ("iShares Silver Trust", "Silver"),
        "USO": ("United States Oil Fund", "Crude Oil"),
        "UNG": ("United States Natural Gas Fund", "Natural Gas"),
        "DBA": ("Invesco DB Agriculture Fund", "Agriculture"),
        "PDBC": ("Invesco Optimum Yield Diversified", "Commodities"),
        "COPX": ("Global X Copper Miners ETF", "Copper"),
    }
    
    # Bond/Fixed Income ETFs
    BOND_ETFS = {
        "TLT": ("iShares 20+ Year Treasury", "Long-Term Bonds"),
        "IEF": ("iShares 7-10 Year Treasury", "Intermediate Bonds"),
        "SHY": ("iShares 1-3 Year Treasury", "Short-Term Bonds"),
        "LQD": ("iShares Investment Grade Corporate", "Corporate Bonds"),
        "HYG": ("iShares High Yield Corporate", "High Yield Bonds"),
        "AGG": ("iShares Core US Aggregate Bond", "Total Bond Market"),
    }
    
    # Volatility Products
    VOLATILITY = {
        "VIX": ("CBOE Volatility Index", "Volatility Index"),
        "VXX": ("iPath Series B S&P 500 VIX", "VIX Futures"),
        "UVXY": ("ProShares Ultra VIX Short-Term", "Leveraged VIX"),
        "SVXY": ("ProShares Short VIX Short-Term", "Inverse VIX"),
    }
    
    # Popular High-Volume Stocks
    HIGH_VOLUME_MOVERS = {
        "SOFI": ("SoFi Technologies", "Fintech"),
        "PLTR": ("Palantir Technologies", "Data Analytics"),
        "NIO": ("NIO Inc", "Electric Vehicles"),
        "RIVN": ("Rivian Automotive", "Electric Vehicles"),
        "LCID": ("Lucid Group", "Electric Vehicles"),
        "F": ("Ford Motor", "Automotive"),
        "GM": ("General Motors", "Automotive"),
        "AAL": ("American Airlines", "Airlines"),
        "DAL": ("Delta Air Lines", "Airlines"),
        "UAL": ("United Airlines", "Airlines"),
        "CCL": ("Carnival Corp", "Cruise Lines"),
        "NCLH": ("Norwegian Cruise Line", "Cruise Lines"),
        "RCL": ("Royal Caribbean", "Cruise Lines"),
        "AMC": ("AMC Entertainment", "Entertainment"),
        "GME": ("GameStop Corp", "Retail"),
    }


# =============================================================================
# WATCHLIST MANAGER
# =============================================================================

class WatchlistManager:
    """
    Manages multiple watchlists with search, filtering, and persistence
    
    For Brokers:
    -----------
    - Pre-built lists for major indices and sectors
    - Create custom watchlists
    - Search across all lists
    - Filter by sector, market cap, index membership
    - Tag and favorite symbols
    - Export/import watchlists
    
    For Programmers:
    ---------------
    CRUD operations on watchlists with JSON persistence.
    """
    
    def __init__(self, data_dir: str = "./watchlist_data"):
        self.data_dir = data_dir
        self.watchlists: Dict[str, Watchlist] = {}
        self.all_symbols: Dict[str, WatchlistSymbol] = {}  # Master symbol registry
        
        # Initialize with built-in lists
        self._initialize_builtin_lists()
    
    def _initialize_builtin_lists(self):
        """Create built-in watchlists from IndexData"""
        
        # Index ETFs
        self._create_builtin_list(
            "Index ETFs",
            "Major index tracking ETFs",
            IndexData.INDEX_ETFS,
            WatchlistCategory.INDEX.value
        )
        
        # Sector ETFs
        self._create_builtin_list(
            "Sector ETFs",
            "S&P Sector Select SPDRs",
            IndexData.SECTOR_ETFS,
            WatchlistCategory.SECTOR_ETF.value
        )
        
        # Industry ETFs
        self._create_builtin_list(
            "Industry ETFs",
            "Thematic and industry-focused ETFs",
            IndexData.INDUSTRY_ETFS,
            WatchlistCategory.INDUSTRY_ETF.value
        )
        
        # DOW 30
        symbols = []
        for sym, (name, sector, in_sp, in_ndx, in_dow) in IndexData.DOW_30.items():
            symbols.append(WatchlistSymbol(
                symbol=sym, name=name, category="Dow 30", sector=sector,
                market_cap="mega", in_sp500=in_sp, in_nasdaq100=in_ndx, in_dow30=in_dow
            ))
        self.watchlists["Dow 30"] = Watchlist(
            name="Dow 30",
            description="Dow Jones Industrial Average components",
            symbols=symbols,
            is_builtin=True
        )
        self._register_symbols(symbols)
        
        # Nasdaq 100 Top 50
        symbols = []
        for sym, (name, sector) in IndexData.NASDAQ_100_TOP.items():
            symbols.append(WatchlistSymbol(
                symbol=sym, name=name, category="Nasdaq 100", sector=sector,
                market_cap="large", in_nasdaq100=True,
                in_sp500=sym in IndexData.SP500_TOP
            ))
        self.watchlists["Nasdaq 100 Top"] = Watchlist(
            name="Nasdaq 100 Top",
            description="Top 50 Nasdaq 100 components by weight",
            symbols=symbols,
            is_builtin=True
        )
        self._register_symbols(symbols)
        
        # S&P 500 Top 50
        symbols = []
        for sym, (name, sector, cap) in IndexData.SP500_TOP.items():
            symbols.append(WatchlistSymbol(
                symbol=sym, name=name, category="S&P 500", sector=sector,
                market_cap=cap, in_sp500=True,
                in_dow30=sym in IndexData.DOW_30,
                in_nasdaq100=sym in IndexData.NASDAQ_100_TOP
            ))
        self.watchlists["S&P 500 Top"] = Watchlist(
            name="S&P 500 Top",
            description="Top 50 S&P 500 components by weight",
            symbols=symbols,
            is_builtin=True
        )
        self._register_symbols(symbols)
        
        # Mega Cap Tech
        self._create_builtin_list(
            "Mega Cap Tech",
            "Magnificent 7 and other mega-cap tech",
            IndexData.MEGA_CAP_TECH,
            WatchlistCategory.MEGA_CAP.value
        )
        
        # Crypto Related
        self._create_builtin_list(
            "Crypto Related",
            "Cryptocurrency and blockchain stocks",
            IndexData.CRYPTO_RELATED,
            WatchlistCategory.CRYPTO.value
        )
        
        # Commodity ETFs
        self._create_builtin_list(
            "Commodities",
            "Commodity tracking ETFs",
            IndexData.COMMODITY_ETFS,
            WatchlistCategory.COMMODITY.value
        )
        
        # Volatility
        self._create_builtin_list(
            "Volatility",
            "VIX and volatility products",
            IndexData.VOLATILITY,
            "Volatility"
        )
        
        # High Volume Movers
        self._create_builtin_list(
            "High Volume",
            "Popular high-volume stocks",
            IndexData.HIGH_VOLUME_MOVERS,
            "High Volume"
        )
        
        # Create empty Favorites list
        self.watchlists["Favorites"] = Watchlist(
            name="Favorites",
            description="Your favorite symbols",
            symbols=[],
            is_builtin=False
        )
        
        # Create empty Custom list
        self.watchlists["Custom"] = Watchlist(
            name="Custom",
            description="Your custom watchlist",
            symbols=[],
            is_builtin=False
        )
    
    def _create_builtin_list(self, 
                              name: str, 
                              description: str,
                              data: Dict, 
                              category: str):
        """Helper to create a built-in watchlist from data dict"""
        symbols = []
        for sym, info in data.items():
            if len(info) == 2:
                sym_name, sector = info
            else:
                sym_name = info[0]
                sector = info[1] if len(info) > 1 else ""
            
            symbols.append(WatchlistSymbol(
                symbol=sym,
                name=sym_name,
                category=category,
                sector=sector
            ))
        
        self.watchlists[name] = Watchlist(
            name=name,
            description=description,
            symbols=symbols,
            is_builtin=True
        )
        self._register_symbols(symbols)
    
    def _register_symbols(self, symbols: List[WatchlistSymbol]):
        """Add symbols to master registry"""
        for sym in symbols:
            if sym.symbol not in self.all_symbols:
                self.all_symbols[sym.symbol] = sym
            else:
                # Merge info
                existing = self.all_symbols[sym.symbol]
                if sym.in_sp500:
                    existing.in_sp500 = True
                if sym.in_nasdaq100:
                    existing.in_nasdaq100 = True
                if sym.in_dow30:
                    existing.in_dow30 = True
    
    # =========================================================================
    # CRUD OPERATIONS
    # =========================================================================
    
    def create_watchlist(self, name: str, description: str = "") -> Watchlist:
        """Create a new custom watchlist"""
        if name in self.watchlists:
            raise ValueError(f"Watchlist '{name}' already exists")
        
        watchlist = Watchlist(name=name, description=description, is_builtin=False)
        self.watchlists[name] = watchlist
        return watchlist
    
    def delete_watchlist(self, name: str) -> bool:
        """Delete a watchlist (cannot delete built-in lists)"""
        if name not in self.watchlists:
            return False
        
        if self.watchlists[name].is_builtin:
            raise ValueError(f"Cannot delete built-in watchlist '{name}'")
        
        del self.watchlists[name]
        return True
    
    def add_symbol(self, 
                   watchlist_name: str, 
                   symbol: str,
                   name: str = "",
                   category: str = "Custom",
                   sector: str = "",
                   tags: List[str] = None) -> WatchlistSymbol:
        """Add a symbol to a watchlist"""
        if watchlist_name not in self.watchlists:
            raise ValueError(f"Watchlist '{watchlist_name}' not found")
        
        watchlist = self.watchlists[watchlist_name]
        symbol = symbol.upper()
        
        # Check if already exists
        for s in watchlist.symbols:
            if s.symbol == symbol:
                return s  # Already exists
        
        # Check master registry for existing info
        if symbol in self.all_symbols:
            sym_obj = self.all_symbols[symbol]
            # Create a copy with potential overrides
            new_sym = WatchlistSymbol(
                symbol=symbol,
                name=name or sym_obj.name,
                category=category or sym_obj.category,
                sector=sector or sym_obj.sector,
                market_cap=sym_obj.market_cap,
                in_sp500=sym_obj.in_sp500,
                in_nasdaq100=sym_obj.in_nasdaq100,
                in_dow30=sym_obj.in_dow30,
                tags=tags or []
            )
        else:
            new_sym = WatchlistSymbol(
                symbol=symbol,
                name=name,
                category=category,
                sector=sector,
                tags=tags or []
            )
            self.all_symbols[symbol] = new_sym
        
        watchlist.symbols.append(new_sym)
        watchlist.modified_date = datetime.now().isoformat()
        
        return new_sym
    
    def remove_symbol(self, watchlist_name: str, symbol: str) -> bool:
        """Remove a symbol from a watchlist"""
        if watchlist_name not in self.watchlists:
            return False
        
        watchlist = self.watchlists[watchlist_name]
        symbol = symbol.upper()
        
        for i, s in enumerate(watchlist.symbols):
            if s.symbol == symbol:
                watchlist.symbols.pop(i)
                watchlist.modified_date = datetime.now().isoformat()
                return True
        
        return False
    
    def toggle_symbol(self, watchlist_name: str, symbol: str) -> bool:
        """Toggle a symbol's enabled state"""
        if watchlist_name not in self.watchlists:
            return False
        
        for s in self.watchlists[watchlist_name].symbols:
            if s.symbol == symbol.upper():
                s.enabled = not s.enabled
                return True
        
        return False
    
    def toggle_favorite(self, symbol: str) -> bool:
        """Toggle favorite status and add/remove from Favorites list"""
        symbol = symbol.upper()
        
        if symbol in self.all_symbols:
            sym = self.all_symbols[symbol]
            sym.is_favorite = not sym.is_favorite
            
            # Update Favorites list
            fav_list = self.watchlists.get("Favorites")
            if fav_list:
                if sym.is_favorite:
                    # Add to favorites if not there
                    if not any(s.symbol == symbol for s in fav_list.symbols):
                        fav_list.symbols.append(sym)
                else:
                    # Remove from favorites
                    fav_list.symbols = [s for s in fav_list.symbols if s.symbol != symbol]
            
            return True
        return False
    
    def is_in_watchlist(self, symbol: str) -> bool:
        """Check if a symbol is in any watchlist"""
        return symbol.upper() in self.all_symbols
    
    def get_symbol_info(self, symbol: str) -> Optional[WatchlistSymbol]:
        """Get watchlist info for a symbol if it exists"""
        return self.all_symbols.get(symbol.upper())
    
    def get_symbol_lists(self, symbol: str) -> List[str]:
        """Get names of all watchlists containing this symbol"""
        symbol = symbol.upper()
        lists = []
        for name, wl in self.watchlists.items():
            for sym in wl.symbols:
                if sym.symbol == symbol:
                    lists.append(name)
                    break
        return lists
    
    # =========================================================================
    # SEARCH AND FILTER
    # =========================================================================
    
    def search(self, 
               query: str,
               watchlist_name: str = None,
               limit: int = 50) -> List[WatchlistSymbol]:
        """
        Search for symbols by symbol or name
        
        Args:
            query: Search string (symbol or partial name)
            watchlist_name: Limit search to specific watchlist (None = all)
            limit: Max results
        
        Returns:
            List of matching WatchlistSymbol objects
        """
        query = query.upper().strip()
        results = []
        
        if watchlist_name:
            # Search specific watchlist
            if watchlist_name not in self.watchlists:
                return []
            symbols = self.watchlists[watchlist_name].symbols
        else:
            # Search all symbols
            symbols = list(self.all_symbols.values())
        
        for sym in symbols:
            if query in sym.symbol or query in sym.name.upper():
                results.append(sym)
                if len(results) >= limit:
                    break
        
        # Sort by relevance (exact symbol match first)
        results.sort(key=lambda s: (
            0 if s.symbol == query else 1,
            0 if s.symbol.startswith(query) else 1,
            len(s.symbol)
        ))
        
        return results
    
    def filter_symbols(self,
                       watchlist_name: str = None,
                       sector: str = None,
                       category: str = None,
                       market_cap: str = None,
                       in_sp500: bool = None,
                       in_nasdaq100: bool = None,
                       in_dow30: bool = None,
                       enabled_only: bool = False,
                       favorites_only: bool = False,
                       tags: List[str] = None) -> List[WatchlistSymbol]:
        """
        Filter symbols by various criteria
        
        Returns:
            List of matching WatchlistSymbol objects
        """
        if watchlist_name:
            if watchlist_name not in self.watchlists:
                return []
            symbols = self.watchlists[watchlist_name].symbols
        else:
            symbols = list(self.all_symbols.values())
        
        results = []
        
        for sym in symbols:
            # Apply filters
            if enabled_only and not sym.enabled:
                continue
            if favorites_only and not sym.is_favorite:
                continue
            if sector and sym.sector != sector:
                continue
            if category and sym.category != category:
                continue
            if market_cap and sym.market_cap != market_cap:
                continue
            if in_sp500 is not None and sym.in_sp500 != in_sp500:
                continue
            if in_nasdaq100 is not None and sym.in_nasdaq100 != in_nasdaq100:
                continue
            if in_dow30 is not None and sym.in_dow30 != in_dow30:
                continue
            if tags and not any(t in sym.tags for t in tags):
                continue
            
            results.append(sym)
        
        return results
    
    def get_by_index(self, index: str) -> List[WatchlistSymbol]:
        """Get all symbols in a specific index"""
        index = index.upper()
        
        if index in ["SPY", "SPX", "SP500", "S&P500", "S&P 500"]:
            return self.filter_symbols(in_sp500=True)
        elif index in ["QQQ", "NDX", "NASDAQ100", "NASDAQ 100"]:
            return self.filter_symbols(in_nasdaq100=True)
        elif index in ["DIA", "DJIA", "DOW", "DOW30", "DOW 30"]:
            return self.filter_symbols(in_dow30=True)
        else:
            return []
    
    def get_by_sector(self, sector: str) -> List[WatchlistSymbol]:
        """Get all symbols in a specific sector"""
        return self.filter_symbols(sector=sector)
    
    def get_all_sectors(self) -> List[str]:
        """Get list of all unique sectors"""
        sectors = set()
        for sym in self.all_symbols.values():
            if sym.sector:
                sectors.add(sym.sector)
        return sorted(list(sectors))
    
    def get_all_categories(self) -> List[str]:
        """Get list of all unique categories"""
        categories = set()
        for sym in self.all_symbols.values():
            if sym.category:
                categories.add(sym.category)
        return sorted(list(categories))
    
    # =========================================================================
    # WATCHLIST RETRIEVAL
    # =========================================================================
    
    def get_watchlist(self, name: str) -> Optional[Watchlist]:
        """Get a watchlist by name"""
        return self.watchlists.get(name)
    
    def get_all_watchlists(self) -> List[Watchlist]:
        """Get all watchlists"""
        return list(self.watchlists.values())
    
    def get_watchlist_names(self) -> List[str]:
        """Get names of all watchlists"""
        return list(self.watchlists.keys())
    
    def get_enabled_symbols(self, watchlist_name: str = None) -> List[str]:
        """Get list of enabled symbol strings for scanning"""
        if watchlist_name:
            if watchlist_name not in self.watchlists:
                return []
            symbols = self.watchlists[watchlist_name].symbols
        else:
            # All unique enabled symbols across all lists
            seen = set()
            symbols = []
            for wl in self.watchlists.values():
                for s in wl.symbols:
                    if s.symbol not in seen:
                        symbols.append(s)
                        seen.add(s.symbol)
        
        return [s.symbol for s in symbols if s.enabled]
    
    # =========================================================================
    # PERSISTENCE
    # =========================================================================
    
    def save(self, filename: str = "watchlists.json"):
        """Save all watchlists to JSON file"""
        os.makedirs(self.data_dir, exist_ok=True)
        filepath = os.path.join(self.data_dir, filename)
        
        data = {
            "watchlists": {},
            "all_symbols": {}
        }
        
        for name, wl in self.watchlists.items():
            data["watchlists"][name] = {
                "name": wl.name,
                "description": wl.description,
                "symbols": [asdict(s) for s in wl.symbols],
                "created_date": wl.created_date,
                "modified_date": wl.modified_date,
                "is_builtin": wl.is_builtin
            }
        
        for sym, obj in self.all_symbols.items():
            data["all_symbols"][sym] = asdict(obj)
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filepath
    
    def load(self, filename: str = "watchlists.json") -> bool:
        """Load watchlists from JSON file"""
        filepath = os.path.join(self.data_dir, filename)
        
        if not os.path.exists(filepath):
            return False
        
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Load symbols
            for sym, obj in data.get("all_symbols", {}).items():
                self.all_symbols[sym] = WatchlistSymbol(**obj)
            
            # Load watchlists
            for name, wl_data in data.get("watchlists", {}).items():
                symbols = [WatchlistSymbol(**s) for s in wl_data.get("symbols", [])]
                self.watchlists[name] = Watchlist(
                    name=wl_data["name"],
                    description=wl_data.get("description", ""),
                    symbols=symbols,
                    created_date=wl_data.get("created_date", ""),
                    modified_date=wl_data.get("modified_date", ""),
                    is_builtin=wl_data.get("is_builtin", False)
                )
            
            return True
        except Exception as e:
            print(f"Error loading watchlists: {e}")
            return False
    
    def export_watchlist(self, watchlist_name: str, filename: str) -> str:
        """Export a single watchlist to file"""
        if watchlist_name not in self.watchlists:
            raise ValueError(f"Watchlist '{watchlist_name}' not found")
        
        wl = self.watchlists[watchlist_name]
        data = {
            "name": wl.name,
            "description": wl.description,
            "symbols": [asdict(s) for s in wl.symbols],
            "exported_date": datetime.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        return filename
    
    def import_watchlist(self, filename: str, new_name: str = None) -> Watchlist:
        """Import a watchlist from file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        name = new_name or data["name"]
        if name in self.watchlists:
            name = f"{name}_imported"
        
        symbols = [WatchlistSymbol(**s) for s in data.get("symbols", [])]
        
        watchlist = Watchlist(
            name=name,
            description=data.get("description", ""),
            symbols=symbols,
            is_builtin=False
        )
        
        self.watchlists[name] = watchlist
        self._register_symbols(symbols)
        
        return watchlist
    
    # =========================================================================
    # DISPLAY
    # =========================================================================
    
    def print_summary(self) -> str:
        """Print summary of all watchlists"""
        lines = []
        lines.append("=" * 60)
        lines.append("WATCHLIST SUMMARY")
        lines.append("=" * 60)
        
        total_symbols = len(self.all_symbols)
        lines.append(f"Total unique symbols: {total_symbols}")
        lines.append(f"Total watchlists: {len(self.watchlists)}")
        lines.append("")
        
        # Group by type
        builtin = [wl for wl in self.watchlists.values() if wl.is_builtin]
        custom = [wl for wl in self.watchlists.values() if not wl.is_builtin]
        
        lines.append("BUILT-IN WATCHLISTS:")
        lines.append("-" * 40)
        for wl in sorted(builtin, key=lambda x: x.name):
            lines.append(f"  {wl.name:<25} ({wl.symbol_count} symbols)")
        
        lines.append("")
        lines.append("CUSTOM WATCHLISTS:")
        lines.append("-" * 40)
        for wl in sorted(custom, key=lambda x: x.name):
            lines.append(f"  {wl.name:<25} ({wl.symbol_count} symbols)")
        
        # Index membership
        lines.append("")
        lines.append("INDEX MEMBERSHIP:")
        lines.append("-" * 40)
        sp500 = sum(1 for s in self.all_symbols.values() if s.in_sp500)
        ndx = sum(1 for s in self.all_symbols.values() if s.in_nasdaq100)
        dow = sum(1 for s in self.all_symbols.values() if s.in_dow30)
        lines.append(f"  S&P 500 components:    {sp500}")
        lines.append(f"  Nasdaq 100 components: {ndx}")
        lines.append(f"  Dow 30 components:     {dow}")
        
        lines.append("=" * 60)
        
        return "\n".join(lines)
    
    def print_watchlist(self, name: str) -> str:
        """Print details of a specific watchlist"""
        if name not in self.watchlists:
            return f"Watchlist '{name}' not found"
        
        wl = self.watchlists[name]
        lines = []
        
        lines.append("=" * 70)
        lines.append(f"WATCHLIST: {wl.name}")
        lines.append(f"Description: {wl.description}")
        lines.append(f"Symbols: {wl.symbol_count} ({wl.enabled_count} enabled)")
        lines.append("=" * 70)
        
        lines.append(f"\n{'Symbol':<8} {'Name':<30} {'Sector':<20} {'Enabled'}")
        lines.append("-" * 70)
        
        for s in sorted(wl.symbols, key=lambda x: x.symbol):
            enabled = "✓" if s.enabled else "✗"
            fav = "★" if s.is_favorite else " "
            lines.append(f"{fav}{s.symbol:<7} {s.name[:29]:<30} {s.sector[:19]:<20} {enabled}")
        
        return "\n".join(lines)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_scan_list(manager: WatchlistManager, list_type: str) -> List[str]:
    """
    Get a quick list of symbols for common scan types
    
    Args:
        manager: WatchlistManager instance
        list_type: One of "indices", "sectors", "mega", "dow", "nasdaq", "sp500", "all"
    
    Returns:
        List of symbol strings
    """
    list_type = list_type.lower()
    
    if list_type == "indices":
        return manager.get_enabled_symbols("Index ETFs")
    elif list_type == "sectors":
        return manager.get_enabled_symbols("Sector ETFs")
    elif list_type == "mega" or list_type == "magnificent":
        return manager.get_enabled_symbols("Mega Cap Tech")
    elif list_type == "dow" or list_type == "dow30":
        return manager.get_enabled_symbols("Dow 30")
    elif list_type == "nasdaq" or list_type == "ndx":
        return manager.get_enabled_symbols("Nasdaq 100 Top")
    elif list_type == "sp500" or list_type == "spy":
        return manager.get_enabled_symbols("S&P 500 Top")
    elif list_type == "favorites":
        return manager.get_enabled_symbols("Favorites")
    elif list_type == "all":
        return manager.get_enabled_symbols()
    else:
        # Try as watchlist name
        return manager.get_enabled_symbols(list_type)


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    # Create manager
    manager = WatchlistManager()
    
    # Print summary
    print(manager.print_summary())
    
    # Search demo
    print("\n🔍 Search for 'AAPL':")
    results = manager.search("AAPL")
    for r in results[:5]:
        print(f"  {r.symbol}: {r.name} - {r.sector}")
    
    print("\n🔍 Search for 'NVIDIA':")
    results = manager.search("NVIDIA")
    for r in results[:5]:
        print(f"  {r.symbol}: {r.name} - {r.sector}")
    
    # Filter demo
    print("\n📊 Technology stocks in S&P 500:")
    tech_sp500 = manager.filter_symbols(sector="Technology", in_sp500=True)
    for s in tech_sp500[:10]:
        print(f"  {s.symbol}: {s.name}")
    
    # Dow 30
    print("\n" + manager.print_watchlist("Dow 30"))
    
    # Add custom symbol
    manager.add_symbol("Custom", "CUSTOM1", name="My Custom Stock", sector="Test")
    print("\nAdded custom symbol to Custom watchlist")
    
    # Quick scan lists
    print("\n🎯 Quick Scan Lists:")
    print(f"  Indices: {quick_scan_list(manager, 'indices')}")
    print(f"  Sectors: {quick_scan_list(manager, 'sectors')}")
    print(f"  Mega Cap: {quick_scan_list(manager, 'mega')}")
