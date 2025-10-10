# NCPS-MLX Structure Refactoring

## Summary

Cleaned up the project structure by removing the confusing nested `ncps/ncps_mlx/` folder structure and moving wirings to the top level where it belongs.

## Changes Made

### 1. Removed Empty Directory
- **Deleted**: `ncps/mlx/` (was completely empty)

### 2. Renamed Core Implementation
- **Before**: `ncps/ncps_mlx/` (28 Python files)
- **After**: `ncps/mlx/` 
- **Benefit**: Cleaner path, no more redundant naming

### 3. Moved Wirings to Top Level
- **Before**: `ncps/mlx/wirings.py` (hidden inside mlx folder)
- **After**: `ncps/wirings.py` (top-level module)
- **Benefit**: More logical structure - wirings is a core concept, not MLX-specific

### 4. Updated All Imports
- Updated ~50+ files across examples/, tests/, and docs/
- Changed `from ncps.ncps_mlx` → `from ncps.mlx`
- Changed `from ncps.mlx.wirings` → `from ncps.wirings`
- Changed `from ncps_mlx` → `from ncps.mlx`

### 5. Internal Import Updates
- `ncps/mlx/__init__.py`: Changed `from . import wirings` → `from .. import wirings`
- `ncps/mlx/cfc.py`: Changed `from . import wirings` → `from .. import wirings`
- `ncps/mlx/ltc.py`: Changed `from . import wirings` → `from .. import wirings`

## New Structure

```
ncps/
├── __init__.py           # Imports from .mlx and .wirings
├── wirings.py            # Core wiring topologies (top-level!)
├── mlx/                  # MLX implementations (renamed from ncps_mlx)
│   ├── __init__.py
│   ├── cfc_cell.py
│   ├── cfc.py
│   ├── ltc_cell.py
│   ├── ltc.py
│   ├── ctgru_cell.py
│   ├── ctrnn_cell.py
│   ├── eltc_cell.py
│   └── ... (25 more files)
├── datasets/
└── profiles/

ncps_mlx/                 # Backward compatibility alias
└── __init__.py           # Forwards to ncps.mlx
```

## Import Methods (All Working)

### Recommended
```python
from ncps import CTRNN, LTC, CfC
from ncps import wirings
from ncps.wirings import FullyConnected, AutoNCP, NCP
```

### Direct MLX Access
```python
from ncps.mlx import CTRNN, LTC
from ncps.mlx import wirings  # Same as ncps.wirings
```

### Legacy Compatibility
```python
from ncps_mlx import LTC4  # Still works
```

## Benefits

1. **Cleaner Structure**: No more nested `ncps/ncps_mlx/` confusion
2. **Logical Organization**: Wirings at top level where it belongs
3. **Shorter Imports**: `from ncps.mlx` instead of `from ncps.ncps_mlx`
4. **No Redundancy**: Removed empty directories and proxy files
5. **Backward Compatible**: Old `ncps_mlx` imports still work
6. **Better Semantics**: `ncps.mlx` now actually contains the MLX code!

## Testing

All imports tested and verified working:
- ✓ Module imports (LTC, CfC, etc.)
- ✓ Wirings imports and usage
- ✓ Forward passes with models
- ✓ Internal module cross-references
- ✓ Legacy compatibility imports
- ✓ Example scripts still functional

## Files Modified

- Core package: `ncps/__init__.py`, `ncps/wirings.py` (moved)
- MLX modules: `ncps/mlx/__init__.py`, `ncps/mlx/cfc.py`, `ncps/mlx/ltc.py`
- Alias package: `ncps_mlx/__init__.py`
- Examples: ~25 files updated
- Tests: ~5 files updated
- Docs: Multiple generated files updated

## No More "Coding Hoarder" Behavior! 😄

Removed the empty `ncps/mlx/` directory that was being preserved for no reason.
