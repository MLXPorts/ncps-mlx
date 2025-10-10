# Dependency Cleanup & Reorganization

## Dataset & Examples Reorganization

### Problem
- Datasets were inside the package (`ncps/datasets/` and `ncps/mlx/datasets/`)
- This causes package bloat - datasets shouldn't be bundled with library
- Two different implementations (numpy vs MLX) causing confusion
- Examples already at root, but datasets were nested

### Solution
Moved datasets and utilities outside the package to project root level.

## Changes Made

### 1. Dataset Consolidation
```
BEFORE:
ncps/datasets/
├── __init__.py
├── utils.py
└── icra2020_lidar_collision_avoidance.py (numpy version)

ncps/mlx/datasets/
├── __init__.py
└── icra2020_lidar_collision_avoidance.py (MLX version)

AFTER:
datasets/  (at project root)
├── __init__.py (new)
├── utils.py
├── icra2020_lidar_collision_avoidance.py (MLX version)
├── data/
└── icra_lds/
```

### 2. Import Updates
Updated all imports throughout the codebase:

**Before:**
```python
from ncps.mlx.datasets.icra2020_lidar_collision_avoidance import load_data
from ncps.datasets.icra2020_lidar_collision_avoidance import load_data
```

**After:**
```python
from datasets.icra2020_lidar_collision_avoidance import load_data
from datasets.utils import download_and_unzip
```

### 3. Package Cleanup
- Removed `from . import datasets` from `ncps/__init__.py`
- Removed `"datasets"` from `__all__` in `ncps/__init__.py`
- Deleted `ncps/datasets/` directory
- Deleted `ncps/mlx/datasets/` directory

## New Project Structure

```
ncps-mlx/
├── ncps/              # Pure library code (importable package)
│   ├── __init__.py
│   ├── wirings.py
│   ├── mlx/           # MLX implementations
│   └── profiles/
│
├── datasets/          # Dataset loaders (project utility)
│   ├── __init__.py
│   ├── utils.py
│   ├── icra2020_lidar_collision_avoidance.py
│   ├── data/          # Actual data files
│   └── icra_lds/
│
├── examples/          # Example scripts (project level)
│   ├── sine_example.py
│   ├── icra_lidar_mlx.py
│   └── ...
│
├── tests/             # Test suite
└── docs/              # Documentation
```

## Benefits

### 1. Clean Separation of Concerns
- **ncps/** = Library code only (what gets installed via pip)
- **datasets/** = Data loaders and utilities (project-level)
- **examples/** = Demo scripts (project-level)

### 2. No Package Bloat
- Dataset loaders not bundled when installing package
- Examples not bundled with package
- Smaller package size, faster installs

### 3. Clearer Organization
- Easy to find dataset loaders (top-level `datasets/`)
- Easy to find examples (top-level `examples/`)
- Package only contains core library functionality

### 4. Better Developer Experience
```python
# Import library code (from package)
from ncps import LTC, CfC, wirings
from ncps.wirings import FullyConnected

# Import dataset loaders (project-level utility)
from datasets.icra2020_lidar_collision_avoidance import load_data

# Run examples (just execute)
python examples/sine_example.py
```

### 5. Resolved Duplication
- Eliminated duplicate dataset implementations
- Using MLX version (better for pure MLX project)
- Single source of truth for dataset loading

## Files Updated

### Modified
- `ncps/__init__.py` - Removed datasets import
- `examples/icra_lidar_eval.py` - Updated import
- `examples/icra_lidar_pygame.py` - Updated import
- `examples/icra_lidar_mlx.py` - Updated import
- `examples/icra_lidar_infer.py` - Updated import
- `tests/examples/test_icra_inference.py` - Updated import

### Created
- `datasets/__init__.py` - New package initialization

### Moved
- `ncps/mlx/datasets/icra2020_lidar_collision_avoidance.py` → `datasets/`
- `ncps/datasets/utils.py` → `datasets/`

### Deleted
- `ncps/datasets/` - Entire directory
- `ncps/mlx/datasets/` - Entire directory

## Testing

All imports verified working:
- ✓ `from datasets.icra2020_lidar_collision_avoidance import load_data`
- ✓ `from datasets.utils import download_and_unzip`
- ✓ `from ncps import LTC, CfC, wirings`
- ✓ `ncps.datasets` properly removed from package

## Migration Guide

If you have code importing datasets from the old location:

**Old:**
```python
from ncps.datasets.icra2020_lidar_collision_avoidance import load_data
# or
from ncps.mlx.datasets.icra2020_lidar_collision_avoidance import load_data
```

**New:**
```python
from datasets.icra2020_lidar_collision_avoidance import load_data
```

Note: This only affects project-level code (examples, tests). If you were using ncps as an installed package, you likely weren't importing datasets anyway.

---
Date: $(date)
