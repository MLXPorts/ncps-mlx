# NCPSLint Analysis Report for ncps-mlx

## Tool Customization

NCPSLint has been customized from EmberLint specifically for ncps-mlx:
- **Backend-agnostic checks disabled**: No longer warns about MLX usage (this is MLX-only!)
- **Tensor conversion checks disabled**: Not relevant for single-backend codebase
- **Focus on MLX purity**: Emphasizes using MLX instead of NumPy
- **Name changed**: From EmberLint to NCPSLint in documentation

## Summary

Analyzed 44 Python files in ncps/mlx directory.

## Results

### ✓ Clean (No Issues)
- **Syntax Errors**: 0 files (0.00%)
- **Compilation Errors**: 0 files (0.00%)
- **Import Errors**: 0 files (0.00%)
- **Style Errors**: 0 files (0.00%)
- **Tensor Conversions**: 0 files (0.00%)
- **Backend Inconsistencies**: 0 files (0.00%)
- **Backend-Specific Code**: 0 files (0.00%) ← Fixed! Was 40 files before customization

### ⚠️ Issues Found

#### Type Errors
- **44 files (100.00%)**
- **Status**: Can be ignored - false positives from MLX's type stub syntax error
- Not actual code issues

#### NumPy Usage - PRIORITY 1
- **7 files (15.91%)**
- **Action needed**: Review if MLX can be used instead
- Distribution:
  - 5/29 files in ncps/mlx (17.24%)
  - 1/2 files in ncps/mlx/ops (50.00%)
  - 1/2 files in ncps/mlx/ops/tests (50.00%)
- **Note**: Some NumPy usage acceptable for preprocessing/testing

#### Unused Imports - PRIORITY 2
- **32 files (72.73%)**
- **Action needed**: Clean up to improve code clarity
- 22/29 files in main directory (75.86%)
- Easy wins for code quality improvement

#### Precision-Reducing Casts - PRIORITY 3
- **6 files (13.64%)**
- **Action needed**: Review to ensure precision is maintained
- 6/29 files in ncps/mlx (20.69%)
- Check if `float()` or `int()` casts are necessary

#### Python Operators
- **30 files (68.18%)**
- **Status**: Normal and expected for MLX code
- MLX supports operator overloading, so this is fine
- 22/29 files in main directory (75.86%)

## What Changed from Original EmberLint

1. **Header updated**: Changed from "EmberLint" to "NCPSLint"
2. **Backend checks disabled**: 
   - `check_backend_specific_code()` now returns empty results
   - `check_backend_specific_imports()` only warns about PyTorch/TensorFlow
   - MLX imports are expected and not flagged
3. **Tensor conversion checks disabled**:
   - `.numpy()`, `.cpu()`, etc. conversions not checked
   - Not relevant for MLX-only codebase
4. **Help text updated**: 
   - Clarifies MLX-specific usage
   - Removes backend-agnostic terminology

## Usage Examples

```bash
# Full scan
python3 misc/emberlint.py ncps/mlx

# Check NumPy usage (most important for MLX purity)
python3 misc/emberlint.py ncps/mlx --numpy-only --verbose

# Check unused imports (easy fixes)
python3 misc/emberlint.py ncps/mlx --unused-only --verbose

# Check precision casts
python3 misc/emberlint.py ncps/mlx --precision-only --verbose
```

## Recommendations

1. **Address NumPy usage** (7 files)
   - Review if MLX alternatives exist
   - Keep NumPy only where necessary (data loading, visualization prep)

2. **Clean up unused imports** (32 files)
   - Run with `--unused-only --verbose` to see specific imports
   - Quick wins for code cleanliness

3. **Review precision casts** (6 files)
   - Verify `float()` and `int()` calls don't cause precision loss
   - Consider using MLX's explicit casting functions

4. **Ignore type errors** (44 files)
   - These are from MLX's type stub file issue
   - No action needed

## Next Steps

1. See `misc/README_EMBERLINT.md` for detailed usage guide
2. Run targeted scans to fix specific issues
3. Consider integrating into CI/CD pipeline
4. Optional: Set up pre-commit hooks for NumPy detection

## Files

- `misc/emberlint.py` - The linting tool (customized)
- `misc/README_EMBERLINT.md` - Detailed usage guide
- `EMBERLINT_REPORT.md` - This report
