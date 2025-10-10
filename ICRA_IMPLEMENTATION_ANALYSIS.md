# ICRA 2020 Implementation Analysis

## Original Paper Implementation (/tmp/icra_lds)

### Architecture
The original ICRA 2020 paper "Gershgorin Loss Stabilizes the Recurrent Neural Network Compartment of an End-To-End Robot Learning Scheme" uses:

1. **Conv1D Feature Extractor** for LIDAR data:
   ```
   Conv1D(12, k=5, s=3, relu) 
   → Conv1D(16, k=5, s=3, relu)
   → Conv1D(24, k=5, s=2, relu) 
   → Conv1D(1, k=1, s=1)
   → Flatten
   ```

2. **State Processing**: 
   - 2D vehicle state (velocity, angular_velocity)
   - No explicit processing, concatenated with LIDAR features

3. **RNN Options**:
   - LSTM (baseline)
   - CT-RNN (baseline)
   - LDS without Gershgorin (baseline)
   - LDS with Gershgorin (best performer)

### Training Data
- **Source**: Human demonstrations from real robot
- **Format**: 29 CSV trace files in `training_data/`
- **Structure**: `[timestamp, state_1, state_2, lidar_0...lidar_N, steering]`
  - state_1, state_2: vehicle state (2 dims)
  - lidar_0...lidar_N: LIDAR measurements (typically 181 or 543 bins)
  - steering: ground truth steering command (last column)

- **Data Augmentation**:
  - Mirror LIDAR readings (flip left/right)
  - Negate state values
  - Negate steering command
  - Effectively doubles the dataset

- **Packed Format**: `icra2020_imitation_data_packed.npz`
  - Preprocessed, augmented, and packed for faster loading

### Training Process
1. Load CSV traces or packed NPZ
2. Sample random sequences of length `seq_len` (default 32)
3. Apply data augmentation
4. Train with MSE loss on steering prediction
5. Use batch size 32, typically 100 epochs

## Our MLX Implementation

### ✅ What We Have Correctly

1. **Conv1D Backbone** (`ncps/mlx/icra_cfc_cell.py`):
   ```python
   self.head = nn.Sequential(
       nn.Conv1d(1, 12, kernel_size=5, stride=3), nn.ReLU(),
       nn.Conv1d(12, 16, kernel_size=5, stride=3), nn.ReLU(),
       nn.Conv1d(16, 24, kernel_size=5, stride=2), nn.ReLU(),
       nn.Conv1d(24, 1, kernel_size=1, stride=1),
   )
   ```
   ✅ Matches original exactly

2. **Packed NPZ Data** (`datasets/icra2020_imitation_data_packed.npz`):
   ✅ We have the preprocessed data file

3. **Data Loader** (`datasets/icra2020_lidar_collision_avoidance.py`):
   ✅ Loads packed NPZ
   ✅ Applies augmentation (mirror + negate)
   ✅ Returns MLX arrays

4. **Training Script** (`examples/icra_lidar_mlx.py`):
   ✅ Uses IcraCfCCell
   ✅ MSE loss
   ✅ Adam optimizer

### ⚠️ Potential Issues / Differences

#### 1. Dataset Loading
**Original**: Loads individual CSV files, samples random subsequences
**Ours**: Loads packed NPZ which may have pre-windowed sequences

**Check Needed**: Are the sequences in the NPZ already windowed? Or full traces?

#### 2. Training Methodology
**Original** (`train_imitator.py`):
- Random sampling of sequences during training
- Continuous shuffling per epoch
- Validation on held-out sequences

**Ours** (`examples/icra_lidar_mlx.py`):
- Uses pre-split train/test from packed NPZ
- May not have the same random sampling pattern

#### 3. Raw CSV Traces
**Original**: Has access to 29 raw CSV files for flexible resampling
**Ours**: Only uses packed NPZ - may have lost some flexibility

### 🔍 What to Verify

1. **Check NPZ Contents**:
   ```python
   import numpy as np
   data = np.load('datasets/icra2020_imitation_data_packed.npz')
   print(data.files)  # What keys?
   print(data['train_x'].shape)  # Are these full traces or windows?
   ```

2. **Compare Data Shapes**:
   - Original CSV: Variable length traces
   - Our NPZ: Check if fixed-length windows or full traces

3. **Verify Augmentation**:
   - Does our loader properly mirror LIDAR and negate steering?
   - Check the `_augment_combined` function

4. **Training Hyperparameters**:
   - Original: batch_size=32, seq_len=32, typically 100 epochs
   - Ours: Check default parameters in training script

### 📋 Action Items

#### High Priority
1. ✅ Verify we have Conv1D backbone (DONE - we do!)
2. ⚠️  Inspect NPZ file structure and contents
3. ⚠️  Confirm data augmentation is working correctly
4. ⚠️  Check if we need to implement dynamic sequence sampling like the original

#### Medium Priority
5. Compare training curves with original paper
6. Verify hyperparameters match (learning rate, batch size, etc.)
7. Test if raw CSV loading would improve results

#### Low Priority  
8. Implement Gershgorin loss if not already present
9. Add CT-RNN and LSTM baselines for comparison
10. Create training data visualization tools

## Conclusion

Our implementation has the correct architecture (Conv1D + CfC), but may differ in:
1. **Data loading strategy** (packed windows vs. dynamic sampling)
2. **Training dynamics** (shuffling, sequence selection)
3. **Hyperparameters** (need to verify)

The maze navigation demo likely needs the ICRA-trained weights to work properly,
not weights from a different training setup.

---
Generated: $(date)
