"""Test importing all modules."""

def test_imports():
    """Test that all modules can be imported."""
    print("Testing imports...")
    
    print("\nImporting MLX modules...")
    from ncps.mlx import CfC as MLXCfC
    print("MLX CfC imported successfully")
    
    print("\nImporting TensorFlow modules...")
    from ncps.tf import CfC as TFCfC
    print("TensorFlow CfC imported successfully")
    
    print("\nImporting PaddlePaddle modules...")
    from ncps.paddle import LTCCell
    print("PaddlePaddle LTCCell imported successfully")
    
    print("\nAll imports successful!")

if __name__ == "__main__":
    test_imports()