#!/usr/bin/env python3
"""
Fix Python operators in MLX code to use explicit MLX functions.
This prevents breaking the computation graph.
"""

import ast
import sys
from pathlib import Path
from typing import Set


class OperatorFixer(ast.NodeTransformer):
    """AST transformer to replace Python operators with MLX functions."""
    
    def __init__(self):
        self.modified = False
        self.in_subscript = False
    
    def visit_Subscript(self, node):
        """Don't transform operators inside subscripts (array indexing is OK)."""
        old_in_subscript = self.in_subscript
        self.in_subscript = True
        result = self.generic_visit(node)
        self.in_subscript = old_in_subscript
        return result
    
    def visit_BinOp(self, node):
        """Transform binary operations to MLX functions."""
        # First visit children
        node = self.generic_visit(node)
        
        # Skip if we're in a subscript
        if self.in_subscript:
            return node
        
        # Check if either operand involves mx.array (heuristic: check for Name or Call nodes)
        # We'll be conservative and transform all numeric operations
        
        # Map operators to MLX functions
        op_map = {
            ast.Add: 'mx.add',
            ast.Sub: 'mx.subtract',
            ast.Mult: 'mx.multiply',
            ast.Div: 'mx.divide',
            ast.FloorDiv: 'mx.floor_divide',
            ast.Mod: 'mx.remainder',
            ast.Pow: 'mx.power',
            ast.MatMult: 'mx.matmul',
        }
        
        if type(node.op) not in op_map:
            return node
        
        # For unary minus on left side, handle specially
        if isinstance(node.left, ast.UnaryOp) and isinstance(node.left.op, ast.USub):
            # -a * b => mx.multiply(mx.negative(a), b)
            pass  # Will be handled by visit_UnaryOp
        
        func_name = op_map[type(node.op)]
        self.modified = True
        
        # Create function call: mx.add(left, right)
        func_parts = func_name.split('.')
        if len(func_parts) == 2:
            func_node = ast.Attribute(
                value=ast.Name(id=func_parts[0], ctx=ast.Load()),
                attr=func_parts[1],
                ctx=ast.Load()
            )
        else:
            func_node = ast.Name(id=func_name, ctx=ast.Load())
        
        return ast.Call(
            func=func_node,
            args=[node.left, node.right],
            keywords=[]
        )
    
    def visit_UnaryOp(self, node):
        """Transform unary operations to MLX functions."""
        # First visit children
        node = self.generic_visit(node)
        
        # Skip if we're in a subscript
        if self.in_subscript:
            return node
        
        # Map unary operators
        if isinstance(node.op, ast.USub):
            # -x => mx.negative(x)
            self.modified = True
            return ast.Call(
                func=ast.Attribute(
                    value=ast.Name(id='mx', ctx=ast.Load()),
                    attr='negative',
                    ctx=ast.Load()
                ),
                args=[node.operand],
                keywords=[]
            )
        
        return node


def fix_file(filepath: Path) -> bool:
    """Fix operators in a single file. Returns True if modified."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            source = f.read()
        
        # Parse the AST
        tree = ast.parse(source, filename=str(filepath))
        
        # Transform it
        fixer = OperatorFixer()
        new_tree = fixer.visit(tree)
        
        if not fixer.modified:
            return False
        
        # Convert back to source
        ast.fix_missing_locations(new_tree)
        import astor
        new_source = astor.to_source(new_tree)
        
        # Write back
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(new_source)
        
        print(f"Fixed: {filepath}")
        return True
        
    except Exception as e:
        print(f"Error processing {filepath}: {e}", file=sys.stderr)
        return False


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python fix_operators.py <file_or_directory>")
        sys.exit(1)
    
    path = Path(sys.argv[1])
    
    if path.is_file():
        files = [path]
    elif path.is_dir():
        files = list(path.rglob('*.py'))
    else:
        print(f"Error: {path} is not a file or directory")
        sys.exit(1)
    
    modified_count = 0
    for filepath in files:
        if fix_file(filepath):
            modified_count += 1
    
    print(f"\nModified {modified_count} files")


if __name__ == '__main__':
    main()
