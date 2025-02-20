#!/usr/bin/env python3
"""
RST syntax fixer for NCPS documentation.
Fixes common RST syntax issues and code block indentation.
"""

import os
import re
from pathlib import Path
from typing import List, Tuple

class RSTFixer:
    def __init__(self, docs_dir: str = "."):
        self.docs_dir = Path(docs_dir).resolve()
        
    def fix_all_files(self):
        """Fix all RST files in the documentation."""
        print("Fixing RST syntax in documentation files...")
        
        for rst_file in self.docs_dir.rglob("*.rst"):
            if "_build" not in str(rst_file):
                self.fix_file(rst_file)
                
    def fix_file(self, file_path: Path):
        """Fix RST syntax in a single file."""
        print(f"Fixing {file_path}")
        
        with open(file_path, 'r') as f:
            content = f.read()
            
        # Fix code blocks
        content = self._fix_code_blocks(content)
        
        # Fix section headers
        content = self._fix_section_headers(content)
        
        # Fix cross-references
        content = self._fix_cross_references(content)
        
        # Fix list formatting
        content = self._fix_lists(content)
        
        with open(file_path, 'w') as f:
            f.write(content)
            
    def _fix_code_blocks(self, content: str) -> str:
        """Fix code block syntax and indentation."""
        # Find all code blocks
        code_block_pattern = r'(::?\n\n)( +)(.+?)(?=\n\n|\Z)'
        
        def fix_block(match):
            directive = match.group(1)
            indent = match.group(2)
            code = match.group(3)
            
            # Determine if this is a Python code block
            if re.search(r'(import|def|class|print|return)', code):
                directive = ".. code-block:: python\n\n"
                
            # Fix indentation (ensure 4 spaces)
            lines = code.split('\n')
            fixed_lines = []
            for line in lines:
                if line.strip():
                    fixed_lines.append("    " + line.lstrip())
                else:
                    fixed_lines.append("")
                    
            return f"{directive}{os.linesep.join(fixed_lines)}"
            
        content = re.sub(
            code_block_pattern,
            fix_block,
            content,
            flags=re.MULTILINE | re.DOTALL
        )
        
        return content
        
    def _fix_section_headers(self, content: str) -> str:
        """Fix section header underlining."""
        lines = content.split('\n')
        fixed_lines = []
        
        for i, line in enumerate(lines):
            fixed_lines.append(line)
            
            # Check if this line might be a header
            if i < len(lines) - 1 and lines[i+1].strip() and \
               all(c in '=-~' for c in lines[i+1].strip()):
                # Fix underlining length
                next_line = lines[i+1].strip()[0] * len(line)
                fixed_lines.append(next_line)
                # Skip the original underlining
                i += 1
            
        return '\n'.join(fixed_lines)
        
    def _fix_cross_references(self, content: str) -> str:
        """Fix cross-reference syntax."""
        # Fix :ref: syntax
        content = re.sub(
            r':ref:`([^`]+)`',
            lambda m: f':ref:`{m.group(1).lower().replace(" ", "-")}`',
            content
        )
        
        # Fix :doc: syntax
        content = re.sub(
            r':doc:`([^`]+)`',
            lambda m: f':doc:`{m.group(1).lstrip("/")}` ',
            content
        )
        
        return content
        
    def _fix_lists(self, content: str) -> str:
        """Fix list formatting."""
        lines = content.split('\n')
        fixed_lines = []
        in_list = False
        
        for i, line in enumerate(lines):
            # Check for list items
            if re.match(r'^[-*]\s', line.lstrip()):
                if not in_list:
                    # Add blank line before list
                    if fixed_lines and fixed_lines[-1].strip():
                        fixed_lines.append('')
                    in_list = True
            elif line.strip() and in_list:
                # Add blank line after list
                if fixed_lines[-1].strip():
                    fixed_lines.append('')
                in_list = False
                
            fixed_lines.append(line)
            
        return '\n'.join(fixed_lines)
        
def main():
    fixer = RSTFixer()
    fixer.fix_all_files()
    
if __name__ == '__main__':
    main()