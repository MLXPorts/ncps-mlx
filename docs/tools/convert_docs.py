#!/usr/bin/env python3
"""
Documentation conversion tool for NCPS.
Converts markdown files to RST and organizes them according to our Sphinx structure.
"""

import os
import re
import shutil
from pathlib import Path
from typing import List, Tuple, Optional

class DocConverter:
    def __init__(self, root_dir: str = ".."):
        self.root_dir = Path(root_dir).resolve()
        self.docs_dir = self.root_dir / "docs"
        
    def convert_all(self):
        """Convert all markdown files to RST."""
        print("Converting documentation...")
        
        # Convert markdown files
        for md_file in self.docs_dir.rglob("*.md"):
            if not any(p in str(md_file) for p in ['_build', 'venv']):
                self.convert_file(md_file)
                
    def convert_file(self, md_file: Path):
        """Convert a markdown file to RST."""
        print(f"Converting {md_file}")
        
        # Determine output path
        rel_path = md_file.relative_to(self.docs_dir)
        rst_path = self.docs_dir / rel_path.with_suffix('.rst')
        
        # Create output directory if needed
        rst_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            # Use pandoc for conversion
            os.system(f'pandoc -f markdown -t rst -o "{rst_path}" "{md_file}"')
            
            # Post-process the file
            self._post_process_rst(rst_path)
            
            print(f"Created {rst_path}")
            
        except Exception as e:
            print(f"Error converting {md_file}: {e}")
            
    def _post_process_rst(self, rst_file: Path):
        """Post-process RST file to fix common conversion issues."""
        with open(rst_file, 'r') as f:
            content = f.read()
            
        # Fix code block syntax
        content = re.sub(
            r'::?\n\n    (?=\S)',
            r'.. code-block:: python\n\n    ',
            content
        )
        
        # Fix internal links
        content = re.sub(
            r'`([^`]+)\s*<([^>]+)>`_',
            r':doc:`\1 </\2>`',
            content
        )
        
        # Fix section headers
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
                
        content = '\n'.join(fixed_lines)
        
        with open(rst_file, 'w') as f:
            f.write(content)
            
    def organize_files(self):
        """Organize documentation files into proper structure."""
        print("Organizing documentation files...")
        
        # Create necessary directories
        for dir_name in ['api', 'guides', 'examples']:
            (self.docs_dir / dir_name).mkdir(exist_ok=True)
            
        # Move files to appropriate directories
        self._move_api_docs()
        self._move_guide_docs()
        self._move_example_docs()
        
    def _move_api_docs(self):
        """Move API documentation."""
        api_dir = self.docs_dir / "api"
        patterns = ['*api*.rst', '*reference*.rst']
        self._move_matching_files(patterns, api_dir)
        
    def _move_guide_docs(self):
        """Move guide documentation."""
        guides_dir = self.docs_dir / "guides"
        patterns = ['*guide*.rst', '*tutorial*.rst', '*howto*.rst']
        self._move_matching_files(patterns, guides_dir)
        
    def _move_example_docs(self):
        """Move example documentation."""
        examples_dir = self.docs_dir / "examples"
        patterns = ['*example*.rst', '*demo*.rst']
        self._move_matching_files(patterns, examples_dir)
        
    def _move_matching_files(self, patterns: List[str], target_dir: Path):
        """Move files matching patterns to target directory."""
        for pattern in patterns:
            for file in self.docs_dir.glob(pattern):
                if file.is_file():
                    target_file = target_dir / file.name
                    print(f"Moving {file} to {target_file}")
                    shutil.move(str(file), str(target_file))
                    
def main():
    converter = DocConverter()
    converter.convert_all()
    converter.organize_files()
    
if __name__ == '__main__':
    main()