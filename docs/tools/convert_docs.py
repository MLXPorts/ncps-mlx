#!/usr/bin/env python3
"""
Documentation conversion tool for NCPS.
Converts markdown files to RST and organizes them according to our Sphinx structure.
"""

import os
import sys
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional

class DocConverter:
    def __init__(self, root_dir: str = ".."):
        self.root_dir = Path(root_dir).resolve()
        self.docs_dir = self.root_dir / "docs"
        self.architecture_dir = self.docs_dir / "architecture"
        
    def convert_file(self, md_file: Path, output_file: Optional[Path] = None) -> Path:
        """Convert a markdown file to RST."""
        if output_file is None:
            output_file = md_file.with_suffix('.rst')
            
        print(f"Converting {md_file} to {output_file}")
        
        # Use pandoc for conversion
        try:
            subprocess.run([
                "pandoc",
                str(md_file),
                "-f", "markdown",
                "-t", "rst",
                "-o", str(output_file)
            ], check=True)
            
            # Post-process the file
            self._post_process_rst(output_file)
            
            return output_file
        except subprocess.CalledProcessError as e:
            print(f"Error converting {md_file}: {e}")
            raise
            
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
        
        # Add proper headers
        if not re.match(r'^=+\n', content):
            title = rst_file.stem.replace('_', ' ').title()
            content = f"{title}\n{'=' * len(title)}\n\n{content}"
            
        with open(rst_file, 'w') as f:
            f.write(content)
            
    def organize_files(self):
        """Organize documentation files into proper structure."""
        # Create necessary directories
        for dir_name in ['abstractions', 'implementation', 'knowledge', 'design', 'research']:
            (self.architecture_dir / dir_name).mkdir(exist_ok=True)
            
        # Move files to appropriate directories
        self._move_abstraction_docs()
        self._move_implementation_docs()
        self._move_knowledge_docs()
        self._move_design_docs()
        self._move_research_docs()
        
    def _move_abstraction_docs(self):
        """Move abstraction-related documentation."""
        abstractions_dir = self.architecture_dir / "abstractions"
        patterns = ['*abstraction*.md', '*abstraction*.rst']
        self._move_matching_files(patterns, abstractions_dir)
        
    def _move_implementation_docs(self):
        """Move implementation-related documentation."""
        impl_dir = self.architecture_dir / "implementation"
        patterns = ['*implementation*.md', '*implementation*.rst', '*phase*.md', '*phase*.rst']
        self._move_matching_files(patterns, impl_dir)
        
    def _move_knowledge_docs(self):
        """Move knowledge base documentation."""
        knowledge_dir = self.architecture_dir / "knowledge"
        patterns = ['*insight*.md', '*insight*.rst', '*knowledge*.md', '*knowledge*.rst']
        self._move_matching_files(patterns, knowledge_dir)
        
    def _move_design_docs(self):
        """Move design documentation."""
        design_dir = self.architecture_dir / "design"
        patterns = ['*design*.md', '*design*.rst']
        self._move_matching_files(patterns, design_dir)
        
    def _move_research_docs(self):
        """Move research documentation."""
        research_dir = self.architecture_dir / "research"
        patterns = ['*research*.md', '*research*.rst', '*exploration*.md', '*exploration*.rst']
        self._move_matching_files(patterns, research_dir)
        
    def _move_matching_files(self, patterns: List[str], target_dir: Path):
        """Move files matching patterns to target directory."""
        for pattern in patterns:
            for file in self.architecture_dir.glob(pattern):
                if file.is_file():
                    target_file = target_dir / file.name
                    print(f"Moving {file} to {target_file}")
                    shutil.move(str(file), str(target_file))
                    
    def create_index_files(self):
        """Create index.rst files in each directory."""
        for dir_path in self.architecture_dir.glob('*/'):
            if dir_path.is_dir():
                self._create_index_file(dir_path)
                
    def _create_index_file(self, directory: Path):
        """Create an index.rst file for a directory."""
        index_file = directory / 'index.rst'
        if index_file.exists():
            return
            
        title = directory.name.title()
        content = [
            f"{title}",
            "=" * len(title),
            "",
            ".. toctree::",
            "   :maxdepth: 2",
            "   :caption: Contents:",
            ""
        ]
        
        # Add references to all rst files
        for rst_file in sorted(directory.glob('*.rst')):
            if rst_file.name != 'index.rst':
                content.append(f"   {rst_file.stem}")
                
        with open(index_file, 'w') as f:
            f.write('\n'.join(content))
            
def main():
    converter = DocConverter()
    
    # Convert markdown files
    for md_file in converter.architecture_dir.rglob('*.md'):
        if not any(p in str(md_file) for p in ['_build', 'venv']):
            converter.convert_file(md_file)
            
    # Organize files
    converter.organize_files()
    
    # Create index files
    converter.create_index_files()
    
if __name__ == '__main__':
    main()