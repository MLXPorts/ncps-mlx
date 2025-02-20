#!/usr/bin/env python3
"""
Documentation conversion tool for NCPS.
Converts markdown files to RST and organizes them according to our Sphinx structure.
"""

import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Tuple, Optional

class DocConverter:
    def __init__(self, root_dir: str = ".."):
        self.root_dir = Path(root_dir).resolve()
        self.docs_dir = self.root_dir / "docs"
        
    def check_dependencies(self):
        """Check if required dependencies are installed."""
        try:
            subprocess.run(['pandoc', '--version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
            raise RuntimeError("pandoc is not installed. Please install it first.")
            
    def convert_all(self):
        """Convert all markdown files to RST."""
        print("Converting documentation...")
        self.check_dependencies()
        
        # Convert markdown files
        for md_file in self.docs_dir.rglob("*.md"):
            if not any(p in str(md_file) for p in ['_build', 'venv', 'node_modules']):
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
            # Use pandoc for conversion with extended options
            cmd = [
                'pandoc',
                '-f', 'markdown',
                '-t', 'rst',
                '--wrap=none',  # Prevent line wrapping
                '--columns=100000',  # Prevent table reformatting
                '-o', str(rst_path),
                str(md_file)
            ]
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            
            # Post-process the file
            self._post_process_rst(rst_path)
            
            print(f"Created {rst_path}")
            
        except subprocess.CalledProcessError as e:
            print(f"Error converting {md_file}: {e.stderr}")
        except Exception as e:
            print(f"Error converting {md_file}: {e}")
            
    def _post_process_rst(self, rst_file: Path):
        """Post-process RST file to fix common conversion issues."""
        with open(rst_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Fix code block syntax
        def fix_code_block(match):
            block = match.group(0)
            lines = block.split('\n')
            
            # Detect language based on content and file extension
            code_content = '\n'.join(lines[1:])
            language = self._detect_language(code_content)
            
            # Format the code block with proper directive
            fixed_lines = [f'.. code-block:: {language}\n']
            for line in lines[1:]:  # Skip the :: line
                if line.strip():
                    # Ensure consistent indentation
                    fixed_lines.append('    ' + line.lstrip())
                else:
                    fixed_lines.append('')
            return '\n'.join(fixed_lines)
            
        # Find and fix code blocks
        content = re.sub(
            r'::?\n\n(?:[ ]{4}[^\n]*\n)+',
            fix_code_block,
            content,
            flags=re.MULTILINE
        )
        
        # Fix internal links
        content = re.sub(
            r'`([^`]+)\s*<([^>]+)>`_',
            lambda m: self._fix_link(m.group(1), m.group(2)),
            content
        )
        
        # Fix image references
        content = re.sub(
            r'!\[(.*?)\]\((.*?)\)',
            lambda m: self._fix_image(m.group(1), m.group(2)),
            content
        )
        
        # Fix section headers
        lines = content.split('\n')
        fixed_lines = []
        header_levels = ['=', '-', '~', '^', '"']
        current_level = 0
        
        for i, line in enumerate(lines):
            # Skip empty lines
            if not line.strip():
                fixed_lines.append(line)
                continue
                
            # Check if this line might be a header
            if i < len(lines) - 1 and lines[i+1].strip() and \
               all(c in '=-~^"' for c in lines[i+1].strip()):
                # Remove any leading/trailing whitespace from header
                line = line.strip()
                # Determine header level
                current_marker = lines[i+1].strip()[0]
                level = header_levels.index(current_marker)
                # Add header with proper underlining
                fixed_lines.append(line)
                fixed_lines.append(header_levels[level] * len(line))
                # Skip the original underlining
                i += 1
            else:
                fixed_lines.append(line)
                
        content = '\n'.join(fixed_lines)
        
        # Fix blank lines around directives
        content = re.sub(
            r'(\n\.\. [^\n]+)(\n[^\n])',
            r'\1\n\2',
            content
        )
        
        with open(rst_file, 'w', encoding='utf-8') as f:
            f.write(content)
            
    def _detect_language(self, code_content: str) -> str:
        """Detect the programming language of a code block."""
        # Common language patterns
        patterns = {
            'python': r'\b(def|class|import|from|print|return|if|for|while)\b',
            'javascript': r'\b(function|const|let|var|=>|require|import|export)\b',
            'bash': r'\b(#!/bin/bash|apt-get|yum|brew|chmod|mkdir|cd|ls|grep)\b',
            'json': r'^\s*[{\[](.|[\r\n])*[}\]]\s*$',
            'html': r'<[^>]+>',
            'css': r'[.#][^{]+{[^}]*}',
            'yaml': r'^\s*[^:]+:\s*[^:]+$'
        }
        
        for lang, pattern in patterns.items():
            if re.search(pattern, code_content, re.MULTILINE):
                return lang
        return 'text'  # Default to plain text
        
    def _fix_link(self, text: str, target: str) -> str:
        """Fix internal and external links."""
        if target.startswith(('http://', 'https://', 'ftp://')):
            return f'`{text} <{target}>`_'
        elif target.endswith(('.rst', '.md')):
            # Convert to document reference
            clean_target = target.replace('.md', '').replace('.rst', '')
            return f':doc:`{text} </{clean_target}>`'
        else:
            # Assume it's a reference to a section
            return f':ref:`{text} <{target}>`'
            
    def _fix_image(self, alt_text: str, image_path: str) -> str:
        """Convert markdown image to RST image directive."""
        # Handle both relative and absolute paths
        if image_path.startswith(('http://', 'https://')):
            return f'\n.. image:: {image_path}\n   :alt: {alt_text}\n'
        else:
            # Convert relative path to be relative to docs directory
            rel_path = os.path.relpath(image_path, str(self.docs_dir))
            return f'\n.. image:: {rel_path}\n   :alt: {alt_text}\n'
            
    def organize_files(self):
        """Organize documentation files into proper structure."""
        print("Organizing documentation files...")
        
        # Create necessary directories
        for dir_name in ['api', 'guides', 'examples']:
            (self.docs_dir / dir_name).mkdir(exist_ok=True)
            
        # First convert all files, then organize them
        self.convert_all()
        
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
                if file.is_file() and target_dir not in file.parents:
                    target_file = target_dir / file.name
                    print(f"Moving {file} to {target_file}")
                    shutil.move(str(file), str(target_file))
                    
def main():
    converter = DocConverter()
    converter.organize_files()
    
if __name__ == '__main__':
    main()