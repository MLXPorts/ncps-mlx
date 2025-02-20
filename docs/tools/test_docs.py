#!/usr/bin/env python3
"""
Documentation testing tool for NCPS.
Tests documentation builds, checks links, and validates RST syntax.
"""

import os
import re
import sys
import ast
import subprocess
from pathlib import Path
from typing import List, Dict, Set
from concurrent.futures import ThreadPoolExecutor
import docutils.core
import docutils.parsers.rst

class DocTester:
    def __init__(self, docs_dir: str = "."):
        self.docs_dir = Path(docs_dir).resolve()
        self.build_dir = self.docs_dir / "_build"
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
    def run_all_tests(self) -> bool:
        """Run all documentation tests."""
        success = True
        
        # Clean build directory
        self._clean_build()
        
        # Run tests
        tests = [
            self.test_rst_syntax,
            self.test_build,
            self.test_links,
            self.test_code_blocks,
            self.test_cross_references
        ]
        
        for test in tests:
            try:
                if not test():
                    success = False
            except Exception as e:
                self.errors.append(f"Error in {test.__name__}: {str(e)}")
                success = False
                
        # Print report
        self._print_report()
        
        return success
        
    def test_rst_syntax(self) -> bool:
        """Test RST syntax in all documentation files."""
        print("Testing RST syntax...")
        success = True
        
        for rst_file in self.docs_dir.rglob("*.rst"):
            try:
                with open(rst_file) as f:
                    content = f.read()
                    
                # Use docutils to parse RST
                docutils.core.publish_string(
                    content,
                    writer_name='null',
                    settings_overrides={'warning_stream': None}
                )
            except Exception as e:
                self.errors.append(f"RST syntax error in {rst_file}: {str(e)}")
                success = False
                
        return success
        
    def test_build(self) -> bool:
        """Test documentation build."""
        print("Testing documentation build...")
        try:
            # First try sphinx-build
            result = subprocess.run(
                ["sphinx-build", "-W", "-b", "html", ".", "_build/html"],
                cwd=self.docs_dir,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                # Try python -m sphinx as fallback
                result = subprocess.run(
                    ["python", "-m", "sphinx", "-W", "-b", "html", ".", "_build/html"],
                    cwd=self.docs_dir,
                    capture_output=True,
                    text=True
                )
                
            if result.returncode != 0:
                self.errors.append(f"Build failed: {result.stderr}")
                return False
                
            return True
        except Exception as e:
            self.errors.append(f"Build error: {str(e)}")
            return False
            
    def test_links(self) -> bool:
        """Test all links in documentation."""
        print("Testing links...")
        success = True
        
        try:
            # First try sphinx-build
            result = subprocess.run(
                ["sphinx-build", "-b", "linkcheck", ".", "_build/linkcheck"],
                cwd=self.docs_dir,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                # Try python -m sphinx as fallback
                result = subprocess.run(
                    ["python", "-m", "sphinx", "-b", "linkcheck", ".", "_build/linkcheck"],
                    cwd=self.docs_dir,
                    capture_output=True,
                    text=True
                )
                
            if result.returncode != 0:
                self.errors.append(f"Link check failed: {result.stderr}")
                success = False
                
            # Parse output file
            output_file = self.build_dir / "linkcheck" / "output.txt"
            if output_file.exists():
                with open(output_file) as f:
                    for line in f:
                        if "[broken]" in line:
                            self.errors.append(f"Broken link: {line.strip()}")
                            success = False
                            
        except Exception as e:
            self.errors.append(f"Link check error: {str(e)}")
            success = False
            
        return success
        
    def test_code_blocks(self) -> bool:
        """Test code blocks in documentation."""
        print("Testing code blocks...")
        success = True
        
        for rst_file in self.docs_dir.rglob("*.rst"):
            try:
                with open(rst_file) as f:
                    content = f.read()
                    
                # Find all code blocks
                code_blocks = re.finditer(
                    r'.. code-block:: python\n\n((?:    .+\n)+)',
                    content,
                    re.MULTILINE
                )
                
                for block in code_blocks:
                    code = block.group(1)
                    # Remove indentation
                    code = "\n".join(line[4:] for line in code.splitlines())
                    
                    try:
                        # Parse code to check syntax
                        ast.parse(code)
                    except SyntaxError as e:
                        self.errors.append(
                            f"Syntax error in code block in {rst_file}:"
                            f" {str(e)}"
                        )
                        success = False
                        
            except Exception as e:
                self.errors.append(f"Error checking code blocks in {rst_file}: {str(e)}")
                success = False
                
        return success
        
    def test_cross_references(self) -> bool:
        """Test cross-references in documentation."""
        print("Testing cross-references...")
        success = True
        
        try:
            # First try sphinx-build
            result = subprocess.run(
                ["sphinx-build", "-n", "-W", "-b", "html", ".", "_build/html"],
                cwd=self.docs_dir,
                capture_output=True,
                text=True
            )
            if result.returncode != 0:
                # Try python -m sphinx as fallback
                result = subprocess.run(
                    ["python", "-m", "sphinx", "-n", "-W", "-b", "html", ".", "_build/html"],
                    cwd=self.docs_dir,
                    capture_output=True,
                    text=True
                )
                
            if result.returncode != 0:
                # Parse output for reference errors
                for line in result.stderr.splitlines():
                    if "undefined label:" in line or "unknown document:" in line:
                        self.errors.append(f"Cross-reference error: {line.strip()}")
                        success = False
                        
        except Exception as e:
            self.errors.append(f"Cross-reference check error: {str(e)}")
            success = False
            
        return success
        
    def _clean_build(self):
        """Clean the build directory."""
        if self.build_dir.exists():
            try:
                subprocess.run(["rm", "-rf", str(self.build_dir)])
            except Exception as e:
                self.warnings.append(f"Could not clean build directory: {str(e)}")
                
    def _print_report(self):
        """Print test results report."""
        print("\nDocumentation Test Report")
        print("=" * 30)
        
        if self.errors:
            print("\nErrors:")
            for error in self.errors:
                print(f"- {error}")
                
        if self.warnings:
            print("\nWarnings:")
            for warning in self.warnings:
                print(f"- {warning}")
                
        if not self.errors and not self.warnings:
            print("\nAll tests passed successfully!")
            
def main():
    tester = DocTester()
    success = tester.run_all_tests()
    sys.exit(0 if success else 1)
    
if __name__ == '__main__':
    main()