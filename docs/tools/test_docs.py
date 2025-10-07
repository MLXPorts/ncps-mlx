#!/usr/bin/env python3
"""
Documentation testing tool for NCPS.
Tests documentation builds, checks links, and validates RST syntax.
"""

import os
import re
import sys
import ast
import json
import logging
import subprocess
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Set, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import docutils.core
import docutils.parsers.rst

class DocTester:
    def __init__(self, docs_dir: str = "."):
        self.docs_dir = Path(docs_dir).resolve()
        self.build_dir = self.docs_dir / "_build"
        self.logs_dir = self.docs_dir / "logs"
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
        # Create logs directory if it doesn't exist
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Language-specific syntax checkers
        self.syntax_checkers = {
            'python': self._check_python_syntax,
            'javascript': self._check_javascript_syntax,
            'typescript': self._check_typescript_syntax,
            'json': self._check_json_syntax,
            'yaml': self._check_yaml_syntax,
            'bash': self._check_bash_syntax
        }
        
    def setup_logging(self):
        """Setup logging configuration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = logging.getLogger("doc_tester")
        self.logger.setLevel(logging.DEBUG)
        
        # Create a general log file
        general_handler = logging.FileHandler(self.logs_dir / f"doc_test_{timestamp}.log")
        general_handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        general_handler.setFormatter(formatter)
        self.logger.addHandler(general_handler)
        
    def get_file_logger(self, file_path: Path) -> logging.Logger:
        """Create a logger for a specific file."""
        file_name = file_path.stem
        logger = logging.getLogger(f"doc_tester.{file_name}")
        
        # Create file-specific log
        log_file = self.logs_dir / f"{file_name}.log"
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
        
    def run_all_tests(self) -> bool:
        """Run all documentation tests in parallel where possible."""
        self.logger.info("Starting documentation tests")
        success = True
        
        # Clean build directory
        self._clean_build()
        
        # Tests that can run in parallel
        parallel_tests = [
            self.test_rst_syntax,
            self.test_code_blocks,
            self.test_images,
            self.test_section_hierarchy
        ]
        
        # Run parallel tests
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(test): test.__name__ for test in parallel_tests}
            for future in as_completed(futures):
                test_name = futures[future]
                try:
                    if not future.result():
                        success = False
                        self.logger.error(f"{test_name} failed")
                except Exception as e:
                    success = False
                    self.errors.append(f"Error in {test_name}: {str(e)}")
                    self.logger.error(f"Error in {test_name}: {str(e)}", exc_info=True)
        
        # Sequential tests that depend on build
        sequential_tests = [
            self.test_build,
            self.test_links,
            self.test_cross_references
        ]
        
        for test in sequential_tests:
            try:
                self.logger.info(f"Running {test.__name__}")
                if not test():
                    success = False
            except Exception as e:
                self.errors.append(f"Error in {test.__name__}: {str(e)}")
                self.logger.error(f"Error in {test.__name__}: {str(e)}", exc_info=True)
                success = False
                
        # Print report
        self._print_report()
        
        return success
        
    def test_rst_syntax(self) -> bool:
        """Test RST syntax in all documentation files."""
        print("Testing RST syntax...")
        success = True
        
        for rst_file in self.docs_dir.rglob("*.rst"):
            file_logger = self.get_file_logger(rst_file)
            file_logger.info(f"Testing RST syntax in {rst_file}")
            
            try:
                with open(rst_file, encoding='utf-8') as f:
                    content = f.read()
                    
                # Use docutils to parse RST
                settings = {
                    'warning_stream': None,
                    'report_level': 2  # Report warnings and errors
                }
                docutils.core.publish_string(
                    content,
                    writer_name='null',
                    settings_overrides=settings
                )
                file_logger.debug(f"Successfully parsed {rst_file}")
            except Exception as e:
                self.errors.append(f"RST syntax error in {rst_file}: {str(e)}")
                file_logger.error(f"RST syntax error: {str(e)}", exc_info=True)
                success = False
                
        return success
        
    def test_build(self) -> bool:
        """Test documentation build."""
        print("Testing documentation build...")
        self.logger.info("Testing documentation build")
        try:
            # First try sphinx-build
            result = subprocess.run(
                ["sphinx-build", "-W", "-b", "html", ".", "_build/html"],
                cwd=self.docs_dir,
                capture_output=True,
                text=True
            )
            
            self.logger.debug(f"sphinx-build stdout:\n{result.stdout}")
            self.logger.debug(f"sphinx-build stderr:\n{result.stderr}")
            
            if result.returncode != 0:
                # Try python -m sphinx as fallback
                self.logger.info("sphinx-build failed, trying python -m sphinx")
                result = subprocess.run(
                    ["python", "-m", "sphinx", "-W", "-b", "html", ".", "_build/html"],
                    cwd=self.docs_dir,
                    capture_output=True,
                    text=True
                )
                self.logger.debug(f"python -m sphinx stdout:\n{result.stdout}")
                self.logger.debug(f"python -m sphinx stderr:\n{result.stderr}")
                
            if result.returncode != 0:
                self.errors.append(f"Build failed: {result.stderr}")
                self.logger.error(f"Build failed: {result.stderr}")
                return False
                
            return True
        except Exception as e:
            self.errors.append(f"Build error: {str(e)}")
            self.logger.error("Build error", exc_info=True)
            return False
            
    def test_links(self) -> bool:
        """Test all links in documentation."""
        print("Testing links...")
        self.logger.info("Testing links")
        success = True
        
        try:
            # First try sphinx-build
            result = subprocess.run(
                ["sphinx-build", "-b", "linkcheck", ".", "_build/linkcheck"],
                cwd=self.docs_dir,
                capture_output=True,
                text=True
            )
            
            self.logger.debug(f"linkcheck stdout:\n{result.stdout}")
            self.logger.debug(f"linkcheck stderr:\n{result.stderr}")
            
            if result.returncode != 0:
                # Try python -m sphinx as fallback
                self.logger.info("sphinx-build linkcheck failed, trying python -m sphinx")
                result = subprocess.run(
                    ["python", "-m", "sphinx", "-b", "linkcheck", ".", "_build/linkcheck"],
                    cwd=self.docs_dir,
                    capture_output=True,
                    text=True
                )
                
            if result.returncode != 0:
                self.errors.append(f"Link check failed: {result.stderr}")
                self.logger.error(f"Link check failed: {result.stderr}")
                success = False
                
            # Parse output file
            output_file = self.build_dir / "linkcheck" / "output.txt"
            if output_file.exists():
                with open(output_file) as f:
                    for line in f:
                        if "[broken]" in line:
                            self.errors.append(f"Broken link: {line.strip()}")
                            self.logger.error(f"Broken link: {line.strip()}")
                            success = False
                            
        except Exception as e:
            self.errors.append(f"Link check error: {str(e)}")
            self.logger.error("Link check error", exc_info=True)
            success = False
            
        return success
        
    def test_code_blocks(self) -> bool:
        """Test code blocks in documentation."""
        print("Testing code blocks...")
        success = True
        
        for rst_file in self.docs_dir.rglob("*.rst"):
            file_logger = self.get_file_logger(rst_file)
            file_logger.info(f"Testing code blocks in {rst_file}")
            
            try:
                with open(rst_file, encoding='utf-8') as f:
                    content = f.read()
                    
                # Find all code blocks with their language
                code_blocks = re.finditer(
                    r'.. code-block:: (\w+)\n\n((?:    .+\n)+)',
                    content,
                    re.MULTILINE
                )
                
                for block in code_blocks:
                    language = block.group(1)
                    code = block.group(2)
                    # Remove indentation
                    code = "\n".join(line[4:] for line in code.splitlines())
                    
                    # Check syntax based on language
                    checker = self.syntax_checkers.get(language.lower())
                    if checker:
                        try:
                            checker(code)
                            file_logger.debug(f"Successfully parsed {language} code block")
                        except Exception as e:
                            self.errors.append(
                                f"Syntax error in {language} code block in {rst_file}:"
                                f" {str(e)}"
                            )
                            file_logger.error(f"Syntax error in {language} code block: {str(e)}")
                            success = False
                            
            except Exception as e:
                self.errors.append(f"Error checking code blocks in {rst_file}: {str(e)}")
                file_logger.error(f"Error checking code blocks: {str(e)}", exc_info=True)
                success = False
                
        return success
        
    def test_cross_references(self) -> bool:
        """Test cross-references in documentation."""
        print("Testing cross-references...")
        self.logger.info("Testing cross-references")
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
                # Parse output for reference errors
                for line in result.stderr.splitlines():
                    if "undefined label:" in line or "unknown document:" in line:
                        self.errors.append(f"Cross-reference error: {line.strip()}")
                        self.logger.error(f"Cross-reference error: {line.strip()}")
                        success = False
                        
        except Exception as e:
            self.errors.append(f"Cross-reference check error: {str(e)}")
            self.logger.error("Cross-reference check error", exc_info=True)
            success = False
            
        return success
        
    def test_images(self) -> bool:
        """Test image references in documentation."""
        print("Testing image references...")
        success = True
        
        for rst_file in self.docs_dir.rglob("*.rst"):
            file_logger = self.get_file_logger(rst_file)
            file_logger.info(f"Testing image references in {rst_file}")
            
            try:
                with open(rst_file, encoding='utf-8') as f:
                    content = f.read()
                    
                # Find all image directives
                image_refs = re.finditer(
                    r'.. image:: ([^\n]+)',
                    content
                )
                
                for ref in image_refs:
                    image_path = ref.group(1).strip()
                    if image_path.startswith(('http://', 'https://')):
                        continue  # Skip external images
                        
                    # Check if image exists
                    full_path = (rst_file.parent / image_path).resolve()
                    if not full_path.exists():
                        self.errors.append(
                            f"Missing image in {rst_file}: {image_path}"
                        )
                        file_logger.error(f"Missing image: {image_path}")
                        success = False
                        
            except Exception as e:
                self.errors.append(f"Error checking images in {rst_file}: {str(e)}")
                file_logger.error(f"Error checking images: {str(e)}", exc_info=True)
                success = False
                
        return success
        
    def test_section_hierarchy(self) -> bool:
        """Test section header hierarchy in documentation."""
        print("Testing section hierarchy...")
        success = True
        
        for rst_file in self.docs_dir.rglob("*.rst"):
            file_logger = self.get_file_logger(rst_file)
            file_logger.info(f"Testing section hierarchy in {rst_file}")
            
            try:
                with open(rst_file, encoding='utf-8') as f:
                    content = f.read()
                    
                # Track section levels
                current_level = 0
                header_chars = ['=', '-', '~', '^', '"']
                last_header = None
                
                lines = content.split('\n')
                for i in range(len(lines) - 1):
                    if i > 0 and set(lines[i]) <= set('=-~^"'):
                        header = lines[i-1].strip()
                        level_char = lines[i][0]
                        
                        if not header:
                            continue
                            
                        try:
                            level = header_chars.index(level_char)
                        except ValueError:
                            self.warnings.append(
                                f"Unknown section character in {rst_file}: {level_char}"
                            )
                            continue
                            
                        # Check level hierarchy
                        if last_header is None:
                            if level != 0:
                                self.errors.append(
                                    f"First header in {rst_file} must use '=' underline"
                                )
                                success = False
                        elif level > current_level + 1:
                            self.errors.append(
                                f"Invalid header hierarchy in {rst_file}: "
                                f"'{header}' (skipped level)"
                            )
                            success = False
                            
                        current_level = level
                        last_header = header
                        
            except Exception as e:
                self.errors.append(f"Error checking section hierarchy in {rst_file}: {str(e)}")
                file_logger.error(f"Error checking section hierarchy: {str(e)}", exc_info=True)
                success = False
                
        return success
        
    def _check_python_syntax(self, code: str):
        """Check Python code syntax."""
        ast.parse(code)
        
    def _check_javascript_syntax(self, code: str):
        """Check JavaScript code syntax."""
        result = subprocess.run(
            ['node', '--check'],
            input=code,
            text=True,
            capture_output=True
        )
        if result.returncode != 0:
            raise SyntaxError(result.stderr)
            
    def _check_typescript_syntax(self, code: str):
        """Check TypeScript code syntax."""
        result = subprocess.run(
            ['tsc', '--noEmit'],
            input=code,
            text=True,
            capture_output=True
        )
        if result.returncode != 0:
            raise SyntaxError(result.stderr)
            
    def _check_json_syntax(self, code: str):
        """Check JSON syntax."""
        json.loads(code)
        
    def _check_yaml_syntax(self, code: str):
        """Check YAML syntax."""
        import yaml
        yaml.safe_load(code)
        
    def _check_bash_syntax(self, code: str):
        """Check bash script syntax."""
        result = subprocess.run(
            ['bash', '-n'],
            input=code,
            text=True,
            capture_output=True
        )
        if result.returncode != 0:
            raise SyntaxError(result.stderr)
            
    def _clean_build(self):
        """Clean the build directory."""
        self.logger.info("Cleaning build directory")
        if self.build_dir.exists():
            try:
                subprocess.run(["rm", "-rf", str(self.build_dir)])
                self.logger.debug("Successfully cleaned build directory")
            except Exception as e:
                self.warnings.append(f"Could not clean build directory: {str(e)}")
                self.logger.warning(f"Could not clean build directory: {str(e)}", exc_info=True)
                
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