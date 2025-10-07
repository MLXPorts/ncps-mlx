#!/usr/bin/env python3
"""
RST syntax fixer for NCPS documentation.
Fixes common RST syntax issues and code block indentation.
"""

import os
import re
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

try:
    import yaml
    YAML_AVAILABLE = True
except ImportError:
    YAML_AVAILABLE = False

try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False

class RSTFixer:
    def __init__(self, docs_dir: str = "."):
        self.docs_dir = Path(docs_dir).resolve()
        self.logs_dir = self.docs_dir / "logs"
        self.errors: List[str] = []
        self.warnings: List[str] = []
        
        # Create logs directory if needed
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
        
        # Language-specific formatters
        self.code_formatters = {
            'python': self._format_python_code,
            'javascript': self._format_javascript_code,
            'typescript': self._format_typescript_code,
            'json': self._format_json_code,
            'yaml': self._format_yaml_code,
            'bash': self._format_bash_code,
            'html': self._format_html_code,
            'css': self._format_css_code,
            'sql': self._format_sql_code
        }
        
    def setup_logging(self) -> None:
        """Setup logging configuration."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.logger = logging.getLogger("rst_fixer")
        self.logger.setLevel(logging.DEBUG)
        
        # Create log file
        log_file = self.logs_dir / f"rst_fix_{timestamp}.log"
        handler = logging.FileHandler(log_file)
        handler.setLevel(logging.DEBUG)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        
    def fix_all_files(self) -> bool:
        """Fix all RST files in the documentation."""
        print("Fixing RST syntax in documentation files...")
        self.logger.info("Starting RST syntax fixes")
        
        for rst_file in self.docs_dir.rglob("*.rst"):
            if not any(p in str(rst_file) for p in ['_build', '.venv', 'node_modules']):
                try:
                    self.fix_file(rst_file)
                except Exception as e:
                    self.errors.append(f"Error fixing {rst_file}: {str(e)}")
                    self.logger.error(f"Error fixing {rst_file}", exc_info=True)
                    
        self._print_report()
        return len(self.errors) == 0
        
    def fix_file(self, file_path: Path) -> None:
        """Fix RST syntax in a single file."""
        print(f"Fixing {file_path}")
        self.logger.info(f"Fixing {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                
            # Apply fixes in order
            fixes = [
                ('Section Titles', self._fix_section_titles),
                ('Code Blocks', self._fix_code_blocks),
                ('Cross References', self._fix_cross_references),
                ('Lists', self._fix_lists),
                ('Tables', self._fix_tables),
                ('Images', self._fix_images),
                ('Role Directives', self._fix_role_directives),
                ('TOC Tree', self._fix_toctree_directives),
                ('Spacing', self._fix_spacing),
                ('Indentation', self._fix_indentation)
            ]
            
            for fix_name, fix_func in fixes:
                try:
                    content = fix_func(content)
                    self.logger.debug(f"Applied {fix_name} fix")
                except Exception as e:
                    self.warnings.append(f"Warning in {fix_name} fix for {file_path}: {str(e)}")
                    self.logger.warning(f"Warning in {fix_name} fix", exc_info=True)
                    
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
                
        except Exception as e:
            raise Exception(f"Failed to fix {file_path}: {str(e)}")
            
    def _fix_code_blocks(self, content: str) -> str:
        """Fix code block syntax and indentation for multiple languages."""
        def fix_code_block(match: re.Match) -> str:
            directive = match.group(1)
            language = match.group(2)
            code = match.group(3)
            
            # Get language-specific formatter
            formatter = self.code_formatters.get(language.lower(), self._format_generic_code)
            
            try:
                fixed_code = formatter(code)
            except Exception as e:
                self.warnings.append(f"Warning formatting {language} code: {str(e)}")
                fixed_code = self._format_generic_code(code)
                
            return f"{directive}{language}\n\n{fixed_code}"
            
        # Find and fix all code blocks
        return re.sub(
            r'(\.\.[ ]code-block::[ ])(\w+)\n\n((?:[ ]*[^\n]*\n)*)',
            fix_code_block,
            content,
            flags=re.MULTILINE | re.DOTALL
        )
        
    def _format_python_code(self, code: str) -> str:
        """Format Python code with proper indentation."""
        try:
            import black
            mode = black.FileMode()
            formatted = black.format_str(code, mode=mode)
        except ImportError:
            # Fallback to basic formatting if black is not available
            formatted = self._basic_code_format(code)
            
        # Ensure proper indentation
        lines = formatted.split('\n')
        return '\n'.join('    ' + line if line.strip() else '' for line in lines)
        
    def _format_javascript_code(self, code: str) -> str:
        """Format JavaScript code."""
        try:
            import jsbeautifier
            opts = jsbeautifier.default_options()
            formatted = jsbeautifier.beautify(code, opts)
        except ImportError:
            formatted = self._basic_code_format(code)
            
        return self._indent_code(formatted)
        
    def _format_typescript_code(self, code: str) -> str:
        """Format TypeScript code."""
        return self._format_javascript_code(code)  # Use same formatting as JavaScript
        
    def _format_json_code(self, code: str) -> str:
        """Format JSON code."""
        try:
            parsed = json.loads(code)
            formatted = json.dumps(parsed, indent=4)
        except json.JSONDecodeError:
            formatted = self._basic_code_format(code)
            
        return self._indent_code(formatted)
        
    def _format_yaml_code(self, code: str) -> str:
        """Format YAML code."""
        if YAML_AVAILABLE:
            try:
                parsed = yaml.safe_load(code)
                formatted = yaml.dump(parsed, default_flow_style=False)
            except yaml.YAMLError:
                formatted = self._basic_code_format(code)
        else:
            formatted = self._basic_code_format(code)
            
        return self._indent_code(formatted)
        
    def _format_bash_code(self, code: str) -> str:
        """Format bash script."""
        # Basic bash formatting (no external dependency)
        return self._indent_code(self._basic_code_format(code))
        
    def _format_html_code(self, code: str) -> str:
        """Format HTML code."""
        if BS4_AVAILABLE:
            try:
                formatted = BeautifulSoup(code, 'html.parser').prettify()
            except Exception:
                formatted = self._basic_code_format(code)
        else:
            formatted = self._basic_code_format(code)
            
        return self._indent_code(formatted)
        
    def _format_css_code(self, code: str) -> str:
        """Format CSS code."""
        try:
            import cssbeautifier
            formatted = cssbeautifier.beautify(code)
        except ImportError:
            formatted = self._basic_code_format(code)
            
        return self._indent_code(formatted)
        
    def _format_sql_code(self, code: str) -> str:
        """Format SQL code."""
        try:
            import sqlparse
            formatted = sqlparse.format(
                code,
                reindent=True,
                keyword_case='upper'
            )
        except ImportError:
            formatted = self._basic_code_format(code)
            
        return self._indent_code(formatted)
        
    def _format_generic_code(self, code: str) -> str:
        """Basic code formatting for unknown languages."""
        return self._indent_code(self._basic_code_format(code))
        
    def _basic_code_format(self, code: str) -> str:
        """Basic code formatting with consistent indentation."""
        lines = code.split('\n')
        formatted_lines = []
        indent_level = 0
        
        for line in lines:
            stripped = line.strip()
            if not stripped:
                formatted_lines.append('')
                continue
                
            # Adjust indent level based on brackets
            if any(stripped.endswith(c) for c in '({['):
                formatted_lines.append('    ' * indent_level + stripped)
                indent_level += 1
            elif any(stripped.startswith(c) for c in ')}]'):
                indent_level = max(0, indent_level - 1)
                formatted_lines.append('    ' * indent_level + stripped)
            else:
                formatted_lines.append('    ' * indent_level + stripped)
                
        return '\n'.join(formatted_lines)
        
    def _indent_code(self, code: str) -> str:
        """Add proper RST indentation to code block."""
        return '\n'.join('    ' + line if line.strip() else '' for line in code.split('\n'))
        
    def _fix_section_titles(self, content: str) -> str:
        """Fix section title formatting and hierarchy."""
        lines = content.split('\n')
        fixed_lines = []
        header_levels = ['=', '-', '~', '^', '"']
        current_level = -1
        
        i = 0
        while i < len(lines):
            line = lines[i].rstrip()
            
            # Skip empty lines
            if not line:
                fixed_lines.append(line)
                i += 1
                continue
                
            # Check if this line is a title
            if i < len(lines) - 1:
                next_line = lines[i + 1].rstrip()
                if next_line and all(c == next_line[0] for c in next_line):
                    # This is a title with underlining
                    title = line.strip()
                    
                    # Determine appropriate level
                    marker = next_line[0]
                    try:
                        level = header_levels.index(marker)
                    except ValueError:
                        level = len(header_levels) - 1
                        
                    # Ensure proper hierarchy
                    if current_level == -1:
                        level = 0
                    elif level > current_level + 1:
                        level = current_level + 1
                        
                    current_level = level
                    marker = header_levels[level]
                    
                    # Add title with proper underlining
                    fixed_lines.append(title)
                    fixed_lines.append(marker * len(title))
                    i += 2
                    continue
                    
            fixed_lines.append(line)
            i += 1
            
        return '\n'.join(fixed_lines)
        
    def _fix_cross_references(self, content: str) -> str:
        """Fix cross-reference syntax."""
        # Fix :ref: syntax
        content = re.sub(
            r':ref:`([^`<>]+)\s*(?:<([^>]+)>)?`',
            lambda m: f':ref:`{m.group(1)} <{m.group(2) or m.group(1).lower().replace(" ", "-")}>`',
            content
        )
        
        # Fix :doc: syntax
        content = re.sub(
            r':doc:`([^`<>]+)\s*(?:<([^>]+)>)?`',
            lambda m: f':doc:`{m.group(1)} <{m.group(2) or m.group(1).lstrip("/")}>`',
            content
        )
        
        return content
        
    def _fix_lists(self, content: str) -> str:
        """Fix list formatting and spacing."""
        lines = content.split('\n')
        fixed_lines = []
        in_list = False
        list_indent = 0
        
        for i, line in enumerate(lines):
            stripped = line.lstrip()
            indent = len(line) - len(stripped)
            
            # Check for list items
            list_match = re.match(r'^[-*]\s|^\d+\.\s', stripped)
            if list_match:
                if not in_list:
                    # Add blank line before list
                    if fixed_lines and fixed_lines[-1].strip():
                        fixed_lines.append('')
                    in_list = True
                    list_indent = indent
                # Ensure consistent list item indentation
                fixed_lines.append(' ' * list_indent + stripped)
            elif stripped and in_list:
                if indent > list_indent:
                    # This is a continuation of a list item
                    fixed_lines.append(' ' * (list_indent + 2) + stripped)
                else:
                    # End of list
                    if fixed_lines[-1].strip():
                        fixed_lines.append('')
                    in_list = False
                    fixed_lines.append(line)
            else:
                fixed_lines.append(line)
                
        return '\n'.join(fixed_lines)
        
    def _fix_tables(self, content: str) -> str:
        """Fix table formatting."""
        lines = content.split('\n')
        fixed_lines = []
        in_table = False
        table_lines = []
        
        for line in lines:
            if re.match(r'^[\s+=-]+$', line) and not in_table:
                # Start of table
                in_table = True
                table_lines = [line]
            elif in_table:
                if not line.strip():
                    # End of table
                    fixed_lines.extend(self._format_table(table_lines))
                    fixed_lines.append('')
                    in_table = False
                    table_lines = []
                else:
                    table_lines.append(line)
            else:
                fixed_lines.append(line)
                
        # Handle table at end of content
        if in_table:
            fixed_lines.extend(self._format_table(table_lines))
            
        return '\n'.join(fixed_lines)
        
    def _format_table(self, table_lines: List[str]) -> List[str]:
        """Format a table with proper column widths."""
        if not table_lines:
            return []
            
        # Find column positions
        separator_line = next((l for l in table_lines if re.match(r'^[\s+=-]+$', l)), '')
        if not separator_line:
            return table_lines
            
        # Parse column positions
        col_starts = [i for i, c in enumerate(separator_line) if c != ' ' and (i == 0 or separator_line[i-1] == ' ')]
        col_ends = [i for i, c in enumerate(separator_line) if c != ' ' and (i == len(separator_line)-1 or separator_line[i+1] == ' ')]
        
        # Get maximum width for each column
        col_widths = []
        for start, end in zip(col_starts, col_ends):
            width = end - start + 1
            max_width = max(
                len(line[start:end+1].strip())
                for line in table_lines
                if len(line) > start and not re.match(r'^[\s+=-]+$', line)
            )
            col_widths.append(max(width, max_width))
            
        # Format table
        formatted = []
        for line in table_lines:
            if re.match(r'^[\s+=-]+$', line):
                # Separator line
                formatted.append('+'.join('-' * width for width in col_widths))
            else:
                # Content line
                cols = []
                for i, (start, end) in enumerate(zip(col_starts, col_ends)):
                    if len(line) > start:
                        content = line[start:end+1].strip()
                        cols.append(content.ljust(col_widths[i]))
                    else:
                        cols.append(' ' * col_widths[i])
                formatted.append('|'.join(cols))
                
        return formatted
        
    def _fix_images(self, content: str) -> str:
        """Fix image directive formatting."""
        def fix_image(match: re.Match) -> str:
            directive = match.group(1)
            path = match.group(2)
            options = match.group(3) or ''
            
            # Normalize path
            path = path.strip().replace('\\', '/')
            
            # Format options
            if options:
                options = '\n   ' + '\n   '.join(
                    line.strip()
                    for line in options.strip().split('\n')
                    if line.strip()
                )
            
            return f"{directive}{path}{options}\n"
            
        return re.sub(
            r'(\.\.[ ]image::[ ]*)([^\n]+)(\n(?:[ ]*:[^:]+:.*\n)*)',
            fix_image,
            content,
            flags=re.MULTILINE
        )
        
    def _fix_role_directives(self, content: str) -> str:
        """Fix role directive formatting."""
        def fix_role(match: re.Match) -> str:
            role_type = match.group(1)
            name = match.group(2)
            options = match.group(3)
            
            fixed = f".. role:: {name}"
            if options:
                fixed += '\n   ' + '\n   '.join(
                    line.strip()
                    for line in options.strip().split('\n')
                    if line.strip()
                )
            return fixed + '\n\n'
            
        return re.sub(
            r'\.\.[ ]role::[ ]*(\w+)[ ]*(\w+)((?:\n[ ]*:[^:]+:.*)*)',
            fix_role,
            content,
            flags=re.MULTILINE
        )
        
    def _fix_toctree_directives(self, content: str) -> str:
        """Fix toctree directive formatting."""
        def fix_toctree(match: re.Match) -> str:
            directive = match.group(0)
            # Ensure proper spacing around options and entries
            directive = re.sub(r'\n[ ]*:', '\n   :', directive)
            directive = re.sub(r'\n(?![ ]|$)', '\n   ', directive)
            if not directive.endswith('\n\n'):
                directive += '\n'
            return directive
            
        return re.sub(
            r'\.\.[ ]toctree::\n(?:\n[ ]*:[^:]+:[ ]*[^\n]*\n)*(?:\n[ ]+[^\n]+\n)*',
            fix_toctree,
            content,
            flags=re.MULTILINE
        )
        
    def _fix_spacing(self, content: str) -> str:
        """Fix spacing issues."""
        # Remove multiple blank lines
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        # Ensure blank lines around directives
        content = re.sub(r'([^\n])\n(\.\.[ ][^\n]+)', r'\1\n\n\2', content)
        content = re.sub(r'(\.\.[ ][^\n]+\n)([^\n])', r'\1\n\2', content)
        
        return content
        
    def _fix_indentation(self, content: str) -> str:
        """Fix indentation issues."""
        lines = content.split('\n')
        fixed_lines = []
        in_directive = False
        directive_indent = 0
        
        for line in lines:
            if not line.strip():
                fixed_lines.append(line)
                continue
                
            # Check for directives
            if line.lstrip().startswith('.. '):
                in_directive = True
                directive_indent = len(line) - len(line.lstrip())
                fixed_lines.append(line)
            elif in_directive:
                if line.lstrip().startswith(':'):
                    # Directive option
                    fixed_lines.append(' ' * (directive_indent + 3) + line.lstrip())
                elif len(line) - len(line.lstrip()) <= directive_indent:
                    # End of directive
                    in_directive = False
                    fixed_lines.append(line)
                else:
                    # Directive content
                    fixed_lines.append(' ' * (directive_indent + 3) + line.lstrip())
            else:
                fixed_lines.append(line)
                
        return '\n'.join(fixed_lines)
        
    def _print_report(self) -> None:
        """Print fixing results report."""
        print("\nRST Fixing Report")
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
            print("\nAll files fixed successfully!")
            
def main() -> int:
    fixer = RSTFixer()
    success = fixer.fix_all_files()
    return 0 if success else 1
    
if __name__ == '__main__':
    import sys
    sys.exit(main())