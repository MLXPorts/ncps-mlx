#!/usr/bin/env python3
"""
Documentation build log parser and viewer.
Captures build output, categorizes issues, and provides paginated viewing.
"""

import os
import re
import sys
import json
from pathlib import Path
from typing import List, Dict, Optional

class LogParser:
    def __init__(self, log_dir: str = "../logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.current_log = None
        self.bookmark_file = self.log_dir / "bookmark.json"
        self.page_size = 10
        
    def capture_build_output(self, command: str) -> str:
        """Capture build output to a log file."""
        timestamp = self._get_timestamp()
        log_file = self.log_dir / f"build_{timestamp}.log"
        
        # Run build command and capture output
        os.system(f"{command} 2>&1 | tee {log_file}")
        
        self.current_log = log_file
        return str(log_file)
        
    def parse_log(self, log_file: Optional[str] = None) -> Dict:
        """Parse log file and categorize issues."""
        log_file = log_file or self.current_log
        if not log_file:
            raise ValueError("No log file specified")
            
        issues = {
            "warnings": [],
            "errors": [],
            "missing_refs": [],
            "syntax_issues": []
        }
        
        with open(log_file, 'r') as f:
            for line in f:
                line = line.strip()
                
                # Categorize issues
                if "WARNING:" in line:
                    if "missing" in line.lower():
                        issues["missing_refs"].append(line)
                    else:
                        issues["warnings"].append(line)
                elif "ERROR:" in line:
                    issues["errors"].append(line)
                elif "SEVERE:" in line or "SYNTAX:" in line:
                    issues["syntax_issues"].append(line)
                    
        return issues
        
    def view_log(self, log_file: Optional[str] = None, category: Optional[str] = None):
        """View log contents with pagination."""
        log_file = log_file or self.current_log
        if not log_file:
            raise ValueError("No log file specified")
            
        # Load bookmark
        bookmark = self._load_bookmark()
        start_line = bookmark.get(str(log_file), 0)
        
        # Parse and filter issues
        issues = self.parse_log(log_file)
        if category:
            lines = issues.get(category, [])
        else:
            lines = []
            for issue_list in issues.values():
                lines.extend(issue_list)
                
        # Display current page
        end_line = min(start_line + self.page_size, len(lines))
        for line in lines[start_line:end_line]:
            print(line)
            
        # Save position
        bookmark[str(log_file)] = end_line
        self._save_bookmark(bookmark)
        
        # Show navigation info
        remaining = len(lines) - end_line
        if remaining > 0:
            print(f"\n{remaining} more lines. Run again to see next page.")
        else:
            print("\nEnd of log.")
            # Reset bookmark when reaching the end
            bookmark[str(log_file)] = 0
            self._save_bookmark(bookmark)
            
    def _get_timestamp(self) -> str:
        """Get current timestamp for log filename."""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def _load_bookmark(self) -> Dict:
        """Load reading position bookmark."""
        if self.bookmark_file.exists():
            with open(self.bookmark_file, 'r') as f:
                return json.load(f)
        return {}
        
    def _save_bookmark(self, bookmark: Dict):
        """Save reading position bookmark."""
        with open(self.bookmark_file, 'w') as f:
            json.dump(bookmark, f)

def main():
    parser = LogParser()
    
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Capture build: log_parser.py build 'make html'")
        print("  View log: log_parser.py view [category]")
        print("Categories: warnings, errors, missing_refs, syntax_issues")
        sys.exit(1)
        
    command = sys.argv[1]
    
    if command == "build":
        if len(sys.argv) < 3:
            print("Please provide build command")
            sys.exit(1)
        build_cmd = sys.argv[2]
        log_file = parser.capture_build_output(build_cmd)
        print(f"Build output saved to: {log_file}")
        
    elif command == "view":
        category = sys.argv[2] if len(sys.argv) > 2 else None
        parser.view_log(category=category)
        
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)

if __name__ == "__main__":
    main()