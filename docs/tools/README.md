# Documentation Tools

This directory contains tools for managing NCPS documentation.

## Quick Start

1. Set up documentation environment:
```bash
# Make setup script executable
chmod +x setup.sh

# Run setup
./setup.sh

# Activate documentation environment
source ../activate_docs
```

2. Use documentation tools:
```bash
# Show available commands
doctools help

# Build documentation
doctools build

# Preview in browser
doctools preview
```

## Available Tools

### 1. convert_docs.py
Converts markdown files to RST and organizes them into the Sphinx structure.

Features:
- Multi-language code block support
- Improved image handling
- Parallel processing for faster conversion
- Better error handling and logging

```bash
# Convert and organize all documentation
doctools convert

# The script will:
# 1. Convert MD to RST with proper formatting
# 2. Handle code blocks in multiple languages
# 3. Process images and fix references
# 4. Organize files into appropriate directories
# 5. Create and update index files
```

### 2. test_docs.py
Tests documentation builds and checks for common issues.

Features:
- Parallel testing for faster validation
- Enhanced code block validation
- Image reference checking
- Cross-reference validation
- Comprehensive error reporting

```bash
# Run all documentation tests
doctools test

# Tests include:
# - RST syntax validation
# - Documentation build
# - Link checking
# - Code block validation
# - Cross-reference verification
# - Image reference validation
# - Section hierarchy checking
```

### 3. create_doc.py
Creates new documentation files using templates.

Features:
- Multiple template types
- Metadata support
- Automatic index updates
- Template customization
- Better error handling

```bash
# Create a guide
doctools create "My Guide Title" --template guide --directory guides

# Create API documentation
doctools create "MyClass API" --template api --directory api \
    --module-path ncps.module --class-name MyClass

# Create architecture documentation
doctools create "Design Document" --template architecture \
    --directory architecture/design

# Create research documentation
doctools create "Research Topic" --template research \
    --directory architecture/research
```

### 4. fix_rst.py
Fixes common RST syntax issues and formatting.

Features:
- Multi-language code formatting
- Table formatting
- Image directive fixes
- Role directive handling
- Proper indentation
- Comprehensive error reporting

```bash
# Fix RST syntax in all documentation
doctools fix
```

### 5. log_parser.py
Manages documentation build output with improved readability and issue tracking.

Features:
- Captures build output to log files
- Categorizes issues by type
- Paginated viewing of results
- Saves reading position
- Filters by issue category

```bash
# View all build issues
doctools logs

# View specific categories
doctools logs warnings     # View only warnings
doctools logs errors      # View only errors
doctools logs missing-refs # View missing references
doctools logs syntax      # View syntax issues

# Build with logging
doctools build           # Automatically captures output
```

## Templates

### Guide Template
- Overview
- Prerequisites
- Getting Started
- Detailed Instructions
- Advanced Usage
- Troubleshooting

### API Template
- Overview
- Classes
- Examples
- Functions
- See Also

### Architecture Template
- Overview
- Design Goals
- System Components
- Integration Points
- Performance Considerations
- Security Considerations
- Deployment Strategy
- Future Considerations

### Research Template
- Abstract
- Background
- Methodology
- Results
- Discussion
- Future Work
- References

### Implementation Template
- Overview
- Prerequisites
- Implementation Details
- Code Examples
- Testing
- Deployment

### Visualization Template
- Overview
- Basic Usage
- Advanced Features
- Best Practices
- Export Options

## Usage Examples

### Converting Documentation
```bash
# Convert all markdown files
doctools convert

# Test the conversion
doctools test
```

### Creating New Documentation
```bash
# Create a new guide
doctools create "Installation Guide" \
    --template guide \
    --directory guides

# Create API documentation
doctools create "TensorAbstraction API" \
    --template api \
    --directory api \
    --module-path ncps.abstractions \
    --class-name TensorAbstraction

# Create with metadata
doctools create "Research Paper" \
    --template research \
    --directory research \
    --metadata '{"author": "John Doe", "date": "2025-02-20"}'
```

### Testing Documentation
```bash
# Run all tests
doctools test

# Build documentation after changes
doctools build

# Build specific format
doctools build --pdf

# Build all formats
doctools build --all

# View build issues
doctools logs
```

## Best Practices

1. Documentation Creation
   - Use appropriate templates
   - Follow RST syntax guidelines
   - Include practical examples
   - Add proper cross-references
   - Include metadata where appropriate

2. Testing
   - Run tests before committing
   - Fix all reported issues
   - Verify links work
   - Check code examples
   - Validate image references

3. Organization
   - Use correct directories
   - Update index files
   - Maintain proper hierarchy
   - Follow naming conventions
   - Keep related documents together

4. Code Examples
   - Use language-specific formatting
   - Include complete, working examples
   - Add proper comments
   - Follow style guidelines
   - Test all examples

## Maintenance

1. Regular Tasks
   - Run tests periodically
   - Update broken links
   - Verify code examples
   - Check cross-references
   - Update dependencies

2. Updates
   - Keep templates current
   - Update test scripts
   - Maintain conversion tools
   - Add new features as needed
   - Monitor build performance

## Getting Help

1. Tool Help:
```bash
doctools help
```

2. Documentation:
   - Check Sphinx documentation: https://www.sphinx-doc.org/
   - Review existing docs for examples
   - Use test output for debugging
   - Check log files in logs directory

3. Issues:
   - Run `doctools test` for validation
   - Check error messages
   - Review build output
   - Check log files
   - Use --verbose flag for more details

These tools are designed to make documentation management easier and more consistent. Use them regularly to maintain high-quality documentation.