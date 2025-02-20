# Documentation Tools

This directory contains tools for managing NCPS documentation.

## Quick Start

The `doctools` script provides a simple interface to all documentation tools:

```bash
# Make tools executable (if needed)
chmod +x tools/doctools tools/*.py

# Add to your PATH (optional)
export PATH=$PATH:/path/to/ncps/docs/tools

# Show available commands
./doctools help

# Common commands
./doctools convert    # Convert markdown to RST
./doctools test      # Run doc tests
./doctools build     # Build HTML docs
./doctools preview   # Build and open in browser
```

## Available Commands

1. `convert` - Convert and organize documentation
   ```bash
   ./doctools convert
   ```

2. `test` - Run documentation tests
   ```bash
   ./doctools test
   ```

3. `create` - Create new documentation
   ```bash
   ./doctools create "My Guide" --template guide --directory guides
   ```

4. `build` - Build documentation
   ```bash
   ./doctools build
   ```

5. `preview` - Build and preview in browser
   ```bash
   ./doctools preview
   ```

6. `clean` - Clean build directory
   ```bash
   ./doctools clean
   ```

## Individual Tools

### 1. convert_docs.py
Converts markdown files to RST and organizes them into the Sphinx structure.

```bash
# Convert and organize all documentation
python convert_docs.py

# The script will:
# 1. Convert MD to RST
# 2. Organize files into appropriate directories
# 3. Create index files
```

### 2. test_docs.py
Tests documentation builds and checks for common issues.

```bash
# Run all documentation tests
python test_docs.py

# Tests include:
# - RST syntax validation
# - Documentation build
# - Link checking
# - Code block validation
# - Cross-reference verification
```

### 3. create_doc.py
Creates new documentation files using templates.

```bash
# Create a guide
python create_doc.py "My Guide Title" --template guide --directory guides

# Create API documentation
python create_doc.py "MyClass API" --template api --directory api \
    --module-path ncps.module --class-name MyClass
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
- Implementation Details
- Integration Points
- Performance Considerations
- Future Considerations

### Research Template
- Overview
- Background
- Methodology
- Results
- Conclusions
- Future Work
- References

## Best Practices

1. Documentation Creation
   - Use appropriate templates
   - Follow RST syntax guidelines
   - Include practical examples
   - Add proper cross-references

2. Testing
   - Run tests before committing
   - Fix all reported issues
   - Verify links work
   - Check code examples

3. Organization
   - Use correct directories
   - Update index files
   - Maintain proper hierarchy
   - Follow naming conventions

## Maintenance

1. Regular Tasks
   - Run tests periodically
   - Update broken links
   - Verify code examples
   - Check cross-references

2. Updates
   - Keep templates current
   - Update test scripts
   - Maintain conversion tools
   - Add new features as needed

## Getting Help

- Check Sphinx documentation
- Review existing docs
- Use test output for debugging
- Consult team members

These tools are designed to make documentation management easier and more consistent. Use them regularly to maintain high-quality documentation.