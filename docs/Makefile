# Minimal makefile for Sphinx documentation

# You can set these variables from the command line, and also
# from the environment for the first two.
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SOURCEDIR     = .
BUILDDIR      = _build

# Put it first so that "make" without argument is like "make help".
help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

.PHONY: help Makefile clean livehtml

# Catch-all target: route all unknown targets to Sphinx using the new
# "make mode" option.  $(O) is meant as a shortcut for $(SPHINXOPTS).
%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Custom targets

# Clean build directory
clean:
	rm -rf $(BUILDDIR)/*

# Live preview with auto-rebuild
livehtml:
	sphinx-autobuild "$(SOURCEDIR)" "$(BUILDDIR)/html" $(SPHINXOPTS) $(O)

# Convert markdown to RST
md2rst:
	@echo "Converting Markdown files to RST..."
	@find . -name "*.md" -not -path "./_build/*" -not -path "./venv/*" -exec sh -c 'pandoc "$$0" -f markdown -t rst -o "$${0%.md}.rst"' {} \;

# Check broken links
linkcheck:
	@$(SPHINXBUILD) -b linkcheck "$(SOURCEDIR)" "$(BUILDDIR)/linkcheck" $(SPHINXOPTS) $(O)

# Build PDF documentation
pdf:
	@$(SPHINXBUILD) -M latexpdf "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Build EPUB documentation
epub:
	@$(SPHINXBUILD) -M epub "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Run doctests
doctest:
	@$(SPHINXBUILD) -M doctest "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Build coverage report
coverage:
	@$(SPHINXBUILD) -M coverage "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Build documentation and open in browser
preview: html
	@python -c "import webbrowser, os; webbrowser.open('file://' + os.path.realpath('$(BUILDDIR)/html/index.html'))"

# Create new RST file from template
newpage:
	@read -p "Enter page name (without .rst): " name; \
	echo "$$name\n$$(printf '=%.0s' $$(seq 1 $${#name}))\n\nWrite your content here." > "$(SOURCEDIR)/$$name.rst"

# Show TODOs
todos:
	@$(SPHINXBUILD) -M todo "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

# Build and check documentation
check: clean html linkcheck doctest

# Full build with all formats
full: clean html pdf epub

# Development build with live preview
dev: clean livehtml