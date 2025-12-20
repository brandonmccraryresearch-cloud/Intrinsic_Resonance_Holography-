#!/bin/bash
# Build script for IRH Desktop .deb package
set -euo pipefail
# 
# This script builds a Debian package for the IRH Desktop application.
# The package can be installed on Debian/Ubuntu systems.
#
# Usage:
#   ./build-deb.sh           # Build using dpkg-deb (simple)
#   ./build-deb.sh --full    # Build using debuild (complete, requires devscripts)
#
# Requirements:
#   - dpkg-deb (dpkg package)
#   - Python 3.10+
#   - Optional: debhelper, dh-python, debuild (for --full build)

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

VERSION="21.0.0"
PACKAGE_NAME="irh-desktop"
ARCHITECTURE="all"
OUTPUT_DIR="${SCRIPT_DIR}/dist"

echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║           IRH Desktop .deb Package Builder                        ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║ Version: ${VERSION}                                                     ║"
echo "║ Package: ${PACKAGE_NAME}                                              ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
echo ""

# Check for required tools
if ! command -v dpkg-deb &> /dev/null; then
    echo "ERROR: dpkg-deb not found. Install with: sudo apt-get install dpkg"
    exit 1
fi

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Clean previous builds
rm -rf "${OUTPUT_DIR}/${PACKAGE_NAME}_${VERSION}"*

if [ "$1" == "--full" ]; then
    # Full Debian build using debuild
    echo "[INFO] Building with debuild (full Debian packaging)..."
    
    if ! command -v debuild &> /dev/null; then
        echo "WARNING: debuild not found. Install with: sudo apt-get install devscripts"
        echo "Falling back to simple dpkg-deb build..."
    else
        debuild -us -uc -b
        mv ../*.deb "$OUTPUT_DIR/" || true
        echo ""
        echo "✓ Build complete. Package available in: $OUTPUT_DIR"
        exit 0
    fi
fi

# Simple build using dpkg-deb
echo "[INFO] Building with dpkg-deb (simple packaging)..."

BUILD_ROOT="${OUTPUT_DIR}/${PACKAGE_NAME}_${VERSION}-1_${ARCHITECTURE}"

# Create package directory structure
echo "[INFO] Creating package structure..."
mkdir -p "${BUILD_ROOT}/DEBIAN"
mkdir -p "${BUILD_ROOT}/opt/irh/desktop/lib/python"
mkdir -p "${BUILD_ROOT}/opt/irh/desktop/bin"
mkdir -p "${BUILD_ROOT}/usr/bin"
mkdir -p "${BUILD_ROOT}/usr/share/applications"
mkdir -p "${BUILD_ROOT}/usr/share/doc/${PACKAGE_NAME}"

# Copy control files
echo "[INFO] Copying Debian control files..."
cp debian/control "${BUILD_ROOT}/DEBIAN/"
cp debian/postinst "${BUILD_ROOT}/DEBIAN/"
chmod 755 "${BUILD_ROOT}/DEBIAN/postinst"

# Create prerm script
cat > "${BUILD_ROOT}/DEBIAN/prerm" << 'PRERM'
#!/bin/bash
set -e

case "$1" in
    remove|upgrade|deconfigure)
        echo "[INFO] Removing IRH Desktop..."
        ;;
    failed-upgrade)
        ;;
    *)
        echo "prerm called with unknown argument '$1'" >&2
        exit 1
        ;;
esac

exit 0
PRERM
chmod 755 "${BUILD_ROOT}/DEBIAN/prerm"

# Create postrm script
cat > "${BUILD_ROOT}/DEBIAN/postrm" << 'POSTRM'
#!/bin/bash
set -e

case "$1" in
    purge)
        echo "[INFO] Purging IRH Desktop configuration..."
        rm -rf /opt/irh/config 2>/dev/null || true
        rm -rf /var/log/irh 2>/dev/null || true
        ;;
    remove|upgrade|failed-upgrade|abort-install|abort-upgrade|disappear)
        ;;
    *)
        echo "postrm called with unknown argument '$1'" >&2
        exit 1
        ;;
esac

exit 0
POSTRM
chmod 755 "${BUILD_ROOT}/DEBIAN/postrm"

# Copy Python source files
echo "[INFO] Copying Python application..."
cp -r src/irh_desktop "${BUILD_ROOT}/opt/irh/desktop/lib/python/"

# Create launcher script
cat > "${BUILD_ROOT}/opt/irh/desktop/bin/irh-desktop" << 'LAUNCHER'
#!/bin/bash
# IRH Desktop Application Launcher
#
# This script launches the IRH Desktop GUI application.

export PYTHONPATH="/opt/irh/desktop/lib/python:${PYTHONPATH}"

# Check for Python
if command -v python3 &> /dev/null; then
    PYTHON=python3
elif command -v python &> /dev/null; then
    PYTHON=python
else
    echo "ERROR: Python not found. Please install Python 3.10 or later."
    exit 1
fi

exec "$PYTHON" -m irh_desktop.main "$@"
LAUNCHER
chmod 755 "${BUILD_ROOT}/opt/irh/desktop/bin/irh-desktop"

# Create symlink in /usr/bin
ln -sf /opt/irh/desktop/bin/irh-desktop "${BUILD_ROOT}/usr/bin/irh-desktop"

# Copy desktop file
cp debian/irh-desktop.desktop "${BUILD_ROOT}/usr/share/applications/"

# Copy documentation
cp README.md "${BUILD_ROOT}/usr/share/doc/${PACKAGE_NAME}/"

# Create copyright file
cat > "${BUILD_ROOT}/usr/share/doc/${PACKAGE_NAME}/copyright" << 'COPYRIGHT'
Format: https://www.debian.org/doc/packaging-manuals/copyright-format/1.0/
Upstream-Name: irh-desktop
Upstream-Contact: Brandon D. McCrary <brandon@irhresearch.org>
Source: https://github.com/brandonmccraryresearch-cloud/Intrinsic_Resonance_Holography-

Files: *
Copyright: 2024-2025 Brandon D. McCrary
License: MIT

License: MIT
 Permission is hereby granted, free of charge, to any person obtaining a
 copy of this software and associated documentation files (the "Software"),
 to deal in the Software without restriction, including without limitation
 the rights to use, copy, modify, merge, publish, distribute, sublicense,
 and/or sell copies of the Software, and to permit persons to whom the
 Software is furnished to do so, subject to the following conditions:
 .
 The above copyright notice and this permission notice shall be included
 in all copies or substantial portions of the Software.
 .
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS
 OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
COPYRIGHT

# Build the package
echo "[INFO] Building .deb package..."
dpkg-deb --build "${BUILD_ROOT}"

# Move to output directory with correct name
DEB_FILE="${PACKAGE_NAME}_${VERSION}-1_${ARCHITECTURE}.deb"
if [ "${BUILD_ROOT}.deb" != "${OUTPUT_DIR}/${DEB_FILE}" ]; then
    mv "${BUILD_ROOT}.deb" "${OUTPUT_DIR}/${DEB_FILE}"
fi

# Verify the package
echo "[INFO] Verifying package..."
dpkg-deb --info "${OUTPUT_DIR}/${DEB_FILE}"
echo ""
dpkg-deb --contents "${OUTPUT_DIR}/${DEB_FILE}" | head -20

echo ""
echo "╔══════════════════════════════════════════════════════════════════╗"
echo "║                    Build Complete!                                ║"
echo "╠══════════════════════════════════════════════════════════════════╣"
echo "║ Package: ${OUTPUT_DIR}/${DEB_FILE}"
echo "║                                                                   ║"
echo "║ To install:                                                       ║"
echo "║   sudo dpkg -i ${DEB_FILE}"
echo "║   sudo apt-get install -f  # to fix dependencies                  ║"
echo "╚══════════════════════════════════════════════════════════════════╝"
