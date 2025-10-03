#!/bin/bash
set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}===============================================${NC}"
echo -e "${GREEN}  Stable-Retro Installation Script for ARM${NC}"
echo -e "${GREEN}===============================================${NC}"
echo ""

# Check if running on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo -e "${RED}Error: This script is designed for macOS only${NC}"
    exit 1
fi

# Check if running on ARM
ARCH=$(uname -m)
if [[ "$ARCH" != "arm64" ]]; then
    echo -e "${YELLOW}Warning: This script is optimized for Apple Silicon (ARM). Detected: $ARCH${NC}"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo -e "${GREEN}[1/5] Checking dependencies...${NC}"

# Check for Homebrew
if ! command -v brew &> /dev/null; then
    echo -e "${RED}Homebrew not found. Please install it first:${NC}"
    echo "  /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
    exit 1
fi

# Check for required tools
echo "  Checking cmake..."
if ! command -v cmake &> /dev/null; then
    echo "  Installing cmake..."
    brew install cmake
fi

echo "  Checking pkg-config..."
if ! command -v pkg-config &> /dev/null; then
    echo "  Installing pkg-config..."
    brew install pkg-config
fi

# Install other dependencies
echo "  Installing additional dependencies..."
brew list lua@5.3 &> /dev/null || brew install lua@5.3
brew list libzip &> /dev/null || brew install libzip
brew list capnp &> /dev/null || brew install capnp

# Check for Python and pip
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python 3 not found. Please install Python 3.8 or higher.${NC}"
    exit 1
fi

echo "  Installing Python build dependencies..."
pip3 install cmake wheel --quiet

echo -e "${GREEN}[2/5] Fixing zlib compatibility issues for ARM...${NC}"

# Function to fix zutil.h files
fix_zutil() {
    local file=$1
    if [[ ! -f "$file" ]]; then
        echo -e "${YELLOW}  Warning: $file not found, skipping${NC}"
        return
    fi

    echo "  Patching $file..."

    # Check if already patched
    if grep -q "Modern macOS has fdopen, don't redefine it" "$file"; then
        echo "    Already patched, skipping"
        return
    fi

    # Create backup
    cp "$file" "${file}.backup"

    # Fix MACOS/TARGET_OS_MAC section
    if grep -q "define fdopen(fd,mode) NULL" "$file"; then
        # Use perl for more reliable multi-line replacements
        perl -i -pe '
            BEGIN { undef $/; }
            s/#if defined\(MACOS\) \|\| defined\(TARGET_OS_MAC\)\n#  define OS_CODE  (?:7|0x07)\n#  ifndef Z_SOLO\n#    if defined\(__MWERKS__\) && __dest_os != __be_os && __dest_os != __win32_os\n#      include <unix.h> \/\* for fdopen \*\/\n#    else\n#      ifndef fdopen\n#        define fdopen\(fd,mode\) NULL \/\* No fdopen\(\) \*\/\n#      endif\n#    endif\n#  endif\n#endif/#if defined(MACOS) || defined(TARGET_OS_MAC)\n#  define OS_CODE  7\n#  ifndef Z_SOLO\n#    if defined(__MWERKS__) && __dest_os != __be_os && __dest_os != __win32_os\n#      include <unix.h> \/* for fdopen *\/\n#    else\n       \/* Modern macOS has fdopen, don'\''t redefine it *\/\n#      if !defined(__APPLE__) && !defined(fdopen)\n#        define fdopen(fd,mode) NULL \/* No fdopen() *\/\n#      endif\n#    endif\n#  endif\n#endif/g;
            s/#if defined\(_BEOS_\) \|\| defined\(RISCOS\)\n#  define fdopen\(fd,mode\) NULL \/\* No fdopen\(\) \*\/\n#endif/\/* Disabled for modern macOS compatibility *\/\n\/* #if defined(_BEOS_) || defined(RISCOS)\n#  define fdopen(fd,mode) NULL\n#endif *\//g;
        ' "$file"
    fi

    echo "    ✓ Patched successfully"
}

# Fix all zutil.h files
fix_zutil "cores/genesis/core/cd_hw/libchdr/deps/zlib/zutil.h"
fix_zutil "cores/32x/pico/cd/libchdr/deps/zlib-1.2.12/zutil.h"
fix_zutil "cores/32x/pico/cd/libchdr/deps/zlib-1.2.11/zutil.h"
fix_zutil "cores/gba/src/third-party/zlib/zutil.h"
fix_zutil "cores/pce/deps/zlib/zutil.h"

echo -e "${GREEN}[3/5] Setting up build environment...${NC}"

# Set SDKROOT for macOS SDK
export SDKROOT=$(xcrun --sdk macosx --show-sdk-path)
echo "  SDKROOT set to: $SDKROOT"

# Add Qt to PATH if needed
if [[ -d "/opt/homebrew/opt/qt@5/bin" ]]; then
    export PATH="/opt/homebrew/opt/qt@5/bin:$PATH"
    echo "  Qt@5 added to PATH"
fi

echo -e "${GREEN}[4/5] Compiling stable-retro...${NC}"

# Clean previous builds
if [[ -f "Makefile" ]]; then
    echo "  Cleaning previous build..."
    make clean &> /dev/null || true
fi

# Run CMake
echo "  Running CMake..."
cmake . -G "Unix Makefiles" || {
    echo -e "${RED}CMake configuration failed${NC}"
    exit 1
}

# Compile
echo "  Compiling (this may take a few minutes)..."
make -j$(sysctl -n hw.ncpu) retro || {
    echo -e "${RED}Compilation failed${NC}"
    exit 1
}

echo -e "${GREEN}[5/5] Installing stable-retro...${NC}"

# Install with pip
pip3 install -e . || {
    echo -e "${RED}Installation failed${NC}"
    exit 1
}

echo ""
echo -e "${GREEN}===============================================${NC}"
echo -e "${GREEN}  Installation Complete!${NC}"
echo -e "${GREEN}===============================================${NC}"
echo ""
echo "Testing installation..."
python3 -c "import retro; print('✓ stable-retro version:', retro.__version__)" || {
    echo -e "${RED}Installation test failed${NC}"
    exit 1
}

echo ""
echo -e "${GREEN}Success! Stable-retro is now installed.${NC}"
echo ""
echo "To import ROMs, run:"
echo "  python3 -m retro.import /path/to/your/roms"
echo ""
echo "To list available games:"
echo "  python3 -c \"import retro; print(retro.data.list_games())\""
echo ""
echo "Example usage:"
echo "  python3 play_mario_random.py"
echo ""
