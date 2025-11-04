#!/bin/bash
# TinyReact Multiverse Startup Script

echo "🌌 Starting TinyReact Multiverse..."
echo ""

# Check for Python
if command -v python3 &> /dev/null; then
    echo "✅ Python 3 found"
    echo "🚀 Starting server on http://localhost:8000"
    echo ""
    echo "Press Ctrl+C to stop the server"
    echo "In your browser, press Alt+D to open the Multiverse Dev Pane"
    echo ""
    python3 -m http.server 8000
elif command -v python &> /dev/null; then
    echo "✅ Python found"
    echo "🚀 Starting server on http://localhost:8000"
    echo ""
    echo "Press Ctrl+C to stop the server"
    echo "In your browser, press Alt+D to open the Multiverse Dev Pane"
    echo ""
    python -m SimpleHTTPServer 8000
elif command -v npx &> /dev/null; then
    echo "✅ Node.js found"
    echo "🚀 Starting server on http://localhost:3000"
    echo ""
    echo "Press Ctrl+C to stop the server"
    echo "In your browser, press Alt+D to open the Multiverse Dev Pane"
    echo ""
    npx serve -l 3000
else
    echo "❌ No suitable HTTP server found"
    echo ""
    echo "Please install one of the following:"
    echo "  - Python 3: https://www.python.org/"
    echo "  - Node.js: https://nodejs.org/"
    echo ""
    echo "Or use any other static file server"
    exit 1
fi
