"""Simple HTTP server with CORS support for Label Studio."""
import http.server
import socketserver
import sys
import os

class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', '*')
        super().end_headers()

    def do_OPTIONS(self):
        self.send_response(200)
        self.end_headers()

if __name__ == '__main__':
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8888
    directory = sys.argv[2] if len(sys.argv) > 2 else '.'

    os.chdir(directory)

    with socketserver.TCPServer(("", port), CORSRequestHandler) as httpd:
        print(f"Serving {directory} at http://localhost:{port} with CORS enabled")
        httpd.serve_forever()
