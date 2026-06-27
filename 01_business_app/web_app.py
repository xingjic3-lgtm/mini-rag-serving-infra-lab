from __future__ import annotations

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from urllib.parse import urlparse

from rag_pipeline import DEFAULT_DOCUMENT, DEFAULT_QUESTION, run_rag


HOST = "127.0.0.1"
PORT = 8001


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Mini RAG Serving Infra Lab</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f4f6f8;
      --panel: #ffffff;
      --text: #17202a;
      --muted: #5d6d7e;
      --line: #d6dde5;
      --accent: #126a72;
      --accent-strong: #0b4f55;
      --code: #101820;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      background: var(--bg);
      color: var(--text);
      font-family: Segoe UI, Arial, sans-serif;
      font-size: 15px;
    }
    header {
      padding: 18px 24px 12px;
      border-bottom: 1px solid var(--line);
      background: var(--panel);
    }
    h1 {
      margin: 0 0 4px;
      font-size: 22px;
      font-weight: 650;
      letter-spacing: 0;
    }
    header p {
      margin: 0;
      color: var(--muted);
    }
    main {
      display: grid;
      grid-template-columns: minmax(320px, 0.9fr) minmax(360px, 1.1fr);
      gap: 16px;
      padding: 16px;
      max-width: 1360px;
      margin: 0 auto;
    }
    section {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 14px;
      min-width: 0;
    }
    h2 {
      margin: 0 0 10px;
      font-size: 15px;
      font-weight: 650;
    }
    label {
      display: block;
      margin: 12px 0 6px;
      color: var(--muted);
      font-size: 13px;
    }
    textarea, input {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 4px;
      padding: 10px;
      font: inherit;
      line-height: 1.45;
      background: #fff;
      color: var(--text);
    }
    textarea {
      min-height: 280px;
      resize: vertical;
    }
    input { min-height: 42px; }
    button {
      margin-top: 12px;
      width: 100%;
      min-height: 42px;
      border: 0;
      border-radius: 4px;
      background: var(--accent);
      color: white;
      font: inherit;
      font-weight: 650;
      cursor: pointer;
    }
    button:hover { background: var(--accent-strong); }
    button:disabled {
      cursor: wait;
      background: #8aa6aa;
    }
    .status {
      min-height: 22px;
      margin-top: 10px;
      color: var(--muted);
      font-size: 13px;
    }
    .output {
      display: grid;
      gap: 12px;
    }
    pre {
      margin: 0;
      padding: 12px;
      min-height: 86px;
      max-height: 330px;
      overflow: auto;
      border-radius: 4px;
      background: var(--code);
      color: #eef3f7;
      white-space: pre-wrap;
      word-break: break-word;
      line-height: 1.45;
      font-family: Consolas, ui-monospace, monospace;
      font-size: 13px;
    }
    @media (max-width: 860px) {
      main { grid-template-columns: 1fr; }
      textarea { min-height: 220px; }
    }
  </style>
</head>
<body>
  <header>
    <h1>Mini RAG Serving Infra Lab</h1>
    <p>Phase 01 local document QA business loop</p>
  </header>
  <main>
    <section>
      <h2>Request</h2>
      <label for="document">Document</label>
      <textarea id="document"></textarea>
      <label for="question">Question</label>
      <input id="question" type="text">
      <button id="ask">Run Local LLM</button>
      <div class="status" id="status"></div>
    </section>
    <section class="output">
      <div>
        <h2>Retrieved Context</h2>
        <pre id="context"></pre>
      </div>
      <div>
        <h2>Prompt</h2>
        <pre id="prompt"></pre>
      </div>
      <div>
        <h2>LLM Answer</h2>
        <pre id="answer"></pre>
      </div>
    </section>
  </main>
  <script>
    const defaults = __DEFAULTS__;
    const documentInput = document.getElementById("document");
    const questionInput = document.getElementById("question");
    const button = document.getElementById("ask");
    const statusBox = document.getElementById("status");
    const contextBox = document.getElementById("context");
    const promptBox = document.getElementById("prompt");
    const answerBox = document.getElementById("answer");

    documentInput.value = defaults.document;
    questionInput.value = defaults.question;

    async function ask() {
      button.disabled = true;
      statusBox.textContent = "Running local model. First request may take a few seconds.";
      contextBox.textContent = "";
      promptBox.textContent = "";
      answerBox.textContent = "";

      try {
        const response = await fetch("/api/ask", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            document: documentInput.value,
            question: questionInput.value
          })
        });

        const data = await response.json();
        if (!response.ok) {
          throw new Error(data.error || "Request failed");
        }

        contextBox.textContent = data.retrieved_context;
        promptBox.textContent = data.prompt;
        answerBox.textContent = data.answer;
        statusBox.textContent = "Done";
      } catch (error) {
        statusBox.textContent = error.message;
      } finally {
        button.disabled = false;
      }
    }

    button.addEventListener("click", ask);
  </script>
</body>
</html>
"""


class RagRequestHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if urlparse(self.path).path != "/":
            self._send_json({"error": "Not found"}, status=404)
            return

        defaults = json.dumps({"document": DEFAULT_DOCUMENT.strip(), "question": DEFAULT_QUESTION})
        html = HTML.replace("__DEFAULTS__", defaults)
        self._send_text(html, content_type="text/html; charset=utf-8")

    def do_POST(self):
        if urlparse(self.path).path != "/api/ask":
            self._send_json({"error": "Not found"}, status=404)
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
            document = str(payload.get("document", "")).strip()
            question = str(payload.get("question", "")).strip()

            if not document or not question:
                self._send_json({"error": "Document and question are required."}, status=400)
                return

            result = run_rag(document, question)
            self._send_json(
                {
                    "document": result.document,
                    "question": result.question,
                    "retrieved_context": result.retrieved_context,
                    "prompt": result.prompt,
                    "answer": result.answer,
                }
            )
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

    def log_message(self, format, *args):
        return

    def _send_text(self, body: str, content_type: str = "text/plain; charset=utf-8", status: int = 200):
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, body: dict, status: int = 200):
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main():
    server = ThreadingHTTPServer((HOST, PORT), RagRequestHandler)
    print(f"Mini RAG app running at http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
