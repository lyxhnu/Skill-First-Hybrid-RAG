from __future__ import annotations

import uvicorn


def run() -> None:
    uvicorn.run("rag_graph.api.main:app", host="0.0.0.0", port=8000, reload=False)


if __name__ == "__main__":
    run()

