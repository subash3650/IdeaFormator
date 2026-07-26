from __future__ import annotations

import uvicorn

from phase5.api.app import create_app

app = create_app()

if __name__ == "__main__":
    uvicorn.run("phase5.api.__main__:app", host="0.0.0.0", port=8000, reload=True)
