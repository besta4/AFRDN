"""Start the Jatayu backend from the project root."""

from pathlib import Path
import sys

BACKEND_DIR = Path(__file__).resolve().parent / "backend"
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from backend.start import main  # noqa: E402


if __name__ == "__main__":
    main()
