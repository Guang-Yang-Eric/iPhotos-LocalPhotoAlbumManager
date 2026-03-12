"""Entry point — ``python -m demo.video``."""

import sys
from PySide6.QtWidgets import QApplication
from ui import VideoEditor


def main():
    app = QApplication(sys.argv)
    window = VideoEditor()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
