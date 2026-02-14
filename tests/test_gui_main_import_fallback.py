from pathlib import Path


def test_gui_main_has_iPhoto_import_fallback():
    main_py = Path("src/iPhoto/gui/main.py").read_text(encoding="utf-8")
    assert 'if exc.name != "iPhoto":' in main_py
    assert "sys.path.insert(0, str(Path(__file__).resolve().parents[2]))" in main_py
