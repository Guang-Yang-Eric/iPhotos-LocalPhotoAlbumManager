import importlib.util
from pathlib import Path
import sys


def _load_gl_shader_manager():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "src/iPhoto/gui/ui/widgets/gl_shader_manager.py"
    spec = importlib.util.spec_from_file_location("test_gl_shader_manager_module", module_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules["test_gl_shader_manager_module"] = module
    spec.loader.exec_module(module)
    return module


load_shader_source = _load_gl_shader_manager().load_shader_source


def test_load_shader_source_reads_sibling_file(tmp_path: Path):
    module_file = tmp_path / "module.py"
    module_file.write_text("# marker", encoding="utf-8")
    shader_file = tmp_path / "demo.frag"
    shader_file.write_text("void main(){}", encoding="utf-8")

    loaded = load_shader_source(str(module_file), "demo.frag")

    assert loaded == "void main(){}"
