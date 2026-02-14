import importlib.util
import os
import sys
import types
from unittest.mock import MagicMock


def setup_dummy_packages():
    packages = [
        "iPhoto",
        "iPhoto.infrastructure",
        "iPhoto.infrastructure.services",
        "iPhoto.gui",
        "iPhoto.gui.ui",
        "iPhoto.gui.ui.widgets",
    ]
    for pkg in packages:
        if pkg not in sys.modules:
            module = types.ModuleType(pkg)
            module.__path__ = []
            sys.modules[pkg] = module


def load_module_from_file(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None:
        raise ImportError(f"Could not load spec for {file_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


setup_dummy_packages()
this_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(this_dir)

gpu_pipeline_path = os.path.join(
    project_root, "src", "iPhoto", "infrastructure", "services", "gpu_pipeline.py"
)
gl_texture_manager_path = os.path.join(
    project_root, "src", "iPhoto", "gui", "ui", "widgets", "gl_texture_manager.py"
)

load_module_from_file("iPhoto.infrastructure.services.gpu_pipeline", gpu_pipeline_path)
gl_texture_manager_mod = load_module_from_file(
    "iPhoto.gui.ui.widgets.gl_texture_manager", gl_texture_manager_path
)
TextureManager = gl_texture_manager_mod.TextureManager


class _Bits(bytearray):
    def setsize(self, _size):
        return None


class _FakeImage:
    def __init__(self, width: int, height: int):
        self._width = width
        self._height = height
        self._bytes_per_line = width * 4
        self._data = _Bits(b"\x00" * (self._bytes_per_line * self._height))

    def isNull(self):
        return False

    def convertToFormat(self, _fmt):
        return self

    def width(self):
        return self._width

    def height(self):
        return self._height

    def constBits(self):
        return self._data

    def sizeInBytes(self):
        return len(self._data)

    def bytesPerLine(self):
        return self._bytes_per_line


def _mock_gl(module):
    module.QImage = type(
        "MockQImage",
        (),
        {"Format": type("Format", (), {"Format_RGBA8888": 0})},
    )
    module.gl.GL_TEXTURE_2D = 3553
    module.gl.GL_RGBA8 = 32856
    module.gl.GL_RGBA = 6408
    module.gl.GL_UNSIGNED_BYTE = 5121
    module.gl.GL_UNPACK_ALIGNMENT = 3317
    module.gl.GL_UNPACK_ROW_LENGTH = 3314
    module.gl.GL_TEXTURE_MIN_FILTER = 10241
    module.gl.GL_TEXTURE_MAG_FILTER = 10240
    module.gl.GL_TEXTURE_WRAP_S = 10242
    module.gl.GL_TEXTURE_WRAP_T = 10243
    module.gl.GL_LINEAR = 9729
    module.gl.GL_CLAMP_TO_EDGE = 33071
    module.gl.GL_NO_ERROR = 0
    module.gl.glDeleteTextures = MagicMock()
    module.gl.glGenTextures = MagicMock(return_value=7)
    module.gl.glBindTexture = MagicMock()
    module.gl.glTexImage2D = MagicMock()
    module.gl.glPixelStorei = MagicMock()
    module.gl.glTexSubImage2D = MagicMock()
    module.gl.glTexParameteri = MagicMock()
    module.gl.glGetError = MagicMock(return_value=0)


def test_upload_texture_streams_large_images_in_chunks():
    _mock_gl(gl_texture_manager_mod)
    manager = TextureManager()

    manager.upload_texture(_FakeImage(256, 600))

    assert gl_texture_manager_mod.gl.glTexSubImage2D.call_count == 3
    heights = [call.args[5] for call in gl_texture_manager_mod.gl.glTexSubImage2D.call_args_list]
    y_offsets = [call.args[3] for call in gl_texture_manager_mod.gl.glTexSubImage2D.call_args_list]
    assert heights == [256, 256, 88]
    assert y_offsets == [0, 256, 512]


def test_upload_texture_keeps_single_upload_for_small_images():
    _mock_gl(gl_texture_manager_mod)
    manager = TextureManager()

    manager.upload_texture(_FakeImage(128, 64))

    assert gl_texture_manager_mod.gl.glTexSubImage2D.call_count == 1
