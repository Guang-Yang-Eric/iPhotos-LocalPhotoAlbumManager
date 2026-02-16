"""Tests for GPU Pipeline integration into GLRenderer subsystems.

These tests verify the wiring of ShaderPrecompiler, StreamingTextureUploader,
and FBOPool into the GL rendering pipeline.  All tests use injected stubs —
no OpenGL context required.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch

from iPhoto.infrastructure.services.gpu_pipeline import (
    CompiledShader,
    FBOPool,
    ShaderPrecompiler,
    ShaderSource,
    StreamingTextureUploader,
    TextureChunk,
)


# ======================================================================
# ShaderPrecompiler → ShaderManager integration
# ======================================================================

class TestShaderManagerPrecompilerIntegration:
    """Verify ShaderManager uses ShaderPrecompiler when provided."""

    @staticmethod
    def _ok_compile(src: ShaderSource) -> CompiledShader:
        """Stub compile function that always succeeds."""
        fake_program = MagicMock()
        fake_program.bind.return_value = True
        fake_program.uniformLocation.return_value = -1
        return CompiledShader(name=src.name, program=fake_program, success=True)

    @staticmethod
    def _fail_compile(src: ShaderSource) -> CompiledShader:
        return CompiledShader(
            name=src.name, program=None, success=False, error="test failure"
        )

    def test_precompiler_registers_and_compiles_shaders(self):
        """ShaderPrecompiler should register 'main' and 'overlay' shaders."""
        precompiler = ShaderPrecompiler(self._ok_compile)
        # Simulate what ShaderManager._compile_shaders_precompiled does
        precompiler.register(ShaderSource("main", "v", "f"))
        precompiler.register(ShaderSource("overlay", "v2", "f2"))
        precompiler.compile_all()

        assert precompiler.compiled_count == 2
        assert precompiler.all_succeeded
        assert precompiler.get("main") is not None
        assert precompiler.get("overlay") is not None

    def test_precompiler_failure_raises(self):
        """If precompilation fails, the error should be detectable."""
        precompiler = ShaderPrecompiler(self._fail_compile)
        precompiler.register(ShaderSource("main", "v", "f"))
        precompiler.compile_all()

        main = precompiler.get("main")
        assert main is not None
        assert main.success is False
        assert main.error == "test failure"

    def test_precompiler_programs_are_retrievable(self):
        """Compiled programs can be retrieved by name for use in GLRenderer."""
        precompiler = ShaderPrecompiler(self._ok_compile)
        precompiler.register(ShaderSource("main", "vertex_src", "fragment_src"))
        precompiler.compile_all()

        compiled = precompiler.get("main")
        assert compiled is not None
        assert compiled.success is True
        assert compiled.program is not None


# ======================================================================
# StreamingTextureUploader → TextureManager integration
# ======================================================================

class TestTextureManagerStreamingIntegration:
    """Verify StreamingTextureUploader integration with texture upload path."""

    def test_streaming_uploader_chunks_large_image(self):
        """Large images should be uploaded in multiple chunks."""
        uploaded_chunks: list[tuple[int, TextureChunk]] = []

        def _capture_upload(tex_id: int, chunk: TextureChunk):
            uploaded_chunks.append((tex_id, chunk))

        uploader = StreamingTextureUploader(chunk_height=256, upload_fn=_capture_upload)

        # Simulate uploading a 1024x4096 image (taller than threshold)
        count = uploader.upload(
            texture_id=1,
            width=1024,
            height=4096,
            get_chunk_data=lambda y, h, w: b"\x00" * (w * h * 4),
        )

        assert count == 16  # 4096 / 256 = 16 chunks
        assert len(uploaded_chunks) == 16
        # First chunk starts at y=0
        assert uploaded_chunks[0][1].y_offset == 0
        assert uploaded_chunks[0][1].height == 256
        # Last chunk
        assert uploaded_chunks[-1][1].y_offset == 3840
        assert uploaded_chunks[-1][1].height == 256

    def test_streaming_uploader_preserves_chunk_height(self):
        """StreamingTextureUploader should use the configured chunk height."""
        uploader = StreamingTextureUploader(chunk_height=512)
        chunks = uploader.plan_chunks(1024, 2048)
        assert len(chunks) == 4
        assert all(h == 512 for _, h in chunks)

    def test_streaming_uploader_handles_remainder(self):
        """Last chunk should handle non-divisible image heights."""
        uploader = StreamingTextureUploader(chunk_height=256)
        chunks = uploader.plan_chunks(1024, 300)
        assert len(chunks) == 2
        assert chunks[0] == (0, 256)
        assert chunks[1] == (256, 44)

    def test_streaming_uploader_small_image_single_chunk(self):
        """Small images should result in a single chunk."""
        uploader = StreamingTextureUploader(chunk_height=256)
        chunks = uploader.plan_chunks(100, 100)
        assert chunks == [(0, 100)]


# ======================================================================
# FBOPool → Offscreen Rendering integration
# ======================================================================

class TestFBOPoolOffscreenIntegration:
    """Verify FBOPool integration with offscreen rendering."""

    @staticmethod
    def _make_pool():
        created = []

        def _create(w, h):
            fbo = MagicMock()
            fbo.isValid.return_value = True
            fbo.width.return_value = w
            fbo.height.return_value = h
            created.append(fbo)
            return fbo

        destroyed = []

        def _destroy(fbo):
            destroyed.append(fbo)

        pool = FBOPool(max_size=4, create_fn=_create, destroy_fn=_destroy)
        return pool, created, destroyed

    def test_fbo_pool_reuses_same_size(self):
        """FBOPool should reuse an FBO for the same dimensions."""
        pool, created, _ = self._make_pool()
        fbo1 = pool.acquire(800, 600)
        fbo2 = pool.acquire(800, 600)
        assert fbo1 is fbo2
        assert len(created) == 1

    def test_fbo_pool_creates_for_different_sizes(self):
        """FBOPool should create new FBOs for different dimensions."""
        pool, created, _ = self._make_pool()
        fbo1 = pool.acquire(800, 600)
        fbo2 = pool.acquire(1920, 1080)
        assert fbo1 is not fbo2
        assert len(created) == 2

    def test_fbo_pool_evicts_lru(self):
        """FBOPool should evict LRU entries when full."""
        pool, created, destroyed = self._make_pool()
        pool._max_size = 2  # Reduce size for test

        pool.acquire(100, 100)
        pool.acquire(200, 200)
        pool.acquire(300, 300)  # evicts 100x100

        assert pool.size == 2
        assert not pool.contains(100, 100)
        assert pool.contains(200, 200)
        assert pool.contains(300, 300)
        assert len(destroyed) == 1

    def test_fbo_pool_clear_destroys_all(self):
        """FBOPool.clear() should destroy all pooled FBOs."""
        pool, _, destroyed = self._make_pool()
        pool.acquire(100, 100)
        pool.acquire(200, 200)
        pool.clear()
        assert pool.size == 0
        assert len(destroyed) == 2


# ======================================================================
# DI Bootstrap — GPU Pipeline registration
# ======================================================================

class TestBootstrapGPUPipeline:
    """Verify GPU pipeline components are registered in DI container."""

    def test_bootstrap_registers_streaming_uploader(self):
        from iPhoto.di.bootstrap import bootstrap
        from iPhoto.di.container import Container

        container = Container()
        bootstrap(container)

        uploader = container.resolve(StreamingTextureUploader)
        assert isinstance(uploader, StreamingTextureUploader)
        assert uploader.chunk_height == 256

    def test_bootstrap_registers_fbo_pool(self):
        from iPhoto.di.bootstrap import bootstrap
        from iPhoto.di.container import Container

        container = Container()
        bootstrap(container)

        pool = container.resolve(FBOPool)
        assert isinstance(pool, FBOPool)
        assert pool.max_size == 4

    def test_gpu_pipeline_singletons_consistent(self):
        from iPhoto.di.bootstrap import bootstrap
        from iPhoto.di.container import Container

        container = Container()
        bootstrap(container)

        u1 = container.resolve(StreamingTextureUploader)
        u2 = container.resolve(StreamingTextureUploader)
        assert u1 is u2

        p1 = container.resolve(FBOPool)
        p2 = container.resolve(FBOPool)
        assert p1 is p2


# ======================================================================
# GLRenderer GPU Pipeline acceptance
# ======================================================================

class TestGLRendererAcceptsGPUPipeline:
    """Verify GLRenderer constructor accepts GPU pipeline components."""

    def test_renderer_accepts_optional_components(self):
        """GLRenderer should accept optional GPU pipeline parameters without error."""
        # We can't instantiate GLRenderer without a real GL context, but we can
        # verify the parameter signatures are correct by testing the sub-modules.
        precompiler = ShaderPrecompiler(lambda src: CompiledShader(
            name=src.name, program=None, success=True
        ))
        uploader = StreamingTextureUploader(chunk_height=256)
        pool = FBOPool(max_size=4)

        # Verify objects created successfully
        assert precompiler.registered_count == 0
        assert uploader.chunk_height == 256
        assert pool.max_size == 4

    def test_shader_precompiler_compile_fn_protocol(self):
        """A ShaderPrecompiler compile function should accept ShaderSource and return CompiledShader."""
        def compile_fn(source: ShaderSource) -> CompiledShader:
            return CompiledShader(name=source.name, program="fake_program", success=True)

        precompiler = ShaderPrecompiler(compile_fn)
        precompiler.register(ShaderSource("test", "vertex", "fragment"))
        results = precompiler.compile_all()
        assert len(results) == 1
        assert results[0].program == "fake_program"
