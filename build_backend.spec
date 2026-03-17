# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

from PyInstaller.utils.hooks import copy_metadata, collect_data_files
import os, sys, tempfile

datas = [
    ('pantheon/factory/templates', 'pantheon/factory/templates'),
    ('pantheon/toolsets/database_api/schemas', 'pantheon/toolsets/database_api/schemas'),
    ('pantheon/chatroom/nats-ws.conf', 'pantheon/chatroom'),
    ('pantheon/toolsets/knowledge/config.yaml', 'pantheon/toolsets/knowledge'),
]
datas += copy_metadata('fastmcp')
datas += collect_data_files('litellm', includes=['**/*.json'])
# fakeredis: model/_command_info.py loads os.path.join(dirname(__file__), '..', 'commands.json')
# PyInstaller must include the JSON so the relative path resolves at runtime.
datas += collect_data_files('fakeredis', includes=['commands.json'])

# Runtime hook: ensure fakeredis/model/ dir exists so ../commands.json resolves.
_rthook = tempfile.NamedTemporaryFile('w', suffix='.py', delete=False)
_rthook.write(
    "import sys, os\n"
    "d = getattr(sys, '_MEIPASS', None)\n"
    "if d:\n"
    "    os.makedirs(os.path.join(d, 'fakeredis', 'model'), exist_ok=True)\n"
)
_rthook.close()

a = Analysis(
    ['pantheon/chatroom/__main__.py'],
    pathex=[],
    binaries=[],
    datas=datas,
    hiddenimports=[
        'pantheon',
        'pantheon.chatroom',
        'pantheon.endpoint',
        'pantheon.remote',
        'pantheon.toolsets',
        # All toolsets (dynamically imported via importlib)
        'pantheon.toolsets.code',
        'pantheon.toolsets.shell',
        'pantheon.toolsets.python',
        'pantheon.toolsets.file',
        'pantheon.toolsets.file_transfer',
        'pantheon.toolsets.notebook',
        'pantheon.toolsets.image',
        'pantheon.toolsets.database_api',
        'pantheon.toolsets.browser_use',
        'pantheon.toolsets.evolution',
        'pantheon.toolsets.task',
        'pantheon.toolsets.rag',
        'pantheon.toolsets.scfm',
        'nats',
        'litellm',
        'openai',
        'anthropic',
        'fastmcp',
        'fastmcp.server',
        'fastmcp.client',
        'lupa', 'lupa.lua51',
        'importlib.metadata',
        'importlib_metadata',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[_rthook.name],
    excludes=[
        # ── Knowledge / vector DB (lancedb only used in RAG toolset, lazy import) ──
        'lancedb', 'lance', 'pyarrow', 'llama_index', 'qdrant_client',
        # ── Heavy transitive deps not directly imported ──
        'scipy', 'scipy.libs',           # ~140MB, only SCFM toolset (lazy)
        'patchright',                     # ~133MB, crawl4ai transitive
        'networkx',                       # ~18MB, transitive
        'fontTools',                      # ~27MB, matplotlib transitive
        'shapely', 'shapely.libs',        # ~12MB, transitive
        'alphashape',                     # transitive
        'nltk', 'nltk_data',             # ~13MB, transitive
        # ── Dev / build tools ──
        'debugpy',                        # ~23MB, dev only
        'pytest', 'pytest_asyncio',
        'pip', 'setuptools', 'pkg_resources',
        'PyInstaller', '_pyinstaller_hooks_contrib',
        # ── Unused heavy modules ──
        'tkinter', '_tkinter',
        'torch', 'tensorflow', 'sklearn', 'cv2',
        'IPython', 'ipywidgets',
        'sympy',
        'pandas',                         # ~45MB, not imported by backend code
        'litellm.proxy',                  # ~19MB, proxy server not used (SDK only)
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    name='pantheon-backend-exe',
    debug=False,
    bootloader_ignore_signals=False,
    strip=sys.platform != 'win32',  # MinGW strip corrupts MSVC-built DLLs on Windows
    upx=True,
    upx_exclude=[
        'python3*.dll',
        'vcruntime*.dll',
        'msvcp*.dll',
        'ucrtbase.dll',
        'api-ms-win-*.dll',
    ],
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# COLLECT renames the exe back to 'pantheon-backend' inside the output directory
coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=sys.platform != 'win32',
    upx=True,
    upx_exclude=[
        'python3*.dll',
        'vcruntime*.dll',
        'msvcp*.dll',
        'ucrtbase.dll',
        'api-ms-win-*.dll',
    ],
    name='pantheon-backend',
)
