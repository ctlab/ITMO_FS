from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import pytest

import ITMO_FS
from ITMO_FS import __version__
from ITMO_FS.base import BaseTransformer, BaseWrapper
from ITMO_FS.utils.base_transformer import BaseTransformer as LegacyBaseTransformer
from ITMO_FS.utils.base_wrapper import BaseWrapper as LegacyBaseWrapper


pytestmark = pytest.mark.smoke


def test_package_exposes_version():
    assert isinstance(__version__, str)
    assert __version__
    assert ITMO_FS.__version__ == __version__


def test_version_matches_version_file():
    version_path = Path(ITMO_FS.__file__).resolve().with_name("VERSION")
    assert version_path.read_text(encoding="utf-8").strip() == __version__


def test_legacy_base_imports_still_point_to_same_classes():
    assert LegacyBaseTransformer is BaseTransformer
    assert LegacyBaseWrapper is BaseWrapper


def test_about_module_can_be_loaded_without_importing_package_root():
    package_dir = Path(ITMO_FS.__file__).resolve().parent
    about_path = package_dir / "__about__.py"
    spec = spec_from_file_location("itmo_fs_about", about_path)
    module = module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module.__version__ == __version__
