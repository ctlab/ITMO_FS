import importlib.util
from pathlib import Path

import ITMO_FS
import pytest
from ITMO_FS.base import BaseTransformer, BaseWrapper
from ITMO_FS.utils import BaseTransformer as UtilsBaseTransformer
from ITMO_FS.utils import BaseWrapper as UtilsBaseWrapper
from ITMO_FS.utils.base_transformer import BaseTransformer as LegacyBaseTransformer
from ITMO_FS.utils.base_wrapper import BaseWrapper as LegacyBaseWrapper


pytestmark = pytest.mark.synthetic


def test_version_matches_version_file():
    version_path = Path(__file__).resolve().parents[2] / "ITMO_FS" / "VERSION"
    assert ITMO_FS.__version__ == version_path.read_text(encoding="utf-8").strip()


def test_base_class_imports_are_backward_compatible():
    assert BaseTransformer is UtilsBaseTransformer is LegacyBaseTransformer
    assert BaseWrapper is UtilsBaseWrapper is LegacyBaseWrapper


def test_package_import_is_loadable_from_init():
    assert hasattr(ITMO_FS, "UnivariateFilter")
    assert hasattr(ITMO_FS, "__version__")


def test_about_module_can_be_loaded_without_importing_package():
    about_path = Path(__file__).resolve().parents[2] / "ITMO_FS" / "__about__.py"
    spec = importlib.util.spec_from_file_location("itmo_fs_about", about_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module.__version__ == ITMO_FS.__version__
