STR_VERSION = '1.0.1'


def test_version():
    import layerlumos
    assert layerlumos.__version__ == STR_VERSION

def test_version_setup():
    import importlib
    assert importlib.metadata.version('layerlumos') == STR_VERSION
