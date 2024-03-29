def test_import_layerlumos():
    import layerlumos

def test_import():
    from layerlumos import stackrt
    from layerlumos import stackrt0

    assert callable(stackrt)
    assert callable(stackrt0)

    from layerlumos.layerlumos import stackrt
    from layerlumos.layerlumos import stackrt0

    assert callable(stackrt)
    assert callable(stackrt0)

    import layerlumos.utils_spectra
    import layerlumos.utils_materials
