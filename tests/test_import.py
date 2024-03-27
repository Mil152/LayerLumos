def test_import_layerlumos():
    import layerlumos

def test_import():
    from layerlumos import stackrt
    from layerlumos import stackrt0

    assert callable(stackrt)
    assert callable(stackrt0)
