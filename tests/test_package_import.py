def test_import_project_modules():
    import src.config
    import src.data.verify_local_data
    import src.models.common
    import src.evaluation.evaluate_models

    assert src.config.DATA_DIR.name == "data"
    assert hasattr(src.data.verify_local_data, "verify_local_data")
    assert hasattr(src.models.common, "get_classifier")
    assert hasattr(src.evaluation.evaluate_models, "evaluate_model")
