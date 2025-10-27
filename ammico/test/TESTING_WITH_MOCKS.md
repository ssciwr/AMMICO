# Testing with Mock Models

This document explains how to use the mock model fixture to write fast unit tests that don't require loading the actual model.

## Mock Model Fixture

A `mock_model` fixture has been added to `conftest.py` that creates a lightweight mock of the `MultimodalSummaryModel` class. This fixture:

- **Does not load any actual models** (super fast)
- **Mocks all necessary methods** (processor, tokenizer, model.generate, etc.)
- **Returns realistic tensor shapes** (so the code doesn't crash)
- **Can be used for fast unit tests** that don't need actual model inference

## Usage

Simply use `mock_model` instead of `model` in your test fixtures:

```python
def test_my_feature(mock_model):
    detector = ImageSummaryDetector(summary_model=mock_model, subdict={})
    # Your test code here
    pass
```

## When to Use Mock vs Real Model

### Use `mock_model` when:
- Testing utility functions (like `_clean_list_of_questions`)
- Testing input validation logic
- Testing data processing methods
- Testing class initialization
- **Any test that doesn't need actual model inference**

### Use `model` (real model) when:
- Testing end-to-end functionality
- Testing actual caption generation quality
- Testing actual question answering
- Integration tests that verify model behavior
- **Any test marked with `@pytest.mark.long`**

## Example Tests Added

The following new tests use the mock model:

1. `test_image_summary_detector_init_mock` - Tests initialization
2. `test_load_pil_if_needed_string` - Tests image loading
3. `test_is_sequence_but_not_str` - Tests utility methods
4. `test_validate_analysis_type` - Tests validation logic

All of these run quickly without loading the model.

## Running Tests

### Run only fast tests (with mocks):
```bash
pytest ammico/test/test_image_summary.py -v
```

### Run only long tests (with real model):
```bash
pytest ammico/test/test_image_summary.py -m long -v
```

### Run all tests:
```bash
pytest ammico/test/test_image_summary.py -v
```

## Customizing the Mock

If you need to customize the mock's behavior for specific tests, you can override its methods:

```python
def test_custom_behavior(mock_model):
    # Customize the mock's return value
    mock_model.tokenizer.batch_decode.return_value = ["custom", "output"]
    
    detector = ImageSummaryDetector(summary_model=mock_model, subdict={})
    # Test with custom behavior
    pass
```

