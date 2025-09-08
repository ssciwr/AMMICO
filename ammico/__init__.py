try:
    from importlib import metadata
except ImportError:
    # Running on pre-3.8 Python; use importlib-metadata package
    import importlib_metadata as metadata  # type: ignore
from ammico.display import AnalysisExplorer
from ammico.faces import EmotionDetector, ethical_disclosure
from ammico.text import TextDetector, TextAnalyzer, privacy_disclosure
from ammico.utils import find_files, get_dataframe

# Export the version defined in project metadata
__version__ = metadata.version(__package__)
del metadata

__all__ = [
    "AnalysisExplorer",
    "EmotionDetector",
    "TextDetector",
    "TextAnalyzer",
    "find_files",
    "get_dataframe",
    "ethical_disclosure",
    "privacy_disclosure",
]
