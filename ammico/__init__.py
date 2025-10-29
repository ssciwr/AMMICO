from ammico.display import AnalysisExplorer
from ammico.faces import EmotionDetector, ethical_disclosure
from ammico.model import MultimodalSummaryModel
from ammico.text import TextDetector, TextAnalyzer, privacy_disclosure
from ammico.image_summary import ImageSummaryDetector
from ammico.utils import find_files, get_dataframe

# Export the version defined in project metadata
try:
    from importlib.metadata import version

    __version__ = version("ammico")
except ImportError:
    __version__ = "unknown"

__all__ = [
    "AnalysisExplorer",
    "EmotionDetector",
    "MultimodalSummaryModel",
    "TextDetector",
    "TextAnalyzer",
    "ImageSummaryDetector",
    "find_files",
    "get_dataframe",
    "ethical_disclosure",
    "privacy_disclosure",
]
