from ammico.display import AnalysisExplorer
from ammico.model import MultimodalSummaryModel
from ammico.text import TextDetector, TextAnalyzer, privacy_disclosure
from ammico.image_summary import ImageSummaryDetector
from ammico.utils import find_files, get_dataframe, AnalysisType, find_videos
from ammico.video_summary import VideoSummaryDetector

# Export the version defined in project metadata
try:
    from importlib.metadata import version

    __version__ = version("ammico")
except ImportError:
    __version__ = "unknown"

__all__ = [
    "AnalysisType",
    "AnalysisExplorer",
    "MultimodalSummaryModel",
    "TextDetector",
    "TextAnalyzer",
    "ImageSummaryDetector",
    "VideoSummaryDetector",
    "find_files",
    "find_videos",
    "get_dataframe",
    "privacy_disclosure",
]
