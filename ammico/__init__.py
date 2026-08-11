from ammico.display import AnalysisExplorer
from ammico.image_summary import ImageSummaryDetector
from ammico.inference import AudioTranscriptionModel, InferenceModel
from ammico.model import MultimodalEmbeddingsModel
from ammico.multimodal_search import MultimodalSearch
from ammico.text import TextAnalyzer, TextDetector, privacy_disclosure
from ammico.utils import AnalysisType, find_files, find_videos, get_dataframe
from ammico.video_summary import VideoSummaryDetector

# Export the version defined in project metadata
try:
    from importlib.metadata import version

    __version__ = version("ammico")
except ImportError:
    __version__ = "unknown"

__all__ = [
    "AnalysisExplorer",
    "AnalysisType",
    "AudioTranscriptionModel",
    "ImageSummaryDetector",
    "InferenceModel",
    "MultimodalEmbeddingsModel",
    "MultimodalSearch",
    "TextAnalyzer",
    "TextDetector",
    "VideoSummaryDetector",
    "find_files",
    "find_videos",
    "get_dataframe",
    "privacy_disclosure",
]
