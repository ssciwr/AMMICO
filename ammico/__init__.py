try:
    from importlib import metadata
except ImportError:
    # Running on pre-3.8 Python; use importlib-metadata package
    import importlib_metadata as metadata  # type: ignore
from ammico.cropposts import crop_media_posts, crop_posts_from_refs
from ammico.display import AnalysisExplorer
from ammico.faces import EmotionDetector
from ammico.multimodal_search import MultimodalSearch
from ammico.objects import ObjectDetector
from ammico.summary import SummaryDetector
from ammico.text import TextDetector, PostprocessText
from ammico.utils import find_files, get_dataframe

# Export the version defined in project metadata
__version__ = metadata.version(__package__)
del metadata

__all__ = [
    "crop_media_posts",
    "crop_posts_from_refs",
    "AnalysisExplorer",
    "EmotionDetector",
    "MultimodalSearch",
    "ObjectDetector",
    "SummaryDetector",
    "TextDetector",
    "PostprocessText",
    "find_files",
    "get_dataframe",
]
