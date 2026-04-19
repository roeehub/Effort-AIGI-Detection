# Data module for training pipeline
# Contains augmentations, splitting strategies, batching, data loaders, and data sources

from . import augmentations
from . import splitting
from . import batching
from . import sources

# Re-export the main data source factory for convenience
from .sources import create_data_pipeline, DataPipelineResult

__all__ = [
    'augmentations', 
    'splitting', 
    'batching', 
    'sources',
    'create_data_pipeline',
    'DataPipelineResult',
]
