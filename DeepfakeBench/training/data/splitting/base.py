"""
Base classes for data splitting.

Defines the Splitter interface and SplitResult dataclass that all
splitter implementations must use.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List, Dict, Any, Union, Tuple

from .video_info import VideoInfo


@dataclass
class SplitResult:
    """
    Result of a data splitting operation.
    
    Attributes:
        train_data: Training data (format depends on splitter)
        val_in_dist: In-distribution validation videos
        val_holdout: Holdout validation videos (unseen methods)
        stats: Statistics about the split
    """
    train_data: Union[List[VideoInfo], List[Dict[str, Any]]]
    val_in_dist: List[VideoInfo]
    val_holdout: List[VideoInfo]
    stats: Dict[str, Any] = field(default_factory=dict)
    
    def __iter__(self):
        """Allow unpacking: train, val_in, val_out, stats = split_result"""
        return iter((self.train_data, self.val_in_dist, self.val_holdout, self.stats))


class Splitter(ABC):
    """
    Abstract base class for data splitters.
    
    Implementations must provide a split() method that takes configuration
    and returns a SplitResult with train/val data.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the splitter with configuration.
        
        Args:
            config: Data configuration dictionary containing:
                - gcp.bucket_name: GCS bucket for data
                - data_params.seed: Random seed
                - data_params.val_split_ratio: Validation split ratio
                - methods.*: Method selection configuration
        """
        self.config = config
        self.seed = config.get('data_params', {}).get('seed', 42)
    
    @abstractmethod
    def split(self) -> SplitResult:
        """
        Perform the data split.
        
        Returns:
            SplitResult containing train/val data and statistics
        """
        pass
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Name of this splitter strategy."""
        pass
    
    def _get_method_sets(self) -> Tuple[set, set, set, set]:
        """
        Extract method sets from configuration.
        
        Returns:
            Tuple of (real_methods, train_fake_methods, val_fake_methods, val_only_real_methods)
        """
        methods = self.config.get('methods', {}) or self.config.get('dataset_methods', {})
        
        real_methods = set(methods.get('use_real_sources', []))
        train_fake_methods = set(methods.get('use_fake_methods_for_training', []))
        val_fake_methods = set(methods.get('use_fake_methods_for_validation', []))
        val_only_real_methods = set(methods.get('use_real_methods_for_validation_only', []))
        
        return real_methods, train_fake_methods, val_fake_methods, val_only_real_methods
