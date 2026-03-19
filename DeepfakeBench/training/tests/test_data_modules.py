"""
Tests for data modules (splitting, batching, augmentations).
"""
import pytest


class TestSplitting:
    """Tests for the data.splitting module."""
    
    def test_get_splitter_exists(self):
        """Test get_splitter factory function exists."""
        from data.splitting import get_splitter
        assert callable(get_splitter)
    
    def test_splitter_strategies(self):
        """Test available splitter strategies."""
        from data.splitting.splitters import LegacySplitter, PropertyBasedSplitter
        
        assert LegacySplitter is not None
        assert PropertyBasedSplitter is not None
    
    def test_split_result_dataclass(self):
        """Test SplitResult dataclass."""
        from data.splitting import SplitResult
        
        result = SplitResult(
            train_data=[{'a': 1}],
            val_in_dist=[],
            val_holdout=[],
            stats={'test': 1}
        )
        
        assert result.train_data == [{'a': 1}]
        assert result.stats['test'] == 1
    
    def test_split_result_unpacking(self):
        """Test SplitResult can be unpacked."""
        from data.splitting import SplitResult
        
        result = SplitResult(
            train_data=['train'],
            val_in_dist=['val_in'],
            val_holdout=['val_out'],
            stats={'count': 3}
        )
        
        train, val_in, val_out, stats = result
        
        assert train == ['train']
        assert val_in == ['val_in']
        assert val_out == ['val_out']
        assert stats == {'count': 3}
    
    def test_video_info_creation(self):
        """Test VideoInfo creation."""
        from data.splitting import VideoInfo
        
        # VideoInfo requires: label, method, video_id, frame_paths, identity
        video = VideoInfo(
            label='real',
            method='FaceForensics++',
            video_id='video_001',
            frame_paths=['gs://bucket/frame1.jpg', 'gs://bucket/frame2.jpg'],
            identity=123,
        )
        
        assert video.label == 'real'
        assert video.method == 'FaceForensics++'
        assert len(video.frame_paths) == 2
        assert video.identity == 123
        assert video.label_id == 0  # real = 0
    
    def test_video_info_fake_label(self):
        """Test VideoInfo with fake label."""
        from data.splitting import VideoInfo
        
        video = VideoInfo(
            label='fake',
            method='Deepfakes',
            video_id='video_002',
            frame_paths=['gs://bucket/frame1.jpg'],
            identity=456,
        )
        
        assert video.label == 'fake'
        assert video.label_id == 1  # fake = 1
    
    def test_video_info_invalid_label_raises(self):
        """Test VideoInfo raises on invalid label."""
        from data.splitting import VideoInfo
        
        with pytest.raises(ValueError):
            VideoInfo(
                label='invalid',  # Should be 'real' or 'fake'
                method='method',
                video_id='vid',
                frame_paths=['gs://bucket/frame.jpg'],
                identity=0,
            )
    
    def test_video_info_empty_frames_raises(self):
        """Test VideoInfo raises on empty frame_paths."""
        from data.splitting import VideoInfo
        
        with pytest.raises(ValueError):
            VideoInfo(
                label='real',
                method='method',
                video_id='vid',
                frame_paths=[],  # Empty
                identity=0,
            )
    
    def test_video_info_num_frames(self):
        """Test VideoInfo.num_frames property."""
        from data.splitting import VideoInfo
        
        video = VideoInfo(
            label='real',
            method='method',
            video_id='vid',
            frame_paths=['f1.jpg', 'f2.jpg', 'f3.jpg'],
            identity=0,
        )
        
        assert video.num_frames == 3


class TestBatching:
    """Tests for the data.batching module."""
    
    def test_batching_imports(self):
        """Test batching module imports work."""
        from data.batching import (
            BatchingStrategy,
            BatchingStrategyConfig,
            get_batching_strategy,
            create_dataloaders,
        )
        
        assert BatchingStrategy is not None
        assert BatchingStrategyConfig is not None
        assert callable(get_batching_strategy)
        assert callable(create_dataloaders)
    
    def test_strategy_classes_exist(self):
        """Test strategy classes are importable."""
        from data.batching import (
            PerMethodStrategy,
            VideoLevelStrategy,
            FrameLevelStrategy,
            PropertyBalancedStrategy,
        )
        
        assert PerMethodStrategy is not None
        assert VideoLevelStrategy is not None
        assert FrameLevelStrategy is not None
        assert PropertyBalancedStrategy is not None
    
    def test_datapipes_exist(self):
        """Test custom datapipes are importable."""
        from data.batching import (
            CustomRoundRobinDataPipe,
            CustomSampleMultiplexerDataPipe,
            MateFinderDataPipe,
        )
        
        assert CustomRoundRobinDataPipe is not None
        assert CustomSampleMultiplexerDataPipe is not None
        assert MateFinderDataPipe is not None
    
    def test_loader_functions_exist(self):
        """Test loader functions are importable."""
        from data.batching import (
            load_and_process_frame_batch,
            load_and_process_property_batch,
            load_and_process_video,
            collate_fn,
            collate_fn_detailed,
        )
        
        assert callable(load_and_process_frame_batch)
        assert callable(load_and_process_property_batch)
        assert callable(load_and_process_video)
        assert callable(collate_fn)
        assert callable(collate_fn_detailed)


class TestAugmentations:
    """Tests for the data.augmentations module."""
    
    def test_augmentation_imports(self):
        """Test augmentation module imports work."""
        from data.augmentations import (
            get_pipeline,
            PIPELINE_REGISTRY,
            register_pipeline,
            CustomUnsharpMask,
            NoOp,
        )
        
        assert callable(get_pipeline)
        assert isinstance(PIPELINE_REGISTRY, dict)
        assert callable(register_pipeline)
        assert CustomUnsharpMask is not None
        assert NoOp is not None
    
    def test_pipeline_registry_has_versions(self):
        """Test pipeline registry has expected versions."""
        from data.augmentations import PIPELINE_REGISTRY
        
        # Check for numeric versions (3, 4, 5, 6, 7) and string versions
        expected_keys = [3, 4, 5, 6, 7, 'surgical']
        
        for key in expected_keys:
            assert key in PIPELINE_REGISTRY, f"Missing pipeline: {key}"
    
    def test_get_pipeline_basic(self):
        """Test get_pipeline returns callable for basic versions."""
        from data.augmentations import get_pipeline
        
        # V3 should return an albumentations pipeline
        pipeline = get_pipeline(version=3)
        assert pipeline is not None
    
    def test_custom_unsharp_mask(self):
        """Test CustomUnsharpMask transform."""
        from data.augmentations import CustomUnsharpMask
        
        transform = CustomUnsharpMask()
        assert transform is not None
        assert hasattr(transform, 'get_params')
    
    def test_no_op_transform(self):
        """Test NoOp transform."""
        from data.augmentations import NoOp
        
        transform = NoOp()
        assert transform is not None


class TestUtils:
    """Tests for the utils module."""
    
    def test_init_seed(self):
        """Test init_seed function."""
        from utils import init_seed
        
        # Should not raise
        config = {'manualSeed': 42, 'cuda': False}
        init_seed(config)
    
    def test_choose_metric(self):
        """Test choose_metric function."""
        from utils import choose_metric
        
        # choose_metric expects a config dict with 'metric_scoring' key
        config_auc = {'metric_scoring': 'auc'}
        result = choose_metric(config_auc)
        assert result == 'auc'
        
        config_acc = {'metric_scoring': 'acc'}
        result = choose_metric(config_acc)
        assert result == 'acc'
    
    def test_choose_optimizer(self):
        """Test choose_optimizer function."""
        from utils import choose_optimizer
        
        assert callable(choose_optimizer)
    
    def test_choose_scheduler(self):
        """Test choose_scheduler function."""
        from utils import choose_scheduler
        
        assert callable(choose_scheduler)
