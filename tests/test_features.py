from pathlib import Path

import numpy as np

from cell_tracking.features import sliding_window_features


def test_sliding_window_features_shape(tmp_path: Path):
    """Test that enhanced scikit-image features have correct shape."""
    image = np.arange(25, dtype=np.uint8).reshape(5, 5).astype(np.float32)
    # Enhanced features: 25 raw pixels + 1 gradient magnitude = 26 features
    features = sliding_window_features(image, window_size=3, use_enhanced_features=True)
    assert features.shape == (25, 26)
    
    # Test raw pixel features (original behavior)
    raw_features = sliding_window_features(image, window_size=3, use_enhanced_features=False)
    assert raw_features.shape == (25, 9)
