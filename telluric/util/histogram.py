import numpy as np


class HistogramStretchingError(Exception):
    """Base class for exceptions in histogram stretching."""

    pass


def stretch_histogram(img, dark_clip_percentile=None, bright_clip_percentile=None,
                      dark_clip_value=None, bright_clip_value=None, ignore_zero=True):
    """Stretch img histogram.

    2 possible modes: by percentile (pass dark/bright_clip_percentile), or by value (pass dark/bright_clip_value)
    :param dark_clip_percentile: percent of pixels that will be saturated to min_value
    :param bright_clip_percentile: percent of pixels that will be saturated to max_value
    :param dark_clip_value: all values below this will be saturated to min_value
    :param bright_clip_value: all values above this will be saturated to max_value
    :param ignore_zero: if true, pixels with value 0 are ignored in stretch calculation
    :returns image (same shape as 'img')
    """
    # verify stretching method is specified:
    if (dark_clip_percentile is not None and dark_clip_value is not None) or \
            (bright_clip_percentile is not None and bright_clip_value is not None):
        raise KeyError('Provided parameters for both by-percentile and by-value stretch, need only one of those.')

    # the default stretching:
    if dark_clip_percentile is None and dark_clip_value is None:
        dark_clip_percentile = 0.001
    if bright_clip_percentile is None and bright_clip_value is None:
        bright_clip_percentile = 0.001

    if dark_clip_percentile is not None:
        dark_clip_value = np.percentile(img[img != 0] if ignore_zero else img, 100 * dark_clip_percentile)
    if bright_clip_percentile is not None:
        bright_clip_value = np.percentile(img[img != 0] if ignore_zero else img, 100 * (1 - bright_clip_percentile))

    dst_min = np.iinfo(img.dtype).min
    dst_max = np.iinfo(img.dtype).max

    if bright_clip_value == dark_clip_value:
        raise HistogramStretchingError
    gain = (dst_max - dst_min) / (bright_clip_value - dark_clip_value)
    offset = -gain * dark_clip_value + dst_min

    stretched = np.empty_like(img, dtype=img.dtype)
    if len(img.shape) == 2:
        stretched[:, :] = np.clip(gain * img[:, :].astype(np.float32) + offset, dst_min, dst_max).astype(img.dtype)
    else:
        for band in range(img.shape[0]):
            stretched[band, :, :] = np.clip(gain * img[band, :, :].astype(np.float32) + offset,
                                            dst_min, dst_max).astype(img.dtype)
    return stretched
