from collections import namedtuple

AVX_8x1x1 = namedtuple("AVX_8x1x1", ["x1", "y1", "z1", "w1", "x2", "y2", "z2", "w2"])
AVX_4x2x1 = namedtuple("AVX_4x2x1", ["x", "y", "z", "w"])
