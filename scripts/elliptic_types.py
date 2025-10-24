from collections import namedtuple

Curve = namedtuple("Curve", ["A", "C", "A24", "C24"])

PointX = namedtuple("PointX", ["X", "Z"])
BasisX = namedtuple("BasisX", ["PX", "PZ", "QX", "QZ", "PmQX", "PmQZ"])

PointJ = namedtuple("PointJ", ["X", "Y", "Z"])
PointW = namedtuple("PointW", ["X", "Y", "Z"])
