from sqlalchemy.util import OrderedSet


def append_intersection(a, b):
    return list(OrderedSet(a).difference(b)) + list(OrderedSet(a).intersection(b))
