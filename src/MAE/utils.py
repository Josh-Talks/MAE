from typing import Sequence, Tuple, Union


def ensure_tuple(x: Union[Sequence[int], int], dim: int) -> Tuple[int, ...]:
    """Ensure that the input is a tuple of given dimension.

    If the input is an integer, it will be converted to a tuple with the integer
    repeated `dim` times. If the input is already a sequence, it will be converted
    to a tuple.

    Args:
        x (Union[Sequence[int], int]): Input value to ensure as tuple.
        dim (int): Desired dimension of the output tuple.

    Returns:
        Tuple[int, ...]: Tuple of integers with length `dim`.
    """
    if isinstance(x, int):
        return (x,) * dim
    elif isinstance(x, Sequence):
        if len(x) != dim:
            raise ValueError(f"Input sequence must have length {dim}, but got {len(x)}")
        return tuple(x)
    else:
        raise TypeError("Input must be an integer or a sequence of integers.")
