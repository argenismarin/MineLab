"""3D regular block model class and grade-tonnage analysis.

Provides the BlockModel class for managing block-based data and
functions for grade-tonnage curve computation.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from minelab.utilities.validators import validate_positive


class BlockModel:
    """3D regular block model.

    Parameters
    ----------
    origin : array-like
        Origin coordinates (x_min, y_min, z_min).
    block_size : array-like
        Block dimensions (dx, dy, dz).
    n_blocks : array-like
        Number of blocks per dimension (nx, ny, nz).

    Attributes
    ----------
    origin : np.ndarray
    block_size : np.ndarray
    n_blocks : np.ndarray
    n_total : int
        Total number of blocks.

    Examples
    --------
    >>> bm = BlockModel([0, 0, 0], [10, 10, 5], [10, 10, 5])
    >>> bm.n_total
    500
    >>> centers = bm.block_centers()
    >>> centers.shape
    (500, 3)

    References
    ----------
    .. [1] Sinclair, A.J. & Blackwell, G.H. (2002). "Applied Mineral Inventory
       Estimation." Cambridge University Press.
    """

    def __init__(
        self,
        origin: list | np.ndarray,
        block_size: list | np.ndarray,
        n_blocks: list | np.ndarray,
    ) -> None:
        self.origin = np.asarray(origin, dtype=float)
        self.block_size = np.asarray(block_size, dtype=float)
        self.n_blocks = np.asarray(n_blocks, dtype=int)

        if len(self.origin) != 3 or len(self.block_size) != 3 or len(self.n_blocks) != 3:
            raise ValueError("origin, block_size, n_blocks must each have 3 elements.")

        for i, sz in enumerate(self.block_size):
            validate_positive(sz, f"block_size[{i}]")
        for i, nb in enumerate(self.n_blocks):
            if nb < 1:
                raise ValueError(f"n_blocks[{i}] must be >= 1, got {nb}.")

        self.n_total: int = int(np.prod(self.n_blocks))
        self._variables: dict[str, np.ndarray] = {}

    def block_centers(self) -> np.ndarray:
        """Return the center coordinates of all blocks.

        Returns
        -------
        np.ndarray
            Shape (n_total, 3).
        """
        axes = []
        for d in range(3):
            axes.append(self.origin[d] + self.block_size[d] * (np.arange(self.n_blocks[d]) + 0.5))
        gx, gy, gz = np.meshgrid(*axes, indexing="ij")
        return np.column_stack([gx.ravel(), gy.ravel(), gz.ravel()])

    def add_variable(self, name: str, data: np.ndarray) -> None:
        """Add a variable (grade, density, etc.) to the block model.

        Parameters
        ----------
        name : str
            Variable name.
        data : np.ndarray
            Values, shape (n_total,).

        Raises
        ------
        ValueError
            If data length doesn't match n_total.
        """
        data = np.asarray(data, dtype=float)
        if len(data) != self.n_total:
            raise ValueError(f"Data length {len(data)} != n_total {self.n_total}.")
        self._variables[name] = data

    def get_variable(self, name: str) -> np.ndarray:
        """Retrieve a variable by name.

        Parameters
        ----------
        name : str
            Variable name.

        Returns
        -------
        np.ndarray
            Values, shape (n_total,).

        Raises
        ------
        KeyError
            If variable not found.
        """
        if name not in self._variables:
            raise KeyError(f"Variable '{name}' not found in block model.")
        return self._variables[name]

    def filter_blocks(self, variable: str, condition: str, threshold: float) -> np.ndarray:
        """Return indices of blocks matching a condition.

        Parameters
        ----------
        variable : str
            Variable name to filter on.
        condition : str
            One of ``">"``, ``">="``, ``"<"``, ``"<="``, ``"=="``.
        threshold : float
            Threshold value.

        Returns
        -------
        np.ndarray
            Array of block indices matching the condition.
        """
        data = self.get_variable(variable)
        ops = {
            ">": np.greater,
            ">=": np.greater_equal,
            "<": np.less,
            "<=": np.less_equal,
            "==": np.equal,
        }
        if condition not in ops:
            raise ValueError(f"Unknown condition '{condition}'. Use: {list(ops.keys())}")
        mask = ops[condition](data, threshold)
        return np.where(mask)[0]

    @property
    def variables(self) -> list[str]:
        """List of variable names stored in the block model."""
        return list(self._variables.keys())


def block_grade_tonnage(
    block_model: BlockModel,
    grade_var: str,
    density_var: str,
    cutoffs: np.ndarray | list[float],
) -> pd.DataFrame:
    """Compute grade-tonnage curve from a block model.

    Parameters
    ----------
    block_model : BlockModel
        Block model with grade and density variables.
    grade_var : str
        Name of the grade variable.
    density_var : str
        Name of the density variable (t/m³).
    cutoffs : array-like
        Cutoff grades to evaluate.

    Returns
    -------
    pd.DataFrame
        Columns: ``"cutoff"``, ``"tonnage"``, ``"mean_grade"``, ``"metal"``.

    Examples
    --------
    >>> bm = BlockModel([0, 0, 0], [10, 10, 10], [5, 5, 2])
    >>> rng = np.random.default_rng(42)
    >>> bm.add_variable("grade", rng.lognormal(0, 0.5, bm.n_total))
    >>> bm.add_variable("density", np.full(bm.n_total, 2.7))
    >>> gt = block_grade_tonnage(bm, "grade", "density", [0.5, 1.0, 1.5])
    >>> gt["tonnage"].is_monotonic_decreasing
    True

    References
    ----------
    .. [1] Standard mining engineering practice.
    """
    grades = block_model.get_variable(grade_var)
    density = block_model.get_variable(density_var)
    cutoffs = np.asarray(cutoffs, dtype=float)

    block_vol = float(np.prod(block_model.block_size))
    block_tonnes = density * block_vol

    rows = []
    for cog in cutoffs:
        mask = grades >= cog
        tonnes = float(np.sum(block_tonnes[mask]))
        if np.sum(mask) > 0:
            mean_grade = float(np.average(grades[mask], weights=block_tonnes[mask]))
        else:
            mean_grade = 0.0
        metal = tonnes * mean_grade
        rows.append(
            {
                "cutoff": float(cog),
                "tonnage": tonnes,
                "mean_grade": mean_grade,
                "metal": metal,
            }
        )

    return pd.DataFrame(rows)
