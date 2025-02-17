import collections
import collections.abc as cabc
import copy
import enum
import inspect
import types
import typing as typ
import warnings

import numpy as np
import scipy.sparse.linalg as spsl

import pyxu.info.deps as pxd
import pyxu.info.ptype as pxt
import pyxu.info.warning as pxw
import pyxu.runtime as pxrt
import pyxu.util as pxu


class Property(enum.Enum):
    """
    Mathematical properties.
    """

    CAN_EVAL = enum.auto()
    FUNCTIONAL = enum.auto()
    PROXIMABLE = enum.auto()
    DIFFERENTIABLE = enum.auto()
    DIFFERENTIABLE_FUNCTION = enum.auto()
    LINEAR = enum.auto()
    LINEAR_SQUARE = enum.auto()
    LINEAR_NORMAL = enum.auto()
    LINEAR_IDEMPOTENT = enum.auto()
    LINEAR_SELF_ADJOINT = enum.auto()
    LINEAR_POSITIVE_DEFINITE = enum.auto()
    LINEAR_UNITARY = enum.auto()
    QUADRATIC = enum.auto()

    def arithmetic_methods(self) -> cabc.Set[str]:
        "Instance methods affected by arithmetic operations."
        data = collections.defaultdict(list)
        data[self.CAN_EVAL].extend(
            [
                "apply",
                "__call__",
                "estimate_lipschitz",
                "_expr",
            ]
        )
        data[self.FUNCTIONAL].append("asloss")
        data[self.PROXIMABLE].append("prox")
        data[self.DIFFERENTIABLE].extend(
            [
                "jacobian",
                "estimate_diff_lipschitz",
            ]
        )
        data[self.DIFFERENTIABLE_FUNCTION].append("grad")
        data[self.LINEAR].extend(
            [
                "adjoint",
                "asarray",
                "svdvals",
                "pinv",
                "gram",
                "cogram",
            ]
        )
        data[self.LINEAR_SQUARE].append("trace")
        data[self.QUADRATIC].append("_quad_spec")

        meth = frozenset(data[self])
        return meth


class Operator:
    """
    Abstract Base Class for Pyxu operators.

    Goals:

    * enable operator arithmetic.
    * cast operators to specialized forms.
    * attach :py:class:`~pyxu.abc.operator.Property` tags encoding certain mathematical properties.
      Each core sub-class **must** have a unique set of properties to be distinguishable from its peers.
    """

    # For `(size-1 ndarray) * OpT` to work, we need to force NumPy's hand and call OpT.__rmul__() in
    # place of ndarray.__mul__() to determine how scaling should be performed.
    # This is achieved by increasing __array_priority__ for all operators.
    __array_priority__ = np.inf

    def __init__(self, shape: pxt.OpShape):
        r"""
        Parameters
        ----------
        shape: OpShape
            (N, M) operator shape.
        """
        shape = pxu.as_canonical_shape(shape)
        assert len(shape) == 2, f"shape: expected {pxt.OpShape}, got {shape}."

        self._shape = shape
        self._name = self.__class__.__name__

    # Public Interface --------------------------------------------------------
    @property
    def shape(self) -> pxt.OpShape:
        r"""
        Return (N, M) operator shape.
        """
        return self._shape

    @property
    def dim(self) -> pxt.Integer:
        r"""
        Return dimension of operator's domain. (M)
        """
        return self.shape[1]

    @property
    def codim(self) -> pxt.Integer:
        r"""
        Return dimension of operator's co-domain. (N)
        """
        return self.shape[0]

    @classmethod
    def properties(cls) -> cabc.Set[Property]:
        "Mathematical properties of the operator."
        return frozenset()

    @classmethod
    def has(cls, prop: typ.Union[Property, cabc.Collection[Property]]) -> bool:
        """
        Verify if operator possesses supplied properties.
        """
        if isinstance(prop, Property):
            prop = (prop,)
        return frozenset(prop) <= cls.properties()

    def asop(self, cast_to: pxt.OpC) -> pxt.OpT:
        r"""
        Recast an :py:class:`~pyxu.abc.operator.Operator` (or subclass thereof) to another
        :py:class:`~pyxu.abc.operator.Operator`.

        Users may call this method if the arithmetic API yields sub-optimal return types.

        This method is a no-op if `cast_to` is a parent class of ``self``.

        Parameters
        ----------
        cast_to: OpC
            Target type for the recast.

        Returns
        -------
        op: OpT
            Operator with the new interface.

            Fails when cast is forbidden.
            (Ex: :py:class:`~pyxu.abc.operator.Map` -> :py:class:`~pyxu.abc.operator.Func` if codim > 1)

        Notes
        -----
        * The interface of `cast_to` is provided via encapsulation + forwarding.
        * If ``self`` does not implement all methods from `cast_to`, then unimplemented methods
          will raise :py:class:`NotImplementedError` when called.
        """
        if cast_to not in _core_operators():
            raise ValueError(f"cast_to: expected a core base-class, got {cast_to}.")

        p_core = frozenset(self.properties())
        p_shell = frozenset(cast_to.properties())
        if p_shell <= p_core:
            # Trying to cast `self` to it's own class or a parent class.
            # Inheritance rules mean the target object already satisfies the intended interface.
            return self
        else:
            # (p_shell > p_core) -> specializing to a sub-class of ``self``
            # OR
            # len(p_shell ^ p_core) > 0 -> specializing to another branch of the class hierarchy.
            op = cast_to(shape=self.shape)
            op._core = self  # for debugging

            # Forward shared arithmetic fields from core to shell.
            for p in p_shell & p_core:
                for m in p.arithmetic_methods():
                    m_core = getattr(self, m)
                    setattr(op, m, m_core)
            return op

    # Operator Arithmetic -----------------------------------------------------
    def __add__(self, other: pxt.OpT) -> pxt.OpT:
        """
        Add two operators.

        Parameters
        ----------
        self: OpT
            (A, B) Left operand.
        other: OpT
            (C, D) Right operand.

        Returns
        -------
        op: OpT
            Composite operator ``self + other``

        Notes
        -----
        Operand shapes must be `consistent`, i.e.:

            * have `identical shape`, or
            * be `range-broadcastable`, i.e. functional + map works.

        See Also
        --------
        :py:class:`~pyxu.abc.arithmetic.AddRule`
        """
        import pyxu.abc.arithmetic as arithmetic

        if isinstance(other, Operator):
            return arithmetic.AddRule(lhs=self, rhs=other).op()
        else:
            return NotImplemented

    def __sub__(self, other: pxt.OpT) -> pxt.OpT:
        """
        Subtract two operators.

        Parameters
        ----------
        self: OpT
            (A, B) Left operand.
        other: OpT
            (C, D) Right operand.

        Returns
        -------
        op: OpT
            Composite operator ``self - other``
        """
        import pyxu.abc.arithmetic as arithmetic

        if isinstance(other, Operator):
            return arithmetic.AddRule(lhs=self, rhs=-other).op()
        else:
            return NotImplemented

    def __neg__(self) -> pxt.OpT:
        """
        Negate an operator.

        Returns
        -------
        op: OpT
            Composite operator ``-1 * self``.
        """
        import pyxu.abc.arithmetic as arithmetic

        return arithmetic.ScaleRule(op=self, cst=-1).op()

    def __mul__(self, other: typ.Union[pxt.Real, pxt.OpT]) -> pxt.OpT:
        """
        Compose two operators, or scale an operator by a constant.

        Parameters
        ----------
        self: OpT
            (A, B) Left operand.
        other: Real, OpT
            (1,) scalar, or
            (C, D) Right operand.

        Returns
        -------
        op: OpT
            (A, B) scaled operator, or
            (A, D) composed operator ``self * other``.

        Notes
        -----
        If called with two operators, their shapes must be `consistent`, i.e. B == C.

        See Also
        --------
        :py:class:`~pyxu.abc.arithmetic.ScaleRule`,
        :py:class:`~pyxu.abc.arithmetic.ChainRule`
        """
        import pyxu.abc.arithmetic as arithmetic

        if isinstance(other, Operator):
            return arithmetic.ChainRule(lhs=self, rhs=other).op()
        elif _is_real(other):
            return arithmetic.ScaleRule(op=self, cst=float(other)).op()
        else:
            return NotImplemented

    def __rmul__(self, other: pxt.Real) -> pxt.OpT:
        import pyxu.abc.arithmetic as arithmetic

        if _is_real(other):
            return arithmetic.ScaleRule(op=self, cst=float(other)).op()
        else:
            return NotImplemented

    def __truediv__(self, other: pxt.Real) -> pxt.OpT:
        import pyxu.abc.arithmetic as arithmetic

        if _is_real(other):
            return arithmetic.ScaleRule(op=self, cst=float(1 / other)).op()
        else:
            return NotImplemented

    def __pow__(self, k: pxt.Integer) -> pxt.OpT:
        """
        Exponentiate an operator, i.e. compose it with itself.

        Parameters
        ----------
        k: Integer
            Number of times the operator is composed with itself.

        Returns
        -------
        op: OpT
            Exponentiated operator.

        Notes
        -----
        Exponentiation is only allowed for endomorphisms, i.e. square operators.

        See Also
        --------
        :py:class:`~pyxu.abc.arithmetic.PowerRule`
        """
        import pyxu.abc.arithmetic as arithmetic

        if isinstance(k, pxt.Integer) and (k >= 0):
            return arithmetic.PowerRule(op=self, k=k).op()
        else:
            return NotImplemented

    def __matmul__(self, other) -> pxt.OpT:
        # (op @ NDArray) unsupported
        return NotImplemented

    def __rmatmul__(self, other) -> pxt.OpT:
        # (NDArray @ op) unsupported
        return NotImplemented

    def argscale(self, scalar: pxt.Real) -> pxt.OpT:
        """
        Scale operator's domain.

        Parameters
        ----------
        scalar: Real

        Returns
        -------
        op: OpT
            (N, M) domain-scaled operator.

        See Also
        --------
        :py:class:`~pyxu.abc.arithmetic.ArgScaleRule`
        """
        import pyxu.abc.arithmetic as arithmetic

        assert _is_real(scalar)
        return arithmetic.ArgScaleRule(op=self, cst=float(scalar)).op()

    def argshift(self, shift: typ.Union[pxt.Real, pxt.NDArray]) -> pxt.OpT:
        """
        Shift operator's domain.

        Parameters
        ----------
        shift: Real, NDArray
            (M,) shift value

        Returns
        -------
        op: OpT
            (N, M) domain-shifted operator.

        See Also
        --------
        :py:class:`~pyxu.abc.arithmetic.ArgShiftRule`
        """
        import pyxu.abc.arithmetic as arithmetic

        if _is_real(shift):
            shift = float(shift)
        return arithmetic.ArgShiftRule(op=self, cst=shift).op()

    # Internal Helpers --------------------------------------------------------
    @staticmethod
    def _infer_operator_type(prop: cabc.Collection[Property]) -> pxt.OpC:
        prop = frozenset(prop)
        for op in _core_operators():
            if op.properties() == prop:
                return op
        else:
            raise ValueError(f"No operator found with properties {prop}.")

    def squeeze(self) -> pxt.OpT:
        r"""
        Cast an :py:class:`~pyxu.abc.operator.Operator` to the right core operator sub-type given
        codomain dimension.
        """
        p = set(self.properties())
        if self.codim == 1:
            p.add(Property.FUNCTIONAL)
            if Property.DIFFERENTIABLE in self.properties():
                p.add(Property.DIFFERENTIABLE_FUNCTION)
            if Property.LINEAR in self.properties():
                for p_ in Property:
                    if p_.name.startswith("LINEAR_"):
                        p.discard(p_)
                p.add(Property.PROXIMABLE)
        elif self.codim == self.dim:
            if Property.LINEAR in self.properties():
                p.add(Property.LINEAR_SQUARE)
        klass = self._infer_operator_type(p)
        return self.asop(klass)

    def __repr__(self) -> str:
        klass = self._name
        return f"{klass}{self.shape}"

    def _expr(self) -> tuple:
        r"""
        Show the expression-representation of the operator.

        If overridden, must return a tuple of the form

            (head, \*tail),

        where `head` is the operator (ex: +/\*), and `tail` denotes all the expression's terms.
        If an operator cannot be expanded further, then this method should return (self,).
        """
        return (self,)

    def expr(self, level: int = 0, strip: bool = True) -> str:
        """
        Pretty-Print the expression representation of the operator.

        Useful for debugging arithmetic-induced expressions.

        Example
        -------

        .. code-block:: python3

           import numpy as np
           import pyxu.abc as pxa

           N = 5
           op1 = pxa.LinFunc.from_array(np.arange(N))
           op2 = pxa.LinOp.from_array(np.ones((N, N)))
           op = ((2 * op1) + (op2 ** 3)).argshift(np.full(N, 4))

           print(op.expr())
           # [argshift, ==> DiffMap(5, 5)
           # .[add, ==> SquareOp(5, 5)
           # ..[scale, ==> LinFunc(1, 5)
           # ...LinFunc(1, 5),
           # ...2.0],
           # ..[exp, ==> SquareOp(5, 5)
           # ...LinOp(5, 5),
           # ...3]],
           # .(5,)]
        """
        fmt = lambda obj, lvl: ("." * lvl) + str(obj)
        lines = []

        head, *tail = self._expr()
        if len(tail) == 0:
            head = f"{repr(head)},"
        else:
            head = f"[{head}, ==> {repr(self)}"
        lines.append(fmt(head, level))

        for t in tail:
            if isinstance(t, Operator):
                lines += t.expr(level=level + 1, strip=False).split("\n")
            else:
                t = f"{t},"
                lines.append(fmt(t, level + 1))
        if len(tail) > 0:
            # Drop comma for last tail item, then close the sub-expression.
            lines[-1] = lines[-1][:-1]
            lines[-1] += "],"

        out = "\n".join(lines)
        if strip:
            out = out.strip(",")  # drop comma at top-level tail.
        return out


class Map(Operator):
    r"""
    Base class for real-valued maps :math:`\mathbf{f}: \mathbb{R}^{M} \to \mathbb{R}^{N}`.

    Instances of this class must implement
    :py:meth:`~pyxu.abc.operator.Map.apply`.

    If :math:`\mathbf{f}` is Lipschitz-continuous with known Lipschitz constant :math:`L`, the latter should be stored
    in the :py:attr:`~pyxu.abc.operator.Map.lipschitz` property.
    """

    @classmethod
    def properties(cls) -> cabc.Set[Property]:
        p = set(super().properties())
        p.add(Property.CAN_EVAL)
        return frozenset(p)

    def __init__(self, shape: pxt.OpShape):
        super().__init__(shape=shape)
        self.lipschitz = np.inf

    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        """
        Evaluate operator at specified point(s).

        Parameters
        ----------
        arr: NDArray
            (..., M) input points.

        Returns
        -------
        out: NDArray
            (..., N) output points.
        """
        raise NotImplementedError

    def __call__(self, arr: pxt.NDArray) -> pxt.NDArray:
        """
        Alias for :py:meth:`~pyxu.abc.operator.Map.apply`.
        """
        return self.apply(arr)

    @property
    def lipschitz(self) -> pxt.Real:
        r"""
        Return the last computed Lipschitz constant of :math:`\mathbf{f}`.

        Notes
        -----
        * If a Lipschitz constant is known apriori, it can be stored in the instance as follows:

          .. code-block:: python3

             class TestOp(Map):
                 def __init__(self, shape):
                     super().__init__(shape=shape)
                     self.lipschitz = 2

             op = TestOp(shape=(2, 3))
             op.lipschitz  # => 2


          Alternatively the Lipschitz constant can be set manually after initialization:

          .. code-block:: python3

             class TestOp(Map):
                 def __init__(self, shape):
                     super().__init__(shape=shape)

             op = TestOp(shape=(2,3))
             op.lipschitz  # => inf, since unknown apriori

             op.lipschitz = 2  # post-init specification
             op.lipschitz  # => 2

        * :py:meth:`~pyxu.abc.operator.Map.lipschitz` **never** computes anything:
          call :py:meth:`~pyxu.abc.operator.Map.estimate_lipschitz` manually to *compute* a new Lipschitz estimate:

          .. code-block:: python3

             op.lipschitz = op.estimate_lipschitz()
        """
        if not hasattr(self, "_lipschitz"):
            self._lipschitz = self.estimate_lipschitz()

        return pxrt.coerce(self._lipschitz)

    @lipschitz.setter
    def lipschitz(self, L: pxt.Real):
        assert L >= 0
        self._lipschitz = float(L)

        # If no algorithm available to auto-determine estimate_lipschitz(), then enforce user's choice.
        if not self.has(Property.LINEAR):

            def op_estimate_lipschitz(_, **kwargs) -> pxt.Real:
                return _._lipschitz

            self.estimate_lipschitz = types.MethodType(op_estimate_lipschitz, self)

    def estimate_lipschitz(self, **kwargs) -> pxt.Real:
        r"""
        Compute a Lipschitz constant of the operator.

        Parameters
        ----------
        kwargs: ~collections.abc.Mapping
            Class-specific kwargs to configure Lipschitz estimation.

        Notes
        -----
        * This method should always be callable without specifying any kwargs.

        * A constant :math:`L_{\mathbf{f}} > 0` is said to be a *Lipschitz constant* for a map
          :math:`\mathbf{f}: \mathbb{R}^{M} \to \mathbb{R}^{N}` if:

          .. math::

             \|\mathbf{f}(\mathbf{x}) - \mathbf{f}(\mathbf{y})\|_{\mathbb{R}^{N}}
             \leq
             L_{\mathbf{f}} \|\mathbf{x} - \mathbf{y}\|_{\mathbb{R}^{M}},
             \qquad
             \forall \mathbf{x}, \mathbf{y}\in \mathbb{R}^{M},

          where
          :math:`\|\cdot\|_{\mathbb{R}^{M}}` and
          :math:`\|\cdot\|_{\mathbb{R}^{N}}`
          are the canonical norms on their respective spaces.

          The smallest Lipschitz constant of a map is called the *optimal Lipschitz constant*.
        """
        raise NotImplementedError


class Func(Map):
    r"""
    Base class for real-valued functionals :math:`f: \mathbb{R}^{M} \to \mathbb{R}\cup\{+\infty\}`.

    Instances of this class must implement
    :py:meth:`~pyxu.abc.operator.Map.apply`.

    If :math:`f` is Lipschitz-continuous with known Lipschitz constant :math:`L`, the latter should be stored in the
    :py:attr:`~pyxu.abc.operator.Map.lipschitz` property.
    """

    @classmethod
    def properties(cls) -> cabc.Set[Property]:
        p = set(super().properties())
        p.add(Property.FUNCTIONAL)
        return frozenset(p)

    def __init__(self, shape: pxt.OpShape):
        super().__init__(shape=shape)
        assert self.codim == 1, f"shape: expected (1, n), got {shape}."

    def asloss(self, data: pxt.NDArray = None) -> pxt.OpT:
        """
        Transform a functional into a loss functional.

        The meaning of
        :py:meth:`~pyxu.abc.operator.Func.asloss`
        is operator-dependent: always examine the output of
        :py:meth:`~pyxu.abc.operator.Operator.expr`
        to determine the loss-notion involved.

        Parameters
        ----------
        data: NDArray
            (M,) input.

        Returns
        -------
        op: OpT
            (1, M) loss function.
            If `data` is unspecified, returns ``self``.
        """
        raise NotImplementedError


class DiffMap(Map):
    r"""
    Base class for real-valued differentiable maps :math:`\mathbf{f}: \mathbb{R}^{M} \to \mathbb{R}^{N}`.

    Instances of this class must implement
    :py:meth:`~pyxu.abc.operator.Map.apply` and
    :py:meth:`~pyxu.abc.operator.DiffMap.jacobian`.

    If :math:`\mathbf{f}` is Lipschitz-continuous with known Lipschitz constant :math:`L`, the latter should be stored
    in the :py:attr:`~pyxu.abc.operator.Map.lipschitz` property.

    If :math:`\mathbf{J}_{\mathbf{f}}` is Lipschitz-continuous with known Lipschitz constant :math:`\partial L`, the
    latter should be stored in the :py:attr:`~pyxu.abc.operator.DiffMap.diff_lipschitz` property.
    """

    @classmethod
    def properties(cls) -> cabc.Set[Property]:
        p = set(super().properties())
        p.add(Property.DIFFERENTIABLE)
        return frozenset(p)

    def __init__(self, shape: pxt.OpShape):
        super().__init__(shape=shape)
        self.diff_lipschitz = np.inf

    def jacobian(self, arr: pxt.NDArray) -> pxt.OpT:
        r"""
        Evaluate the Jacobian of :math:`\mathbf{f}` at the specified point.

        Parameters
        ----------
        arr: NDArray
            (M,) evaluation point.

        Returns
        -------
        op: OpT
            (N, M) Jacobian operator at point `arr`.

        Notes
        -----
        Let :math:`\mathbf{f}=[f_{1}, \ldots, f_{N}]: \mathbb{R}^{M} \to \mathbb{R}^{N}` be a differentiable
        multi-dimensional map.
        The *Jacobian* (or *differential*) of :math:`\mathbf{f}` at
        :math:`\mathbf{z} \in \mathbb{R}^{M}` is defined as the best linear approximator of
        :math:`\mathbf{f}` near :math:`\mathbf{z}`, in the following sense:

        .. math::

           \mathbf{f}(\mathbf{x}) - \mathbf{f}(\mathbf{z})
           =
           \mathbf{J}_{\mathbf{f}}(\mathbf{z}) (\mathbf{x} - \mathbf{z})
           +
           o(\| \mathbf{x} - \mathbf{z} \|)
           \quad
           \text{as} \quad \mathbf{x} \to \mathbf{z}.

        The Jacobian admits the following matrix representation:

        .. math::

           [\mathbf{J}_{\mathbf{f}}(\mathbf{x})]_{ij}
           :=
           \frac{\partial f_{i}}{\partial x_{j}}(\mathbf{x}),
           \qquad
           \forall (i,j) \in \{1,\ldots,N\} \times \{1,\ldots,M\}.
        """
        raise NotImplementedError

    @property
    def diff_lipschitz(self) -> pxt.Real:
        r"""
        Return the last computed Lipschitz constant of :math:`\mathbf{J}_{\mathbf{f}}`.

        Notes
        -----
        * If a diff-Lipschitz constant is known apriori, it can be stored in the instance as follows:

          .. code-block:: python3

             class TestOp(DiffMap):
                 def __init__(self, shape):
                     super().__init__(shape=shape)
                     self.diff_lipschitz = 2

             op = TestOp(shape=(2, 3))
             op.diff_lipschitz  # => 2


          Alternatively the diff-Lipschitz constant can be set manually after initialization:

          .. code-block:: python3

             class TestOp(DiffMap):
                 def __init__(self, shape):
                     super().__init__(shape=shape)

             op = TestOp(shape=(2,3))
             op.diff_lipschitz  # => inf, since unknown apriori

             op.diff_lipschitz = 2  # post-init specification
             op.diff_lipschitz  # => 2

        * :py:meth:`~pyxu.abc.operator.DiffMap.diff_lipschitz` **never** computes anything:
          call :py:meth:`~pyxu.abc.operator.DiffMap.estimate_diff_lipschitz` manually to *compute* a new diff-Lipschitz estimate:

          .. code-block:: python3

             op.diff_lipschitz = op.estimate_diff_lipschitz()
        """
        if not hasattr(self, "_diff_lipschitz"):
            self._diff_lipschitz = self.estimate_diff_lipschitz()

        return pxrt.coerce(self._diff_lipschitz)

    @diff_lipschitz.setter
    def diff_lipschitz(self, dL: pxt.Real):
        assert dL >= 0
        self._diff_lipschitz = float(dL)

        # If no algorithm available to auto-determine estimate_diff_lipschitz(), then enforce user's choice.
        if not self.has(Property.QUADRATIC):

            def op_estimate_diff_lipschitz(_, **kwargs) -> pxt.Real:
                return _._diff_lipschitz

            self.estimate_diff_lipschitz = types.MethodType(op_estimate_diff_lipschitz, self)

    def estimate_diff_lipschitz(self, **kwargs) -> pxt.Real:
        r"""
        Compute a Lipschitz constant of :py:meth:`~pyxu.abc.operator.DiffMap.jacobian`.

        Parameters
        ----------
        kwargs: ~collections.abc.Mapping
            Class-specific kwargs to configure diff-Lipschitz estimation.

        Notes
        -----
        * This method should always be callable without specifying any kwargs.

        * A Lipschitz constant :math:`L_{\mathbf{J}_{\mathbf{f}}} > 0` of the Jacobian map
          :math:`\mathbf{J}_{\mathbf{f}}: \mathbb{R}^{M} \to \mathbb{R}^{N \times M}` is such that:

          .. math::

             \|\mathbf{J}_{\mathbf{f}}(\mathbf{x}) - \mathbf{J}_{\mathbf{f}}(\mathbf{y})\|_{\mathbb{R}^{N \times M}}
             \leq
             L_{\mathbf{J}_{\mathbf{f}}} \|\mathbf{x} - \mathbf{y}\|_{\mathbb{R}^{M}},
             \qquad
             \forall \mathbf{x}, \mathbf{y} \in \mathbb{R}^{M},

          where
          :math:`\|\cdot\|_{\mathbb{R}^{N \times M}}` and
          :math:`\|\cdot\|_{\mathbb{R}^{M}}`
          are the canonical norms on their respective spaces.

          The smallest Lipschitz constant of the Jacobian is called the *optimal diff-Lipschitz constant*.
        """
        raise NotImplementedError


class ProxFunc(Func):
    r"""
    Base class for real-valued proximable functionals :math:`f: \mathbb{R}^{M} \to \mathbb{R} \cup \{+\infty\}`.

    A functional :math:`f: \mathbb{R}^{M} \to \mathbb{R} \cup \{+\infty\}` is said *proximable* if its
    **proximity operator** (see :py:meth:`~pyxu.abc.operator.ProxFunc.prox` for a definition) admits
    a *simple closed-form expression*
    **or**
    can be evaluated *efficiently* and with *high accuracy*.

    Instances of this class must implement
    :py:meth:`~pyxu.abc.operator.Map.apply` and
    :py:meth:`~pyxu.abc.operator.ProxFunc.prox`.

    If :math:`f` is Lipschitz-continuous with known Lipschitz constant :math:`L`, the latter should be stored in the
    :py:attr:`~pyxu.abc.operator.Map.lipschitz` property.
    """

    @classmethod
    def properties(cls) -> cabc.Set[Property]:
        p = set(super().properties())
        p.add(Property.PROXIMABLE)
        return frozenset(p)

    def __init__(self, shape: pxt.OpShape):
        super().__init__(shape=shape)

    def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
        r"""
        Evaluate proximity operator of :math:`\tau f` at specified point(s).

        Parameters
        ----------
        arr: NDArray
            (..., M) input points.
        tau: Real
            Positive scale factor.

        Returns
        -------
        out: NDArray
            (..., M) proximal evaluations.

        Notes
        -----
        For :math:`\tau >0`, the *proximity operator* of a scaled functional
        :math:`f: \mathbb{R}^{M} \to \mathbb{R}` is defined as:

        .. math::

           \mathbf{\text{prox}}_{\tau f}(\mathbf{z})
           :=
           \arg\min_{\mathbf{x}\in\mathbb{R}^{M}} f(x)+\frac{1}{2\tau} \|\mathbf{x}-\mathbf{z}\|_{2}^{2},
           \quad
           \forall \mathbf{z} \in \mathbb{R}^{M}.
        """
        raise NotImplementedError

    @pxrt.enforce_precision(i=("arr", "sigma"))
    def fenchel_prox(self, arr: pxt.NDArray, sigma: pxt.Real) -> pxt.NDArray:
        r"""
        Evaluate proximity operator of :math:`\sigma f^{\ast}`, the scaled Fenchel conjugate of :math:`f`, at specified point(s).

        Parameters
        ----------
        arr: NDArray
            (..., M) input points.
        sigma: Real
            Positive scale factor.

        Returns
        -------
        out: NDArray
            (..., M) proximal evaluations.

        Notes
        -----
        For :math:`\sigma > 0`, the *Fenchel conjugate* is defined as:

        .. math::

           f^{\ast}(\mathbf{z})
           :=
           \max_{\mathbf{x}\in\mathbb{R}^{M}} \langle \mathbf{x},\mathbf{z} \rangle - f(\mathbf{x}).

        From *Moreau's identity*, its proximal operator is given by:

        .. math::

           \mathbf{\text{prox}}_{\sigma f^{\ast}}(\mathbf{z})
           =
           \mathbf{z} - \sigma \mathbf{\text{prox}}_{f/\sigma}(\mathbf{z}/\sigma).
        """
        out = self.prox(arr=arr / sigma, tau=1 / sigma)
        out = pxu.copy_if_unsafe(out)
        out *= -sigma
        out += arr
        return out

    def moreau_envelope(self, mu: pxt.Real) -> pxt.OpT:
        r"""
        Approximate proximable functional :math:`f` by its *Moreau envelope* :math:`f^{\mu}`.

        Parameters
        ----------
        mu: Real
            Positive regularization parameter.

        Returns
        -------
        op: OpT
            Differential Moreau envelope.

        Notes
        -----
        Consider a convex non-smooth proximable functional
        :math:`f: \mathbb{R}^{M} \to \mathbb{R} \cup \{+\infty\}`
        and a regularization parameter :math:`\mu > 0`.
        The :math:`\mu`-*Moreau envelope* (or *Moreau-Yoshida envelope*) of :math:`f` is given by

        .. math::

           f^{\mu}(\mathbf{x})
           =
           \min_{\mathbf{z} \in \mathbb{R}^{M}}
           f(\mathbf{z})
           \quad + \quad
           \frac{1}{2\mu} \|\mathbf{x} - \mathbf{z}\|^{2}.

        The parameter :math:`\mu` controls the trade-off between the regularity properties of :math:`f^{\mu}` and
        the approximation error incurred by the Moreau-Yoshida regularization.

        The Moreau envelope inherits the convexity of :math:`f` and is gradient-Lipschitz (with Lipschitz constant
        :math:`\mu^{-1}`), even if :math:`f` is non-smooth.
        Its gradient is moreover given by:

        .. math::

           \nabla f^{\mu}(\mathbf{x})
           =
           \mu^{-1} \left(\mathbf{x} - \text{prox}_{\mu f}(\mathbf{x})\right).

        In addition, :math:`f^{\mu}` envelopes :math:`f` from below:
        :math:`f^{\mu}(\mathbf{x}) \leq f(\mathbf{x})`.
        This envelope becomes tighter as :math:`\mu \to 0`:

        .. math::

           \lim_{\mu \to 0} f^{\mu}(\mathbf{x}) = f(\mathbf{x}).

        Finally, it can be shown that the minimizers of :math:`f` and :math:`f^{\mu}` coincide, and
        that the Fenchel conjugate of :math:`f^{\mu}` is strongly-convex.

        Example
        -------
        Construct and plot the Moreau envelope of the :math:`\ell_{1}`-norm:

        .. plot::

           import numpy as np
           import matplotlib.pyplot as plt
           from pyxu.abc import ProxFunc

           class L1Norm(ProxFunc):
               def __init__(self, dim: int):
                   super().__init__(shape=(1, dim))
               def apply(self, arr):
                   return np.linalg.norm(arr, axis=-1, keepdims=True, ord=1)
               def prox(self, arr, tau):
                   return np.clip(np.abs(arr)-tau, a_min=0, a_max=None) * np.sign(arr)

           N = 512
           l1_norm = L1Norm(dim=N)
           mus = [0.1, 0.5, 1]
           smooth_l1_norms = [l1_norm.moreau_envelope(mu) for mu in mus]

           x = np.linspace(-1, 1, N)[:, None]
           labels=['mu=0']
           labels.extend([f'mu={mu}' for mu in mus])
           plt.figure()
           plt.plot(x, l1_norm(x))
           for f in smooth_l1_norms:
               plt.plot(x, f(x))
           plt.legend(labels)
           plt.title('Moreau Envelope')

           labels=[f'mu={mu}' for mu in mus]
           plt.figure()
           for f in smooth_l1_norms:
               plt.plot(x, f.grad(x))
           plt.legend(labels)
           plt.title('Derivative of Moreau Envelope')
        """
        from pyxu.operator.interop.source import from_source

        assert mu > 0, f"mu: expected positive, got {mu}"

        @pxrt.enforce_precision(i="arr")
        def op_apply(_, arr):
            from pyxu.math.linalg import norm

            x = self.prox(arr, tau=_._mu)
            out = pxu.copy_if_unsafe(self.apply(x))
            out += (0.5 / _._mu) * norm(arr - x, axis=-1, keepdims=True) ** 2
            return out

        @pxrt.enforce_precision(i="arr")
        def op_grad(_, arr):
            x = arr.copy()
            x -= self.prox(arr, tau=_._mu)
            x /= _._mu
            return x

        op = from_source(
            cls=DiffFunc,
            shape=self.shape,
            embed=dict(
                _name="moreau_envelope",
                _mu=mu,
                _diff_lipschitz=float(1 / mu),
            ),
            apply=op_apply,
            grad=op_grad,
            _expr=lambda _: ("moreau_envelope", _, _._mu),
        )
        return op


class DiffFunc(DiffMap, Func):
    r"""
    Base class for real-valued differentiable functionals :math:`f: \mathbb{R}^{M} \to \mathbb{R}`.

    Instances of this class must implement
    :py:meth:`~pyxu.abc.operator.Map.apply` and
    :py:meth:`~pyxu.abc.operator.DiffFunc.grad`.

    If :math:`f` and/or its derivative :math:`f'` are Lipschitz-continuous with known Lipschitz constants :math:`L` and
    :math:`\partial L`, the latter should be stored in the
    :py:attr:`~pyxu.abc.operator.Map.lipschitz`
    and
    :py:attr:`~pyxu.abc.operator.DiffMap.diff_lipschitz`
    properties.
    """

    @classmethod
    def properties(cls) -> cabc.Set[Property]:
        p = set()
        for klass in cls.__bases__:
            p |= klass.properties()
        p.add(Property.DIFFERENTIABLE_FUNCTION)
        return frozenset(p)

    def __init__(self, shape: pxt.OpShape):
        DiffMap.__init__(self, shape)
        Func.__init__(self, shape)

    @pxrt.enforce_precision(i="arr", o=False)
    def jacobian(self, arr: pxt.NDArray) -> pxt.OpT:
        return LinFunc.from_array(self.grad(arr))

    def grad(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Evaluate operator gradient at specified point(s).

        Parameters
        ----------
        arr: NDArray
            (..., M) input points.

        Returns
        -------
        out: NDArray
            (..., M) gradients.

        Notes
        -----
        The gradient of a functional :math:`f: \mathbb{R}^{M} \to \mathbb{R}` is given, for every
        :math:`\mathbf{x} \in \mathbb{R}^{M}`, by

        .. math::

           \nabla f(\mathbf{x})
           :=
           \left[\begin{array}{c}
           \frac{\partial f}{\partial x_{1}}(\mathbf{x}) \\
           \vdots \\
           \frac{\partial f}{\partial x_{M}}(\mathbf{x})
           \end{array}\right].
        """
        raise NotImplementedError


class ProxDiffFunc(ProxFunc, DiffFunc):
    r"""
    Base class for real-valued differentiable *and* proximable functionals
    :math:`f:\mathbb{R}^{M} \to \mathbb{R}`.

    Instances of this class must implement
    :py:meth:`~pyxu.abc.operator.Map.apply`,
    :py:meth:`~pyxu.abc.operator.DiffFunc.grad`, and
    :py:meth:`~pyxu.abc.operator.ProxFunc.prox`.

    If :math:`f` and/or its derivative :math:`f'` are Lipschitz-continuous with known Lipschitz constants :math:`L` and
    :math:`\partial L`, the latter should be stored in the
    :py:attr:`~pyxu.abc.operator.Map.lipschitz`
    and
    :py:attr:`~pyxu.abc.operator.DiffMap.diff_lipschitz`
    properties.
    """

    @classmethod
    def properties(cls) -> cabc.Set[Property]:
        p = set()
        for klass in cls.__bases__:
            p |= klass.properties()
        return frozenset(p)

    def __init__(self, shape: pxt.OpShape):
        ProxFunc.__init__(self, shape)
        DiffFunc.__init__(self, shape)


class QuadraticFunc(ProxDiffFunc):
    # This is a special abstract base class with more __init__ parameters than `shape`.
    r"""
    Base class for quadratic functionals :math:`f: \mathbb{R}^{M} \to \mathbb{R} \cup \{+\infty\}`.

    The quadratic functional is defined as:

    .. math::

       f(\mathbf{x})
       =
       \frac{1}{2} \langle\mathbf{x}, \mathbf{Q}\mathbf{x}\rangle
       +
       \mathbf{c}^T\mathbf{x}
       +
       t,
       \qquad \forall \mathbf{x} \in \mathbb{R}^{M},

    where
    :math:`Q` is a positive-definite operator :math:`\mathbf{Q}:\mathbb{R}^{M} \to \mathbb{R}^{M}`,
    :math:`\mathbf{c} \in \mathbb{R}^{M}`, and
    :math:`t > 0`.

    Its gradient is given by:

    .. math::

       \nabla f(\mathbf{x}) = \mathbf{Q}\mathbf{x} + \mathbf{c}.

    Its proximity operator by:

    .. math::

       \text{prox}_{\tau f}(x)
       =
       \left(
           \mathbf{Q} + \tau^{-1} \mathbf{Id}
       \right)^{-1}
       \left(
           \tau^{-1}\mathbf{x} - \mathbf{c}
       \right).

    In practice the proximity operator is evaluated via :py:class:`~pyxu.opt.solver.cg.CG`.

    The Lipschitz constant :math:`L` of a quadratic on an unbounded domain is unbounded.
    The Lipschitz constant :math:`\partial L` of :math:`\nabla f` is given by the spectral norm of :math:`\mathbf{Q}`.
    """

    @classmethod
    def properties(cls) -> cabc.Set[Property]:
        p = set(super().properties())
        p.add(Property.QUADRATIC)
        return frozenset(p)

    def __init__(
        self,
        shape: pxt.OpShape,
        # required in place of `dim` to have uniform interface with Operator hierarchy.
        Q: "PosDefOp" = None,
        c: "LinFunc" = None,
        t: pxt.Real = 0,
    ):
        r"""
        Parameters
        ----------
        Q: ~pyxu.abc.operator.PosDefOp
            (N, N) positive-definite operator. (Default: Identity)
        c: ~pyxu.abc.operator.LinFunc
            (1, N) linear functional. (Default: NullFunc)
        t: Real
            offset. (Default: 0)
        """
        from pyxu.operator.linop import IdentityOp, NullFunc

        super().__init__(shape=shape)

        # Do NOT access (_Q, _c, _t) directly through `self`:
        # their values may not reflect the true (Q, c, t) parameterization.
        # (Reason: arithmetic propagation.)
        # Always access (Q, c, t) by querying the arithmetic method `_quad_spec()`.
        self._Q = IdentityOp(dim=self.dim) if (Q is None) else Q
        self._c = NullFunc(dim=self.dim) if (c is None) else c
        self._t = t

        # ensure dimensions are consistent if None-initialized
        assert self._Q.shape == (self.dim, self.dim)
        assert self._c.shape == self.shape

    @pxrt.enforce_precision(i="arr")
    def apply(self, arr: pxt.NDArray) -> pxt.NDArray:
        Q, c, t = self._quad_spec()
        out = (arr * Q.apply(arr)).sum(axis=-1, keepdims=True)
        out /= 2
        out += c.apply(arr)
        out += t
        return out

    @pxrt.enforce_precision(i="arr")
    def grad(self, arr: pxt.NDArray) -> pxt.NDArray:
        Q, c, _ = self._quad_spec()
        out = pxu.copy_if_unsafe(Q.apply(arr))
        out += c.grad(arr)
        return out

    @pxrt.enforce_precision(i=("arr", "tau"))
    def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
        from pyxu.operator.linop import HomothetyOp
        from pyxu.opt.solver import CG
        from pyxu.opt.stop import MaxIter

        Q, c, _ = self._quad_spec()
        A = Q + HomothetyOp(cst=1 / tau, dim=Q.dim)
        b = arr.copy()
        b /= tau
        b -= c.grad(arr)

        slvr = CG(A=A, show_progress=False)

        sentinel = MaxIter(n=2 * A.dim)
        stop_crit = slvr.default_stop_crit() | sentinel

        slvr.fit(b=b, stop_crit=stop_crit)
        return slvr.solution()

    def asloss(self, data: pxt.NDArray = None) -> pxt.OpT:
        from pyxu.operator.func.loss import shift_loss

        op = shift_loss(op=self, data=data)
        return op

    def estimate_diff_lipschitz(self, **kwargs) -> pxt.Real:
        Q, *_ = self._quad_spec()
        dL = Q.estimate_lipschitz(**kwargs)
        return dL

    def _quad_spec(self):
        """
        Canonical quadratic parameterization.

        Useful for some internal methods, and overloaded during operator arithmetic.
        """
        return (self._Q, self._c, self._t)


class LinOp(DiffMap):
    r"""
    Base class for real-valued linear operators :math:`\mathbf{A}: \mathbb{R}^{M} \to \mathbb{R}^{N}`.

    Instances of this class must implement
    :py:meth:`~pyxu.abc.operator.Map.apply` and
    :py:meth:`~pyxu.abc.operator.LinOp.adjoint`.

    If known, the Lipschitz constant :math:`L` should be stored in the
    :py:attr:`~pyxu.abc.operator.Map.lipschitz` property.

    The Jacobian of a linear map :math:`\mathbf{A}` is constant.
    """

    # Internal Helpers ------------------------------------
    @staticmethod
    def _warn_vals_sparse_gpu():
        msg = "\n".join(
            [
                "Potential Error:",
                "Sparse GPU-evaluation of svdvals() is known to produce incorrect results. (CuPy-specific + Matrix-Dependant.)",
                "It is advised to cross-check results with CPU-computed results.",
            ]
        )
        warnings.warn(msg, pxw.BackendWarning)

    # -----------------------------------------------------

    @classmethod
    def properties(cls) -> cabc.Set[Property]:
        p = set(super().properties())
        p.add(Property.LINEAR)
        return frozenset(p)

    def __init__(self, shape: pxt.OpShape):
        super().__init__(shape=shape)
        self.diff_lipschitz = 0

    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        r"""
        Evaluate operator adjoint at specified point(s).

        Parameters
        ----------
        arr: NDArray
            (..., N) input points.

        Returns
        -------
        out: NDArray
            (..., M) adjoint evaluations.

        Notes
        -----
        The *adjoint* :math:`\mathbf{A}^{\ast}: \mathbb{R}^{N} \to \mathbb{R}^{M}` of
        :math:`\mathbf{A}: \mathbb{R}^{M} \to \mathbb{R}^{N}` is defined as:

        .. math::

           \langle \mathbf{x}, \mathbf{A}^{\ast}\mathbf{y}\rangle_{\mathbb{R}^{M}}
           :=
           \langle \mathbf{A}\mathbf{x}, \mathbf{y}\rangle_{\mathbb{R}^{N}},
           \qquad
           \forall (\mathbf{x},\mathbf{y})\in \mathbb{R}^{M} \times \mathbb{R}^{N}.
        """
        raise NotImplementedError

    def jacobian(self, arr: pxt.NDArray) -> pxt.OpT:
        return self

    @property
    def T(self) -> pxt.OpT:
        r"""
        Return the (M, N) adjoint of :math:`\mathbf{A}`.
        """
        import pyxu.abc.arithmetic as arithmetic

        return arithmetic.TransposeRule(op=self).op()

    def to_sciop(
        self,
        dtype: pxt.DType = None,
        gpu: bool = False,
    ) -> spsl.LinearOperator:
        r"""
        Cast a :py:class:`~pyxu.abc.operator.LinOp` to a CPU/GPU
        :py:class:`~scipy.sparse.linalg.LinearOperator`, compatible with the matrix-free linear
        algebra routines of :py:mod:`scipy.sparse.linalg`.

        Parameters
        ----------
        dtype: DType
            Working precision of the linear operator.
        gpu: bool
            Operate on CuPy inputs (True) vs. NumPy inputs (False).

        Returns
        -------
        op: ~scipy.sparse.linalg.LinearOperator
            Linear operator object compliant with SciPy's interface.
        """

        def matmat(arr):
            with pxrt.EnforcePrecision(False):
                return self.apply(arr.T).T

        def rmatmat(arr):
            with pxrt.EnforcePrecision(False):
                return self.adjoint(arr.T).T

        if dtype is None:
            dtype = pxrt.getPrecision().value

        if gpu:
            assert pxd.CUPY_ENABLED
            import cupyx.scipy.sparse.linalg as spx
        else:
            spx = spsl
        return spx.LinearOperator(
            shape=self.shape,
            matvec=matmat,
            rmatvec=rmatmat,
            matmat=matmat,
            rmatmat=rmatmat,
            dtype=dtype,
        )

    def estimate_lipschitz(self, **kwargs) -> pxt.Real:
        r"""
        Compute a Lipschitz constant of the operator.

        Parameters
        ----------
        method: "svd" | "trace"

            * If `svd`, compute the optimal Lipschitz constant.
            * If `trace`, compute an upper bound. (Default)

        kwargs:
            Optional kwargs passed on to:

            * `svd`: :py:func:`~pyxu.abc.operator.LinOp.svdvals`
            * `trace`: :py:func:`~pyxu.math.linalg.hutchpp`

        Notes
        -----
        * The tightest Lipschitz constant is given by the spectral norm of the operator :math:`\mathbf{A}`:
          :math:`\|\mathbf{A}\|_{2}`.
          It can be computed via the SVD, which is compute-intensive task for large operators.
          In this setting, it may be advantageous to overestimate the Lipschitz constant with the
          Frobenius norm of :math:`\mathbf{A}` since :math:`\|\mathbf{A}\|_{F} \geq \|\mathbf{A}\|_{2}`.

          :math:`\|\mathbf{A}\|_{F}` can be efficiently approximated by computing the trace of :math:`\mathbf{A}^{\ast} \mathbf{A}`
          (or :math:`\mathbf{A}\mathbf{A}^{\ast}`) via the `Hutch++ stochastic algorithm <https://arxiv.org/abs/2010.09649>`_.

        * :math:`\|\mathbf{A}\|_{F}` is upper-bounded by :math:`\|\mathbf{A}\|_{F} \leq \sqrt{n} \|\mathbf{A}\|_{2}`,
          where the equality is reached (worst-case scenario) when the eigenspectrum of the linear operator is flat.
        """
        method = kwargs.get("method", "trace").lower().strip()

        if method == "svd":
            # svdvals() may have alternative signature in specialized classes, but we must always use
            # the LinOp.svdvals() interface below for kwargs-filtering.
            func, sig_func = self.__class__.svdvals, LinOp.svdvals
            kwargs.update(
                k=1,
                which="LM",
            )
            estimate = lambda: func(self, **kwargs).item()
        elif method == "trace":
            if self.shape == (1, 1):
                # [Sepand] Special case of degenerate LinOps (which are not LinFuncs).
                # Cannot call .gram() below since it will recurse indefinitely.
                # [Reason: asop().squeeze() combinations don't work for (1, 1) degenerate LinOps.]
                func = sig_func = LinFunc.estimate_lipschitz
                estimate = lambda: func(self.squeeze(), **kwargs)
            else:
                from pyxu.math.linalg import hutchpp as func

                sig_func = func

                kwargs.update(
                    op=self.gram() if (self.codim >= self.dim) else self.cogram(),
                    m=kwargs.get("m", 126),
                )
                estimate = lambda: np.sqrt(func(**kwargs)).item()
        else:
            raise NotImplementedError

        # Filter unsupported kwargs
        sig = inspect.Signature.from_callable(sig_func)
        kwargs = {k: v for (k, v) in kwargs.items() if (k in sig.parameters)}

        L = estimate()
        return L

    @pxrt.enforce_precision()
    def svdvals(
        self,
        k: pxt.Integer,
        which: str = "LM",
        gpu: bool = False,
        **kwargs,
    ) -> pxt.NDArray:
        r"""
        Compute singular values of a linear operator.

        Parameters
        ----------
        k: Integer
            Number of singular values to compute.
        which: "LM" | "SM"
            Which `k` singular values to find:

                * `LM`: largest magnitude
                * `SM`: smallest magnitude
        gpu: bool
            If ``True`` the singular value decomposition is performed on the GPU.
        kwargs: ~collections.abc.Mapping
            Additional kwargs accepted by :py:func:`~scipy.sparse.linalg.svds`.

        Returns
        -------
        D: NDArray
            (k,) singular values in ascending order.
        """
        which = which.upper().strip()
        assert which in {"SM", "LM"}

        def _dense_eval():
            if gpu:
                import cupy as xp
                import cupy.linalg as spx
            else:
                import numpy as xp
                import scipy.linalg as spx
            op = self.asarray(xp=xp, dtype=pxrt.getPrecision().value)
            return spx.svd(op, compute_uv=False)

        def _sparse_eval():
            if gpu:
                assert pxd.CUPY_ENABLED
                import cupyx.scipy.sparse.linalg as spx

                self._warn_vals_sparse_gpu()
            else:
                spx = spsl
            op = self.to_sciop(gpu=gpu, dtype=pxrt.getPrecision().value)
            kwargs.update(
                k=k,
                which=which,
                return_singular_vectors=False,
                # random_state=0,  # unsupported by CuPy
            )
            return spx.svds(op, **kwargs)

        if k >= min(self.shape) // 2:
            msg = "Too many svdvals wanted: using matrix-based ops."
            warnings.warn(msg, pxw.DenseWarning)
            D = _dense_eval()
        else:
            D = _sparse_eval()

        # Filter to k largest/smallest magnitude + sorted
        xp = pxu.get_array_module(D)
        D = D[xp.argsort(D)]
        return D[:k] if (which == "SM") else D[-k:]

    def asarray(
        self,
        xp: pxt.ArrayModule = None,
        dtype: pxt.DType = None,
    ) -> pxt.NDArray:
        r"""
        Matrix representation of the linear operator.

        Parameters
        ----------
        xp: ArrayModule
            Which array module to use to represent the output. (Default: NumPy.)
        dtype: DType
            Precision of the array. (Default: current runtime precision.)

        Returns
        -------
        A: NDArray
            (N, M) array-representation of the operator.

        Note
        ----
        This generic implementation assumes the operator is backend-agnostic.
        Thus, when defining a new backend-specific operator,
        :py:meth:`~pyxu.abc.operator.LinOp.asarray`
        may need to be overriden.
        """
        if xp is None:
            xp = pxd.NDArrayInfo.default().module()
        if dtype is None:
            dtype = pxrt.getPrecision().value

        with pxrt.EnforcePrecision(False):
            E = xp.eye(self.dim, dtype=dtype)
            A = self.apply(E).T
        return A

    def gram(self) -> pxt.OpT:
        r"""
        Gram operator :math:`\mathbf{A}^{\ast} \mathbf{A}: \mathbb{R}^{M} \to \mathbb{R}^{M}`.

        Returns
        -------
        op: OpT
            (M, M) Gram operator.

        Note
        ----
        By default the Gram is computed by the composition ``self.T * self``.
        This may not be the fastest way to compute the Gram operator.
        If the Gram can be computed more efficiently (e.g. with a convolution), the user should re-define this method.
        """

        def op_expr(_) -> tuple:
            return ("gram", self)

        op = self.T * self
        op._expr = types.MethodType(op_expr, op)
        return op.asop(SelfAdjointOp).squeeze()

    def cogram(self) -> pxt.OpT:
        r"""
        Co-Gram operator :math:`\mathbf{A}\mathbf{A}^{\ast}:\mathbb{R}^{N} \to \mathbb{R}^{N}`.

        Returns
        -------
        op: OpT
            (N, N) Co-Gram operator.

        Note
        ----
        By default the co-Gram is computed by the composition ``self * self.T``.
        This may not be the fastest way to compute the co-Gram operator.
        If the co-Gram can be computed more efficiently (e.g. with a convolution), the user should re-define this method.
        """

        def op_expr(_) -> tuple:
            return ("cogram", self)

        op = self * self.T
        op._expr = types.MethodType(op_expr, op)
        return op.asop(SelfAdjointOp).squeeze()

    @pxrt.enforce_precision(i=("arr", "damp"))
    def pinv(
        self,
        arr: pxt.NDArray,
        damp: pxt.Real,
        kwargs_init=None,
        kwargs_fit=None,
    ) -> pxt.NDArray:
        r"""
        Evaluate the Moore-Penrose pseudo-inverse :math:`\mathbf{A}^{\dagger}` at specified point(s).

        Parameters
        ----------
        arr: NDArray
            (..., N) input points.
        damp: Real
            Positive dampening factor regularizing the pseudo-inverse.
        kwargs_init: ~collections.abc.Mapping
            Optional kwargs to be passed to :py:meth:`~pyxu.opt.solver.cg.CG`'s ``__init__()`` method.
        kwargs_fit: ~collections.abc.Mapping
            Optional kwargs to be passed to :py:meth:`~pyxu.opt.solver.cg.CG`'s ``fit()`` method.

        Returns
        -------
        out: NDArray
            (..., M) pseudo-inverse(s).

        Notes
        -----
        The Moore-Penrose pseudo-inverse of an operator :math:`\mathbf{A}: \mathbb{R}^{M} \to \mathbb{R}^{N}` is
        defined as the operator :math:`\mathbf{A}^{\dagger}: \mathbb{R}^{N} \to \mathbb{R}^{M}` verifying the
        Moore-Penrose conditions:

            1. :math:`\mathbf{A} \mathbf{A}^{\dagger} \mathbf{A} = \mathbf{A}`,
            2. :math:`\mathbf{A}^{\dagger} \mathbf{A} \mathbf{A}^{\dagger} = \mathbf{A}^{\dagger}`,
            3. :math:`(\mathbf{A}^{\dagger} \mathbf{A})^{\ast} = \mathbf{A}^{\dagger} \mathbf{A}`,
            4. :math:`(\mathbf{A} \mathbf{A}^{\dagger})^{\ast} = \mathbf{A} \mathbf{A}^{\dagger}`.

        This operator exists and is unique for any finite-dimensional linear operator.
        The action of the pseudo-inverse :math:`\mathbf{A}^{\dagger} \mathbf{y}` for every
        :math:`\mathbf{y} \in \mathbb{R}^{N}` can be computed in matrix-free fashion by solving the
        *normal equations*:

        .. math::

           \mathbf{A}^{\ast} \mathbf{A} \mathbf{x} = \mathbf{A}^{\ast} \mathbf{y}
           \quad\Leftrightarrow\quad
           \mathbf{x} = \mathbf{A}^{\dagger} \mathbf{y},
           \quad
           \forall (\mathbf{x}, \mathbf{y}) \in \mathbb{R}^{M} \times \mathbb{R}^{N}.

        In the case of severe ill-conditioning, it is possible to consider the dampened normal
        equations for a numerically-stabler approximation of :math:`\mathbf{A}^{\dagger} \mathbf{y}`:

        .. math::

           (\mathbf{A}^{\ast} \mathbf{A} + \tau I) \mathbf{x} = \mathbf{A}^{\ast} \mathbf{y},

        where :math:`\tau > 0` corresponds to the `damp` parameter.
        """
        from pyxu.operator.linop import HomothetyOp
        from pyxu.opt.solver import CG
        from pyxu.opt.stop import MaxIter

        kwargs_fit = dict() if kwargs_fit is None else kwargs_fit
        kwargs_init = dict() if kwargs_init is None else kwargs_init
        kwargs_init.update(show_progress=kwargs_init.get("show_progress", False))

        if np.isclose(damp, 0):
            A = self.gram()
        else:
            A = self.gram() + HomothetyOp(cst=damp, dim=self.dim)

        cg = CG(A, **kwargs_init)
        if "stop_crit" not in kwargs_fit:
            # .pinv() may not have sufficiently converged given the default CG stopping criteria.
            # To avoid infinite loops, CG iterations are thresholded.
            sentinel = MaxIter(n=20 * A.dim)
            kwargs_fit["stop_crit"] = cg.default_stop_crit() | sentinel

        b = self.adjoint(arr)
        cg.fit(b=b, **kwargs_fit)
        return cg.solution()

    def dagger(
        self,
        damp: pxt.Real,
        kwargs_init=None,
        kwargs_fit=None,
    ) -> pxt.OpT:
        r"""
        Return the Moore-Penrose pseudo-inverse operator :math:`\mathbf{A}^\dagger`.

        Parameters
        ----------
        damp: Real
            Positive dampening factor regularizing the pseudo-inverse.
        kwargs_init: ~collections.abc.Mapping
            Optional kwargs to be passed to :py:meth:`~pyxu.opt.solver.cg.CG`'s ``__init__()`` method.
        kwargs_fit: ~collections.abc.Mapping
            Optional kwargs to be passed to :py:meth:`~pyxu.opt.solver.cg.CG`'s ``fit()`` method.

        Returns
        -------
        op: OpT
            (M, N) Moore-Penrose pseudo-inverse operator.
        """
        from pyxu.operator.interop.source import from_source

        def op_apply(_, arr: pxt.NDArray) -> pxt.NDArray:
            return self.pinv(
                arr,
                damp=_._damp,
                kwargs_init=_._kwargs_init,
                kwargs_fit=_._kwargs_fit,
            )

        def op_adjoint(_, arr: pxt.NDArray) -> pxt.NDArray:
            return self.T.pinv(
                arr,
                damp=_._damp,
                kwargs_init=_._kwargs_init,
                kwargs_fit=_._kwargs_fit,
            )

        kwargs_fit = dict() if kwargs_fit is None else kwargs_fit
        kwargs_init = dict() if kwargs_init is None else kwargs_init

        dagger = from_source(
            cls=SquareOp if (self.dim == self.codim) else LinOp,
            shape=(self.dim, self.codim),
            embed=dict(
                _name="dagger",
                _damp=damp,
                _kwargs_init=copy.copy(kwargs_init),
                _kwargs_fit=copy.copy(kwargs_fit),
            ),
            apply=op_apply,
            adjoint=op_adjoint,
            _expr=lambda _: (_._name, _, _._damp),
        )
        return dagger

    @classmethod
    def from_array(
        cls,
        A: typ.Union[pxt.NDArray, pxt.SparseArray],
        enable_warnings: bool = True,
    ) -> pxt.OpT:
        r"""
        Instantiate a :py:class:`~pyxu.abc.operator.LinOp` from its array representation.

        Parameters
        ----------
        A: NDArray
            (N, M) array
        enable_warnings: bool
            If ``True``, emit a warning in case of precision mis-match issues.

        Returns
        -------
        op: OpT
            (N, M) linear operator
        """
        from pyxu.operator.linop.base import _ExplicitLinOp

        return _ExplicitLinOp(cls, A, enable_warnings)


class SquareOp(LinOp):
    r"""
    Base class for *square* linear operators, i.e. :math:`\mathbf{A}: \mathbb{R}^{M} \to \mathbb{R}^{M}` (endomorphsisms).
    """

    @classmethod
    def properties(cls) -> cabc.Set[Property]:
        p = set(super().properties())
        p.add(Property.LINEAR_SQUARE)
        return frozenset(p)

    def __init__(self, shape: pxt.OpShape):
        super().__init__(shape=shape)
        assert self.dim == self.codim, f"shape: expected (M, M), got {self.shape}."

    @pxrt.enforce_precision()
    def trace(self, **kwargs) -> pxt.Real:
        """
        Compute trace of the operator.

        Parameters
        ----------
        method: "explicit" | "hutchpp"

            * If `explicit`, compute the exact trace.
            * If `hutchpp`, compute an approximation.

        kwargs: ~collections.abc.Mapping
            Optional kwargs passed to:

            * `explicit`: :py:func:`~pyxu.math.linalg.trace`
            * `hutchpp`: :py:func:`~pyxu.math.linalg.hutchpp`

        Returns
        -------
        tr: Real
            Trace estimate.
        """
        from pyxu.math.linalg import hutchpp, trace

        method = kwargs.get("method", "hutchpp").lower().strip()

        if method == "explicit":
            func = sig_func = trace
            estimate = lambda: func(op=self, **kwargs)
        elif method == "hutchpp":
            func = sig_func = hutchpp
            estimate = lambda: func(op=self, **kwargs)
        else:
            raise NotImplementedError

        # Filter unsupported kwargs
        sig = inspect.Signature.from_callable(sig_func)
        kwargs = {k: v for (k, v) in kwargs.items() if (k in sig.parameters)}

        tr = estimate()
        return tr


class NormalOp(SquareOp):
    r"""
    Base class for *normal* operators.

    Notes
    -----
    Normal operators commute with their adjoint, i.e.
    :math:`\mathbf{A} \mathbf{A}^{\ast} = \mathbf{A}^{\ast} \mathbf{A}`.
    It is `possible to show <https://www.wikiwand.com/en/Spectral_theorem#/Normal_matrices>`_ that
    an operator is normal iff it is *unitarily diagonalizable*, i.e.
    :math:`\mathbf{A} = \mathbf{U} \mathbf{D} \mathbf{U}^{\ast}`.
    """

    @classmethod
    def properties(cls) -> cabc.Set[Property]:
        p = set(super().properties())
        p.add(Property.LINEAR_NORMAL)
        return frozenset(p)

    def cogram(self) -> pxt.OpT:
        return self.gram()


class SelfAdjointOp(NormalOp):
    r"""
    Base class for *self-adjoint* operators, i.e.
    :math:`\mathbf{A}^{\ast} = \mathbf{A}`.
    """

    @classmethod
    def properties(cls) -> cabc.Set[Property]:
        p = set(super().properties())
        p.add(Property.LINEAR_SELF_ADJOINT)
        return frozenset(p)

    def adjoint(self, arr: pxt.NDArray) -> pxt.NDArray:
        return self.apply(arr)


class UnitOp(NormalOp):
    r"""
    Base class for *unitary* operators, i.e.
    :math:`\mathbf{A} \mathbf{A}^{\ast} = \mathbf{A}^{\ast} \mathbf{A} = I`.
    """

    @classmethod
    def properties(cls) -> cabc.Set[Property]:
        p = set(super().properties())
        p.add(Property.LINEAR_UNITARY)
        return frozenset(p)

    def __init__(self, shape: pxt.OpShape):
        super().__init__(shape=shape)
        self.lipschitz = UnitOp.estimate_lipschitz(self)

    def estimate_lipschitz(self, **kwargs) -> pxt.Real:
        return 1

    @pxrt.enforce_precision(i=("arr", "damp"))
    def pinv(self, arr: pxt.NDArray, damp: pxt.Real, **kwargs) -> pxt.NDArray:
        out = self.adjoint(arr)
        if not np.isclose(damp, 0):
            out = pxu.copy_if_unsafe(out)
            out /= 1 + damp
        return out

    def dagger(self, damp: pxt.Real, **kwargs) -> pxt.OpT:
        op = self.T / (1 + damp)
        return op

    def gram(self) -> pxt.OpT:
        from pyxu.operator.linop import IdentityOp

        return IdentityOp(dim=self.dim).squeeze()

    def svdvals(self, **kwargs) -> pxt.NDArray:
        gpu = kwargs.get("gpu", False)
        xp = pxd.NDArrayInfo.from_flag(gpu).module()
        width = pxrt.getPrecision()

        D = xp.ones(kwargs["k"], dtype=width.value)
        return D


class ProjOp(SquareOp):
    r"""
    Base class for *projection* operators.

    Projection operators are *idempotent*, i.e.
    :math:`\mathbf{A}^{2} = \mathbf{A}`.
    """

    @classmethod
    def properties(cls) -> cabc.Set[Property]:
        p = set(super().properties())
        p.add(Property.LINEAR_IDEMPOTENT)
        return frozenset(p)


class OrthProjOp(ProjOp, SelfAdjointOp):
    r"""
    Base class for *orthogonal projection* operators.

    Orthogonal projection operators are *idempotent* and *self-adjoint*, i.e.
    :math:`\mathbf{A}^{2} = \mathbf{A}` and :math:`\mathbf{A}^{\ast} = \mathbf{A}`.
    """

    @classmethod
    def properties(cls) -> cabc.Set[Property]:
        p = set()
        for klass in cls.__bases__:
            p |= klass.properties()
        return frozenset(p)

    def __init__(self, shape: pxt.OpShape):
        super().__init__(shape=shape)
        self.lipschitz = OrthProjOp.estimate_lipschitz(self)

    def estimate_lipschitz(self, **kwargs) -> pxt.Real:
        return 1

    def gram(self) -> pxt.OpT:
        return self.squeeze()

    def cogram(self) -> pxt.OpT:
        return self.squeeze()

    @pxrt.enforce_precision(i=("arr", "damp"))
    def pinv(self, arr: pxt.NDArray, damp: pxt.Real, **kwargs) -> pxt.NDArray:
        out = self.apply(arr)
        if not np.isclose(damp, 0):
            out = pxu.copy_if_unsafe(out)
            out /= 1 + damp
        return out

    def dagger(self, damp: pxt.Real, **kwargs) -> pxt.OpT:
        op = self / (1 + damp)
        return op


class PosDefOp(SelfAdjointOp):
    r"""
    Base class for *positive-definite* operators.
    """

    @classmethod
    def properties(cls) -> cabc.Set[Property]:
        p = set(super().properties())
        p.add(Property.LINEAR_POSITIVE_DEFINITE)
        return frozenset(p)


class LinFunc(ProxDiffFunc, LinOp):
    r"""
    Base class for real-valued linear functionals :math:`f: \mathbb{R}^{M} \to \mathbb{R}`.

    Instances of this class must implement
    :py:meth:`~pyxu.abc.operator.Map.apply`, and
    :py:meth:`~pyxu.abc.operator.LinOp.adjoint`.

    If known, the Lipschitz constant :math:`L` should be stored in the
    :py:attr:`~pyxu.abc.operator.Map.lipschitz` property.
    """

    @classmethod
    def properties(cls) -> cabc.Set[Property]:
        p = set()
        for klass in cls.__bases__:
            p |= klass.properties()
        return frozenset(p)

    def __init__(self, shape: pxt.OpShape):
        super().__init__(shape=shape)
        ProxDiffFunc.__init__(self, shape)
        LinOp.__init__(self, shape)

    def jacobian(self, arr: pxt.NDArray) -> pxt.OpT:
        return LinOp.jacobian(self, arr)

    def estimate_lipschitz(self, **kwargs) -> pxt.Real:
        # Try all backends until one works.
        for ndi in pxd.NDArrayInfo:
            try:
                xp = ndi.module()
                g = self.grad(xp.ones(self.dim))
                L = float(xp.linalg.norm(g))
                return L
            except Exception:
                pass

    @pxrt.enforce_precision(i="arr")
    def grad(self, arr: pxt.NDArray) -> pxt.NDArray:
        xp = pxu.get_array_module(arr)
        x = xp.ones((*arr.shape[:-1], 1), dtype=arr.dtype)
        g = self.adjoint(x)
        return g

    @pxrt.enforce_precision(i=("arr", "tau"))
    def prox(self, arr: pxt.NDArray, tau: pxt.Real) -> pxt.NDArray:
        # out = arr - tau * self.grad(arr)
        out = pxu.copy_if_unsafe(self.grad(arr))
        out *= -tau
        out += arr
        return out

    @pxrt.enforce_precision(i=("arr", "sigma"))
    def fenchel_prox(self, arr: pxt.NDArray, sigma: pxt.Real) -> pxt.NDArray:
        return self.grad(arr)

    def cogram(self) -> pxt.OpT:
        from pyxu.operator.linop import HomothetyOp

        L = self.estimate_lipschitz()
        return HomothetyOp(cst=L**2, dim=1)

    def svdvals(self, **kwargs) -> pxt.NDArray:
        gpu = kwargs.get("gpu", False)
        xp = pxd.NDArrayInfo.from_flag(gpu).module()
        width = pxrt.getPrecision()

        L = self.estimate_lipschitz()
        D = xp.array([L], dtype=width.value)
        return D

    def asarray(self, **kwargs) -> pxt.NDArray:
        xp = kwargs.get("xp", pxd.NDArrayInfo.default().module())
        dtype = kwargs.get("dtype", pxrt.getPrecision().value)

        with pxrt.EnforcePrecision(False):
            x = xp.ones((1, 1), dtype=dtype)
            A = self.adjoint(x)
        return A

    @classmethod
    def from_array(
        cls,
        A: typ.Union[pxt.NDArray, pxt.SparseArray],
        enable_warnings: bool = True,
    ) -> pxt.OpT:
        if A.ndim == 1:
            A = A.reshape((1, -1))
        op = super().from_array(A, enable_warnings)
        return op


def _core_operators() -> cabc.Set[pxt.OpC]:
    # Operators which can be sub-classed by end-users and participate in arithmetic rules.
    ops = set()
    for _ in globals().values():
        if inspect.isclass(_) and issubclass(_, Operator):
            ops.add(_)
    ops.remove(Operator)
    return ops


def _is_real(x) -> bool:
    if isinstance(x, pxt.Real):
        return True
    elif isinstance(x, pxd.supported_array_types()) and (x.size == 1):
        return True
    else:
        return False


__all__ = [
    "Operator",
    "Property",
    *map(lambda _: _.__name__, _core_operators()),
]
