from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pytest
from matplotlib.figure import Figure as MplFigure

from pymatgen.util.plotting import (
    add_fig_kwargs,
    format_formula,
    get_ax3d_fig,
    get_ax_fig,
    get_axarray_fig_plt,
    periodic_table_heatmap,
    pretty_plot,
    pretty_plot_two_axis,
    pretty_polyfit_plot,
    van_arkel_triangle,
)
from pymatgen.util.testing import MatSciTest

try:
    import pymatviz
    from plotly.graph_objects import Figure
except ImportError:
    pymatviz = None


class TestFunc(MatSciTest):
    def test_plot_periodic_heatmap(self):
        random_data = {"Te": 0.11083, "Au": 0.75756, "Th": 1.24758, "Ni": -2.0354}
        fig = periodic_table_heatmap(random_data)
        if pymatviz is not None:
            assert isinstance(fig, Figure)
        else:
            assert isinstance(fig, plt.Axes)

        # Test all keywords
        periodic_table_heatmap(
            random_data,
            cmap="plasma",
            max_row=10,
            cbar_label_size=18,
            cmap_range=[0, 1],
            cbar_label="Hello World",
            blank_color="white",
            value_format=".4f",
            edge_color="black",
            value_fontsize=12,
            symbol_fontsize=18,
            readable_fontcolor=True,
        )

    def test_van_arkel_triangle(self):
        random_list = [("Fe", "C"), ("Ni", "F")]
        ax = van_arkel_triangle(random_list)
        assert isinstance(ax, plt.Axes)
        assert ax.get_title() == ""
        assert ax.get_xlabel() == r"$\frac{\chi_{A}+\chi_{B}}{2}$"
        assert ax.get_ylabel() == r"$|\chi_{A}-\chi_{B}|$"
        ax = van_arkel_triangle(random_list, annotate=True)
        assert isinstance(ax, plt.Axes)


class TestPrettyPlot:
    def teardown_method(self):
        plt.close("all")

    def test_default_returns_axes(self):
        ax = pretty_plot()
        assert isinstance(ax, plt.Axes)

    def test_height_defaults_to_golden_ratio(self):
        ax = pretty_plot(width=10)
        fig = ax.figure
        w, h = fig.get_size_inches()
        assert w == 10
        # int(width * golden_ratio) where golden = (sqrt(5)-1)/2 ≈ 0.618
        assert h == int(10 * ((5**0.5 - 1) / 2))

    def test_with_existing_axes_resizes_figure(self):
        fig, ax = plt.subplots()
        out = pretty_plot(width=6, height=4, ax=ax)
        assert out is ax
        assert tuple(fig.get_size_inches()) == (6, 4)


class TestPrettyPlotTwoAxis:
    def teardown_method(self):
        plt.close("all")

    def test_sequence_inputs(self):
        x = [0, 1, 2, 3]
        y1 = [1, 2, 3, 4]
        y2 = [4, 3, 2, 1]
        ax1 = pretty_plot_two_axis(x, y1, y2, xlabel="x", y1label="y1", y2label="y2", dpi=None)
        assert isinstance(ax1, plt.Axes)
        assert ax1.get_xlabel() == "x"
        assert ax1.get_ylabel() == "y1"

    def test_dict_inputs(self):
        x = [0, 1, 2]
        y1 = {"a": [1, 2, 3], "b": [3, 2, 1]}
        y2 = {"c": [0, 1, 0]}
        ax1 = pretty_plot_two_axis(x, y1, y2, dpi=None)
        assert isinstance(ax1, plt.Axes)


class TestPrettyPolyfitPlot:
    def teardown_method(self):
        plt.close("all")

    def test_linear_fit(self):
        x = np.array([1.0, 2.0, 3.0, 4.0])
        y = 2 * x + 1
        ax = pretty_polyfit_plot(x, y, deg=1, xlabel="x", ylabel="y")
        assert isinstance(ax, plt.Axes)
        assert ax.get_xlabel() == "x"
        assert ax.get_ylabel() == "y"

    def test_quadratic_fit(self):
        x = np.linspace(-2, 2, 20)
        y = x**2
        ax = pretty_polyfit_plot(x, y, deg=2)
        assert isinstance(ax, plt.Axes)


class TestFormatFormula:
    @pytest.mark.parametrize(
        ("formula", "expected"),
        [
            ("Li2O", "$Li_{2}O$"),
            ("Fe", "$Fe$"),
            ("LiFePO4", "$LiFePO_{4}$"),
            ("H2O", "$H_{2}O$"),
            ("Li3V2(PO4)3", "$Li_{3}V_{2}(PO_{4})_{3}$"),
        ],
    )
    def test_format(self, formula, expected):
        assert format_formula(formula) == expected


class TestAxFigHelpers:
    def teardown_method(self):
        plt.close("all")

    def test_get_ax_fig_creates_when_none(self):
        ax, fig = get_ax_fig()
        assert isinstance(ax, plt.Axes)
        assert isinstance(fig, MplFigure)

    def test_get_ax_fig_passes_through(self):
        existing_ax = plt.figure().gca()
        ax, _fig = get_ax_fig(ax=existing_ax)
        assert ax is existing_ax

    def test_get_ax3d_fig_creates_when_none(self):
        ax, fig = get_ax3d_fig()
        assert isinstance(fig, MplFigure)
        assert ax.name == "3d"

    def test_get_ax3d_fig_passes_through(self):
        existing_ax = plt.figure().add_subplot(projection="3d")
        ax, _fig = get_ax3d_fig(ax=existing_ax)
        assert ax is existing_ax

    def test_get_axarray_fig_plt_creates(self):
        ax_array, fig, plt_mod = get_axarray_fig_plt(None, nrows=2, ncols=2)
        assert ax_array.shape == (2, 2)
        assert isinstance(fig, MplFigure)
        assert plt_mod is plt

    def test_get_axarray_fig_plt_squeezes_single(self):
        ax_array, _fig, _ = get_axarray_fig_plt(None, nrows=1, ncols=1)
        assert isinstance(ax_array, plt.Axes)

    def test_get_axarray_fig_plt_passthrough(self):
        _fig, axes = plt.subplots(1, 2)
        ax_array, _fig_out, _ = get_axarray_fig_plt(axes, nrows=1, ncols=2)
        assert ax_array.shape == (2,)


class TestAddFigKwargs:
    def teardown_method(self):
        plt.close("all")

    def _make_fn(self):
        @add_fig_kwargs
        def make_fig():
            fig, ax = plt.subplots()
            ax.plot([0, 1], [0, 1])
            return fig

        return make_fig

    def test_returns_figure_with_title(self):
        fig = self._make_fn()(title="hello", show=False)
        assert isinstance(fig, MplFigure)
        assert fig._suptitle.get_text() == "hello"

    def test_size_kwargs(self):
        fig = self._make_fn()(size_kwargs={"w": 6, "h": 4}, show=False)
        assert tuple(fig.get_size_inches()) == (6, 4)

    def test_savefig(self, tmp_path):
        out = tmp_path / "out.png"
        self._make_fn()(savefig=str(out), show=False)
        assert out.exists()

    def test_ax_grid_and_tight_layout(self):
        fig = self._make_fn()(ax_grid=True, tight_layout=True, show=False)
        assert all(any(line.get_visible() for line in ax.get_xgridlines()) for ax in fig.axes)

    def test_ax_annotate(self):
        @add_fig_kwargs
        def make_two_panel():
            fig, _ = plt.subplots(1, 2)
            return fig

        fig = make_two_panel(ax_annotate=True, show=False)
        # Each axis should have one annotation: "(a)", "(b)"
        for ax, expected in zip(fig.axes, ["(a)", "(b)"], strict=True):
            assert any(child.get_text() == expected for child in ax.get_children() if hasattr(child, "get_text"))

    def test_fig_close_returns_fig(self):
        fig = self._make_fn()(show=False, fig_close=True)
        assert isinstance(fig, MplFigure)

    def test_passthrough_when_func_returns_none(self):
        @add_fig_kwargs
        def make_nothing():
            return None

        assert make_nothing(title="x", show=False) is None
