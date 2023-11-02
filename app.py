from __future__ import annotations

from shiny import App, reactive, render, ui, req
import shinyswatch
import shiny.experimental as x

from htmltools import css
import pandas as pd
import numpy as np
import numpy.matlib as m

from scipy.stats import multivariate_normal
import sklearn.discriminant_analysis as da

import matplotlib.pyplot as plt
import copy

COL_GRP1 = "#0a9396"
COL_GRP2 = "#f95738"
COL_LDA_LINE = "#ee9b00"
COL_L1 = "#6c757d"
COL_DIM = "#e9d8a6"
COL_DIST = "#06d6a0"
ALPHA_DIM = 0.4
LINE_DIM = 0.5

# https://icons.getbootstrap.com/icons/question-circle-fill/
question_circle_fill = ui.HTML(
    '<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" fill="gray" class="bi bi-question-circle-fill mb-1" viewBox="0 0 16 16"> \
    <path d="M16 8A8 8 0 1 1 0 8a8 8 0 0 1 16 0zM5.496 6.033h.825c.138 0 .248-.113.266-.25.09-.656.54-1.134 1.342-1.134.686 0 1.314.343 1.314 1.168 0 .635-.374.927-.965 1.371-.673.489-1.206 1.06-1.168 1.987l.003.217a.25.25 0 0 0 .25.246h.811a.25.25 0 0 0 .25-.25v-.105c0-.718.273-.927 1.01-1.486.609-.463 1.244-.977 1.244-2.056 0-1.511-1.276-2.241-2.673-2.241-1.267 0-2.655.59-2.75 2.286a.237.237 0 0 0 .241.247zm2.325 6.443c.61 0 1.029-.394 1.029-.927 0-.552-.42-.94-1.029-.94-.584 0-1.009.388-1.009.94 0 .533.425.927 1.01.927z"/> \
    </svg>'
)


class Point:
    def __init__(self, num_points, center, cov):
        self.num_points = num_points
        self.center = center
        self.cov = cov
        self.projection = None
        self.lda_projection = None

        self.points = multivariate_normal(self.center, self.cov).rvs(self.num_points)

    def __repr__(self):
        return f"Point({self.num_points}, {self.center}, {self.cov})"

    def __str__(self):
        return f"Point({self.num_points}, {self.center}, {self.cov})"

    def __len__(self):
        return self.num_points

    def __getitem__(self, idx):
        return self.points[idx]

    def project(self, slope):
        V = [1, slope]
        self.projection = np.array(
            [
                list(m.dot(m.outer(V, V), self.points[_] / m.inner(V, V)))
                for _ in range(self.num_points)
            ]
        )

    def project_lda(self, scaling):
        scaling = list(scaling.flatten())
        V = [1, scaling[1] / scaling[0]]
        self.lda_projection = np.array(
            [
                list(m.dot(m.outer(V, V), self.points[_] / m.inner(V, V)))
                for _ in range(self.num_points)
            ]
        )


def var(a, b):
    return np.linalg.norm(a - b)


app_ui = ui.page_fluid(
    shinyswatch.theme.cosmo(),
    ui.br(),
    ui.panel_title("How Does Linear Discriminant Analysis Work?"),
    ui.br(),
    ui.p(
        "Linear Discriminant Analysis (LDA) is a method used in statistics, pattern recognition, and machine learning to find a straight line (or plane) that best separates two or more groups of data.",
        " Imagine you have two clusters of dots on a piece of paper, and you want to draw a single straight line so that most of the dots from one group are on one side and most from the other group are on the opposite side.",
        " LDA helps in finding that optimal line. It's like drawing the best boundary between the groups to distinguish them.",
    ),
    ui.p(
        "Let's see this in action.",
        "Here are two groups of points. ",
        "The objective is to find a line which best separates the two groups. ",
    ),
    ui.row(
        ui.column(
            2,
            ui.br(),
            ui.panel_well(
                ui.input_select(
                    id="case",
                    label=x.ui.tooltip(
                        ui.span("Cases ", question_circle_fill),
                        ui.markdown(
                            "Each case here has a different distribution of points.\
                        These points are created using 'multivariate_normal' function from scipy.stats."
                        ),
                    ),
                    multiple=False,
                    choices={
                        "1": "Parallel Ellipses",
                        "2": "Angled Ellipses",
                        "3": "Circles",
                        "4": "Offset Ellipses",
                    },
                ),
                ui.input_slider(
                    "n_points", "Points in each group", 10, 50, 30, step=10
                ),
                ui.input_action_button("generate", "Generate"),
            ),
            ui.br(),
            ui.panel_well(
                ui.input_slider(
                    "angle",
                    x.ui.tooltip(
                        ui.span("Angle of Line L ", question_circle_fill),
                        "The angle of the arbitrary line L",
                    ),
                    0,
                    180,
                    20,
                    step=1,
                    post="°",
                ),
            ),
            ui.br(),
            ui.panel_well(
                ui.input_checkbox("plot_proj_L1", "Projections on L", False),
                ui.input_checkbox(
                    "plot_proj_lda",
                    "Projections on LDA",
                    False,
                ),
            ),
        ),
        ui.column(
            10,
            ui.row(
                ui.column(
                    6,
                    ui.output_plot("plot", height="600px", width="600px"),
                    ui.HTML(
                        f"The plot shows you two lines: an optimal separator selected by <span style='color:{COL_LDA_LINE};'>LDA</span> and an arbitrary <span style='color:{COL_L1};'>line L</span> that is not optimal."
                    ),
                    ui.HTML(
                        f" The numbers in the four corners display the variance of each point cloud when projected on the <span style='color:{COL_LDA_LINE};'>LDA line (indicated by ▼)</span> or the arbitrary <span style='color:{COL_L1};'>line L (indicated by +)</span>. You can turn on the projections using the buttons on the left."
                    ),
                ),
                ui.column(
                    6,
                    ui.output_plot("plot_by_angle", height="600px", width="600px"),
                    ui.HTML(
                        "This graph visualizes the trade-offs between intra-group variance and inter-group distance as the angle of projection changes.",
                    ),
                    ui.HTML(
                        f"  The ideal angle - given by <span style='color:{COL_LDA_LINE};'>LDA</span> - has the best balance between <i>minimizing</i> within-group-variance and <i>maximizing</i> between-group-distance.",
                    ),
                ),
            ),
        ),
    ),
    ui.br(),
    ui.row(
        ui.HTML(
            "<div style='text-align: center; color: gray; font-size:0.9em;'> Created using Shiny for Python | <a href = 'http://www.rsangole.com'>Rahul Sangole</a> | Nov '23</div>"
        )
    ),
)


def server(input, output, session):
    @reactive.Calc
    @reactive.event(input.generate, ignore_none=False)
    def pt1_settings():
        """Point Cloud 2 Settings

        Returns:
            list, list: center, covariance
        """
        if input.case() == "1":
            return [-2, 0], [0.2, 1]
        elif input.case() == "2":
            return [-2, 1], [[0.5, 0.4], [0.4, 0.6]]
        elif input.case() == "3":
            return [-2, -2], [[0.5, 0], [0, 0.5]]
        elif input.case() == "4":
            return [-2, -2], [[0.2, 0], [0, 2]]

    @reactive.Calc
    @reactive.event(input.generate, ignore_none=False)
    def pt2_settings():
        """Point Cloud 2 Settings

        Returns:
            list, list: center, covariance
        """
        if input.case() == "1":
            return [2, 0], [0.2, 1]
        elif input.case() == "2":
            return [2, -1], [[0.5, 0.4], [0.4, 0.6]]
        elif input.case() == "3":
            return [2, 2], [[0.5, 0], [0, 0.5]]
        elif input.case() == "4":
            return [2, 2], [[0.2, 0], [0, 2]]

    @reactive.Calc
    def get_slope():
        return np.tan(np.deg2rad(input.angle()))

    @reactive.Calc
    @reactive.event(input.generate, ignore_none=False)
    def num_points():
        return input.n_points()

    @reactive.Calc
    def Point1():
        center, cov = pt1_settings()
        return Point(num_points(), center, cov)

    @reactive.Calc
    def Point2():
        center, cov = pt2_settings()
        return Point(num_points(), center, cov)

    @reactive.Calc
    def project():
        pt1 = Point1()
        pt2 = Point2()

        pt1.project(get_slope())
        pt2.project(get_slope())

        return (pt1, pt2)

    @reactive.Calc
    def calcs():
        _pt1, _pt2 = project()

        lda = da.LinearDiscriminantAnalysis(n_components=1)
        X = np.vstack([_pt1.points, _pt2.points])
        y = np.hstack([np.zeros(len(_pt1)), np.ones(len(_pt2))])
        lda.fit(X, y)

        _pt1.project_lda(lda.scalings_)
        _pt2.project_lda(lda.scalings_)

        lda_angle = np.rad2deg(np.tan(float(lda.scalings_[1] / lda.scalings_[0])))

        if lda_angle < 0:
            lda_angle += 180

        return (_pt1, _pt2, lda.scalings_, lda_angle)

    @reactive.Calc
    def lda_by_angle():
        angles = list(range(0, 180, 1))
        result = []
        for a in angles:
            slope = np.tan(np.deg2rad(a))
            n_points = num_points()

            p1 = copy.copy(Point1())
            p2 = copy.copy(Point2())
            p1.project(slope)
            p2.project(slope)

            V1 = var(p1.projection, np.mean(p1.projection, axis=0))
            V2 = var(p2.projection, np.mean(p2.projection, axis=0))

            V1_V2 = var(np.mean(p1.projection, axis=0), np.mean(p2.projection, axis=0))

            result += [[a, V1, V2, V1_V2]]

        return result

    @output
    @render.plot
    def plot_by_angle():
        _, _, _, lda_angle = calcs()

        res = pd.DataFrame(
            lda_by_angle(),
            columns=["angle", "V1", "V2", "V1V2"],
        )
        fig, ax = plt.subplots()
        ax.plot(res.angle, res.V1, label="Var(Group1 → L)", color=COL_GRP1)
        ax.plot(res.angle, res.V2, label="Var(Group2 → L)", color=COL_GRP2)
        ax.plot(res.angle, res.V1V2, label="Distance Between Groups", color=COL_DIST)
        ax.vlines(
            input.angle(),
            0,
            10,
            color=COL_L1,
            linestyle="dashed",
            lw=1,
        ),
        ax.vlines(
            lda_angle,
            0,
            10,
            color=COL_LDA_LINE,
            linestyle="dashed",
            lw=1,
        )
        ax.text(
            (lda_angle + 2) if lda_angle < 159 else (lda_angle - 2),
            0.1,
            f"LDA {round(lda_angle,1)}°",
            ha="left" if lda_angle < 159 else "right",
            color=COL_L1,
        )
        ax.text(
            (input.angle() + 2) if input.angle() < 159 else (input.angle() - 2),
            0.4,
            f"L1 {round(input.angle(),1)}°",
            ha="left" if input.angle() < 159 else "right",
            color=COL_L1,
        )
        ax.set_title("Change in Variance & Distance")
        ax.set_xlabel("Angle (°)")
        ax.set_xlim([0, 180])
        ax.set_ylim([0, 10])
        ax.set_ylabel("Variance, Distance")
        ax.legend()
        return fig

    @output
    @render.plot
    def plot():
        _pt1, _pt2, lda_scaling, lda_angle = calcs()

        V1 = var(_pt1.projection, np.mean(_pt1.projection, axis=0))
        V2 = var(_pt2.projection, np.mean(_pt2.projection, axis=0))

        V1_lda = var(_pt1.lda_projection, np.mean(_pt1.lda_projection, axis=0))
        V2_lda = var(_pt2.lda_projection, np.mean(_pt2.lda_projection, axis=0))

        V1_V2 = var(np.mean(_pt1.projection, axis=0), np.mean(_pt2.projection, axis=0))
        V1_lag_V2_lag = var(
            np.mean(_pt1.lda_projection, axis=0), np.mean(_pt2.lda_projection, axis=0)
        )

        if input.plot_proj_L1() | input.plot_proj_lda():
            pt1_color = COL_DIM
            pt2_color = COL_DIM
            alpha = ALPHA_DIM
            alpha_line = LINE_DIM
        else:
            pt1_color = COL_GRP1
            pt2_color = COL_GRP2
            alpha = 1
            alpha_line = 1

        fig, ax = plt.subplots()
        ax.plot(
            _pt1.points[:, 0],
            _pt1.points[:, 1],
            marker="o",
            fillstyle="none",
            linestyle="none",
            color=pt1_color,
            alpha=alpha,
        )
        ax.plot(
            _pt2.points[:, 0],
            _pt2.points[:, 1],
            marker="o",
            fillstyle="none",
            linestyle="none",
            color=pt2_color,
            alpha=alpha,
        )
        ax.axline(
            (0, 0),
            slope=get_slope(),
            color=COL_L1,
            linestyle="dashed",
            lw=1,
            label="L",
            alpha=alpha_line,
        )
        ax.axline(
            (0, 0),
            slope=float(lda_scaling[1] / lda_scaling[0]),
            color=COL_LDA_LINE,
            linestyle="dashed",
            lw=1,
            label="LDA",
            alpha=alpha_line,
        )
        if input.plot_proj_L1():
            ax.plot(
                _pt1.projection[:, 0],
                _pt1.projection[:, 1],
                marker="+",
                markersize=10,
                fillstyle="none",
                linestyle="none",
                color=COL_GRP1,
                alpha=0.9,
            )
            ax.plot(
                _pt2.projection[:, 0],
                _pt2.projection[:, 1],
                marker="+",
                markersize=10,
                fillstyle="none",
                linestyle="none",
                color=COL_GRP2,
                alpha=0.9,
            )
        if input.plot_proj_lda():
            ax.plot(
                _pt1.lda_projection[:, 0],
                _pt1.lda_projection[:, 1],
                marker="v",
                markersize=7,
                fillstyle="none",
                linestyle="none",
                color=COL_GRP1,
            )
            ax.plot(
                _pt2.lda_projection[:, 0],
                _pt2.lda_projection[:, 1],
                marker="v",
                markersize=7,
                fillstyle="none",
                linestyle="none",
                color=COL_GRP2,
            )
        ax.set_xlim([-5, 5])
        ax.set_ylim([-5, 5])
        plt.text(-4.9, -4, f"Group 1", ha="left", color=COL_GRP1)
        plt.text(-4.9, -4.4, f"Var(+): {round(V1,3)}", ha="left", color=COL_L1)
        plt.text(4.9, -4.4, f"Var(+): {round(V2,3)}", ha="right", color=COL_L1)
        plt.text(4.9, -4, f"Group 2", ha="right", color=COL_GRP2)
        plt.text(
            -4.9,
            -4.8,
            f"Var(▼): {round(V1_lda,3)}",
            ha="left",
            color=COL_LDA_LINE,
        )
        plt.text(
            4.9,
            -4.8,
            f"Var(▼): {round(V2_lda,3)}",
            ha="right",
            color=COL_LDA_LINE,
        )
        ax.set_title(f"LDA Optimal Angle: {round(lda_angle, 2)}°")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend(loc="upper left")
        return fig


app = App(app_ui, server)
