from __future__ import annotations

from shiny import App, reactive, render, ui, req
from shinywidgets import render_altair
import shinyswatch
import shiny.experimental as x
import pandas as pd
from htmltools import css
import numpy as np
import altair as alt
from scipy.stats import norm
import plotly.express as px
import plotly.graph_objects as go
# import altair as alt    
# alt.data_transformers.disable_max_rows()
from shinywidgets import output_widget, render_widget

import matplotlib.pyplot as plt

COL_GRtreatment = "#0a9396"
COL_GRcontrol = "#f95738"
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


app_ui = ui.page_fluid(
    ui.panel_title("Why “Statistical Significance” Is Pointless"),
    ui.br(),
    ui.p(
        "Article author: Samuele Mazzanti",
        ui.br(),
        "Article link: https://towardsdatascience.com/why-statistical-significance-is-pointless-a7644be30266"
    ),
    # ui.p(
    #     "Let's see this in action.",
    #     "Here are two groups of points. ",
    #     "The objective is to find a line which best separates the two groups. ",
    # ),
    ui.row(
        ui.column(
            2,
            ui.br(),
            ui.panel_well(
                # ui.input_select(
                #     id="case",
                #     label=x.ui.tooltip(
                #         ui.span("Cases ", question_circle_fill),
                #         ui.markdown(
                #             "Each case here has a different distribution of points.\
                #         These points are created using 'multivariate_normal' function from scipy.stats."
                #         ),
                #     ),
                #     multiple=False,
                #     choices={
                #         "1": "Parallel Ellipses",
                #         "2": "Angled Ellipses",
                #         "3": "Circles",
                #         "4": "Offset Ellipses",
                #     },
                # ),
                ui.input_numeric(
                    id="treatment_mean",
                    label="Treatment Mean",
                    value=10,
                    min=0,
                    max=-20,
                    step=0.1,
                ),
                ui.input_numeric(
                    id="control_mean",
                    label="Control Mean",
                    min=0,
                    max=20,
                    value=10.5,
                    step=0.1,
                ),
                ui.input_numeric(
                    id="treatment_cov",
                    label="Treatment Std Dev",
                    value=1,
                    min=0,
                    max=10,
                    step=0.1,
                ),
                ui.input_numeric(
                    id="control_cov",
                    label="Control Std Dev",
                    value=1,
                    min=0,
                    max=10,
                    step=0.1,
                ),
                ui.input_slider(
                    id="n_points",
                    label="Points in each group",
                    min=10,
                    max=300,
                    value=100,
                    step=10,
                ),
                ui.input_slider(
                    id="n_permutations",
                    label="Permutations",
                    min=1000,
                    max=10000,
                    value=10000,
                    step=1000,
                ),
                ui.hr(),
                ui.input_action_button("generate", "Generate"),
            ),
            ui.br(),
            # ui.panel_well(
            #     ui.input_slider(
            #         "angle",
            #         x.ui.tooltip(
            #             ui.span("Angle of Line L ", question_circle_fill),
            #             "The angle of the arbitrary line L",
            #         ),
            #         0,
            #         180,
            #         20,
            #         step=1,
            #         post="°",
            #     ),
            # ),
            # ui.br(),
            # ui.panel_well(
            #     ui.input_checkbox("plot_proj_L1", "Projections on L", False),
            #     ui.input_checkbox(
            #         "plot_proj_lda",
            #         "Projections on LDA",
            #         False,
            #     ),
            # ),
        ),
        ui.column(
            10,
            ui.row(
                ui.column(
                    6,
                    output_widget(
                        "treatment_control_hist", height="400px", width="400px"
                    ),
                    ui.output_text("txt_pop_dif"),  
                    ui.output_text("txt_sample_dif"),  
                    #         ui.HTML(
                    #             f"The plot shows you two lines: an optimal separator selected by <span style='color:{COL_LDA_LINE};'>LDA</span> and an arbitrary <span style='color:{COL_L1};'>line L</span> that is not optimal."
                    #         ),
                    #         ui.HTML(
                    #             f" The numbers in the four corners display the variance of each point cloud when projected on the <span style='color:{COL_LDA_LINE};'>LDA line (indicated by ▼)</span> or the arbitrary <span style='color:{COL_L1};'>line L (indicated by +)</span>. You can turn on the projections using the buttons on the left."
                    #         ),
                ),
                ui.column(
                    6,
                    output_widget("permutation_hist", height="400px", width="400px"),
                    ui.output_text("txt_perm"),  

                    #         ui.output_plot("plot_by_angle", height="600px", width="600px"),
                    #         ui.HTML(
                    #             "This graph visualizes the trade-offs between intra-group variance and inter-group distance as the angle of projection changes.",
                    #         ),
                    #         ui.HTML(
                    #             f"  The ideal angle - given by <span style='color:{COL_LDA_LINE};'>LDA</span> - has the best balance between <i>minimizing</i> within-group-variance and <i>maximizing</i> between-group-distance.",
                    #         ),
                ),
            ),
        ),
    ),
    ui.br(),
    ui.row(
        ui.HTML(
            "<div style='text-align: center; color: gray; font-size:0.9em;'> Created using Shiny for Python | <a href = 'http://www.rsangole.com'>Rahul Sangole</a> | Dec '24</div>"
        )
    ),

    theme=shinyswatch.theme.cosmo,
)


def server(input, output, session):
    @reactive.Calc
    @reactive.event(input.generate, ignore_none=False)
    def treatment():
        return norm.rvs(input.treatment_mean(), input.treatment_cov(), input.n_points())

    @reactive.Calc
    @reactive.event(input.generate, ignore_none=False)
    def control():
        return norm.rvs(input.control_mean(), input.control_cov(), input.n_points())

    @reactive.Calc
    @reactive.event(input.generate, ignore_none=False)
    def permute():
        combined = np.concatenate([treatment(), control()])
        permutation_results = []
        for _ in range(input.n_permutations()):
            combined = np.random.permutation(combined)
            perm_treatment = combined[: len(treatment())]
            perm_control = combined[-len(treatment()) :]
            permutation_results.append(np.mean(perm_treatment) - np.mean(perm_control))
        return permutation_results

    @render_widget
    def treatment_control_hist():
        res = pd.DataFrame(
            {"Treatment": treatment(), "Control": control()},
            index=range(len(treatment())),
        ).melt()
        fig = (
            px.histogram(
                res,
                x="value",
                color="variable",
                marginal="rug",
                color_discrete_sequence=[COL_GRtreatment, COL_GRcontrol],
                # opacity=0.75,
            )
            .update_layout(
                title={"text": "", "x": 0.5},
                yaxis_title="Count",
                xaxis_title="Treatment, Control Values",
                legend_title="",
                legend=dict(
                    orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
                ),
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
        )
        return fig

    @render_widget
    def permutation_hist():
        res = pd.DataFrame(permute(), columns=["Permutation"]).melt()
        mean_diff = np.mean(res["value"])
        fig = px.histogram(
            res,
            x="value",
            color="variable",
            marginal="rug",
            color_discrete_sequence=[COL_L1],
        ).update_layout(
            title={"text": "", "x": 0.5},
            yaxis_title="Count",
            xaxis_title="Difference in Means",
            legend_title="",
            plot_bgcolor="white",
            paper_bgcolor="white",
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        fig.add_shape(
            type="line",
            x0=mean_diff,
            y0=0,
            x1=mean_diff,
            y1=1,
            xref="x",
            yref="paper",
            line=dict(
                # color="DarkGray",
                width=2,
                dash="dash",
            ),
        )
        fig.add_annotation(
            x=mean_diff,
            y=1,
            xref="x",
            yref="paper",
            text=f"Mean {mean_diff:.3f}",
            showarrow=False,
            yshift=1,
            xanchor="left",
            # font=dict(color="LightGray"),
        )

        return fig
    
    @render.text  
    def txt_pop_dif():
        return f"Population Mean Difference: {input.treatment_mean() - input.control_mean():.3f}"
    
    @render.text  
    def txt_sample_dif():
        return f"Sample Mean Difference: {np.mean(treatment()) - np.mean(control()):.3f}"
    
    @render.text  
    def txt_perm():
        return f"How likely is it to get a result as extreme as {np.mean(treatment()) - np.mean(control()):.3f}? To answer this, we just need to compute the percentage of experiments that had an outcome higher than {np.mean(treatment()) - np.mean(control()):.3f}"


app = App(app_ui, server)
