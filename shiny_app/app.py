from __future__ import annotations
from scipy.stats import norm
from shiny import App, reactive, render, ui
from shinyswatch.theme import cosmo as shiny_theme
from shinywidgets import output_widget, render_widget
import numpy as np
import pandas as pd
import plotly.express as px
import shinyswatch

COL_TXT = "#0081a7"
COL_treatment = "#0081a7"
COL_control = "#00afb9"
COL_permutation = "#c0c0c0"
COL_perm_highlight = "#f07167"

app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.markdown(
            "Based on Samuele Mazzanti's [Medium post](https://towardsdatascience.com/why-statistical-significance-is-pointless-a7644be30266), this app makes interactive the two ideas of statistical significance which Samuele explores."
        ),
        ui.input_slider(
            id="treatment_mean",
            label="Treatment Mean",
            min=1,
            max=20,
            value=10,
            step=0.1,
        ),
        ui.input_slider(
            id="control_mean",
            label="Control Mean",
            min=0,
            max=20,
            value=10.5,
            step=0.1,
        ),
        ui.input_slider(
            id="treatment_cov",
            label="Treatment Std Dev",
            value=2,
            min=0,
            max=10,
            step=0.1,
        ),
        ui.input_slider(
            id="control_cov",
            label="Control Std Dev",
            value=2,
            min=0,
            max=10,
            step=0.1,
        ),
        ui.input_slider(
            id="n_points",
            label="Points per Group",
            min=10,
            max=300,
            value=100,
            step=10,
        ),
        ui.input_slider(
            id="n_permutations",
            label="Number of Permutations",
            min=100,
            max=10000,
            value=1000,
            step=1000,
        ),
        open="always",
        bg="#f8f8f8",
    ),
    ui.navset_tab(
        ui.nav_panel(
            "P-Values",
            ui.column(
                10,
                ui.row(
                    ui.column(
                        6,
                        output_widget(
                            "treatment_control_hist", height="400px", width="400px"
                        ),
                        ui.h5("Simulated Data"),
                        ui.output_ui("txt_pop_dif"),
                        ui.br(),
                        ui.output_ui("txt_sample_dif"),
                    ),
                    ui.column(
                        6,
                        output_widget(
                            "permutation_hist", height="400px", width="400px"
                        ),
                        ui.output_data_frame("pval_df"),
                    ),
                ),
            ),
        ),
        ui.nav_panel(
            "Confidence Intervals",
            ui.em("Coming soon!"),
        )
    ),
    ui.br(),
    ui.HTML(
        "<div style='text-align: center; color: gray; font-size:0.9em;'> Shiny for Python, using ShinyLive | <a href = 'https://rsangole.github.io/shiny-python-statsignif/' target='_blank'>Github Repo</a> | <a href = 'http://www.rsangole.com' target='_blank'>Rahul Sangole</a> | Dec '24</div>"
    ),
    fillable=False,
    title="Why “Statistical Significance” Is Pointless",
    theme=shiny_theme,
)


def server(input, output, session):
    @reactive.Calc
    def treatment():
        return norm.rvs(
            input.treatment_mean(),
            input.treatment_cov(),
            input.n_points(),
            random_state=42,
        )

    @reactive.Calc
    def control():
        return norm.rvs(
            input.control_mean(), 
            input.control_cov(), 
            input.n_points(), 
            random_state=42
        )

    @reactive.Calc
    def sample_mean_diff():
        return np.abs(np.mean(control()) - np.mean(treatment()))

    @reactive.Calc
    def permute():
        combined = np.concatenate([treatment(), control()])
        permutation_results = []
        for _ in range(input.n_permutations()):
            combined = np.random.permutation(combined)
            perm_treatment = combined[: len(treatment())]
            perm_control = combined[-len(control()) :]
            permutation_results.append(np.mean(perm_treatment) - np.mean(perm_control))
        return permutation_results

    @reactive.Calc
    def count_extreme():
        return np.sum(np.array(np.abs(permute())) >= sample_mean_diff())

    @reactive.Calc
    def p_value():
        return count_extreme() / input.n_permutations()

    @render_widget
    def treatment_control_hist():
        res = pd.DataFrame(
            {"Treatment": treatment(), "Control": control()},
            index=range(len(treatment())),
        ).melt()
        fig = px.histogram(
            res,
            x="value",
            color="variable",
            marginal="rug",
            nbins=60,
            color_discrete_sequence=[COL_treatment, COL_control],
            # opacity=0.75,
        ).update_layout(
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
        return fig

    @render_widget
    def permutation_hist():
        res = pd.DataFrame(permute(), columns=["Permutation"])
        res["Highlight"] = np.abs(res["Permutation"]) >= sample_mean_diff()
        fig = px.histogram(
            res,
            x="Permutation",
            color="Highlight",
            marginal="rug",
            color_discrete_sequence=[COL_permutation, COL_perm_highlight],
        ).update_layout(
            title={"text": "", "x": 0.5},
            yaxis_title="Count",
            xaxis_title="Difference in Means",
            legend_title="",
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
            legend=dict(
                orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1
            ),
        )
        fig.add_shape(
            type="line",
            x0=sample_mean_diff(),
            y0=0,
            x1=sample_mean_diff(),
            y1=0.5,
            xref="x",
            yref="paper",
            line=dict(
                width=1,
                dash="dot",
            ),
        )
        fig.add_shape(
            type="line",
            x0=-sample_mean_diff(),
            y0=0,
            x1=-sample_mean_diff(),
            y1=0.5,
            xref="x",
            yref="paper",
            line=dict(
                width=1,
                dash="dot",
            ),
        )
        fig.add_annotation(
            x=sample_mean_diff(),
            y=0.5,
            xref="x",
            yref="paper",
            text=f"{sample_mean_diff():.3f}",
            showarrow=False,
            yshift=1,
            xanchor="left",
        )
        fig.add_annotation(
            x=-sample_mean_diff(),
            y=0.5,
            xref="x",
            yref="paper",
            text=f"-{sample_mean_diff():.3f}",
            showarrow=False,
            yshift=1,
            xanchor="left",
        )

        return fig

    @render.ui
    def txt_pop_dif():
        return ui.HTML(
            f"Diff Population Means: <span style='color:{COL_TXT};'>{input.control_mean()-input.treatment_mean():.3f}</span>"
            )

    @render.ui
    def txt_sample_dif():
        return ui.HTML(
            f"Diff Sample Means: <span style='color:{COL_TXT};'>{sample_mean_diff():.3f}</span>"
        )

    # @render.ui
    # def txt_perm():
    #     return ui.HTML(
    #         f"How likely is it to get a result as extreme as <span style='color:{COL_TXT};'>{sample_mean_diff():.3f}</span>? \
    #             <br>What % of experiments have an outcome > <span style='color:{COL_TXT};'>{sample_mean_diff():.3f}</span>?"
    #     )

    # @render.ui
    # def txt_p_value():
    #     return ui.HTML(
    #         f"<b>p-value: <span style='color:{COL_TXT};'>{p_value():.3f}</span><b>"
    #     )

    @render.data_frame
    def pval_df():
        df = pd.DataFrame(
            {
                "What question are we trying to answer?": [
                    f"What proportion of permutations have an outcome > {sample_mean_diff():.3f} or  < -{sample_mean_diff():.3f}?",
                    f"How likely is it to get a result as extreme as {sample_mean_diff():.3f}?",
                ],
                "Answers": [
                    f"{count_extreme()} out of {input.n_permutations()}",
                    f"{p_value()*100:.2f}%, or a p-value of {p_value():.3f}",
                ],
            }
        )
        return df


app = App(app_ui, server)
