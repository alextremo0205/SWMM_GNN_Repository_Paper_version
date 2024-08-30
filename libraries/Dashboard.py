"""
Interactive dashboard for the visualization of the results of the machine learning metamodel in comparison with the hydraulic model.

@author: Alexander Garzón
@email: j.a.garzondiaz@tudelft.nl
"""

import numpy as np
import pandas as pd
import networkx as nx
import plotly.graph_objs as go
from ipywidgets import widgets
import libraries.utils as utils
from sklearn.metrics import r2_score
from plotly.subplots import make_subplots
from libraries.MetricCalculator import MetricCalculator


class Dashboard:
    def __init__(self, swmm_heads_pd, predicted_heads_pd, G, runoff=None):
        self.swmm_heads_pd = swmm_heads_pd
        self.predicted_heads_pd = predicted_heads_pd

        self.max_heads = self.swmm_heads_pd.max().max()
        self.min_heads = self.swmm_heads_pd.min().min()

        self.G = G
        self.runoff = runoff
        self.node_names = list(self.swmm_heads_pd.columns)
        self.length_event = len(self.swmm_heads_pd)
        self.f = go.FigureWidget(
            make_subplots(
                rows=1,
                cols=3,
                specs=[[{}, {"colspan": 2, "secondary_y": True}, None]],
                subplot_titles=(" ", "Comparison of hydraulic head for node j_9006F"),
            )
        )

    def create_plotly_figure(self):
        self._add_left_map()
        self._add_right_line_plot()
        self._add_selected_point()

        if self.runoff is not None:
            self._add_runoff_trace()

        self.update_plot_style()

        # create our callback function
        def update_point(trace, points, selector):
            try:
                swmm_time_series = self.swmm_heads_pd[
                    self.node_names[points.point_inds[0]]
                ]
                model_time_series = self.predicted_heads_pd[
                    self.node_names[points.point_inds[0]]
                ]

                if self.runoff is not None:
                    runoff_in_considered_time = (
                        self.runoff
                    )  # .iloc[self.steps_behind:, :]
                    runoff_time_series = runoff_in_considered_time[
                        self.node_names[points.point_inds[0]]
                    ]

                self.f.data[2].x = list(swmm_time_series.index)
                self.f.data[3].x = list(model_time_series.index)

                self.f.data[2].y = swmm_time_series.values
                self.f.data[3].y = model_time_series.values

                self.f.data[4].x, self.f.data[4].y = [], []
                self.f.data[4].x, self.f.data[4].y = points.xs, points.ys
                self.f.data[4].marker["opacity"] = 1.0

                if self.runoff is not None:
                    self.f.data[5].x = list(
                        range(len(runoff_time_series))
                    )  # ! Change to trace 5 when selected point is active
                    self.f.data[5].y = (
                        runoff_time_series.values
                    )  # ! Change to trace 5 when selected point is active

                node_name = self.node_names[points.point_inds[0]]

                self.f.layout.annotations[1].update(
                    text=f"Comparison of hydraulic head for node {node_name}"
                )
            except Exception as e:
                print(e)
                pass

        trace = self.f.data[0]
        trace.on_click(update_point)

    def update_plot_style(self):
        self.f.update_layout(
            width=1600,
            height=800,
            title="Distribution of SWMM Head at minute 0",
            font_family="Computer Modern",
        )
        self.f.update_yaxes(scaleanchor="x", scaleratio=1, row=1, col=1)
        self.f.update_xaxes(showticklabels=False, col=1)
        self.f.update_xaxes(title_text="Time [min]", row=1, col=2)
        self.f.update_yaxes(showticklabels=False, col=1)
        self.f.update_yaxes(
            title_text="Head [masl]",
            range=[self.min_heads, self.max_heads],
            row=1,
            col=2,
            secondary_y=False,
        )
        self.f.update_yaxes(
            title_text="Runoff [cms]",
            range=[0.1, -0.01],
            row=1,
            col=2,
            secondary_y=True,
        )

    def _add_selected_point(self):
        coordinates_df = pd.DataFrame.from_dict(
            nx.get_node_attributes(self.G, "pos"), orient="index"
        )

        x_coord = coordinates_df[0].iloc[0]
        y_coord = coordinates_df[1].iloc[0]

        selected_point = go.Scatter(
            x=[x_coord],
            y=[y_coord],
            marker_size=15,
            marker_symbol="x-open",
            marker_line_width=2,
            marker_opacity=0.0,
            marker_color="midnightblue",
            showlegend=False,
        )
        self.f.add_trace(selected_point, row=1, col=1)

    def _add_left_map(self):
        node_trace = self._get_network_node_scatter_trace()
        self.f.add_trace(node_trace, row=1, col=1)

        # Add the edges to the figure
        edge_trace = self._get_edge_trace()
        self.f.add_trace(edge_trace, row=1, col=1)

    def _add_right_line_plot(self):
        series = self.swmm_heads_pd[self.node_names[0]]
        x = list(series.index)
        y = series.values

        scatter_trace = self._get_SWMM_scatter_trace(x, y)
        self.f.add_trace(scatter_trace, row=1, col=2)

        scatter_trace = self._get_predicted_scatter_trace()
        self.f.add_trace(scatter_trace, row=1, col=2)

    def _add_runoff_trace(self):

        runoff_node = self.runoff[self.node_names[0]]
        x_runoff = list(range(len(runoff_node)))
        y_runoff = runoff_node

        bar_trace = self._get_runoff_bar_trace(x_runoff, y_runoff)
        # bar_trace.update(base="below")
        self.f.add_trace(bar_trace, row=1, col=2, secondary_y=True)
        self.f.update_yaxes(title="Runoff", row=1, col=2, secondary_y=True)

    def _get_predicted_scatter_trace(self):
        series = self.predicted_heads_pd[self.node_names[0]]

        x = list(series.index)
        y = series.values

        scatter_trace = go.Scatter(
            x=x,
            y=y,
            name="Our model",
            mode="lines+markers",
            line=dict(width=3),
            marker=dict(size=4, color="#6F1D77"),
        )

        return scatter_trace

    def _get_SWMM_scatter_trace(self, x, y):
        scatter_trace = go.Scatter(
            x=x,
            y=y,
            name="SWMM",
            mode="lines+markers",
            line=dict(width=3),
            marker=dict(size=4, color="#00A6D6"),
        )

        return scatter_trace

    def _get_runoff_bar_trace(self, x, y):
        bar_trace = go.Bar(x=x, y=y, name="Runoff", marker=dict(color="Green"))
        return bar_trace

    def _get_network_node_scatter_trace(self):
        coordinates_df = pd.DataFrame.from_dict(
            nx.get_node_attributes(self.G, "pos"), orient="index"
        )

        x_coord = coordinates_df[0]
        y_coord = coordinates_df[1]

        node_signal = self.swmm_heads_pd.iloc[0, :]
        value = node_signal.values
        sizeref = 2.0 * max(value) / (2**2)
        node_trace = go.Scatter(
            x=x_coord,
            y=y_coord,
            mode="markers",
            name="coordinates",
            hovertemplate="%{text}",
            text=self._get_text(
                value
            ),  # ['<b><br> Node ID: </b> {name} <br> <b>Value:</b> {value:.2f}'.format(name = node_names[i], value = value[i]) for i in range(len(node_names))],
            marker_size=5,  # value-min(value),
            marker=dict(
                color=value,
                sizeref=sizeref,
                sizemin=1,
                colorscale="YlGnBu",
                cmax=1,
                cmin=0,
                showscale=True,
                colorbar=dict(x=-0.05, ticklabelposition="outside"),
                line=dict(width=1, color="DarkSlateGrey"),
            ),
            showlegend=False,
        )

        return node_trace

    def _get_edge_trace(self):
        edge_x = []
        edge_y = []
        for edge in self.G.edges():
            x0, y0 = self.G.nodes[edge[0]]["pos"]
            x1, y1 = self.G.nodes[edge[1]]["pos"]
            edge_x.append(x0)
            edge_x.append(x1)
            edge_x.append(None)
            edge_y.append(y0)
            edge_y.append(y1)
            edge_y.append(None)

        edge_trace = go.Scatter(
            x=edge_x,
            y=edge_y,
            line=dict(width=0.5, color="#888"),
            hoverinfo="none",
            mode="lines",
            showlegend=False,
        )

        return edge_trace

    def _get_text(self, value):
        # print(type(value))
        # print(value.shape)
        return [
            "<b><br> Node ID: </b> {name} <br> <b>Value:</b> {value:.2f}".format(
                name=self.node_names[i], value=value[i]
            )
            for i in range(len(self.node_names))
        ]

    def display_results(self):
        self.create_plotly_figure()
        self.minute_slider = widgets.IntSlider(
            value=0.0,
            min=0.0,
            max=self.length_event - 1,
            step=1.0,
            description="Minute:",
            continuous_update=False,
            layout=widgets.Layout(width="50%"),
        )
        self.textbox = widgets.Dropdown(
            description="Property:   ",
            value="SWMM Head",
            options=[
                "Predicted Head",
                "SWMM Head",
                "Error",
                "RMSE",
                "R2",
                "Wet R2",
                "Dry R2",
            ],
        )

        error_heads_pd = self.swmm_heads_pd - self.predicted_heads_pd
        elevation = np.array(list(nx.get_node_attributes(self.G, "elevation").values()))

        r2 = r2_score(
            self.swmm_heads_pd.to_numpy(),
            self.predicted_heads_pd.to_numpy(),
            multioutput="raw_values",
        )
        clipped_r2 = np.clip(r2, -10, 1)
        inverted_r2 = 1 / (clipped_r2 - min(clipped_r2) + 1e-6)

        mc = MetricCalculator()
        rmse = np.sqrt(
            mc.calculate_metric(
                "MSE", self.swmm_heads_pd.to_numpy(), self.predicted_heads_pd.to_numpy()
            )
        )
        wet_rmse = np.sqrt(
            mc.calculate_metric_state(
                "MSE",
                self.swmm_heads_pd,
                self.predicted_heads_pd,
                elevation=elevation,
                state="wet",
                multioutput="raw_values",
            )
        )

        wet_r2 = utils.wet_r2_per_node(
            self.swmm_heads_pd, self.predicted_heads_pd, elevation
        )
        inverted_wet_r2 = 1 / (wet_r2 - min(wet_r2) + 1e-6)

        dry_r2 = utils.dry_r2_per_node(
            self.swmm_heads_pd, self.predicted_heads_pd, elevation
        )
        inverted_dry_r2 = 1 / (dry_r2 - min(dry_r2) + 1e-6)

        min_value_head = self.swmm_heads_pd.min().min()

        def response(change):

            t = self.minute_slider.value
            property = self.textbox.value

            if property == "SWMM Head":
                self.f.update_layout(title=f"Distribution of {property} at minute {t}")

                swmm_signal_at_t = self.swmm_heads_pd.iloc[t, :]
                value = swmm_signal_at_t.values

                sizeref = 2.0 * max(abs(value)) / (4**2)
                self.f.update_traces(
                    text=self._get_text(value), selector=dict(name="coordinates")
                )
                self.f.update_traces(
                    marker=dict(
                        color=value,
                        size=value - min_value_head,
                        sizeref=sizeref,
                        sizemin=1,
                        colorscale="YlGnBu",
                        cmax=2,
                        cmin=-2,
                    ),
                    selector=dict(name="coordinates"),
                )

                self.f.update_yaxes(title_text="Head [masl]", row=1, col=2)

            if property == "Predicted Head":
                self.f.update_layout(title=f"Distribution of {property} at minute {t}")

                predicted_signal_at_t = self.predicted_heads_pd.iloc[t, :]
                value = predicted_signal_at_t.values

                sizeref = 2.0 * max(abs(value)) / (4**2)
                self.f.update_traces(
                    text=self._get_text(value), selector=dict(name="coordinates")
                )
                self.f.update_traces(
                    marker=dict(
                        color=value,
                        size=value - min_value_head,
                        sizeref=sizeref,
                        sizemin=1,
                        colorscale="YlGnBu",
                        cmax=2,
                        cmin=-2,
                    ),
                    selector=dict(name="coordinates"),
                )
                self.f.update_yaxes(title_text="Head [masl]", row=1, col=2)

            if property == "Error":
                self.f.update_layout(title=f"Distribution of {property} at minute {t}")

                error_signal_at_t = abs(error_heads_pd.iloc[t, :])
                value = error_signal_at_t.values

                sizeref = 2.0 * max(abs(value)) / (5**2)
                self.f.update_traces(
                    marker=dict(
                        color=value,
                        size=value - min(value),
                        sizeref=sizeref,
                        sizemin=1,
                        colorscale="inferno_r",
                    ),
                    selector=dict(name="coordinates"),
                )
                self.f.update_traces(
                    text=self._get_text(value), selector=dict(name="coordinates")
                )
                self.f.update_yaxes(title_text="Error [m]", row=1, col=2)

            if property == "R2":
                self.f.update_layout(title=f"Distribution of {property}")

                sizeref = 2.0 * 6 / (6**2)  # 2. * max(inverted_r2) / (6 ** 2)
                self.f.update_traces(
                    marker=dict(
                        color=clipped_r2,
                        size=6,
                        sizeref=sizeref,
                        sizemin=3,
                        cmax=1.0,
                        cmin=0,
                        colorscale="rdylbu",
                    ),
                    selector=dict(name="coordinates"),
                )
                self.f.update_traces(
                    text=self._get_text(r2), selector=dict(name="coordinates")
                )
                # size = inverted_r2

            if property == "RMSE":
                self.f.update_layout(title=f"Distribution of {property}")

                sizeref = 2.0 * 6 / (6**2)  # 2. * max(inverted_r2) / (6 ** 2)
                self.f.update_traces(
                    marker=dict(
                        color=rmse,
                        size=6,
                        sizeref=sizeref,
                        sizemin=3,
                        cmax=0.3,
                        cmin=0,
                        colorscale="rdylbu_r",
                    ),
                    selector=dict(name="coordinates"),
                )
                self.f.update_traces(
                    text=self._get_text(rmse), selector=dict(name="coordinates")
                )

            if property == "Wet_RMSE":
                self.f.update_layout(title=f"Distribution of {property}")

                sizeref = 2.0 * 6 / (6**2)  # 2. * max(inverted_r2) / (6 ** 2)
                self.f.update_traces(
                    marker=dict(
                        color=wet_rmse,
                        size=6,
                        sizeref=sizeref,
                        sizemin=3,
                        colorscale="rdylbu_r",
                    ),
                    selector=dict(name="coordinates"),
                )
                self.f.update_traces(
                    text=self._get_text(wet_rmse), selector=dict(name="coordinates")
                )

            if property == "Wet R2":
                self.f.update_layout(title=f"Distribution of {property}")

                sizeref = 2.0 * max(inverted_r2) / (6**2)
                self.f.update_traces(
                    marker=dict(
                        color=wet_r2,
                        size=inverted_wet_r2,
                        sizeref=sizeref,
                        sizemin=3,
                        colorscale="RdBu",
                    ),
                    selector=dict(name="coordinates"),
                )
                self.f.update_traces(
                    text=self._get_text(wet_r2), selector=dict(name="coordinates")
                )

            if property == "Dry R2":
                self.f.update_layout(title=f"Distribution of {property}")

                sizeref = 2.0 * max(inverted_r2) / (6**2)
                self.f.update_traces(
                    marker=dict(
                        color=dry_r2,
                        size=inverted_dry_r2,
                        sizeref=sizeref,
                        sizemin=3,
                        colorscale="RdBu",
                    ),
                    selector=dict(name="coordinates"),
                )
                self.f.update_traces(
                    text=self._get_text(dry_r2), selector=dict(name="coordinates")
                )

        self.textbox.observe(response, names="value")
        self.minute_slider.observe(response, names="value")

        return widgets.VBox([self.textbox, self.f, self.minute_slider])

    def obtain_r2_graph(self, height=800, width=400, title="Distribution of NSE"):
        r2 = r2_score(
            self.swmm_heads_pd.to_numpy(),
            self.predicted_heads_pd.to_numpy(),
            multioutput="raw_values",
        )
        clipped_r2 = np.clip(r2, -10, 1)

        self.f.update_layout(title=f"Distribution of {property}")
        sizeref = 2.0 * 6 / (6**2)  # 2. * max(inverted_r2) / (6 ** 2)
        self.f.update_traces(
            marker=dict(
                color=clipped_r2,
                size=6,
                sizeref=sizeref,
                sizemin=3,
                colorscale="rdylbu",
            ),
            selector=dict(name="coordinates"),
        )
        self.f.update_traces(text=self._get_text(r2), selector=dict(name="coordinates"))

        aux_fig = go.Figure(data=self.f.data[0:2])
        aux_fig.update_yaxes(scaleanchor="x", scaleratio=1)  # ,row=1, col=1)
        aux_fig.update_layout(
            width=width,
            height=height,
            title=title,
            font_family="Computer Modern",
            font_size=14,
        )
        aux_fig.update_xaxes(showticklabels=False)
        aux_fig.update_yaxes(showticklabels=False)
        aux_fig.data[0].marker.colorbar.update(thickness=15, x=0, title="NSE")
        aux_fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)"
        )
        aux_fig.update_xaxes(showgrid=False)
        aux_fig.update_yaxes(showgrid=False)

        return aux_fig

    def obtain_timeseries_graph(self, node_name, height=600, width=1200, title=""):
        try:
            swmm_time_series = self.swmm_heads_pd[node_name]
            model_time_series = self.predicted_heads_pd[node_name]

            # if self.runoff is not None:
            #     runoff_time_series = self.runoff[node_name]

            self.f.data[2].x = list(swmm_time_series.index)
            self.f.data[3].x = list(model_time_series.index)

            self.f.data[2].y = swmm_time_series.values
            self.f.data[3].y = model_time_series.values

            # f.data[4].x, f.data[4].y = [], []
            # f.data[4].x, f.data[4].y = points.xs, points.ys
            if self.runoff is not None:
                runoff_time_series = self.runoff[node_name]
                self.f.data[4].x = list(
                    range(len(runoff_time_series))
                )  # ! Change to trace 5 when selected point is active
                self.f.data[4].y = (
                    runoff_time_series.values
                )  # ! Change to trace 5 when selected point is active

            self.f.layout.annotations[1].update(
                text=f"Comparison of hydraulic head for node {node_name}"
            )

        except Exception as e:
            print(e)
            pass
        if self.runoff is not None:
            runoff_node = self.runoff[node_name]
            x_runoff = list(range(len(runoff_node)))
            y_runoff = runoff_node

            bar_trace = self._get_runoff_bar_trace(x_runoff, y_runoff)
            # bar_trace.update(base="below")
            self.f.add_trace(bar_trace, row=1, col=2, secondary_y=True)
            self.f.update_yaxes(title="Runoff", row=1, col=2, secondary_y=True)

        aux_fig = go.FigureWidget(
            make_subplots(
                rows=1, cols=1, specs=[[{"secondary_y": True}]], subplot_titles=("")
            )
        )

        aux_fig.add_trace(self.f.data[2], row=1, col=1, secondary_y=False)
        aux_fig.add_trace(self.f.data[3], row=1, col=1, secondary_y=False)
        aux_fig.add_trace(self.f.data[4], row=1, col=1, secondary_y=True)

        aux_fig.update_layout(
            width=width,
            height=height,
            title=title,
            font_family="Computer Modern",
            font_size=18,
        )
        aux_fig.update_xaxes(title_text="Time [min]", row=1, col=1)
        aux_fig.update_yaxes(
            title_text="Hydraulic head [m.a.s.l]",
            row=1,
            col=1,
            range=[self.min_heads, self.max_heads],
            secondary_y=False,
        )
        aux_fig.update_yaxes(
            title_text="Runoff [cms]",
            range=[0.1, -0.01],
            row=1,
            col=1,
            secondary_y=True,
        )
        return aux_fig
