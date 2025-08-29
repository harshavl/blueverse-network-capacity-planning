import dash
from dash import dcc, html
import plotly.express as px
import pandas as pd
from preprocessing import preprocess_data
from inference import load_model, forecast

app = dash.Dash(__name__)

def create_dashboard():
    df, scaler = preprocess_data()
    tft = load_model()
    forecast_df, recommendations = forecast(tft, df, scaler, router_ids=["router_1"], forecast_horizon=14)
    
    fig = px.line(
        forecast_df,
        x="timestamp",
        y="bandwidth_mbps_forecast",
        color="router_id",
        title="Bandwidth Forecast for Router 1",
        labels={"bandwidth_mbps_forecast": "Bandwidth (Mbps)", "timestamp": "Date"}
    )
    
    app.layout = html.Div([
        html.H1("Network Capacity Planning Dashboard"),
        dcc.DatePickerRange(
            id="date-picker",
            min_date_allowed=forecast_df["timestamp"].min(),
            max_date_allowed=forecast_df["timestamp"].max(),
            initial_visible_month=forecast_df["timestamp"].min(),
            start_date=forecast_df["timestamp"].min(),
            end_date=forecast_df["timestamp"].max()
        ),
        dcc.Graph(id="forecast-graph", figure=fig),
        html.Div([
            html.H3("Capacity Recommendations"),
            html.Ul([html.Li(f"{router}: {rec or 'No capacity issues detected.'}") for router, rec in recommendations.items()])
        ])
    ])
    
    @app.callback(
        dash.dependencies.Output("forecast-graph", "figure"),
        [dash.dependencies.Input("date-picker", "start_date"),
        dash.dependencies.Input("date-picker", "end_date")]
    )
    def update_graph(start_date, end_date):
        filtered_df = forecast_df[
            (forecast_df["timestamp"] >= start_date) & (forecast_df["timestamp"] <= end_date)
        ]
        fig = px.line(
            filtered_df,
            x="timestamp",
            y="bandwidth_mbps_forecast",
            color="router_id",
            title="Bandwidth Forecast for Router 1",
            labels={"bandwidth_mbps_forecast": "Bandwidth (Mbps)", "timestamp": "Date"}
        )
        return fig

if __name__ == "__main__":
    create_dashboard()
    app.run_server(debug=False, host='0.0.0.0', port=8050)
