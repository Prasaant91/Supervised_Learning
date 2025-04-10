import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import os
import dash
from dash import dcc, html
from dash.dependencies import Input, Output

file_path = "data/Generated_synthetic_manufacturing_data.csv"
df = pd.read_csv(file_path)

if "timestamp" in df.columns:
    df["timestamp"] = pd.to_datetime(df["timestamp"])

numeric_columns = df.select_dtypes(include=["float64", "int64"]).columns.tolist()

app = dash.Dash(__name__)
app.title = "Explanatory Data Analysis Dashboard"

app.layout = html.Div([
    html.H1("📊 Feature Distribution Viewer", style={"textAlign": "center"}),

    html.Div([
        html.Label("Select Feature to View:"),
        dcc.Dropdown(
            id="feature-dropdown",
            options=[{"label": col, "value": col} for col in numeric_columns],
            value=numeric_columns[0],
            style={"width": "400px"}
        )
    ], style={"textAlign": "center", "padding": "20px"}),

    dcc.Graph(id="feature-histogram"),
    dcc.Graph(id="time-series-plot")
])

# Callback to update plot based on dropdown
@app.callback(
    [Output("feature-histogram", "figure"),
     Output("time-series-plot", "figure")],
    Input("feature-dropdown", "value")
)
def update_plots(selected_feature):

    hist = go.Figure()
    hist.add_trace(go.Histogram(x=df[selected_feature], nbinsx=50, name='Histogram', marker_color='lightblue', opacity=0.75))
    hist.add_trace(go.Scatter(x=sorted(df[selected_feature]),
                              y=pd.Series(sorted(df[selected_feature])).rolling(100).mean(),
                              mode='lines', name='Smoothed KDE', line=dict(color='black')))
    hist.update_layout(title=f"Distribution of {selected_feature}", barmode='overlay',
                       template="plotly_white", margin={"t": 50, "b": 40, "l": 10, "r": 10})

    # Time series plot if timestamp exists
    if "timestamp" in df.columns:
        time_fig = px.line(df.sort_values("timestamp"), x="timestamp", y=selected_feature,
                           title=f"{selected_feature} Over Time")
        time_fig.update_layout(template="plotly_white", margin={"t": 50, "b": 40, "l": 10, "r": 10})
    else:
        time_fig = go.Figure()
        time_fig.update_layout(title="Timestamp column not available for time-series plot.")

    return hist, time_fig

if __name__ == '__main__':
    app.run_server(debug=True)
