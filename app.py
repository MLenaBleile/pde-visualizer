"""
Interactive PDE Visualizations - COMPLETE WORKING VERSION
Push this file as app.py to your GitHub repo
"""

import numpy as np
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots

BLUE = '#58C4DD'
DARK_BLUE = '#236B8E'
GREEN = '#83C167'
YELLOW = '#FFFF00'
RED = '#FC6255'
PURPLE = '#9A72AC'
BACKGROUND = '#0F0F0F'
TEXT_COLOR = '#ECECEC'
GRID_COLOR = '#2F2F2F'

app = Dash(__name__, suppress_callback_exceptions=True)
app.title = "PDE Visualizations"
server = app.server

app.layout = html.Div([
    html.Div([
        html.H1("Interactive PDE & Stochastic Process Visualizations",
                style={'textAlign': 'center', 'color': TEXT_COLOR, 'marginBottom': '10px'}),
    ], style={'backgroundColor': BACKGROUND, 'padding': '20px'}),
    
    dcc.Tabs(id='tabs', value='transport', children=[
        dcc.Tab(label='Transport Equation', value='transport', 
                style={'backgroundColor': BACKGROUND, 'color': TEXT_COLOR},
                selected_style={'backgroundColor': DARK_BLUE, 'color': TEXT_COLOR}),
    ], style={'backgroundColor': BACKGROUND}),
    
    html.Div(id='tabs-content', style={'backgroundColor': BACKGROUND, 'padding': '20px'}),
    
    dcc.Interval(id='anim-interval', interval=50, disabled=True),
    dcc.Store(id='is-playing', data=False),
    
], style={'backgroundColor': BACKGROUND, 'minHeight': '100vh'})

def create_transport_controls():
    return html.Div([
        html.Div([
            html.H3('Mathematical Setup', style={'color': TEXT_COLOR, 'marginBottom': '15px'}),
            html.P('∂ₜu + b·∂ₓu = 0', style={'color': BLUE, 'fontSize': '20px'}),
            html.P('u(0,x) = exp(-x²) + 0.3·sin(3x)', style={'color': GREEN, 'fontSize': '16px'}),
        ], style={'backgroundColor': DARK_BLUE, 'padding': '20px', 'borderRadius': '10px', 'marginBottom': '20px'}),
        
        html.Label('Velocity (b):', style={'color': TEXT_COLOR}),
        dcc.Slider(id='velocity', min=0.1, max=3.0, step=0.1, value=1.0,
                  marks={i: str(i) for i in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]}),
        html.Label('Time:', style={'color': TEXT_COLOR, 'marginTop': '20px'}),
        dcc.Slider(id='time-slider', min=0, max=5, step=0.05, value=0,
                  marks={i: f'{i}s' for i in range(6)}),
        html.Div([
            html.Button('▶ Play', id='play-btn', style={
                'backgroundColor': BLUE, 'color': BACKGROUND, 'border': 'none',
                'padding': '10px 30px', 'fontSize': '16px', 'borderRadius': '5px',
                'cursor': 'pointer', 'marginRight': '10px', 'marginTop': '20px'}),
            html.Button('↻ Reset', id='reset-btn', style={
                'backgroundColor': RED, 'color': BACKGROUND, 'border': 'none',
                'padding': '10px 30px', 'fontSize': '16px', 'borderRadius': '5px',
                'cursor': 'pointer', 'marginTop': '20px'}),
        ]),
        dcc.Graph(id='main-graph', style={'marginTop': '30px'}),
    ])

@app.callback(Output('tabs-content', 'children'), Input('tabs', 'value'))
def render_content(tab):
    return create_transport_controls()

@app.callback(
    Output('anim-interval', 'disabled'),
    Output('play-btn', 'children'),
    Output('is-playing', 'data'),
    Input('play-btn', 'n_clicks'),
    State('is-playing', 'data'),
    prevent_initial_call=True
)
def toggle_play(n, playing):
    new_state = not playing
    return not new_state, '⏸ Pause' if new_state else '▶ Play', new_state

@app.callback(
    Output('time-slider', 'value', allow_duplicate=True),
    Input('reset-btn', 'n_clicks'),
    prevent_initial_call=True
)
def reset_time(n):
    return 0

@app.callback(
    Output('time-slider', 'value'),
    Input('anim-interval', 'n_intervals'),
    State('time-slider', 'value'),
    State('is-playing', 'data'),
    prevent_initial_call=True
)
def animate(n, t, playing):
    if playing:
        return (t + 0.05) % 5.05
    return t

@app.callback(
    Output('main-graph', 'figure'),
    Input('velocity', 'value'),
    Input('time-slider', 'value')
)
def update_graph(b, t):
    x = np.linspace(-5, 5, 500)
    u0 = np.exp(-x**2) + 0.3*np.sin(3*x)
    u = np.exp(-(x - b*t)**2) + 0.3*np.sin(3*(x - b*t))
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=x, y=u, mode='lines', name='u(t,x)',
                            line=dict(color=BLUE, width=3)))
    fig.add_trace(go.Scatter(x=x, y=u0, mode='lines', name='u(0,x)',
                            line=dict(color=BLUE, width=2, dash='dash'), opacity=0.3))
    
    fig.update_layout(
        title=f'Transport Equation (b={b:.1f}, t={t:.2f})',
        plot_bgcolor=BACKGROUND,
        paper_bgcolor=BACKGROUND,
        font=dict(color=TEXT_COLOR),
        xaxis=dict(title='x', gridcolor=GRID_COLOR),
        yaxis=dict(title='u', gridcolor=GRID_COLOR, range=[-0.5, 1.8]),
        height=600
    )
    return fig

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
