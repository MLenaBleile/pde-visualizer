"""
Interactive PDE Visualizations Web App
Built with Plotly Dash - deployable to web

To run locally:
pip install dash plotly numpy scipy --break-system-packages
python app.py

Then visit http://127.0.0.1:8050 in your browser
"""

import numpy as np
from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter1d

# 3Blue1Brown color palette
BLUE = '#58C4DD'
DARK_BLUE = '#236B8E'
LIGHT_BLUE = '#9CDCEB'
GREEN = '#83C167'
YELLOW = '#FFFF00'
GOLD = '#FFD700'
RED = '#FC6255'
MAROON = '#C55F73'
PURPLE = '#9A72AC'
BACKGROUND = '#0F0F0F'
TEXT_COLOR = '#ECECEC'
GRID_COLOR = '#2F2F2F'

# Initialize the Dash app
app = Dash(__name__)
app.title = "PDE Visualizations"
server = app.server  # Expose the Flask server for deployment

# Layout
app.layout = html.Div([
    html.Div([
        html.H1("Interactive PDE & Stochastic Process Visualizations",
                style={'textAlign': 'center', 'color': TEXT_COLOR, 'marginBottom': '10px'}),
        html.P("Explore the beauty of partial differential equations",
               style={'textAlign': 'center', 'color': TEXT_COLOR, 'opacity': '0.7'}),
    ], style={'backgroundColor': BACKGROUND, 'padding': '20px'}),
    
    # Tabs for different visualizations
    dcc.Tabs(id='tabs', value='transport', children=[
        dcc.Tab(label='Transport Equation', value='transport', 
                style={'backgroundColor': BACKGROUND, 'color': TEXT_COLOR},
                selected_style={'backgroundColor': DARK_BLUE, 'color': TEXT_COLOR}),
        dcc.Tab(label='Transport with Source', value='transport_source',
                style={'backgroundColor': BACKGROUND, 'color': TEXT_COLOR},
                selected_style={'backgroundColor': DARK_BLUE, 'color': TEXT_COLOR}),
        dcc.Tab(label='Laplace Equation', value='laplace',
                style={'backgroundColor': BACKGROUND, 'color': TEXT_COLOR},
                selected_style={'backgroundColor': DARK_BLUE, 'color': TEXT_COLOR}),
        dcc.Tab(label='Brownian Circle', value='brownian',
                style={'backgroundColor': BACKGROUND, 'color': TEXT_COLOR},
                selected_style={'backgroundColor': DARK_BLUE, 'color': TEXT_COLOR}),
    ], style={'backgroundColor': BACKGROUND}),
    
    html.Div(id='tabs-content', style={'backgroundColor': BACKGROUND, 'padding': '20px'}),
    
    # Interval component for animation
    dcc.Interval(id='interval', interval=50, n_intervals=0, disabled=True),
    
    # Store components for data
    dcc.Store(id='animation-data'),
    dcc.Store(id='animation-params'),
    
], style={'backgroundColor': BACKGROUND, 'minHeight': '100vh'})


# ============================================================================
# TRANSPORT EQUATION COMPONENTS
# ============================================================================

def create_transport_controls():
    return html.Div([
        html.Div([
            html.Label('Transport Velocity (b):', style={'color': TEXT_COLOR, 'fontWeight': 'bold'}),
            dcc.Slider(id='transport-velocity', min=0.1, max=3.0, step=0.1, value=1.0,
                      marks={i: str(i) for i in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]},
                      tooltip={"placement": "bottom", "always_visible": True}),
        ], style={'marginBottom': '20px'}),
        
        html.Div([
            html.Label('Time:', style={'color': TEXT_COLOR, 'fontWeight': 'bold'}),
            dcc.Slider(id='transport-time', min=0, max=5, step=0.05, value=0,
                      marks={i: f'{i}s' for i in range(6)},
                      tooltip={"placement": "bottom", "always_visible": True}),
        ], style={'marginBottom': '20px'}),
        
        html.Div([
            html.Button('Play', id='play-button', n_clicks=0,
                       style={'backgroundColor': BLUE, 'color': BACKGROUND, 
                             'border': 'none', 'padding': '10px 30px',
                             'fontSize': '16px', 'fontWeight': 'bold',
                             'borderRadius': '5px', 'cursor': 'pointer',
                             'marginRight': '10px'}),
            html.Button('Reset', id='reset-button', n_clicks=0,
                       style={'backgroundColor': RED, 'color': BACKGROUND, 
                             'border': 'none', 'padding': '10px 30px',
                             'fontSize': '16px', 'fontWeight': 'bold',
                             'borderRadius': '5px', 'cursor': 'pointer'}),
        ], style={'textAlign': 'center', 'marginTop': '20px'}),
        
        dcc.Graph(id='transport-graph', style={'marginTop': '30px'}),
    ])


def create_transport_source_controls():
    return html.Div([
        html.Div([
            html.Label('Transport Velocity (b):', style={'color': TEXT_COLOR, 'fontWeight': 'bold'}),
            dcc.Slider(id='transport-source-velocity', min=0.1, max=3.0, step=0.1, value=0.8,
                      marks={i: str(i) for i in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]},
                      tooltip={"placement": "bottom", "always_visible": True}),
        ], style={'marginBottom': '20px'}),
        
        html.Div([
            html.Label('Source Strength:', style={'color': TEXT_COLOR, 'fontWeight': 'bold'}),
            dcc.Slider(id='source-strength', min=0, max=2.0, step=0.1, value=0.5,
                      marks={i: str(i) for i in [0, 0.5, 1.0, 1.5, 2.0]},
                      tooltip={"placement": "bottom", "always_visible": True}),
        ], style={'marginBottom': '20px'}),
        
        html.Div([
            html.Label('Time:', style={'color': TEXT_COLOR, 'fontWeight': 'bold'}),
            dcc.Slider(id='transport-source-time', min=0, max=8, step=0.08, value=0,
                      marks={i: f'{i}s' for i in range(9)},
                      tooltip={"placement": "bottom", "always_visible": True}),
        ], style={'marginBottom': '20px'}),
        
        html.Div([
            html.Button('Play', id='play-source-button', n_clicks=0,
                       style={'backgroundColor': BLUE, 'color': BACKGROUND, 
                             'border': 'none', 'padding': '10px 30px',
                             'fontSize': '16px', 'fontWeight': 'bold',
                             'borderRadius': '5px', 'cursor': 'pointer',
                             'marginRight': '10px'}),
            html.Button('Reset', id='reset-source-button', n_clicks=0,
                       style={'backgroundColor': RED, 'color': BACKGROUND, 
                             'border': 'none', 'padding': '10px 30px',
                             'fontSize': '16px', 'fontWeight': 'bold',
                             'borderRadius': '5px', 'cursor': 'pointer'}),
        ], style={'textAlign': 'center', 'marginTop': '20px'}),
        
        dcc.Graph(id='transport-source-graph', style={'marginTop': '30px'}),
    ])


# ============================================================================
# LAPLACE EQUATION COMPONENTS
# ============================================================================

def create_laplace_controls():
    return html.Div([
        html.Div([
            html.Label('Boundary Condition:', style={'color': TEXT_COLOR, 'fontWeight': 'bold'}),
            dcc.Dropdown(id='laplace-boundary',
                        options=[
                            {'label': 'Hot Top / Cold Bottom', 'value': 'hot_cold'},
                            {'label': 'Hot Corners', 'value': 'corners'}
                        ],
                        value='hot_cold',
                        style={'backgroundColor': DARK_BLUE, 'color': TEXT_COLOR}),
        ], style={'marginBottom': '20px', 'width': '50%'}),
        
        html.Div([
            html.Label('Iteration:', style={'color': TEXT_COLOR, 'fontWeight': 'bold'}),
            dcc.Slider(id='laplace-iteration', min=0, max=100, step=1, value=0,
                      marks={i: str(i*10) for i in range(11)},
                      tooltip={"placement": "bottom", "always_visible": True}),
        ], style={'marginBottom': '20px'}),
        
        html.Div([
            html.Button('Play', id='play-laplace-button', n_clicks=0,
                       style={'backgroundColor': BLUE, 'color': BACKGROUND, 
                             'border': 'none', 'padding': '10px 30px',
                             'fontSize': '16px', 'fontWeight': 'bold',
                             'borderRadius': '5px', 'cursor': 'pointer',
                             'marginRight': '10px'}),
            html.Button('Reset', id='reset-laplace-button', n_clicks=0,
                       style={'backgroundColor': RED, 'color': BACKGROUND, 
                             'border': 'none', 'padding': '10px 30px',
                             'fontSize': '16px', 'fontWeight': 'bold',
                             'borderRadius': '5px', 'cursor': 'pointer'}),
        ], style={'textAlign': 'center', 'marginTop': '20px'}),
        
        dcc.Graph(id='laplace-graph', style={'marginTop': '30px'}),
    ])


# ============================================================================
# BROWNIAN MOTION COMPONENTS
# ============================================================================

def create_brownian_controls():
    return html.Div([
        html.Div([
            html.Label('Number of Particles:', style={'color': TEXT_COLOR, 'fontWeight': 'bold'}),
            dcc.Slider(id='brownian-particles', min=100, max=1000, step=100, value=500,
                      marks={i: str(i) for i in [100, 300, 500, 700, 1000]},
                      tooltip={"placement": "bottom", "always_visible": True}),
        ], style={'marginBottom': '20px'}),
        
        html.Div([
            html.Label('Diffusion Rate (σ):', style={'color': TEXT_COLOR, 'fontWeight': 'bold'}),
            dcc.Slider(id='brownian-sigma', min=0.1, max=1.0, step=0.1, value=0.3,
                      marks={i/10: str(i/10) for i in range(1, 11)},
                      tooltip={"placement": "bottom", "always_visible": True}),
        ], style={'marginBottom': '20px'}),
        
        html.Div([
            html.Label('Time Step:', style={'color': TEXT_COLOR, 'fontWeight': 'bold'}),
            dcc.Slider(id='brownian-time', min=0, max=500, step=5, value=0,
                      marks={i: str(i) for i in [0, 100, 200, 300, 400, 500]},
                      tooltip={"placement": "bottom", "always_visible": True}),
        ], style={'marginBottom': '20px'}),
        
        html.Div([
            html.Button('Play', id='play-brownian-button', n_clicks=0,
                       style={'backgroundColor': BLUE, 'color': BACKGROUND, 
                             'border': 'none', 'padding': '10px 30px',
                             'fontSize': '16px', 'fontWeight': 'bold',
                             'borderRadius': '5px', 'cursor': 'pointer',
                             'marginRight': '10px'}),
            html.Button('Reset', id='reset-brownian-button', n_clicks=0,
                       style={'backgroundColor': RED, 'color': BACKGROUND, 
                             'border': 'none', 'padding': '10px 30px',
                             'fontSize': '16px', 'fontWeight': 'bold',
                             'borderRadius': '5px', 'cursor': 'pointer'}),
        ], style={'textAlign': 'center', 'marginTop': '20px'}),
        
        dcc.Graph(id='brownian-graph', style={'marginTop': '30px'}),
    ])


# ============================================================================
# CALLBACKS
# ============================================================================

@app.callback(
    Output('tabs-content', 'children'),
    Input('tabs', 'value')
)
def render_content(tab):
    if tab == 'transport':
        return create_transport_controls()
    elif tab == 'transport_source':
        return create_transport_source_controls()
    elif tab == 'laplace':
        return create_laplace_controls()
    elif tab == 'brownian':
        return create_brownian_controls()


# Transport Equation Callback
@app.callback(
    Output('transport-graph', 'figure'),
    Input('transport-velocity', 'value'),
    Input('transport-time', 'value')
)
def update_transport_graph(b, t):
    x = np.linspace(-5, 5, 500)
    
    # Initial condition
    u0 = np.exp(-x**2) + 0.3*np.sin(3*x)
    
    # Solution at time t
    u = np.exp(-(x - b*t)**2) + 0.3*np.sin(3*(x - b*t))
    
    fig = make_subplots(rows=2, cols=1, 
                        subplot_titles=('Solution u(t,x)', 'Characteristic Lines'),
                        vertical_spacing=0.15,
                        row_heights=[0.6, 0.4])
    
    # Solution plot
    fig.add_trace(go.Scatter(x=x, y=u, mode='lines', name='u(t,x)',
                            line=dict(color=BLUE, width=3)), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=u0, mode='lines', name='u(0,x)',
                            line=dict(color=BLUE, width=2, dash='dash'),
                            opacity=0.3), row=1, col=1)
    
    # Characteristic lines
    x0_vals = np.linspace(-5, 5, 15)
    t_line = np.linspace(0, 5, 100)
    for x0 in x0_vals:
        x_char = x0 + b * t_line
        fig.add_trace(go.Scatter(x=x_char, y=t_line, mode='lines',
                                line=dict(color=GREEN, width=1),
                                opacity=0.3, showlegend=False), row=2, col=1)
    
    # Current time marker
    fig.add_trace(go.Scatter(x=x[::20], y=[t]*len(x[::20]), mode='markers',
                            marker=dict(color=RED, size=8),
                            name='Current time', showlegend=False), row=2, col=1)
    
    fig.update_xaxes(title_text="x", row=1, col=1, gridcolor=GRID_COLOR)
    fig.update_xaxes(title_text="x", row=2, col=1, gridcolor=GRID_COLOR)
    fig.update_yaxes(title_text="u", row=1, col=1, gridcolor=GRID_COLOR, range=[-0.5, 1.8])
    fig.update_yaxes(title_text="t", row=2, col=1, gridcolor=GRID_COLOR, range=[0, 5])
    
    fig.update_layout(
        title=dict(text=f'Transport Equation: ∂ₜu + b·∂ₓu = 0  (b={b:.1f}, t={t:.2f})',
                  font=dict(size=18, color=TEXT_COLOR)),
        plot_bgcolor=BACKGROUND,
        paper_bgcolor=BACKGROUND,
        font=dict(color=TEXT_COLOR),
        height=700,
        showlegend=True
    )
    
    return fig


# Transport with Source Callback
@app.callback(
    Output('transport-source-graph', 'figure'),
    Input('transport-source-velocity', 'value'),
    Input('source-strength', 'value'),
    Input('transport-source-time', 'value')
)
def update_transport_source_graph(b, strength, t):
    x = np.linspace(-5, 5, 500)
    
    # Initial condition
    u0 = np.exp(-x**2) + 0.3*np.sin(3*x)
    
    # Homogeneous solution
    u = np.exp(-(x - b*t)**2) + 0.3*np.sin(3*(x - b*t))
    
    # Add source contribution (simplified)
    dt = 0.01
    n_steps = int(t / dt)
    for i in range(n_steps):
        s = i * dt
        source = strength * np.exp(-(x - 0.5)**2 / 0.1) * (np.sin(3*s) + 1) / 2
        u += source * np.exp(-((x - 0.5) - b*(t - s))**2 / 0.1) * dt * 0.5
    
    # Current source
    source_now = strength * np.exp(-(x - 0.5)**2 / 0.1) * (np.sin(3*t) + 1) / 2
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=x, y=u, mode='lines', name='u(t,x)',
                            line=dict(color=BLUE, width=3)))
    fig.add_trace(go.Scatter(x=x, y=u0, mode='lines', name='u(0,x)',
                            line=dict(color=BLUE, width=2, dash='dash'),
                            opacity=0.3))
    
    # Source visualization
    fig.add_trace(go.Scatter(x=x, y=source_now*3, mode='lines', name='Source f(t,x)',
                            fill='tozeroy', fillcolor=f'rgba(255, 255, 0, 0.3)',
                            line=dict(color=YELLOW, width=2)))
    
    fig.update_xaxes(title_text="x", gridcolor=GRID_COLOR)
    fig.update_yaxes(title_text="u", gridcolor=GRID_COLOR, range=[-0.5, 2.5])
    
    fig.update_layout(
        title=dict(text=f'Transport with Source: ∂ₜu + b·∂ₓu = f(t,x)  (b={b:.1f}, t={t:.2f})',
                  font=dict(size=18, color=TEXT_COLOR)),
        plot_bgcolor=BACKGROUND,
        paper_bgcolor=BACKGROUND,
        font=dict(color=TEXT_COLOR),
        height=600,
        showlegend=True
    )
    
    return fig


# Laplace Equation Callback
@app.callback(
    Output('laplace-graph', 'figure'),
    Input('laplace-boundary', 'value'),
    Input('laplace-iteration', 'value')
)
def update_laplace_graph(boundary_type, iteration):
    n = 80
    x = np.linspace(0, 1, n)
    y = np.linspace(0, 1, n)
    X, Y = np.meshgrid(x, y)
    
    u = np.zeros((n, n))
    
    # Apply boundary conditions
    if boundary_type == 'hot_cold':
        u[0, :] = 0
        u[-1, :] = 1
    else:  # corners
        u[0, :] = 0
        u[-1, :] = 0
        u[:, 0] = 0
        u[:, -1] = 0
        u[0, 0] = 1
        u[0, -1] = 1
        u[-1, 0] = 1
        u[-1, -1] = 1
    
    # Jacobi iteration
    for _ in range(iteration * 10):
        u_old = u.copy()
        u[1:-1, 1:-1] = 0.25 * (u_old[:-2, 1:-1] + u_old[2:, 1:-1] + 
                                 u_old[1:-1, :-2] + u_old[1:-1, 2:])
        
        # Reapply boundary conditions
        if boundary_type == 'hot_cold':
            u[0, :] = 0
            u[-1, :] = 1
        else:
            u[0, :] = 0
            u[-1, :] = 0
            u[:, 0] = 0
            u[:, -1] = 0
            u[0, 0] = 1
            u[0, -1] = 1
            u[-1, 0] = 1
            u[-1, -1] = 1
    
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Temperature Field', 'Isotherms'),
                        specs=[[{'type': 'heatmap'}, {'type': 'contour'}]])
    
    # Heatmap
    fig.add_trace(go.Heatmap(z=u, x=x, y=y, colorscale='RdYlBu_r',
                            zmin=0, zmax=1, colorbar=dict(x=0.45)), row=1, col=1)
    
    # Contour plot
    fig.add_trace(go.Contour(z=u, x=x, y=y, colorscale='RdYlBu_r',
                            contours=dict(start=0, end=1, size=0.1),
                            line=dict(width=2),
                            showscale=False), row=1, col=2)
    
    fig.update_xaxes(title_text="x", row=1, col=1, gridcolor=GRID_COLOR)
    fig.update_xaxes(title_text="x", row=1, col=2, gridcolor=GRID_COLOR)
    fig.update_yaxes(title_text="y", row=1, col=1, gridcolor=GRID_COLOR)
    fig.update_yaxes(title_text="y", row=1, col=2, gridcolor=GRID_COLOR)
    
    fig.update_layout(
        title=dict(text=f'Laplace Equation: ∇²u = 0  (Iteration: {iteration*10})',
                  font=dict(size=18, color=TEXT_COLOR)),
        plot_bgcolor=BACKGROUND,
        paper_bgcolor=BACKGROUND,
        font=dict(color=TEXT_COLOR),
        height=500
    )
    
    return fig


# Brownian Motion Callback
@app.callback(
    Output('brownian-graph', 'figure'),
    Input('brownian-particles', 'value'),
    Input('brownian-sigma', 'value'),
    Input('brownian-time', 'value')
)
def update_brownian_graph(n_particles, sigma, time_step):
    dt = 0.01
    
    # Initial positions
    np.random.seed(42)  # For reproducibility
    theta = np.random.normal(0, 0.3, n_particles) % (2*np.pi)
    
    # Simulate to time_step
    for _ in range(time_step):
        dtheta = np.random.normal(0, sigma * np.sqrt(dt), n_particles)
        theta = (theta + dtheta) % (2*np.pi)
    
    # Convert to Cartesian for polar plot
    r = np.ones(n_particles)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    
    fig = make_subplots(rows=1, cols=2,
                        subplot_titles=('Particles on Circle', 'Angular Distribution'),
                        specs=[[{'type': 'scatter'}, {'type': 'histogram'}]])
    
    # Circle plot
    circle_theta = np.linspace(0, 2*np.pi, 100)
    fig.add_trace(go.Scatter(x=np.cos(circle_theta), y=np.sin(circle_theta),
                            mode='lines', line=dict(color=GREEN, width=2, dash='dash'),
                            name='Circle', showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=x, y=y, mode='markers',
                            marker=dict(color=BLUE, size=4, opacity=0.6),
                            name='Particles', showlegend=False), row=1, col=1)
    
    # Histogram
    fig.add_trace(go.Histogram(x=theta, nbinsx=20, marker=dict(color=BLUE, opacity=0.7),
                              name='Distribution', showlegend=False), row=1, col=2)
    
    # Uniform reference line
    uniform_level = n_particles / 20
    fig.add_trace(go.Scatter(x=[0, 2*np.pi], y=[uniform_level, uniform_level],
                            mode='lines', line=dict(color=GREEN, width=2, dash='dash'),
                            name='Uniform', showlegend=False), row=1, col=2)
    
    fig.update_xaxes(title_text="", row=1, col=1, showgrid=False, zeroline=False,
                    showticklabels=False, range=[-1.2, 1.2])
    fig.update_yaxes(title_text="", row=1, col=1, showgrid=False, zeroline=False,
                    showticklabels=False, range=[-1.2, 1.2])
    fig.update_xaxes(title_text="Angle θ", row=1, col=2, gridcolor=GRID_COLOR,
                    range=[0, 2*np.pi])
    fig.update_yaxes(title_text="Count", row=1, col=2, gridcolor=GRID_COLOR)
    
    fig.update_layout(
        title=dict(text=f'Brownian Motion on Circle: dθ = σdW  (Time: {time_step}, σ={sigma:.1f})',
                  font=dict(size=18, color=TEXT_COLOR)),
        plot_bgcolor=BACKGROUND,
        paper_bgcolor=BACKGROUND,
        font=dict(color=TEXT_COLOR),
        height=500
    )
    
    return fig


# ============================================================================
# RUN SERVER
# ============================================================================

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
