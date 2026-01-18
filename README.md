# Interactive PDE Visualizations

Beautiful, interactive visualizations of partial differential equations and stochastic processes, inspired by 3Blue1Brown's aesthetic.

## Features

- **Transport Equation**: Visualize pure advection (∂ₜu + b·∂ₓu = 0) with characteristic lines
- **Transport with Source**: See how sources accumulate along characteristics
- **Laplace Equation**: Watch the relaxation to steady-state equilibrium
- **Brownian Motion on Circle**: Observe diffusion converging to uniform distribution

## Running Locally

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Run the app:
```bash
python app.py
```

3. Open your browser to `http://127.0.0.1:8050`

## Deployment Options

### Option 1: Render (Recommended - Free)

1. Push this code to a GitHub repository
2. Go to [render.com](https://render.com) and sign up
3. Click "New +" → "Web Service"
4. Connect your GitHub repository
5. Configure:
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:server`
   - **Environment**: Python 3
6. Click "Create Web Service"
7. Your app will be live at `https://your-app-name.onrender.com`

### Option 2: Heroku

1. Install Heroku CLI
2. Create a `Procfile`:
```
web: gunicorn app:server
```

3. Deploy:
```bash
heroku create your-app-name
git push heroku main
```

### Option 3: Railway

1. Push to GitHub
2. Go to [railway.app](https://railway.app)
3. Click "New Project" → "Deploy from GitHub"
4. Select your repository
5. Railway auto-detects Python and deploys

### Option 4: PythonAnywhere

1. Upload files to PythonAnywhere
2. Set up a web app pointing to your WSGI file
3. Configure to use `gunicorn`

## Modifying the App

The main app logic is in `app.py`. Key sections:

- **Color palette**: Lines 15-26 (change to match your site's theme)
- **Layout**: `app.layout` starting at line 30
- **Equations**: Each callback function implements the math
- **Sliders**: Adjust ranges and steps in the control functions

## Embedding in Your Website

Once deployed, you can embed using an iframe:

```html
<iframe src="https://your-app-url.com" 
        width="100%" 
        height="800px" 
        frameborder="0">
</iframe>
```

Or link directly from your site.

## Customization Ideas

1. **Match your site colors**: Edit the color constants at the top
2. **Add equations**: Fork and add your own PDE visualizations
3. **Export data**: Add download buttons for the computed solutions
4. **3D visualizations**: Extend to 3D using Plotly's 3D scatter/surface plots
5. **Parameter presets**: Add buttons for "interesting" parameter combinations

## Technical Notes

- Built with Plotly Dash (pure Python, no JavaScript required)
- Responsive design works on mobile
- All computations happen in browser after initial load
- No database needed - stateless app

## License

MIT License - feel free to use and modify!

## Author

Created by [Your Name]
Inspired by 3Blue1Brown's visualization style
