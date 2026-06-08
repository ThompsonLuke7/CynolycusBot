# Shared Plotting

Use this package for new Python plots so research charts, audit charts, and model
diagnostics share the same candle colors, axes theme, time ticks, and save logic.

Recommended imports:

```python
from shared_plotting import (
    DEFAULT_THEME,
    apply_mpl_defaults,
    plot_candles,
    plot_candles_from_frame,
    plot_direction_probabilities,
    save_figure,
    style_figure,
)
```

Keep offline/report plots in Python for reproducible PNG artifacts. Use
TradingView Lightweight Charts for browser/live-dashboard charts when
interactivity matters.
