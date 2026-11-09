"""Charts for inspecting a training run."""

from html import escape

import altair as alt
import numpy as np
import pandas as pd

TEAL = "#007f78"
ORANGE = "#b95021"


def history_chart(data, value, focus, group=None):
    base = alt.Chart(data).encode(
        x=alt.X("Iteration:Q", axis=alt.Axis(tickMinStep=1)),
        y=alt.Y(f"{value}:Q", scale=alt.Scale(zero=False)),
    )
    if group:
        base = base.encode(
            color=alt.Color(
                f"{group}:N", scale=alt.Scale(range=[TEAL, ORANGE]),
                legend=alt.Legend(title=None, orient="top"),
            ),
            strokeDash=alt.StrokeDash(f"{group}:N", legend=None),
        )
    line = base.mark_line(strokeWidth=2.5, color=TEAL)
    points = base.transform_filter(alt.datum.Iteration == focus).mark_point(
        filled=True, size=75, color=TEAL,
    )
    tooltip = ["Iteration:Q", alt.Tooltip(f"{value}:Q", format=".5f")]
    if group:
        tooltip.append(f"{group}:N")
    hover = alt.selection_point(nearest=True, on="pointerover", fields=["Iteration"], empty=False)
    selector = base.mark_point(opacity=0).encode(tooltip=tooltip).add_params(hover)
    guide = alt.Chart(pd.DataFrame({"Iteration": [focus]})).mark_rule(
        color="#94a3b8", strokeDash=[3, 3],
    ).encode(x="Iteration:Q")
    return (guide + line + points + selector).properties(height=245)


def activation_chart(name, z, a, activate):
    extent = max(4, min(1e12, abs(z) * 1.1))
    values = np.linspace(-extent, extent, 201)
    line = alt.Chart(pd.DataFrame({"Weighted sum": values, "Activation": activate(name, values)}))
    point = alt.Chart(pd.DataFrame({"Weighted sum": [z], "Activation": [a]}))
    return (
        line.mark_line(color=TEAL, strokeWidth=2.5).encode(x="Weighted sum:Q", y="Activation:Q")
        + point.mark_circle(color=ORANGE, size=130).encode(
            x="Weighted sum:Q", y="Activation:Q", tooltip=["Weighted sum:Q", "Activation:Q"],
        )
    ).properties(height=235)


def network_svg(labels, snapshot, selected):
    weights, masks = snapshot["weights"], snapshot["masks"]
    x = np.linspace(115, 780, len(labels))
    height = max(210, max(map(len, labels)) * 29 + 64)
    ys = [height / 2 + (np.arange(len(layer)) - (len(layer) - 1) / 2) * 29 for layer in labels]
    max_weight = max(1e-9, max(np.max(np.abs(w)) for w in weights))
    svg = [
        f'<svg viewBox="0 0 864 {height}" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Neural network weights and dropped connections" style="width:100%;display:block">',
        f'<rect x="0.5" y="0.5" width="863" height="{height - 1}" rx="18" fill="#f5f9fc" stroke="#dae5ee"/>',
    ]
    for layer, (w, mask) in enumerate(zip(weights, masks)):
        for (i, j), value in np.ndenumerate(w):
            active = mask[i, j]
            highlighted = selected == (layer, i, j)
            strength = np.sqrt(abs(value) / max_weight)
            color = (TEAL if value >= 0 else ORANGE) if active else "#64748b"
            opacity = 0.22 + 0.56 * strength if active else 0.28
            dash = ' stroke-dasharray="3 5"' if not active else ""
            status = "kept" if active else "dropped"
            label = escape(f"{labels[layer][i]} → {labels[layer + 1][j]}. Weight {value:+.5f}. {status}.")
            svg.append(
                f'<line x1="{x[layer]:.1f}" y1="{ys[layer][i]:.1f}" x2="{x[layer+1]:.1f}" y2="{ys[layer+1][j]:.1f}" '
                f'stroke="{color}" stroke-opacity="{1 if highlighted else opacity:.2f}" '
                f'stroke-width="{3.6 if highlighted else 0.65 + 1.8 * strength:.2f}"{dash}><title>{label}</title></line>'
            )
    for layer, names in enumerate(labels):
        heading = "INPUTS" if layer == 0 else "OUTPUT" if layer == len(labels) - 1 else f"HIDDEN {layer}"
        svg.append(f'<text x="{x[layer]:.1f}" y="21" fill="#38526a" text-anchor="middle" font-size="11" font-weight="700" letter-spacing="1.4">{heading}</text>')
        for index, label in enumerate(names):
            y = ys[layer][index]
            svg.append(f'<circle cx="{x[layer]:.1f}" cy="{y:.1f}" r="9" fill="#fff" stroke="#223b53" stroke-width="1.5"><title>{escape(label)}</title></circle>')
            if layer in (0, len(labels) - 1):
                offset, anchor = (-15, "end") if layer == 0 else (15, "start")
                svg.append(f'<text x="{x[layer] + offset:.1f}" y="{y + 4:.1f}" text-anchor="{anchor}" font-size="10" fill="#223b53">{escape(label)}</text>')
            else:
                svg.append(f'<text x="{x[layer]:.1f}" y="{y + 3:.1f}" text-anchor="middle" font-size="8" font-weight="700" fill="#223b53">{index + 1}</text>')
    return '<div style="font-family:system-ui,sans-serif">' + "".join(svg) + "</svg></div>"
