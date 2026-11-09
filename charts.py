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


def network_svg(labels, snapshot, selected, values, sums, activations, input_labels, raw, output, focus_only):
    weights, masks = snapshot["weights"], snapshot["masks"]
    width = 220 + 225 * (len(labels) - 1)
    x = np.linspace(110, width - 110, len(labels))
    half_widths = [91, *([49] * (len(labels) - 2)), 76]
    height = max(214, max(map(len, labels)) * 47 + 72)
    ys = [(height + 42) / 2 + (np.arange(len(layer)) - (len(layer) - 1) / 2) * 47 for layer in labels]
    max_weight = max(1e-9, max(np.max(np.abs(w)) for w in weights))
    edges, incoming_edges, badges, nodes = [], [], [], []
    selected_layer, _, selected_node = selected
    for layer, (w, mask) in enumerate(zip(weights, masks)):
        for (i, j), value in np.ndenumerate(w):
            incoming = layer == selected_layer and j == selected_node
            if focus_only and not incoming:
                continue
            active = mask[i, j]
            highlighted = selected == (layer, i, j)
            strength = np.sqrt(abs(value) / max_weight)
            color = (TEAL if value >= 0 else ORANGE) if active else "#64748b"
            opacity = 0.92 if incoming else 0.16 + 0.36 * strength
            dash = ' stroke-dasharray="4 5"' if not active else ""
            status = "Available" if active else "Dropped during this update"
            contribution = values[layer][0, i] * value
            label = escape(
                f"{labels[layer][i]} → {labels[layer + 1][j]}\n"
                f"Weight {value:+.6g} · {status}\n"
                f"Prediction contribution {contribution:+.6g}"
            )
            x1, x2 = x[layer] + half_widths[layer], x[layer + 1] - half_widths[layer + 1]
            y1, y2 = ys[layer][i], ys[layer + 1][j]
            line = (
                f'<g><title>{label}</title>'
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" stroke="transparent" stroke-width="9"/>'
                f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
                f'stroke="{color}" stroke-opacity="{opacity:.2f}" '
                f'stroke-width="{3.2 if highlighted else 0.7 + 1.7 * strength:.2f}"{dash}/></g>'
            )
            (incoming_edges if incoming else edges).append(line)
            if incoming:
                bx, by = x1 + 0.44 * (x2 - x1), y1 + 0.44 * (y2 - y1)
                badges.append(
                    f'<g><title>{label}</title>'
                    f'<rect x="{bx - 29:.1f}" y="{by - 9:.1f}" width="58" height="18" rx="5" '
                    f'fill="{"#e4f4f2" if highlighted else "#ffffff"}" stroke="{color}" '
                    f'stroke-width="{1.6 if highlighted else 0.65}"{dash}/>'
                    f'<text x="{bx:.1f}" y="{by + 3.5:.1f}" text-anchor="middle" fill="{color}" '
                    f'font-size="10" font-weight="700">{value:+.3g}</text></g>'
                )
    for layer, names in enumerate(labels):
        is_input = layer == 0
        is_output = layer == len(labels) - 1
        heading = "INPUTS" if is_input else "OUTPUT" if is_output else f"HIDDEN {layer}"
        subtitle = "Entered → model value" if is_input else activations[layer - 1]
        cx, half_width = x[layer], half_widths[layer]
        nodes.append(
            f'<text x="{cx:.1f}" y="23" fill="#38526a" text-anchor="middle" font-size="11" font-weight="750" letter-spacing="1.4">{heading}</text>'
            f'<text x="{cx:.1f}" y="41" fill="#475569" text-anchor="middle" font-size="11">{subtitle}</text>'
        )
        for index, label in enumerate(names):
            y = ys[layer][index]
            value = values[layer][0, index]
            if is_input:
                name = f"{label} · {input_labels[index]}"
                number = f"{raw[index]:,.5g} → {value:+.3g}"
                detail = f"{name}\nEntered value {raw[index]:.8g}\nStandardized model input {value:+.8g}"
            else:
                name, number = output if is_output else (label, f"a = {value:+.3g}")
                z = sums[layer - 1][0, index]
                bias = snapshot["biases"][layer - 1][index]
                detail = (
                    f"{name}\n{activations[layer - 1]} activation {value:+.8g}\n"
                    f"Weighted sum {z:+.8g} · bias {bias:+.8g}"
                )
            focused = layer == selected_layer + 1 and index == selected_node
            nodes.append(
                f'<g><title>{escape(detail)}</title>'
                f'<rect x="{cx - half_width:.1f}" y="{y - 19:.1f}" width="{2 * half_width}" height="38" rx="8" '
                f'fill="{"#e4f4f2" if focused else "#ffffff"}" stroke="{TEAL if focused else "#a7bacb"}" '
                f'stroke-width="{2 if focused else 1}"/>'
                f'<text x="{cx:.1f}" y="{y - 3:.1f}" text-anchor="middle" fill="#223b53" font-size="11" font-weight="700">{escape(name)}</text>'
                f'<text x="{cx:.1f}" y="{y + 12:.1f}" text-anchor="middle" fill="#38526a" font-size="11">{escape(number)}</text></g>'
            )
    return (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<title>Network snapshot diagram</title><style>body{margin:0}</style></head><body>'
        '<div style="overflow-x:auto;font-family:system-ui,sans-serif">'
        f'<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg" role="img" '
        'aria-label="Network diagram with example inputs, labeled incoming weights, activations, and prediction" '
        f'style="width:100%;min-width:{max(580, int(width * 0.8))}px;max-width:{max(900, width)}px;margin:auto;display:block">'
        f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="18" fill="#f5f9fc" stroke="#dae5ee"/>'
        + "".join(edges + incoming_edges + nodes + badges) + "</svg></div></body></html>"
    )
