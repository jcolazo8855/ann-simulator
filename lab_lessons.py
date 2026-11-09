"""Fixed backpropagation example and a single-neuron OLS comparison."""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from lab_charts import ORANGE, TEAL
from lab_network import Network

BACKPROP_TAB = "Backpropagation demo"
OLS_TAB = "OLS comparison"

STEPS = (
    (
        "Make the prediction",
        "Follow the input through the hidden ReLU neuron and then through the linear output.",
        r"\begin{aligned}z_1&=w_1x=0.5(2)=1\\h&=\operatorname{ReLU}(1)=1\\"
        r"\hat y&=w_2h=2(1)=2\end{aligned}",
    ),
    (
        "Measure the error and start backward",
        "The prediction is one unit below actual sales. The loss slope at the prediction is −1.",
        r"\begin{aligned}e&=\hat y-y=2-3=-1\\L&=\tfrac12e^2=0.5\\"
        r"\frac{\partial L}{\partial\hat y}&=\hat y-y=-1\end{aligned}",
    ),
    (
        "Find the output-weight gradient",
        "The output weight multiplies the hidden value. Multiply that local slope by the loss slope at the prediction.",
        r"\begin{aligned}\frac{\partial\hat y}{\partial w_2}&=h=1\\"
        r"\frac{\partial L}{\partial w_2}&="
        r"\frac{\partial L}{\partial\hat y}\frac{\partial\hat y}{\partial w_2}=(-1)(1)=-1\end{aligned}",
    ),
    (
        "Send the gradient through the hidden neuron",
        "First go backward through the output weight. Then go through ReLU. Its input is positive, so its local slope is 1.",
        r"\begin{aligned}\frac{\partial L}{\partial h}&="
        r"\frac{\partial L}{\partial\hat y}\frac{\partial\hat y}{\partial h}=(-1)(2)=-2\\"
        r"\frac{\partial L}{\partial z_1}&="
        r"\frac{\partial L}{\partial h}\operatorname{ReLU}'(1)=(-2)(1)=-2\end{aligned}",
    ),
    (
        "Find the hidden-weight gradient",
        "The input weight multiplies x. The four factors below are the local slopes along the whole path from that weight to the loss.",
        r"\begin{aligned}\frac{\partial L}{\partial w_1}"
        r"&=\frac{\partial L}{\partial\hat y}\frac{\partial\hat y}{\partial h}"
        r"\frac{\partial h}{\partial z_1}\frac{\partial z_1}{\partial w_1}\\"
        r"&=(-1)(2)(1)(2)=-4\end{aligned}",
    ),
    (
        "Update both weights and predict again",
        "Both gradients were calculated at the starting weights. Subtract the learning rate times each gradient, then run a new forward pass.",
        r"\begin{aligned}w_1^{\mathrm{new}}&=0.5-0.05(-4)=0.70\\"
        r"w_2^{\mathrm{new}}&=2-0.05(-1)=2.05\\"
        r"\hat y_{\mathrm{new}}&=2.05\,\operatorname{ReLU}(0.70\cdot 2)=2.87\\"
        r"L_{\mathrm{new}}&=\tfrac12(2.87-3)^2=0.00845\end{aligned}",
    ),
)


def _backprop_flow(step):
    final = step == len(STEPS) - 1
    nodes = (
        ("Input", "x = 2", "Sales predictor"),
        ("Hidden · ReLU", "h = 1.4" if final else "h = 1", "z₁ = 1.4" if final else "z₁ = 1"),
        ("Linear output", "ŷ = 2.87" if final else "ŷ = 2", "Predicted sales"),
        ("Squared loss × ½", "L = 0.00845" if final else "L = 0.5" if step else "y = 3", "Actual sales = 3"),
    )
    links = ("w₁ = 0.70" if final else "w₁ = 0.50", "w₂ = 2.05" if final else "w₂ = 2", "Compare")
    highlighted = ((0, 1, 2), (2, 3), (1, 2), (1, 2), (0, 1), (0, 1, 2, 3))[step]
    cells = []
    for i, (title, value, note) in enumerate(nodes):
        cells.append(
            f'<div class="bp-node {"bp-active" if i in highlighted else ""}">'
            f'<small>{title}</small><strong>{value}</strong><small>{note}</small></div>'
        )
        if i < len(links):
            cells.append(f'<div class="bp-link">{links[i]}<span aria-hidden="true">⟶</span></div>')
    backward = []
    if 1 <= step <= 4:
        backward.append('<div class="bp-signal" style="grid-row:2;grid-column:6 / 8">← From loss to prediction · ∂L/∂ŷ = −1</div>')
    if 3 <= step <= 4:
        backward.append('<div class="bp-signal" style="grid-row:2;grid-column:3 / 6">← Through w₂ · ∂L/∂h = −2 · Through ReLU · ∂L/∂z₁ = −2</div>')
    if 2 <= step <= 4:
        backward.append('<div class="bp-gradient" style="grid-row:3;grid-column:4 / 6">∂L/∂w₂ = −1</div>')
    if step == 4:
        backward.append('<div class="bp-gradient" style="grid-row:3;grid-column:1 / 4">∂L/∂w₁ = −4</div>')
    return (
        """
        <style>
        .bp-frame {border:1px solid #d8e6ee;border-radius:14px;background:#f5f9fc;padding:12px;overflow-x:auto;}
        .bp-grid {display:grid;min-width:620px;grid-template-columns:1.25fr .62fr 1.25fr .62fr 1.25fr .62fr 1.25fr;
                  align-items:center;gap:8px;text-align:center;color:#223b53;}
        .bp-node {display:grid;gap:2px;background:#fff;border:1px solid #b7cad9;border-radius:9px;padding:8px;}
        .bp-node strong {font-size:22px;color:#142942;}
        .bp-active {border-color:#007f78;background:#e6f5f2;}
        .bp-link {font-size:12px;font-weight:650;color:#007f78;}
        .bp-link span {display:block;line-height:1;font-size:25px;}
        .bp-signal,.bp-gradient {font-size:12px;border-radius:6px;padding:5px;font-weight:650;}
        .bp-signal {color:#944019;background:#faece4;}
        .bp-gradient {color:#125750;background:#dff0ec;}
        </style>
        <div class="bp-frame" role="group" aria-label="Forward path and backward loss sensitivities">
        <div class="bp-grid">
        """
        + "".join(cells + backward) + "</div></div>"
    )


def render_backprop_demo():
    st.subheader("Backpropagation, one step at a time")
    st.caption(
        "One toy sales observation · input 2 · actual sales 3 · zero biases · learning rate 0.05. "
        "This example runs separately from the simulator’s training session."
    )
    step = st.session_state.get("backprop_step", 0)

    def go_to(index):
        st.session_state.backprop_step = index

    with st.container(horizontal=True):
        st.button("Previous step", on_click=go_to, args=(step - 1,), disabled=step == 0)
        st.button("Next step", on_click=go_to, args=(step + 1,), disabled=step == len(STEPS) - 1, type="primary")
        st.button("Restart walkthrough", on_click=go_to, args=(0,), disabled=step == 0)

    title, explanation, equation = STEPS[step]
    st.progress((step + 1) / len(STEPS), text=f"Step {step + 1} of {len(STEPS)}")
    st.markdown(f"#### {title}")
    st.html(_backprop_flow(step))
    st.write(explanation)
    st.latex(equation)


def ols_parameters(case):
    """Fit in original units and express the same fit in standardized model units."""
    train_raw = case.train_x * case.scale + case.center
    design = np.column_stack((np.ones(len(train_raw)), train_raw))
    coefficients = np.linalg.lstsq(design, case.decode_target(case.train_y).ravel(), rcond=None)[0]
    slopes = coefficients[1:]
    weights = slopes * case.scale / case.target_scale
    bias = (coefficients[0] + case.center @ slopes - case.target_center) / case.target_scale
    return coefficients, weights, bias


def render_ols_demo(case, raw):
    st.subheader("Ordinary least squares with one neuron")
    st.write(
        "Fit ordinary least squares (OLS) to the sales training rows. A single linear neuron with "
        "the corresponding weights and bias produces the same predictions."
    )
    st.caption(
        f"{len(case.train_x)} training rows · {len(case.eval_x)} holdout rows · {len(case.features)} selected inputs. "
        "Change the number of inputs to refit OLS. Open Edit example inputs above to compare a different prediction."
    )
    coefficients, weights, bias = ols_parameters(case)
    examples = np.vstack((raw, case.raw_eval))
    ols_predictions = coefficients[0] + examples @ coefficients[1:]
    neuron = Network((len(weights), 1), (), "Linear", "Mean squared error", 0, case)
    activations, _ = neuron.forward(case.encode(examples), [weights[:, None]], [np.array([bias])])
    neuron_predictions = case.decode_target(activations[-1].ravel())
    gap = np.abs(ols_predictions[1:] - neuron_predictions[1:]).max()

    cards = st.columns(3)
    cards[0].metric("OLS example prediction", "$" + f"{ols_predictions[0]:,.2f}")
    cards[1].metric("Neuron example prediction", "$" + f"{neuron_predictions[0]:,.2f}")
    cards[2].metric("Largest holdout gap", "$" + f"{gap:.2e}")
    st.latex(r"\hat y_{\mathrm{model}}=b+\sum_j w_j x_{j,\mathrm{standardized}}")
    st.caption(
        "The OLS coefficients use original units. The neuron weights and bias express the same fit "
        "using standardized inputs and a standardized output, which the app converts back to dollars."
    )
    st.dataframe(pd.DataFrame({
        "Term": ["Intercept / bias", *[feature.name for feature in case.features]],
        "OLS coefficient": coefficients,
        "Neuron parameter": np.round(np.r_[bias, weights], 6) + 0.0,
    }), hide_index=True, width="stretch")
    with st.expander("How OLS coefficients become neuron parameters"):
        st.write("Use the training-set means and standard deviations for each input and for sales.")
        st.latex(
            r"w_j=\beta_j\frac{\sigma_j}{\sigma_y},\qquad "
            r"b=\frac{\beta_0+\sum_j\beta_j\mu_j-\mu_y}{\sigma_y}"
        )
        st.latex(r"\hat y_{\mathrm{dollars}}=\mu_y+\sigma_y\hat y_{\mathrm{model}}")

    st.markdown("#### Holdout predictions")
    comparison = pd.DataFrame({
        "OLS prediction ($)": ols_predictions[1:],
        "Neuron prediction ($)": neuron_predictions[1:],
        "Actual sales ($)": case.decode_target(case.eval_y.ravel()),
    })
    low, high = float(ols_predictions[1:].min()), float(ols_predictions[1:].max())
    x, y = "OLS prediction ($)", "Neuron prediction ($)"
    axes = {"x": alt.X(f"{x}:Q", scale=alt.Scale(zero=False)), "y": alt.Y(f"{y}:Q", scale=alt.Scale(zero=False))}
    diagonal = alt.Chart(pd.DataFrame({x: [low, high], y: [low, high]})).mark_line(
        color=ORANGE, strokeDash=[4, 4],
    ).encode(**axes)
    dots = alt.Chart(comparison).mark_circle(color=TEAL, opacity=0.65, size=44).encode(
        **axes, tooltip=[alt.Tooltip(f"{column}:Q", format=",.2f") for column in comparison],
    )
    st.altair_chart((diagonal + dots).properties(height=270), width="stretch")
    st.caption(
        "Points on the diagonal show agreement between the neuron and OLS. "
        "The actual sales values in the tooltips let you check prediction accuracy separately."
    )
