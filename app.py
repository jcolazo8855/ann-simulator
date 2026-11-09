"""Streamlit classroom interface for the neural network simulator."""

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from cases import LESSONS, make_case
from charts import ORANGE, TEAL, activation_chart, history_chart, network_svg
from network import ACTIVATIONS, CLASSIFICATION_LOSSES, MAX_ITERATIONS, REGRESSION_LOSSES, Network, activate, train

st.set_page_config(page_title="Neural Network Lab", page_icon="🧠", layout="wide")
st.html("""
<style>
.nn-hero {background:linear-gradient(115deg,#142942,#164861 70%,#08766e);color:#fff;
    border-radius:20px;padding:24px clamp(16px,4vw,38px);margin-bottom:18px;}
.nn-hero h1 {color:#fff;font-size:clamp(28px,4vw,46px);line-height:1.15;letter-spacing:-.035em;padding:4px 0;}
.nn-hero p {color:#dbf0f1;margin:6px 0 0;font-size:15px;}
.nn-hero .nn-eyebrow {color:#a1f5dc;letter-spacing:.16em;font-size:11px;font-weight:750;}
[data-testid="stMetric"] {background:#eef5fa;border-radius:12px;padding:10px 14px;}
[data-testid="stMetricValue"] {font-size:clamp(19px,3vw,29px);}
.block-container {padding-top:3.35rem;}
</style>
<header class="nn-hero">
  <div class="nn-eyebrow">BUSINESS ANALYTICS · INTERACTIVE LAB</div>
  <h1>Neural Network Lab</h1>
  <p>Build a network, watch it learn, and follow a prediction through a single neuron.</p>
</header>
""")

case_name = st.segmented_control(
    "Choose a use case", list(LESSONS), default="Customer churn", key="lesson",
    format_func=lambda name: f"{LESSONS[name].icon}  {name}", selection_mode="single", required=True,
)
lesson = LESSONS[case_name]
st.write(lesson.description)

with st.sidebar:
    st.header("Build your network")
    input_count = st.slider("Number of inputs", 1, 10, lesson.inputs, key=f"inputs:{case_name}")
    hidden_count = st.slider("Hidden layers", 1, 3, 1, key=f"layers:{case_name}")
    st.caption("A single output node follows your hidden layers.")
    hidden_sizes, activations = [], []
    for layer in range(1, hidden_count + 1):
        st.markdown(f"**Hidden layer {layer}**")
        hidden_sizes.append(st.slider(
            f"Nodes in layer {layer}", 1, 10, lesson.nodes, key=f"nodes:{case_name}:{layer}",
        ))
        activations.append(st.selectbox(
            f"Activation for layer {layer}", ACTIVATIONS, key=f"activation:{case_name}:{layer}",
        ))
    case = make_case(case_name, input_count)
    output_activation = "Linear" if case.regression else "Sigmoid"
    st.caption(f"Output activation · {output_activation}")
    st.divider()
    st.subheader("Training settings")
    loss = st.selectbox(
        "Cost function", REGRESSION_LOSSES if case.regression else CLASSIFICATION_LOSSES,
        key=f"loss:{case_name}",
    )
    rate = st.number_input(
        "Learning rate", min_value=0.001, max_value=1.0, value=lesson.rate,
        step=0.01, format="%.3f", key=f"rate:{case_name}",
        help="How far the model moves along the gradient at each update.",
    )
    drop = st.slider(
        "Connection drop probability (%)", 0, 90, 0,
        help="Every connection independently has this chance of dropping during each update.",
    ) / 100
    with st.expander("Advanced settings"):
        seed = st.number_input("Random seed", min_value=0, max_value=2**31 - 1, value=23, step=1)
        budget = st.number_input("Maximum auto-train steps", 100, 3000, 2000, step=100)
        tolerance = st.number_input(
            "Minimum cost improvement", 0.00001, 0.1, 0.0005, step=0.0001, format="%.5f",
            help="Auto-train stops after 100 updates without at least this much improvement.",
        )

sizes = (input_count, *hidden_sizes, 1)
signature = (case_name, sizes, tuple(activations), loss, seed)
if st.session_state.get("signature") != signature:
    st.session_state.signature = signature
    st.session_state.model = Network(sizes, activations, output_activation, loss, seed, case)
    st.session_state.pop("view_iteration", None)

model = st.session_state.model


def reset_model():
    st.session_state.pop("signature", None)


st.subheader("Train the model")
buttons = st.columns(4)
at_cap = model.iteration >= MAX_ITERATIONS
step = buttons[0].button("Step once", width="stretch", disabled=at_cap)
batch = buttons[1].button("Run 25 steps", width="stretch", disabled=at_cap)
auto = buttons[2].button("Train to plateau", width="stretch", type="primary", disabled=at_cap)
buttons[3].button("Reset weights", width="stretch", on_click=reset_model)
if step or batch or auto:
    bar = st.progress(0.0, text="Training…") if auto else None
    reason, count = train(
        model, budget if auto else 25 if batch else 1, rate, drop, tolerance if auto else None,
        (lambda fraction: bar.progress(fraction, text="Training…")) if bar else None,
    )
    if bar:
        bar.empty()
    st.session_state.view_iteration = model.iteration
    if reason == "unstable":
        st.warning(f"Training became unstable after {count} updates. The last valid weights are shown. Try lowering the learning rate or drop probability.")
    elif reason == "plateau":
        st.success(f"Training cost plateaued after {count:,} updates.")
    elif reason == "cap":
        st.info(f"The run reached {MAX_ITERATIONS:,} updates. Reset weights to start a new run.")
    elif auto:
        st.info(f"Completed the {count:,}-update auto-train budget.")

st.caption(
    "Each step updates weights once using the training set. Changing the case, architecture, "
    "cost function, or seed starts a new run. Learning rate and connection dropping apply to future updates."
)
if model.iteration:
    focus = st.slider("Inspect iteration", 0, model.iteration, key="view_iteration")
else:
    focus = 0
snapshot = model.history[focus]
eval_name = case.evaluation_name
train_cost = snapshot["training_cost"]
eval_cost = snapshot["evaluation_cost"]
weights, biases = snapshot["weights"], snapshot["biases"]
predictions = model.forward(case.eval_x, weights, biases)[0][-1].ravel()
actual = case.eval_y.ravel()
metric = np.mean(np.abs(case.decode_target(predictions) - case.decode_target(actual))) if case.regression else np.mean((predictions >= 0.5) == actual)
cards = st.columns(4)
cards[0].metric("Iteration", f"{focus:,} / {model.iteration:,}")
cards[1].metric("Training cost", f"{train_cost:.5f}", f"{train_cost - model.history[0]['training_cost']:+.5f}" if focus else None, delta_color="inverse")
cards[2].metric(f"{eval_name} cost", f"{eval_cost:.5f}")
cards[3].metric(f"{eval_name} {'MAE' if case.regression else 'accuracy'}", f"${metric:,.0f}" if case.regression else f"{metric:.1%}")

overview, neuron_tab, predictions_tab = st.tabs(["Training & weights", "Neuron microscope", "Predictions & activity"])
labels = [tuple(f"x{i + 1}" for i in range(input_count))]
labels.extend(tuple(f"H{layer}.{node + 1}" for node in range(size)) for layer, size in enumerate(hidden_sizes, 1))
labels.append(("ŷ",))

with overview:
    st.subheader("Cost by iteration")
    cost_rows = pd.DataFrame({
        "Iteration": range(len(model.history)),
        "Training": [row["training_cost"] for row in model.history],
        eval_name: [row["evaluation_cost"] for row in model.history],
    }).melt("Iteration", var_name="Sample", value_name="Cost")
    # Altair's default row limit is 5,000. Keep the current iteration and the endpoints.
    if len(cost_rows) > 5000:
        cost_rows = cost_rows[
            (cost_rows["Iteration"] % 2 == 0) | cost_rows["Iteration"].isin([focus, model.iteration])
        ]
    st.altair_chart(history_chart(cost_rows, "Cost", focus, "Sample"), width="stretch")
    st.caption(
        f"{loss} evaluated with all connections active. "
        + ("Sales cost uses standardized dollars. The MAE tile reports dollars." if case.regression else "Accuracy uses a 0.50 probability cutoff.")
    )
    st.subheader("Follow a connection")
    connections = [
        (layer, i, j) for layer, w in enumerate(weights) for i, j in np.ndindex(w.shape)
    ]
    selected = st.selectbox(
        "Connection to trace", connections,
        format_func=lambda edge: f"{labels[edge[0]][edge[1]]} → {labels[edge[0] + 1][edge[2]]}",
        key=f"trace:{sizes}",
    )
    st.html(network_svg(labels, snapshot, selected))
    dropped = sum(mask.size - np.count_nonzero(mask) for mask in snapshot["masks"])
    st.caption(
        f"Teal = positive weight · orange = negative weight · dashed gray = dropped during this update. "
        f"{dropped} of {len(connections)} connections dropped"
        + (f" with {snapshot['drop_rate']:.0%} probability per connection." if focus else " at initialization.")
    )
    layer, i, j = selected
    weight_history = np.array([row["weights"][layer][i, j] for row in model.history])
    mask_history = [row["masks"][layer][i, j] for row in model.history]
    st.altair_chart(history_chart(pd.DataFrame({
        "Iteration": range(len(weight_history)), "Weight": weight_history,
    }), "Weight", focus), width="stretch")
    change = weight_history[focus] - weight_history[max(0, focus - 1)]
    status = "Initial weight" if focus == 0 else "Kept for update" if mask_history[focus] else "Dropped for update"
    st.caption(f"{status} · weight {weight_history[focus]:+.5f} · change from preceding iteration {change:+.5f}")
    with st.expander("View all connection weights at this iteration"):
        st.dataframe(pd.DataFrame([
            {
                "From": labels[l][src], "To": labels[l + 1][dest],
                "Weight": weights[l][src, dest],
                "Change": weights[l][src, dest] - model.history[max(0, focus - 1)]["weights"][l][src, dest],
                "Used in update": bool(snapshot["masks"][l][src, dest]) if focus else None,
            }
            for l, src, dest in connections
        ]), hide_index=True, width="stretch")
    st.caption(
        "A new mask is sampled on every step. Dropped weights receive zero update. "
        "Kept weights are scaled by 1 ÷ (1 − drop probability) during training. Predictions use the full network."
    )

with neuron_tab:
    st.subheader("Send an example through the network")
    st.caption("Edit the inputs to see a prediction and trace the math at the selected iteration.")
    controls = st.columns(2)
    raw = []
    for index, feature in enumerate(case.features):
        initial = feature.low + round((float(case.center[index]) - feature.low) / feature.step) * feature.step
        if case_name == "XOR challenge":
            initial = float(index == 1)
        raw.append(controls[index % 2].number_input(
            f"x{index + 1} · {feature.name}", float(feature.low), float(feature.high),
            float(round(initial, 6)), float(feature.step), key=f"example:{case_name}:{index}",
        ))
    model_inputs = case.encode(np.array(raw, dtype=float))
    values, sums = model.forward(model_inputs[None, :], weights, biases)
    output = float(values[-1][0, 0])
    if case.regression:
        st.metric("Predicted weekly sales", f"${case.decode_target(output):,.0f}")
    else:
        label = "Predicted churn probability" if case_name == "Customer churn" else "Predicted probability of XOR = 1"
        st.metric(label, f"{output:.1%}")
    left, right = st.columns(2)
    inspected_layer = left.selectbox(
        "Layer to inspect", range(len(weights)),
        format_func=lambda i: f"Hidden layer {i + 1}" if i < hidden_count else "Output layer",
        key=f"inspect-layer:{sizes}",
    )
    inspected_node = right.selectbox(
        "Neuron to inspect", range(weights[inspected_layer].shape[1]),
        format_func=lambda i: labels[inspected_layer + 1][i], key=f"inspect-node:{sizes}:{inspected_layer}",
    )
    inputs = values[inspected_layer][0]
    incoming = weights[inspected_layer][:, inspected_node]
    bias = biases[inspected_layer][inspected_node]
    z = sums[inspected_layer][0, inspected_node]
    a = values[inspected_layer + 1][0, inspected_node]
    activation = model.activations[inspected_layer]
    st.latex(r"z = b + \\sum_i w_i a_i \\qquad a = f(z)")
    detail, curve = st.columns(2)
    detail.dataframe(pd.DataFrame({
        "Source": labels[inspected_layer],
        "Incoming value": inputs, "Weight": incoming, "Contribution": inputs * incoming,
    }).round(5), hide_index=True, width="stretch")
    detail.markdown(f"**Bias** {bias:+.5f} · **Weighted sum** {z:+.5f} · **{activation} output** {a:+.5f}")
    detail.caption("First-layer inputs are standardized using the training-set means and standard deviations.")
    curve.altair_chart(activation_chart(activation, z, a, activate), width="stretch")

with predictions_tab:
    st.subheader(eval_name + " predictions")
    st.info(f"**Try it.** {lesson.prompt}")
    frame = pd.DataFrame(case.raw_eval, columns=[feature.name for feature in case.features])
    if case.regression:
        frame["Actual sales ($)"] = case.decode_target(actual)
        frame["Predicted sales ($)"] = case.decode_target(predictions)
        dots = alt.Chart(frame).mark_circle(color=TEAL, size=54, opacity=0.65).encode(
            x=alt.X("Actual sales ($):Q", scale=alt.Scale(zero=False)),
            y=alt.Y("Predicted sales ($):Q", scale=alt.Scale(zero=False)),
            tooltip=list(frame.columns[:input_count]) + ["Actual sales ($)", "Predicted sales ($)"],
        )
        low = frame[["Actual sales ($)", "Predicted sales ($)"]].min().min()
        high = frame[["Actual sales ($)", "Predicted sales ($)"]].max().max()
        diagonal = alt.Chart(pd.DataFrame({"Actual sales ($)": [low, high], "Predicted sales ($)": [low, high]})).mark_line(
            color=ORANGE, strokeDash=[4, 4],
        ).encode(x="Actual sales ($):Q", y="Predicted sales ($):Q")
        st.altair_chart((diagonal + dots).properties(height=260), width="stretch")
    else:
        frame["Actual class"] = actual.astype(int)
        frame["Predicted probability"] = predictions
        frame["Predicted class"] = (predictions >= 0.5).astype(int)
        if case_name == "Customer churn":
            true_positive = int(np.sum((predictions >= 0.5) & (actual == 1)))
            false_positive = int(np.sum((predictions >= 0.5) & (actual == 0)))
            false_negative = int(np.sum((predictions < 0.5) & (actual == 1)))
            true_negative = int(np.sum((predictions < 0.5) & (actual == 0)))
            st.caption(f"At a 50% cutoff · {true_positive} true positives · {false_positive} false positives · {false_negative} false negatives · {true_negative} true negatives.")
        else:
            st.caption("The four truth-table rows use the training switch patterns. Extra switches, if selected, are set to 0 here.")
            if input_count == 1:
                st.warning("Switch B is missing. With only Switch A, the model sees both possible labels for each input.")
    st.dataframe(frame.round(4), hide_index=True, width="stretch")
    st.caption(f"{len(case.train_y)} training rows · {len(case.eval_y)} {eval_name.lower()} rows · synthetic classroom data")

st.divider()
st.caption("Classroom experiment · Train once with 0% connection dropping, then reset and repeat with 20%. Compare training cost, evaluation cost, and the path of the same weight.")
