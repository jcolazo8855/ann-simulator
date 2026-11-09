"""Streamlit classroom interface for the neural network simulator."""

from time import monotonic

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

from cases import LESSONS, make_case
from lab_charts import ORANGE, TEAL, activation_chart, history_chart, network_svg
from lab_network import ACTIVATIONS, CLASSIFICATION_LOSSES, MAX_ITERATIONS, REGRESSION_LOSSES, Network, activate, train
from teaching_demos import BACKPROP_TAB, OLS_TAB, render_backprop_demo, render_ols_demo

st.set_page_config(page_title="Neural Network Lab", page_icon="🧠", layout="wide")
st.html("""
<style>
.nn-hero {background:linear-gradient(115deg,#142942,#164861 70%,#08766e);color:#fff;
    border-radius:20px;padding:24px clamp(16px,4vw,38px);margin-bottom:18px;}
.nn-hero h1 {color:#fff;font-size:clamp(28px,4vw,46px);line-height:1.15;letter-spacing:-.035em;padding:4px 0;}
.nn-hero p {color:#dbf0f1;margin:6px 0 0;font-size:15px;}
.nn-hero .nn-eyebrow {color:#a1f5dc;letter-spacing:.16em;font-size:11px;font-weight:750;}
[data-testid="stMetric"] {background:#eef5fa;border-radius:12px;padding:10px 14px;container-type:inline-size;}
[data-testid="stMetricValue"] {font-size:clamp(15px,18cqi,29px);}
[data-testid="stMetricLabel"] .stMarkdownEllipsis,
[data-testid="stMetricValue"] .stMarkdownEllipsis {white-space:normal;overflow-wrap:anywhere;}
.block-container,[data-testid="stBottomBlockContainer"] {padding-inline:clamp(1rem,4vw,5rem);}
[data-testid="stBottomBlockContainer"] {padding-bottom:1rem;}
.block-container {padding-top:3.35rem;}
</style>
<header class="nn-hero">
  <div class="nn-eyebrow">BUSINESS ANALYTICS · INTERACTIVE LAB</div>
  <h1>Neural Network Lab</h1>
  <p>Build a network, watch it learn, and follow a prediction through a single neuron.</p>
</header>
""")

case_name = st.segmented_control(
    "Choose a dataset", list(LESSONS), default="Customer churn", key="lesson",
    format_func=lambda name: f"{LESSONS[name].icon}  {name}", selection_mode="single", required=True,
)
lesson = LESSONS[case_name]
outcome = "Continuous outcome · regression" if case_name == "Sales prediction" else "Binary outcome · classification"
st.write(f"**{outcome}** · {lesson.description}")

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


def pause_playback():
    st.session_state.playing = False
    st.session_state.pop("playback_notice", None)


if st.session_state.get("signature") != signature:
    st.session_state.signature = signature
    st.session_state.model = Network(sizes, activations, output_activation, loss, seed, case)
    st.session_state.pop("view_iteration", None)
    pause_playback()

model = st.session_state.model
if not isinstance(model, Network):
    # An open session can still hold the model class from a previous deployment.
    upgraded = Network(sizes, activations, output_activation, loss, seed, case)
    upgraded.weights, upgraded.biases = model.weights, model.biases
    upgraded.history, upgraded.mask_rng = model.history, model.mask_rng
    st.session_state.model = model = upgraded


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
    pause_playback()
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
with st.expander("Edit example inputs"):
    st.caption("These values feed the network diagram and the Neuron microscope.")
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

LIVE, REPLAY = "Train live", "Replay saved iterations"
mode = st.session_state.get("playback_mode", LIVE)
teaching_tabs = [BACKPROP_TAB, OLS_TAB] if case.regression else [BACKPROP_TAB]
in_teaching_tab = st.session_state.get("lab_view") in teaching_tabs


def reset_playback_clock():
    st.session_state.last_frame_at = monotonic()


def play():
    cursor = st.session_state.get("view_iteration", model.iteration)
    if mode == LIVE:
        st.session_state.view_iteration = model.iteration
    elif cursor == model.iteration:
        st.session_state.view_iteration = 0
    st.session_state.playing = True
    st.session_state.pop("playback_notice", None)
    reset_playback_clock()


def finish_playback(notice):
    st.session_state.playing = False
    st.session_state.playback_notice = notice
    st.rerun()


with st.bottom:
    with st.container(horizontal=True, vertical_alignment="bottom"):
        st.button(
            "Play", icon=":material/play_arrow:", width=85, on_click=play, type="primary",
            disabled=in_teaching_tab or st.session_state.playing or (at_cap if mode == LIVE else model.iteration == 0),
        )
        st.button(
            "Pause", icon=":material/pause:", width=92, on_click=pause_playback,
            disabled=not st.session_state.playing,
        )
        st.selectbox(
            "Playback mode", (LIVE, REPLAY), key="playback_mode", width=200, on_change=pause_playback,
            help="Train live makes a new weight update each frame. Replay starts at the selected iteration and stops at the end of the saved run.",
        )
        speed = st.selectbox(
            "Playback speed", (0.5, 1, 2), index=1, key="playback_speed", width=145,
            format_func=lambda value: f"{value:g} {'step' if value == 1 else 'steps'}/sec",
            on_change=reset_playback_clock,
        )
    if notice := st.session_state.get("playback_notice"):
        st.caption(notice)

interval = 1 / speed


@st.fragment(run_every=interval if st.session_state.playing else None)
def render_run():
    if (st.session_state.get("lab_view") in teaching_tabs) != in_teaching_tab:
        if not in_teaching_tab:
            pause_playback()
        st.rerun()

    if st.session_state.playing and monotonic() - st.session_state.last_frame_at >= interval - 0.05:
        reset_playback_clock()
        if mode == REPLAY:
            st.session_state.view_iteration += 1
            if st.session_state.view_iteration == model.iteration:
                finish_playback("Replay finished at the last saved iteration.")
        else:
            reason, _ = train(model, 1, rate, drop)
            st.session_state.view_iteration = model.iteration
            if reason == "unstable":
                finish_playback("Training became unstable. Try lowering the learning rate or drop probability.")
            if reason == "cap":
                finish_playback(f"Training reached the {MAX_ITERATIONS:,}-step limit.")

    focus = 0
    if model.iteration:
        focus = st.slider(
            "Inspect iteration", 0, model.iteration, key="view_iteration", disabled=st.session_state.playing,
            help="Pause playback to jump to a saved iteration.",
        )
    snapshot = model.history[focus]
    eval_name = case.evaluation_name
    train_cost = snapshot["training_cost"]
    eval_cost = snapshot["evaluation_cost"]
    weights, biases = snapshot["weights"], snapshot["biases"]
    predictions = model.forward(case.eval_x, weights, biases)[0][-1].ravel()
    actual = case.eval_y.ravel()
    if case.regression:
        metric = np.mean(np.abs(case.decode_target(predictions) - case.decode_target(actual)))
    else:
        metric = np.mean((predictions >= 0.5) == actual)

    values, sums = model.forward(model_inputs[None, :], weights, biases)
    output = float(values[-1][0, 0])
    if case.regression:
        output_label, prediction_label = "Weekly sales", "Predicted weekly sales"
        example_prediction = f"${case.decode_target(output):,.0f}"
    elif case_name == "Customer churn":
        output_label, prediction_label = "Churn probability", "Predicted churn probability"
        example_prediction = f"{output:.1%}"
    else:
        output_label, prediction_label = "P(XOR = 1)", "Predicted probability of XOR = 1"
        example_prediction = f"{output:.1%}"

    overview, neuron_tab, predictions_tab, backprop_tab, *ols_tabs = st.tabs(
        ["Network & training", "Neuron microscope", "Predictions & activity", *teaching_tabs],
        key="lab_view", on_change="rerun",
    )
    labels = [tuple(f"x{i + 1}" for i in range(input_count))]
    labels.extend(tuple(f"H{layer}.{node + 1}" for node in range(size)) for layer, size in enumerate(hidden_sizes, 1))
    labels.append(("ŷ",))

    with overview:
        previous_weights = model.history[focus - 1]["weights"] if focus else None
        gradients = snapshot.get("gradients")
        connections = [
            (layer, i, j) for layer, w in enumerate(weights) for i, j in np.ndindex(w.shape)
        ]
        with st.container(border=True):
            st.subheader("Network snapshot")
            st.caption(f"{loss} · Costs use the full network and the corresponding dataset.")
            cards = st.columns(4)
            cards[0].metric("Iteration", f"{focus:,} / {model.iteration:,}")
            cards[1].metric(
                "Training cost", f"{train_cost:.5f}",
                f"{train_cost - model.history[0]['training_cost']:+.5f}" if focus else None,
                delta_color="inverse",
            )
            cards[2].metric(f"{eval_name} cost", f"{eval_cost:.5f}")
            cards[3].metric(
                f"{eval_name} {'MAE' if case.regression else 'accuracy'}",
                f"${metric:,.0f}" if case.regression else f"{metric:.1%}",
            )
            trace, filter_edges = st.columns(2, vertical_alignment="bottom")
            selected = trace.selectbox(
                "Connection to trace", connections,
                format_func=lambda edge: f"{labels[edge[0]][edge[1]]} → {labels[edge[0] + 1][edge[2]]}",
                key=f"trace:{sizes}",
            )
            focus_only = filter_edges.toggle("Only show links into the selected neuron")
            st.iframe(network_svg(
                labels, snapshot, selected, values, sums, model.activations,
                [feature.name for feature in case.features], raw, (output_label, example_prediction), focus_only,
                previous_weights=previous_weights,
            ))
            dropped = sum(mask.size - np.count_nonzero(mask) for mask in snapshot["masks"])
            st.caption(
                "Teal = positive weight · orange = negative weight · dashed gray = dropped during the update. "
                f"{dropped} of {len(connections)} connections dropped"
                + (f" at a {snapshot['drop_rate']:.0%} drop probability." if focus else " at initialization.")
            )
            st.caption(
                "The labeled weights feed the highlighted neuron. Hover over a link for its weight and update "
                "or a neuron for its activation, weighted sum, and bias. The prediction uses the full network."
            )

        layer, i, j = selected
        after = weights[layer][i, j]
        st.subheader("How this weight changed")
        st.caption(f"{labels[layer][i]} → {labels[layer + 1][j]} · Iteration {focus:,}")
        if focus == 0:
            st.markdown(f"**Initial weight** `{after:+.7g}`. Train one step to see its first gradient.")
        else:
            before = previous_weights[layer][i, j]
            status = "Used in this update" if snapshot["masks"][layer][i, j] else "Dropped for this update"
            if gradients is not None:
                gradient = gradients[layer][i, j]
                step_rate = snapshot["learning_rate"]
                st.markdown("**New weight = previous weight − learning rate × gradient**")
                st.markdown(f"`{before:+.7g} − {step_rate:.7g} × ({gradient:+.7g}) ≈ {after:+.7g}`")
                st.caption(f"Gradient {gradient:+.7g} · change {after - before:+.7g} · {status}")
            else:
                st.caption(f"Weight {before:+.7g} → {after:+.7g} · {status}. Train one more step to record a gradient.")
        st.caption(
            "The gradient comes from the training batch before the update, using that step’s connection mask. "
            "The cost chart shows the full network after the update."
        )
        with st.expander("View all weight updates at this iteration"):
            st.dataframe(pd.DataFrame([
                {
                    "From": labels[l][src],
                    "To": labels[l + 1][dest],
                    "Weight before": previous_weights[l][src, dest] if focus else None,
                    "Gradient": gradients[l][src, dest] if gradients is not None else None,
                    "Weight change": weights[l][src, dest] - previous_weights[l][src, dest] if focus else None,
                    "Weight after": weights[l][src, dest],
                    "Used in update": bool(snapshot["masks"][l][src, dest]) if focus else None,
                }
                for l, src, dest in connections
            ]), hide_index=True, width="stretch")

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
        st.subheader("Connection history")
        quantity = st.radio("Plot for the selected connection", ("Weight", "Gradient"), horizontal=True)
        history_key = "weights" if quantity == "Weight" else "gradients"
        series = [
            row[history_key][layer][i, j] if row.get(history_key) is not None else None
            for row in model.history
        ]
        if any(value is not None for value in series):
            st.altair_chart(history_chart(pd.DataFrame({
                "Iteration": range(len(series)), quantity: series,
            }), quantity, focus), width="stretch")
        else:
            st.caption("Train one step to plot the gradient.")
        st.caption(
            "A new mask is sampled on every step. Dropped weights receive zero update. "
            "Kept weights are scaled by 1 ÷ (1 − drop probability) during training. Predictions use the full network."
        )

    with neuron_tab:
        st.subheader("Send an example through the network")
        st.caption("Use Edit example inputs above to change the prediction and trace the math at the selected iteration.")
        st.metric(prediction_label, example_prediction)
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

    if backprop_tab.open:
        with backprop_tab:
            render_backprop_demo()
    if ols_tabs and ols_tabs[0].open:
        with ols_tabs[0]:
            render_ols_demo(case, raw)


render_run()

st.divider()
st.caption("Classroom experiment · Train once with 0% connection dropping, then reset and repeat with 20%. Compare training cost, evaluation cost, and the path of the same weight.")
