# Neural Network Lab

An interactive Streamlit demo for introductory business analytics. Build a network with 1–10 inputs and 1–3 hidden layers of 1–10 nodes each. Choose each hidden layer’s activation, pick a cost function, and watch full-batch gradient descent update the weights.

## Run locally

Use Python 3.12 or 3.13. From this directory, run:

```sh
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Classroom activities

| Use case | Try this |
| --- | --- |
| Customer churn | Train, then change one subscriber’s support tickets and monthly logins. Trace how those inputs contribute to a hidden neuron’s weighted sum. |
| Sales prediction | Train and inspect the held-out predictions in dollars. Compare the predicted effect of raising ad spend for a low-traffic and a high-traffic store. |
| XOR challenge | Compare a Linear hidden layer with a Tanh hidden layer. Check all four predictions in the truth table. |

**Step once** performs one update. **Run 25 steps** advances the same run. **Train to plateau** stops when unmasked training cost fails to improve by the specified tolerance for 100 consecutive updates, or when it hits the step budget. Scrub through completed iterations to see the cost, stored weights, predictions, and connection mask at that point.

Use **Play** in **Train live** mode to watch one new weight update per frame. In **Replay saved iterations** mode, Play resumes from the selected iteration and stops at the end of the saved run. Starting replay at the end plays from iteration zero. **Pause** holds the current frame so you can inspect it or jump to another iteration. **Playback speed** sets the pace from 0.5 to 2 steps per second.

Open **Edit example inputs** to change the values flowing through the network snapshot. Each input shows the entered value and the standardized value the model receives. Choose a connection to label the weights feeding its destination neuron, or focus the diagram on just those links. Hover over a link for its weight or over a hidden or output node for its weighted sum, bias, and activation.

After a training step, **How this weight changed** shows the selected connection’s previous weight, learning rate, gradient, and updated weight. Open **View all weight updates at this iteration** to compare connections, or select **Gradient** under **Connection history** to follow the selected gradient over time. Replay uses the learning rate and gradient recorded at each step.

Each update draws a new independent random mask for the connections. Dropped weights receive zero gradient in that update. Kept weights are scaled by the inverse keep probability during training. The cost chart and predictions use all connections.

Customer churn and sales use fixed synthetic training and holdout sets. The XOR panel checks the four canonical input patterns. Sales cost is computed on standardized targets and the evaluation MAE is converted to dollars. The active run lives in session memory. The app uses synthetic data and requires no credentials.
