import torch
import torch.nn as nn
import torch.optim as optim
from model import Network
import plotly.express as px
import numpy as np
import pandas as pd


def training_loop(n_epochs: int, learning_rate: float, x: torch.Tensor, y: torch.Tensor):
    model = Network(1, 10, 1)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = nn.MSELoss()

    for epoch in range(1, n_epochs+1):
        y_pred = model(x)
        loss: torch.Tensor = loss_fn(y_pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 500 == 0:
            print(f"Epoch: {epoch}, Loss: {loss.item()}")
    print("Done!!!\n")
    
    return model

def draw(x: torch.Tensor, y: torch.Tensor, y_pred: torch.Tensor):
    x = x.view(-1).data.numpy()
    y = y.view(-1).data.numpy()
    y_pred = y_pred.view(-1).data.numpy()
    y_ideal = np.sin(x)

    df1 = pd.DataFrame({"x":x, "y":y, "type":"true"})
    df2 = pd.DataFrame({"x":x, "y":y_pred, "type":"pred"})
    df = pd.concat([df1, df2], axis=0)

    fig = px.line(df, x="x", y="y", color="type", symbol="type", markers=True,
                  title="sine wave", template="simple_white", symbol_sequence=["star", "diamond"])
    fig.update_layout(
        yaxis=dict(showgrid=True, gridcolor='#E6EAF1'),
    )
    fig.add_scatter(x=x, y=y_ideal, mode='lines', name="ideal")
    fig.show()

def draw_mutiple(nums: list, x: torch.Tensor, ys: list[torch.Tensor]):
    x = x.view(-1).data.numpy()
    y_ideal = np.sin(x)
    all_y = []
    
    for idx in range(len(nums)):
        temp_y = ys[idx].view(-1).data
        temp_num = nums[idx]
        temp = pd.DataFrame({"x":x, "y":temp_y, "nums":str(temp_num)})
        all_y.append(temp)
    df = pd.concat(all_y, axis=0)

    fig = px.line(df, x="x", y="y", title="<b>sine wave</b>", markers=True,
                color="nums", symbol="nums", template="simple_white")
    fig.update_layout(
        yaxis=dict(showgrid=True, gridcolor='#E6EAF1'),
    )
    fig.update_xaxes(title_text="<b>x</b>")
    fig.update_yaxes(title_text="<b>y</b>")
    fig.add_scatter(x=x, y=y_ideal, mode='lines', name="ideal")
    fig.show()