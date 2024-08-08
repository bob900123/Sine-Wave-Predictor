import numpy as np
import torch
from utils import training_loop, draw, draw_mutiple

if __name__ == "__main__":
    num_points = 50
    learning_rate = 0.001
    n_epochs = 5000

    x = np.linspace(0, 6.5, num_points)
    # 加入噪聲
    y = np.sin(x) + [np.random.randn() / 10 for noise in range(num_points)]
    
    x = torch.Tensor(x)
    y = torch.Tensor(y)

    x = x.view(-1, 1)
    y = y.view(-1, 1)

    model = training_loop(n_epochs, learning_rate, x, y)

    y_pred = model(x)

    draw(x, y, y_pred)

    '''
        畫出不同訓練次數的圖片
    '''
    epochs = ["true", 1000, 2000, 5000, 10000]
    all_y_pred = [y]
    for epoch in epochs[1:]:
        model = training_loop(epoch, learning_rate, x, y)
        y_pred = model(x)
        all_y_pred.append(y_pred)
    draw_mutiple(epochs, x, all_y_pred)
