import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np
import cv2
# for visualization
import plotly
from plotly.subplots import make_subplots
from plotly.offline import init_notebook_mode
import plotly.graph_objs as go
plotly.offline.init_notebook_mode(connected=True)


def plot_ts(ts_set: np.ndarray, plot_title: str = 'Input Time Series Set'):
    """
    Plot the time series set

    Parameters
    ----------
    ts_set: time series set with shape (ts_number, ts_length)
    plot_title: title of plot
    """

    ts_num, m = ts_set.shape

    fig = go.Figure()

    for i in range(ts_num):
        fig.add_trace(go.Scatter(x=np.arange(m), y=ts_set[i], line=dict(width=3), name="Time series " + str(i)))

    fig.update_xaxes(showgrid=False,
                     title='Time',
                     title_font=dict(size=18, color='black'),
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=16, color='black'),
                     linewidth=1,
                     tickwidth=1)
    fig.update_yaxes(showgrid=False,
                     title='Values',
                     title_font=dict(size=18, color='black'),
                     linecolor='#000',
                     ticks="outside",
                     tickfont=dict(size=16, color='black'),
                     zeroline=False,
                     linewidth=1,
                     tickwidth=1)

    fig.update_layout(title={'text': plot_title, 'x': 0.5, 'y':0.9, 'xanchor': 'center', 'yanchor': 'top'},
                      title_font=dict(size=18, color='black'),
                      plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor='rgba(0,0,0,0)',
                      legend=dict(font=dict(size=16, color='black')),
                      width=1000,
                      height=400
                      )

    fig.show(renderer="colab")


def display_image_matplotlib(img, contour, edge_coordinates, center):
    """
    Функция для отображения изображения с контурами и координатами используя matplotlib
    """
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.imshow(img_rgb)
    
    contour_plot = contour.squeeze()
    ax.plot(contour_plot[:, 0], contour_plot[:, 1], 'g-', linewidth=3, label='Contour')
    
    for coord in edge_coordinates:
        ax.plot([center[0], coord[0]], [center[1], coord[1]], 'm-', linewidth=2, alpha=0.7)
    
    ax.plot(center[0], center[1], 'ro', markersize=10, label='Center')
    
    ax.set_title('Image with Contours and Lines', fontsize=16)
    ax.axis('off')
    ax.legend()
    
    plt.tight_layout()
    plt.show()

def plot_ts_matplotlib(ts, title="Time Series"):
    """
    Функция для отображения временного ряда используя matplotlib
    """
    fig, ax = plt.subplots(figsize=(12, 5))
    
    angles = np.arange(0, 360, 360//len(ts))
    ax.plot(angles, ts, 'b-', linewidth=3, marker='o', markersize=4, markerfacecolor='red')
    
    ax.set_xlabel('Angle (degrees)', fontsize=14)
    ax.set_ylabel('Distance', fontsize=14)
    ax.set_title(title, fontsize=16)
    
    ax.grid(False)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1)
    ax.spines['bottom'].set_linewidth(1)
    
    plt.tight_layout()
    plt.show()
