import numpy as np
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tslearn.shapelets import LearningShapelets
from tslearn.utils import ts_size
import plotly.graph_objects as go

for source_user in ['S1', 'S2', 'S3']:
    for target_user in ['S1', 'S2', 'S3']:
        source_user = str(source_user)
        target_user = str(target_user)
        if source_user == target_user:
            continue
        Sampling_frequency = 30  # HZ
        Num_Seconds = 128
        Window_Overlap_Rate = 0.9
        DATASET_NAME = 'OPPT'
        activities_required = ['Stand', 'Walk', 'Sit', 'Lie']
        # ['lying', 'walking', 'ascending_stairs', 'descending_stairs']
        # /////////////////
        with open(DATASET_NAME + '_all_' + str(source_user) + '_' + str(Num_Seconds) + '_' + str(
                Window_Overlap_Rate) + '_X_raws.npy', 'rb') as f:
            all_source_bags = np.load(f, allow_pickle=True)
        with open(DATASET_NAME + '_all_' + str(target_user) + '_' + str(Num_Seconds) + '_' + str(
                Window_Overlap_Rate) + '_X_raws.npy', 'rb') as f:
            all_target_bags = np.load(f, allow_pickle=True)
        with open(DATASET_NAME + '_all_' + str(source_user) + '_Y_raws_labels.npy', 'rb') as f:
            all_source_labels = np.load(f)
        with open(DATASET_NAME + '_all_' + str(target_user) + '_Y_raws_labels.npy', 'rb') as f:
            all_target_labels = np.load(f)


        ax = all_source_bags[:, 0]
        ay = all_source_bags[:, 1]
        az = all_source_bags[:, 2]
        gx = all_source_bags[:, 3]
        gy = all_source_bags[:, 4]
        gz = all_source_bags[:, 5]
        t = np.arange(len(ax))
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=t, y=ax, mode='lines+markers', name='aX-axis'))
        fig.add_trace(go.Scatter(x=t, y=ay, mode='lines+markers', name='aY-axis'))
        fig.add_trace(go.Scatter(x=t, y=az, mode='lines+markers', name='aZ-axis'))
        fig.add_trace(go.Scatter(x=t, y=gx, mode='lines+markers', name='gX-axis'))
        fig.add_trace(go.Scatter(x=t, y=gy, mode='lines+markers', name='gY-axis'))
        fig.add_trace(go.Scatter(x=t, y=gz, mode='lines+markers', name='gZ-axis'))
        fig.update_layout(title=str(source_user) + '_', xaxis_title='Time', yaxis_title='Sensor value')
        fig.show()
        print()