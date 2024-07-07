import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.axes_grid1 import make_axes_locatable

filename='predictions_map'
vp_threshold=0.9901
lp_threshold=0.9666
vn_threshold=0.00031
ln_threshold=0.0072
report_dir='./'
predictions_df = pd.DataFrame(columns=['prediction_value', 'period', 'expected_class'])
predictions_df = pd.concat([predictions_df, pd.DataFrame.from_dict(
    {"prediction_value": [0.0001, 0.001, 0.01, 0.2, 0.7, 0.92], 'period': [1, 4, 6, 7, 110, 20],
     "expected_class": [0, 0, 1, 0, 1, 1]}, orient='columns')], ignore_index=True)
min_fpp = 0.000001
fig, axs = plt.subplots(1, 1, figsize=(16, 16), constrained_layout=True)
#fig.suptitle("Predictions Map", fontsize=25)
fig.supylabel("Prediction", fontsize=25)
divider = make_axes_locatable(axs)
ax_log_top = divider.append_axes("top", size=5, pad=0, sharex=axs)
ax_log = divider.append_axes("bottom", size=5, pad=0, sharex=axs)
periods_0 = predictions_df.loc[predictions_df['expected_class'] == 0, 'period'].to_numpy()
prediction_values_0 = predictions_df.loc[predictions_df['expected_class'] == 0, 'prediction_value'].to_numpy()
periods_1 = predictions_df.loc[predictions_df['expected_class'] == 1, 'period'].to_numpy()
prediction_values_1 = predictions_df.loc[predictions_df['expected_class'] == 1, 'prediction_value'].to_numpy()
prediction_values_0[prediction_values_0 < np.nanmin(prediction_values_0)] = min_fpp
prediction_values_1[prediction_values_1 < np.nanmin(prediction_values_1)] = min_fpp

division = 0.1
top_division = 0.95
up_indexes = np.argwhere((prediction_values_0 > division) & (prediction_values_0 <= top_division)).flatten()
axs.scatter(periods_0[up_indexes], prediction_values_0[up_indexes], color="red", s=20)
up_indexes = np.argwhere((prediction_values_1 > division) & (prediction_values_1 <= top_division)).flatten()
axs.scatter(periods_1[up_indexes], prediction_values_1[up_indexes], color="blue", s=20)
axs.set_yscale("linear")
axs.set_xscale("log")
axs.set_ylim([division, top_division])
axs.tick_params(axis='both', which='major', labelsize=25)
axs.tick_params(axis='both', which='minor', labelsize=25)
axs.spines["bottom"].set_visible(False)  # hide bottom of box
axs.spines["top"].set_visible(False)  # hide top of box
axs.get_xaxis().set_visible(False)  # hide x-axis entirely
axs.axhline(y=division, color="black", linestyle="dashed", linewidth=1)
axs.axhline(y=top_division, color="black", linestyle="dashed", linewidth=1)

# Acting on the log axis
ax_log_top.set_xscale("log")
ax_log_top.set_yscale("log")
ax_log_top.set_ylim([top_division, 1.0])
down_indexes = np.argwhere(prediction_values_0 > top_division).flatten()
ax_log_top.scatter(periods_0[down_indexes], prediction_values_0[down_indexes], color="red", s=20)
down_indexes = np.argwhere(prediction_values_1 > top_division).flatten()
ax_log_top.scatter(periods_1[down_indexes], prediction_values_1[down_indexes], color="blue", s=20)
ax_log_top.tick_params(axis='both', which='major', labelsize=25)
ax_log_top.tick_params(axis='both', which='minor', labelsize=25)
ax_log_top.axhspan(lp_threshold, 1, color="cyan", label="LP", alpha=0.3)
ax_log_top.axhspan(vp_threshold, 1, color="lightgreen", label="VP", alpha=0.3)
ax_log_top.spines["bottom"].set_visible(False)  # hide bottom of box
ax_log_top.get_xaxis().set_visible(False)

# Acting on the log axis
ax_log.set_ylim([min_fpp, division])
ax_log.set_xscale("log")
ax_log.set_yscale("log")
down_indexes = np.argwhere(prediction_values_0 <= division).flatten()
ax_log.scatter(periods_0[down_indexes], prediction_values_0[down_indexes], color="red", s=20)
down_indexes = np.argwhere(prediction_values_1 <= division).flatten()
ax_log.scatter(periods_1[down_indexes], prediction_values_1[down_indexes], color="blue", s=20)
ax_log.plot(1, min_fpp, markersize=0)
ax_log.axhspan(0, ln_threshold, color="gray", label="VN", alpha=0.3)
ax_log.axhspan(0, vn_threshold, color="lightcoral", label="LN", alpha=0.3)
ax_log.spines["top"].set_visible(False)  # hide top of box
ax_log.tick_params(axis='both', which='major', labelsize=25)
ax_log.tick_params(axis='both', which='minor', labelsize=25)
ax_log.set_xlabel('Period (d)', fontsize=25)
fig.tight_layout()
fig.savefig(report_dir + f"/{filename}.png")
fig.clf()