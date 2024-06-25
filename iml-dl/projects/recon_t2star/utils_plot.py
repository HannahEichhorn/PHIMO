import itertools
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress, wilcoxon
from statsmodels.stats.multitest import multipletests
import matplotlib.patches as mpatches


def sort_key(key):
    """Define a custom sorting order for a given input key."""

    if key == 'uncorr':
        return (0, 0, 0, 0)
    elif key.startswith('moco_mean'):
        return (1, 0, int(key.split('-')[-1]), 0)
    elif key.startswith('moco_weighted'):
        return (2, 0, int(key.split('-')[-1]), 0)
    elif key.startswith('moco_best'):
        return (3, 0, int(key.split('-')[-1]), 0)
    elif key == 'rdcorr':
        return (4, 0, 0, 0)
    else:
        # Sort any other keys in ascending order
        return (5, key)


def rename_labels(labels):
    """Rename labels based on a predefined mapping."""

    dict_names = {"uncorr": "No\nMoCo",
                  "rdcorr": "HR/QR-\nMoCo",
                  "moco_mean-1000": "Mean of\n1000",
                  "moco_best-1000": "PHIMO",
                  "moco_bestmasks-1000": "PHIMO\nmasks",
                  "motion-free": "Motion-\nfree"}
    return [dict_names[s] for s in labels]


def statistical_testing(keys, metric):
    """
    Perform statistical testing using Wilcoxon signed rank tests and multiple
    comparison correction (FDR) on metric values for pairs of keys.

    Parameters
    ----------
    keys : list
        Sorted keys of the metrics dictionary.
    metric : dict
        Dictionary containing metric values for each key.

    Returns
    -------
    tuple
        A tuple containing combinations and p-values for statistical testing.
    """

    combinations = list(itertools.combinations(np.arange(0, len(keys)),
                                               2))
    p_values = []
    for comb in combinations:
        p_values.append(wilcoxon(metric[keys[comb[0]]],
                                 metric[keys[comb[1]]],
                                 alternative='two-sided')[1])
    rej, p_values_cor, _, __ = multipletests(p_values, alpha=0.05,
                                             method='fdr_bh', is_sorted=False,
                                             returnsorted=False)

    insign = np.where(p_values_cor >= 0.05)
    print("###################")
    for ins in insign[0]:
        print(keys[combinations[ins][0]],
              keys[combinations[ins][1]],
              " No significant difference.")

    return combinations, p_values


def make_violin_plots(motion_types, keys, masks, metric_dict, p_vals, combs,
                      bar_values, ylim, ylabel, bar_top=None,
                      leg_loc=False, show_title=False, save=False):
    """
    Generate violin plots for given different motion types, masks, and keys.

    Parameters
    ----------
    motion_types : list
        Motion types for plotting.
    keys : list
        Keys used for plotting.
    masks : list
        Masks used for plotting.
    metric_dict : dict
        Dictionary containing metric values for each motion type, mask,
        and key.
    p_vals : dict
        Dictionary containing p-values for statistical testing.
    combs : dict
        Dictionary containing combinations for statistical testing.
    bar_values : list
        Bar values for showing significance bars.
    ylim : tuple
        y-axis limits for the plots.
    ylabel : str
        Label for the y-axis.
    bar_top : list, optional
        List indicating whether to show bars on the top of the plots. The
        default is None, corresponding to showing the bars on the top.
    leg_loc : bool or str, optional
        Location of the legend on the plots.
    show_title : bool, optional
        Whether to show titles for each motion type.
    save : str or False, optional
        If provided, the figure will be saved with the specified filename.
    """

    if bar_top is None:
        bar_top = [True, True]
    for motion_type in motion_types:
        fig, ax = plt.subplots(figsize=(6 / 5 * 4.2, 4))
        labels = []
        for mask_name, col in zip(masks, ["tab:gray", "tab:blue"]):
            positions = np.arange(0, len(keys)).astype(float)
            if mask_name == "White matter":
                positions += 0.25
            vp = ax.violinplot(
                [metric_dict[motion_type][mask_name][s] for s in keys],
                positions=positions, showmeans=True, showextrema=False,
                widths=0.2
            )
            for v in vp['bodies']:
                v.set_facecolor(col)
                v.set_edgecolor(col)
            vp['cmeans'].set_edgecolor(col)
            add_label(vp, mask_name, labels=labels)
            if mask_name == "Gray matter":
                show_bars(np.array(p_vals[motion_type][mask_name])[::-1],
                          combs[motion_type][mask_name][::-1],
                          positions,
                          [bar_values[0] for s in keys],
                          flexible_dh=True,
                          col='gray',
                          top=bar_top[0])
            if mask_name == "White matter":
                show_bars(np.array(p_vals[motion_type][mask_name]),
                          combs[motion_type][mask_name],
                          positions,
                          [bar_values[1] for s in keys],
                          flexible_dh=True,
                          col='gray',
                          top=bar_top[1])
        plt.ylabel(ylabel, fontsize=15)
        if leg_loc:
            plt.legend(*zip(*labels), loc=leg_loc, fontsize=13)
        if show_title:
            plt.title(motion_type)
        plt.xticks(positions - 0.125,
                   rename_labels(keys),
                   fontsize=14)
        plt.ylim(ylim[0], ylim[1])
        for y_tick in ax.get_yticks():
            ax.axhline(y_tick, color='lightgray', linestyle='--',
                       linewidth=0.5)
        plt.tight_layout()
        if save:
            plt.savefig(save, dpi=300)
        plt.show()


def add_label(violin, label, labels):
    """Function for adding labels to violin plots"""

    if label == "White matter":
        label = "WM"
    if label == "Gray matter":
        label = "GM"

    color = violin["bodies"][0].get_facecolor().flatten()
    labels.append((mpatches.Patch(color=color), label))
    return labels


def barplot_annotate_brackets(num1, num2, data, center, height, yerr=None,
                              dh=.05, barh=.03, fs=None, maxasterix=None,
                              col='dimgrey', above_bars=True):
    """
    Annotate bar plot with p-values.

    Parameters
    ----------
    num1 : int
        Number of the left bar to put the bracket over.
    num2 : int
        Number of the right bar to put the bracket over.
    data : str or number
        String to write or number for generating asterisks.
    center : list
        Centers of all bars (like plt.bar() input).
    height : list
        Heights of all bars (like plt.bar() input).
    yerr : list
        Y-errors of all bars (like plt.bar() input).
    dh : float
        Height offset over bar/bar + yerr in axes coordinates (0 to 1).
    barh : float
        Bar height in axes coordinates (0 to 1).
    fs : int
        Font size.
    maxasterix : int
        Maximum number of asterisks to write (for very small p-values).
    col : str, optional
        Color of the asterisks or text. The default is 'dimgrey'.
    above_bars : bool, optional
        Whether to show the brackets above the bars. The default is True.
    """

    if type(data) is str:
        text = data
    else:
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    if above_bars:
        bary = [y, y + barh, y + barh, y]
        mid = ((lx + rx) / 2, y + barh - 0.2 * barh)
    else:
        bary = [y + barh, y, y, y + barh]
        mid = ((lx + rx) / 2, y - barh - 0.9 * barh)

    plt.plot(barx, bary, c=col)

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    plt.text(*mid, text, **kwargs, c=col)


def show_bars(p_cor, ind, bars, heights, dh=.1,
              flexible_dh=False, col='dimgrey', top=True):
    """
    Function to show brackets with asterisks indicating statistical
    significance

    Parameters
    ----------
    p_cor : array or list
        corrected p-values.
    ind : list of lists
        indices for the comparisons corresponding to the individual p-values.
    bars : list or array
        x position of boxplots.
    heights : list or array
        maximal value visualised in boxplots.
    col : str
        color of the asterixes. Optional, the default is dimgrey.
    top: bool
        Whether to show the brackets above the bars. The default is True.
    """

    bar = p_cor >= 0.05

    all_nrs = []
    fl_dh = .03
    for i in range(0, len(p_cor)):
        nr = ind[i]
        all_nrs.append(nr)
        if flexible_dh:
            dh = fl_dh
        if bar[i]:
            barplot_annotate_brackets(nr[0], nr[1], '', bars, heights,
                                      dh=dh, col=col, barh=0.02,
                                      above_bars=top)
            if top:
                fl_dh += 0.045
            else:
                fl_dh -= 0.045

    return 0
