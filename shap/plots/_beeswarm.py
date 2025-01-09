"""Summary plots of SHAP values across a whole dataset."""

from __future__ import annotations

import warnings
from typing import Literal

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import scipy.cluster
import scipy.sparse
import scipy.spatial
from scipy.stats import gaussian_kde

from .. import Explanation
from ..utils import safe_isinstance
from ..utils._exceptions import DimensionError
from . import colors
from ._labels import labels
from ._utils import (
    convert_color,
    convert_ordering,
    get_sort_order,
    merge_nodes,
    sort_inds,
)


# TODO: Add support for hclustering based explanations where we sort the leaf order by magnitude and then show the dendrogram to the left
def beeswarm(
    shap_values: Explanation, # SHAP値と関連情報をもつExplanationオブジェクト
    max_display: int | None = 10, # 表示する特徴量
    order=Explanation.abs.mean(0),  # type: ignore
    clustering=None, 
    cluster_threshold=0.5, # クラスタリングの閾値
    color=None, # プロットのカラーマップ or 単一色
    axis_color="#333333", # 軸の色
    alpha: float = 1.0, # 透明度
    ax: pl.Axes | None = None, # 描画するAxes
    show: bool = True, # 描画するかどうか
    log_scale: bool = False, # 対数スケール
    color_bar: bool = True, # カラーバー
    s: float = 16, # マーカーサイズ
    plot_size: Literal["auto"] | float | tuple[float, float] | None = "auto",
    color_bar_label: str = labels["FEATURE_VALUE"],
    group_remaining_features: bool = True,
):
    """Create a SHAP beeswarm plot, colored by feature values when they are provided.

    Parameters
    ----------
    shap_values : Explanation
        This is an :class:`.Explanation` object containing a matrix of SHAP values
        (# samples x # features).

    max_display : int
        How many top features to include in the plot (default is 10, or 7 for
        interaction plots).

    ax: matplotlib Axes
        Axes object to draw the plot onto, otherwise uses the current Axes.

    show : bool
        Whether :external+mpl:func:`matplotlib.pyplot.show()` is called before returning.
        Setting this to ``False`` allows the plot to be customized further
        after it has been created, returning the current axis via
        :external+mpl:func:`matplotlib.pyplot.gca()`.

    color_bar : bool
        Whether to draw the color bar (legend).

    s : float
        What size to make the markers. For further information, see ``s`` in
        :external+mpl:func:`matplotlib.pyplot.scatter`.

    plot_size : "auto" (default), float, (float, float), or None
        What size to make the plot. By default, the size is auto-scaled based on the
        number of features that are being displayed. Passing a single float will cause
        each row to be that many inches high. Passing a pair of floats will scale the
        plot by that number of inches. If ``None`` is passed, then the size of the
        current figure will be left unchanged. If ``ax`` is not ``None``, then passing
        ``plot_size`` will raise a :exc:`ValueError`.

    group_remaining_features: bool
        If there are more features than ``max_display``, then plot a row representing
        the sum of SHAP values of all remaining features. Default True.

    Returns
    -------
    ax: matplotlib Axes
        Returns the :external+mpl:class:`~matplotlib.axes.Axes` object with the plot drawn onto it. Only
        returned if ``show=False``.

    Examples
    --------
    See `beeswarm plot examples <https://shap.readthedocs.io/en/latest/example_notebooks/api_examples/plots/beeswarm.html>`_.

    """
    
    # --- 1) 引数が Explanation型かどうか確認 ---
    if not isinstance(shap_values, Explanation):
        emsg = "The beeswarm plot requires an `Explanation` object as the `shap_values` argument."
        raise TypeError(emsg)

    sv_shape = shap_values.shape
    if len(sv_shape) == 1:
        emsg = (
            "The beeswarm plot does not support plotting a single instance, please pass "
            "an explanation matrix with many instances!"
        )
        raise ValueError(emsg)
    elif len(sv_shape) > 2:
        emsg = (
            "The beeswarm plot does not support plotting explanations with instances that have more "
            "than one dimension!"
        )
        raise ValueError(emsg)
    if ax and plot_size:
        emsg = (
            "The beeswarm plot does not support passing an axis and adjusting the plot size. "
            "To adjust the size of the plot, set plot_size to None and adjust the size on the original figure the axes was part of"
        )
        raise ValueError(emsg)

    shap_exp = shap_values
    
    # --- 2) SHAP値と元データを取得 ---
    values = np.copy(shap_exp.values)
    features = shap_exp.data
    feature_names = shap_exp.feature_names
    
    # --- 3) データが疎行列の場合は密行列へ変換 ---
    if scipy.sparse.issparse(features):
        features = features.toarray()
    # if out_names is None: # TODO: waiting for slicer support
    #     out_names = shap_exp.output_names
    order = convert_ordering(order, values)

    # multi_class = False
    # if isinstance(values, list):
    #     multi_class = True
    #     if plot_type is None:
    #         plot_type = "bar" # default for multi-output explanations
    #     assert plot_type == "bar", "Only plot_type = 'bar' is supported for multi-output explanations!"
    # else:
    #     if plot_type is None:
    #         plot_type = "dot" # default for single output explanations
    #     assert len(values.shape) != 1, "Summary plots need a matrix of values, not a vector."

    # color の指定がない場合は、features（特徴量データ）の有無によってデフォルトのカラーマップを選択
    if color is None:
        # 特徴量データがあるなら赤青グラデーションを、
        if features is not None:
            color = colors.red_blue
        else:
            # 特徴量データがない場合は、単色(青系)を採用
            color = colors.blue_rgb

    # color が文字列やタプルなどの場合、それを matplotlib で扱える形式に変換
    color = convert_color(color)

    # idx2cat は 「各特徴量がカテゴリ型かどうか」を格納するためのフラグ一覧 (デフォルトは None)
    idx2cat = None

    # --- 特徴量データ (features) の型に応じて前処理を行う ---
    # pandas.DataFrame の場合は、列名をそのまま feature_names に使い、さらに .values で ndarray化
    if isinstance(features, pd.DataFrame):
        # feature_names が指定されていないなら、DataFrame のカラム名を feature_names に使う
        if feature_names is None:
            feature_names = features.columns
        
        # カテゴリ型や object 型の列がどれかを判定し、 True/False のリストを作成
        idx2cat = features.dtypes.astype(str).isin(["object", "category"]).tolist()
        # DataFrame を numpy配列に変換
        features = features.values

    # features が単なるリストの場合（例: python の list で各列名を渡したなど）
    elif isinstance(features, list):
        # feature_names が指定されていないなら、そのまま features を名前として使う
        if feature_names is None:
            feature_names = features
        # この場合、特徴量の実データはないので None にしておく
        features = None

    # features が 1次元の配列 (shape が (n,) のような構造) で、かつ feature_names が未指定なら
    # その配列を "特徴量名" として解釈し、features は None にして実データは存在しない扱いにする
    elif (features is not None) and len(features.shape) == 1 and feature_names is None:
        feature_names = features
        features = None

    # shap_values.values の列数（特徴量数）を取得
    num_features = values.shape[1]

    # --- features が実際に存在する場合（None でない場合） ---
    if features is not None:
        # エラーメッセージのひな型（特徴量数が合わない場合に使う）
        shape_msg = "The shape of the shap_values matrix does not match the shape of the provided data matrix."

        # SHAP値の列数より features の列数が1多い場合、最後の列が「定数項(オフセット)」である可能性を指摘しつつエラーにする
        if num_features - 1 == features.shape[1]:
            shape_msg += (
                " Perhaps the extra column in the shap_values matrix is the "
                "constant offset? If so, just pass shap_values[:,:-1]."
            )
            raise DimensionError(shape_msg)
        
        # それ以外で純粋に列数が一致しないなら、DimensionError を投げる
        if num_features != features.shape[1]:
            raise DimensionError(shape_msg)

    # feature_names が指定されていなければ、デフォルトとして "Feature 0", "Feature 1", ... という形で作成
    if feature_names is None:
        feature_names = np.array([labels["FEATURE"] % str(i) for i in range(num_features)])

    # --- matplotlib の Axes (描画先) が未指定なら現在の Axes を取得 ---
    if ax is None:
        ax = pl.gca()

    # Axes が属する Figure を取得（必ずあるはずなので None の場合はエラー扱い）
    fig = ax.get_figure()
    assert fig is not None  # mypyなどの型ヒントのための安全策

    # --- log_scale = True の場合、x軸を対数スケール("symlog")にする ---
    if log_scale:
        ax.set_xscale("symlog")

    # --- clustering 引数の解釈 ---
    # clustering が None の場合は、shap_values 内にある "clustering" 属性を探して使う (あるなら)
    if clustering is None:
        partition_tree = getattr(shap_values, "clustering", None)
        # partition_tree が存在し、その分散がすべて 0 なら、tree構造が複数含まれているとみなし[0]だけを使う
        if partition_tree is not None and partition_tree.var(0).sum() == 0:
            partition_tree = partition_tree[0]
        else:
            partition_tree = None
    elif clustering is False:
        # clustering=False なら、クラスタリング情報を無視して None に
        partition_tree = None
    else:
        # それ以外(明示的にクラスタリング情報が渡された場合)は、そのまま使う
        partition_tree = clustering

    # partition_tree (クラスタリング情報) が有効な場合、4列(階層クラスタリングの形式)でなければエラー
    if partition_tree is not None:
        if partition_tree.shape[1] != 4:
            emsg = (
                "The clustering provided by the Explanation object does not seem to "
                "be a partition tree (which is all shap.plots.bar supports)!"
            )
            raise ValueError(emsg)

    # FIXME: introduce beeswarm interaction values as a separate function `beeswarm_interaction()` (?)
    #   In the meantime, users can use the `shap.summary_plot()` function.
    #
    # # plotting SHAP interaction values
    # if len(values.shape) == 3:
    #
    #     if plot_type == "compact_dot":
    #         new_values = values.reshape(values.shape[0], -1)
    #         new_features = np.tile(features, (1, 1, features.shape[1])).reshape(features.shape[0], -1)
    #
    #         new_feature_names = []
    #         for c1 in feature_names:
    #             for c2 in feature_names:
    #                 if c1 == c2:
    #                     new_feature_names.append(c1)
    #                 else:
    #                     new_feature_names.append(c1 + "* - " + c2)
    #
    #         return beeswarm(
    #             new_values, new_features, new_feature_names,
    #             max_display=max_display, plot_type="dot", color=color, axis_color=axis_color,
    #             title=title, alpha=alpha, show=show, sort=sort,
    #             color_bar=color_bar, plot_size=plot_size, class_names=class_names,
    #             color_bar_label="*" + color_bar_label
    #         )
    #
    #     if max_display is None:
    #         max_display = 7
    #     else:
    #         max_display = min(len(feature_names), max_display)
    #
    #     interaction_sort_inds = order#np.argsort(-np.abs(values.sum(1)).sum(0))
    #
    #     # get plotting limits
    #     delta = 1.0 / (values.shape[1] ** 2)
    #     slow = np.nanpercentile(values, delta)
    #     shigh = np.nanpercentile(values, 100 - delta)
    #     v = max(abs(slow), abs(shigh))
    #     slow = -v
    #     shigh = v
    #
    #     pl.figure(figsize=(1.5 * max_display + 1, 0.8 * max_display + 1))
    #     pl.subplot(1, max_display, 1)
    #     proj_values = values[:, interaction_sort_inds[0], interaction_sort_inds]
    #     proj_values[:, 1:] *= 2  # because off diag effects are split in half
    #     beeswarm(
    #         proj_values, features[:, interaction_sort_inds] if features is not None else None,
    #         feature_names=feature_names[interaction_sort_inds],
    #         sort=False, show=False, color_bar=False,
    #         plot_size=None,
    #         max_display=max_display
    #     )
    #     pl.xlim((slow, shigh))
    #     pl.xlabel("")
    #     title_length_limit = 11
    #     pl.title(shorten_text(feature_names[interaction_sort_inds[0]], title_length_limit))
    #     for i in range(1, min(len(interaction_sort_inds), max_display)):
    #         ind = interaction_sort_inds[i]
    #         pl.subplot(1, max_display, i + 1)
    #         proj_values = values[:, ind, interaction_sort_inds]
    #         proj_values *= 2
    #         proj_values[:, i] /= 2  # because only off diag effects are split in half
    #         summary(
    #             proj_values, features[:, interaction_sort_inds] if features is not None else None,
    #             sort=False,
    #             feature_names=["" for i in range(len(feature_names))],
    #             show=False,
    #             color_bar=False,
    #             plot_size=None,
    #             max_display=max_display
    #         )
    #         pl.xlim((slow, shigh))
    #         pl.xlabel("")
    #         if i == min(len(interaction_sort_inds), max_display) // 2:
    #             pl.xlabel(labels['INTERACTION_VALUE'])
    #         pl.title(shorten_text(feature_names[ind], title_length_limit))
    #     pl.tight_layout(pad=0, w_pad=0, h_pad=0.0)
    #     pl.subplots_adjust(hspace=0, wspace=0.1)
    #     if show:
    #         pl.show()
    #     return

    # -----------------------------------------------------------
    # 1) プロットで表示する特徴量数の決定
    # -----------------------------------------------------------

    # max_display が None の場合は、すべての特徴量を表示
    if max_display is None:
        max_display = len(feature_names)

    # 実際に使用する特徴量数を、feature_names 全体数と max_display の小さい方に制限
    num_features = min(max_display, len(feature_names))


    # -----------------------------------------------------------
    # 2) クラスタリング情報を踏まえた「特徴量のマージ」処理
    #    -> 表示する特徴量数を超える分について、ツリー構造を崩さずにまとめる
    # -----------------------------------------------------------

    # orig_inds: 各特徴量（およびマージ後のグループ）が、元々どの特徴量インデックスを含んでいるかを記録するリスト
    # 例: [[0],[1],[2],...,[n-1]] で開始し、マージすると [[0,2],[1],...] のように要素数が増加していく
    orig_inds = [[i] for i in range(len(feature_names))]

    # 後で一部の演算で使うため、values のコピーを残しておく
    orig_values = values.copy()

    # ループ中でクラスタのマージを繰り返す可能性があるので while True で囲む
    while True:
        # feature_order: order 引数に基づいて SHAP値を並べ替えたインデックスを取得
        # convert_ordering(...) は SHAP内部の関数で、たとえば「abs.mean(0)」等に基づくソートを実行
        feature_order = convert_ordering(order, Explanation(np.abs(values)))

        if partition_tree is not None:
            # クラスタリング情報がある場合、まずはこの木構造から導出される順序 clust_order を得る
            clust_order = sort_inds(partition_tree, np.abs(values))

            # get_sort_order で、クラスタリング構造を尊重しながら feature_order を微調整
            # cluster_threshold より上位の結合は崩さないなどの調整が行われる
            dist = scipy.spatial.distance.squareform(scipy.cluster.hierarchy.cophenet(partition_tree))
            feature_order = get_sort_order(dist, clust_order, cluster_threshold, feature_order)

            # 表示したい max_display よりも特徴量が多く、かつ
            # 「max_display番目の特徴量とそれに隣接する特徴量の距離」が threshold 以下なら、
            #   => クラスタリングを壊さずにまとめるためにマージを実行
            if (
                max_display < len(feature_order)
                and dist[feature_order[max_display - 1], feature_order[max_display - 2]] <= cluster_threshold
            ):
                # merge_nodes(...) は特定のクラスタを一つにまとめる処理
                # 戻り値は updated_partition_tree, ind1, ind2 で、ind1 に ind2 をまとめる
                partition_tree, ind1, ind2 = merge_nodes(np.abs(values), partition_tree)

                # values配列を列方向にマージ: ind2 を ind1 に加算し、ind2 列は削除
                for _ in range(len(values)):
                    values[:, ind1] += values[:, ind2]
                    values = np.delete(values, ind2, 1)

                # orig_inds の ind1 と ind2 を統合し、ind2 は削除
                orig_inds[ind1] += orig_inds[ind2]
                del orig_inds[ind2]
            else:
                # これ以上マージの必要がない(しきい値を超えていない)ので終了
                break
        else:
            # partition_tree が無い場合はマージせず、そのままブレイク
            break


    # -----------------------------------------------------------
    # 3) マージ結果を踏まえて、新しい feature_names を再構築
    # -----------------------------------------------------------

    # feature_order の上位 max_display 個を実際に表示する特徴量とする
    feature_inds = feature_order[:max_display]

    # マージ後の「特徴量グループ」を考慮した新しい名前リストを構築
    feature_names_new = []
    for inds in orig_inds:
        # inds 内に 1つしか特徴量がない場合は、そのまま元の名前を採用
        if len(inds) == 1:
            feature_names_new.append(feature_names[inds[0]])
        # 2つなら "X + Y" のように繋ぐ
        elif len(inds) <= 2:
            feature_names_new.append(" + ".join([feature_names[i] for i in inds]))
        else:
            # 3つ以上の場合は、その中で平均絶対SHAP値が最も大きい要素を代表として
            # "代表名 + n other features" の形式にする
            max_ind = np.argmax(np.abs(orig_values).mean(0)[inds])
            feature_names_new.append(feature_names[inds[max_ind]] + " + %d other features" % (len(inds) - 1))

    # feature_names を、上記で再構築したリストに置き換え
    feature_names = feature_names_new


    # -----------------------------------------------------------
    # 4) "group_remaining_features" が有効なら、残りをまとめて "Sum of ... other features" として扱う
    # -----------------------------------------------------------

    # もし (表示可能な特徴量 < 実際の特徴量) かつ group_remaining_features=True なら、
    # 末尾(=最も重要度の低い)の特徴量を一つにまとめる
    include_grouped_remaining = num_features < len(values[0]) and group_remaining_features
    if include_grouped_remaining:
        # (num_features - 1) から後ろの全特徴量を合算し、まとめる数を num_cut としてカウント
        num_cut = np.sum([len(orig_inds[feature_order[i]]) for i in range(num_features - 1, len(values[0]))])

        # 対象となる列（feature_order[num_features - 1]）に、残り全ての列を足し合わせる
        values[:, feature_order[num_features - 1]] = np.sum(
            [values[:, feature_order[i]] for i in range(num_features - 1, len(values[0]))], 0
        )

    # yticklabels として表示する文字列を feature_names から取り出し、
    # 順序は feature_inds に準拠
    yticklabels = [feature_names[i] for i in feature_inds]

    # グループ化が行われた場合は、その最後の行を "Sum of ... other features" に置き換え
    if include_grouped_remaining:
        yticklabels[-1] = "Sum of %d other features" % num_cut


    # -----------------------------------------------------------
    # 5) プロットサイズの設定: 特徴量数に合わせた行数分の高さを確保する
    # -----------------------------------------------------------

    row_height = 0.4
    if plot_size == "auto":
        fig.set_size_inches(8, min(len(feature_order), max_display) * row_height + 1.5)
    elif isinstance(plot_size, (list, tuple)):
        fig.set_size_inches(plot_size[0], plot_size[1])
    elif plot_size is not None:
        fig.set_size_inches(8, min(len(feature_order), max_display) * plot_size + 1.5)

    # x=0 に縦線を引いて、正のSHAPと負のSHAPを視覚的に区切る
    ax.axvline(x=0, color="#999999", zorder=-1)


    # -----------------------------------------------------------
    # 6) 実際の beeswarm プロット（1つ1つの点）の描画
    #    -> ここで SHAP値を使い、y座標をわずかにズラして点が重ならないようにする
    # -----------------------------------------------------------

    for pos, i in enumerate(reversed(feature_inds)):
        # 各行（pos）に対して水平線を引き、視覚的に区分け
        ax.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)

        # 今回プロットする SHAP値 (全サンプル)
        shaps = values[:, i]

        # 対応する元の特徴量値 (色付け用) を取得
        fvalues = None if features is None else features[:, i]

        # サンプルインデックスをランダムにシャッフルし、散布のランダム性を高める
        f_inds = np.arange(len(shaps))
        np.random.shuffle(f_inds)
        if fvalues is not None:
            fvalues = fvalues[f_inds]
        shaps = shaps[f_inds]

        # カテゴリかどうかの判定 (idx2catが Trueなら数値色付けは行わない)
        colored_feature = True
        try:
            if idx2cat is not None and idx2cat[i]:
                colored_feature = False
            else:
                # 数値変換を試してエラーが出なければ数値扱い
                fvalues = np.array(fvalues, dtype=np.float64)
        except Exception:
            colored_feature = False

        # サンプル数
        N = len(shaps)

        # bins(=100) の単位で、SHAP値を区切って「同じビンの点は上下にずらして描画」する
        nbins = 100
        quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8))
        inds_ = np.argsort(quant + np.random.randn(N) * 1e-6)
        layer = 0
        last_bin = -1
        ys = np.zeros(N)

        # 同じビンが続くかぎり layer を増やして y座標をずらすことで散布
        for ind in inds_:
            if quant[ind] != last_bin:
                layer = 0
            ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
            layer += 1
            last_bin = quant[ind]

        # row_height を使ってスケーリング (max(ys+1) で正規化し、点が重なり過ぎないようにする)
        ys *= 0.9 * (row_height / np.max(ys + 1))

        # -------------------------------------------------------
        # 6-A) Colormap を使う場合の描画
        # -------------------------------------------------------
        if safe_isinstance(color, "matplotlib.colors.Colormap") and fvalues is not None and colored_feature is True:
            # 5% と 95% タイルをもとに、カラー範囲 (vmin, vmax) を決定
            vmin = np.nanpercentile(fvalues, 5)
            vmax = np.nanpercentile(fvalues, 95)
            if vmin == vmax:
                vmin = np.nanpercentile(fvalues, 1)
                vmax = np.nanpercentile(fvalues, 99)
                if vmin == vmax:
                    vmin = np.min(fvalues)
                    vmax = np.max(fvalues)
            if vmin > vmax:
                vmin = vmax  # 数値誤差などで発生した場合に合わせる

            # SHAP配列と features の行数が一致しない場合はエラー
            if features is not None and features.shape[0] != len(shaps):
                raise DimensionError("Feature and SHAP matrices must have the same number of rows!")

            # 欠損値を持つサンプルはグレーでプロット
            nan_mask = np.isnan(fvalues)
            ax.scatter(
                shaps[nan_mask],
                pos + ys[nan_mask],
                color="#777777",
                s=s,
                alpha=alpha,
                linewidth=0,
                zorder=3,
                rasterized=len(shaps) > 500,
            )

            # 欠損でないサンプルは colormap に応じた色付け
            cvals = fvalues[~nan_mask].astype(np.float64)
            cvals_imp = cvals.copy()
            cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
            cvals[cvals_imp > vmax] = vmax
            cvals[cvals_imp < vmin] = vmin
            ax.scatter(
                shaps[~nan_mask],
                pos + ys[~nan_mask],
                cmap=color,         # colormap
                vmin=vmin, vmax=vmax,
                s=s, c=cvals,
                alpha=alpha,
                linewidth=0,
                zorder=3,
                rasterized=len(shaps) > 500,
            )

        # -------------------------------------------------------
        # 6-B) 単色で描画する場合
        # -------------------------------------------------------
        else:
            # color が Colormap であっても、ここで color.colors を単に使う実装
            if safe_isinstance(color, "matplotlib.colors.Colormap"):
                color = color.colors

            ax.scatter(
                shaps,
                pos + ys,
                s=s,
                alpha=alpha,
                linewidth=0,
                zorder=3,
                color=color if colored_feature else "#777777",
                rasterized=len(shaps) > 500,
            )


    # -----------------------------------------------------------
    # 7) カラーバーを描画 (colormap + features が有効な場合)
    # -----------------------------------------------------------

    if safe_isinstance(color, "matplotlib.colors.Colormap") and color_bar and features is not None:
        import matplotlib.cm as cm

        m = cm.ScalarMappable(cmap=color)
        m.set_array([0, 1])
        cb = fig.colorbar(m, ax=ax, ticks=[0, 1], aspect=80)
        cb.set_ticklabels([labels["FEATURE_VALUE_LOW"], labels["FEATURE_VALUE_HIGH"]])
        cb.set_label(color_bar_label, size=12, labelpad=0)
        cb.ax.tick_params(labelsize=11, length=0)
        cb.set_alpha(1)
        cb.outline.set_visible(False)  # type: ignore


    # -----------------------------------------------------------
    # 8) 軸やスパイン(枠線)の設定
    # -----------------------------------------------------------

    ax.xaxis.set_ticks_position("bottom")
    ax.yaxis.set_ticks_position("none")
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(color=axis_color, labelcolor=axis_color)

    # y軸ラベルを先ほど作成した yticklabels でセット
    ax.set_yticks(range(len(feature_inds)), list(reversed(yticklabels)), fontsize=13)
    ax.tick_params("y", length=20, width=0.5, which="major")
    ax.tick_params("x", labelsize=11)
    ax.set_ylim(-1, len(feature_inds))

    # X軸ラベルのテキストを設定
    ax.set_xlabel(labels["VALUE"], fontsize=13)

    # -----------------------------------------------------------
    # 9) show が True なら plot を描画して終了、False なら Axes を返す
    # -----------------------------------------------------------
    if show:
        pl.show()
    else:
        return ax

def shorten_text(text, length_limit):
    """
    テキストが length_limit を超える場合に末尾を "..." で省略するユーティリティ関数。
    """
    # text の長さが limit を超えるなら途中で切って "..." を付ける
    if len(text) > length_limit:
        return text[: length_limit - 3] + "..."
    else:
        return text


def is_color_map(color):
    """
    color が matplotlib の Colormap かどうかを判定する関数。
    実装上は safe_isinstance(...) の呼び出しのみ。
    (ただし実際のコードでは結果を返さず終了しているので注意。)
    """
    safe_isinstance(color, "matplotlib.colors.Colormap")


# 以下は旧式の summary プロット関数 (summary_legacy) で、
# SHAP値の可視化方法を一括して扱う大きな実装。
# 将来的には分割や削除が検討されている部分だが、後方互換のために残っている。
def summary_legacy(
    shap_values,
    features=None,
    feature_names=None,
    max_display=None,
    plot_type=None,
    color=None,
    axis_color="#333333",
    title=None,
    alpha=1,
    show=True,
    sort=True,
    color_bar=True,
    plot_size="auto",
    layered_violin_max_num_bins=20,
    class_names=None,
    class_inds=None,
    color_bar_label=labels["FEATURE_VALUE"],
    cmap=colors.red_blue,
    show_values_in_legend=False,
    use_log_scale=False,
):
    """
    古い形式のSHAPサマリープロット関数。ドット・バー・バイオリンなど複数のスタイルをまとめて扱う。
    多クラスや相互作用 (3次元) などのSHAP値にも対応しており、コードが大きく複雑。

    主な引数:
    -----------
    shap_values : numpy.array もしくは list
        - シングル出力のときは (#samples x #features) の2次元配列
        - マルチ出力のときは、リストに分割された複数のSHAP値配列 or 3次元配列

    features : array, DataFrame, list
        - 特徴量データ。または feature_names の簡易指定としてリストを渡す場合も。

    feature_names : list
        - 特徴量の名前(列数と同じ長さ)。未指定なら "Feature 0" など自動生成。

    max_display : int
        - 表示する特徴量の上限数 (デフォルト20)。相互作用プロットなどのときは別の初期値(7)にもなる。

    plot_type : str
        - "dot", "bar", "violin", "compact_dot" など。出力の形式を指定。
        - シングル出力なら "dot"、マルチ出力なら "bar" がデフォルトなど。

    show_values_in_legend : bool
        - multi-output のバー表示時、凡例に SHAP値の平均を表示するかどうか。

    use_log_scale : bool
        - x軸を "symlog" スケールにするかどうか。

    などなど...
    """
    # matplotlib の状態をクリア (figure 内を初期化)
    pl.clf()

    # --- 1) shap_values が Explanation オブジェクトかどうかを判定して対応 ---
    if str(type(shap_values)).endswith("Explanation'>"):
        # shap_values が Explanation型なら、中から実際の values, data, feature_names を取り出す
        shap_exp = shap_values
        shap_values = shap_exp.values
        if features is None:
            features = shap_exp.data
        if feature_names is None:
            feature_names = shap_exp.feature_names

        # マルチ出力 (base_values.shape[1] > 2) の場合、各出力チャンネルごとに配列を分割し、リスト化して扱う
        if len(shap_exp.base_values.shape) == 2 and shap_exp.base_values.shape[1] > 2:
            shap_values = [shap_values[:, :, i] for i in range(shap_exp.base_values.shape[1])]
        # output_names (出力名) は slicing 未対応でTODOコメントが残っている

    # --- 2) multi_class (マルチ出力) かどうかの判定とデフォルト plot_type 設定 ---
    multi_class = False
    if isinstance(shap_values, list):
        # shap_values がリストなら複数クラス or 複数出力と見なす
        multi_class = True
        if plot_type is None:
            # マルチ出力のときのデフォルトは 'bar'
            plot_type = "bar"
        # multi-output で 'bar' 以外が指定された場合はサポート外
        assert plot_type == "bar", "Only plot_type = 'bar' is supported for multi-output explanations!"
    else:
        # シングル出力の場合、デフォルトで 'dot' にする
        if plot_type is None:
            plot_type = "dot"
        # shap_values が1次元(ベクトル)のときはサマリープロットが使えないのでエラー
        assert len(shap_values.shape) != 1, "Summary plots need a matrix of shap_values, not a vector."

        # もし 3次元でかつ最終軸が 2を超える(複数出力)かつ plot_type='bar' なら
        # リストに分解して multi_class として扱う (旧式のマルチ出力サポート)
        if len(shap_values.shape) == 3 and shap_values.shape[2] > 2 and plot_type == "bar":
            shap_values = [shap_values[:, :, i] for i in range(shap_values.shape[2])]
            multi_class = True

    # --- 3) color 引数のデフォルト設定 ---
    if color is None:
        if plot_type == "layered_violin":
            color = "coolwarm"  # layered_violin の場合のデフォルトカラーマップ
        elif multi_class:
            # multi_class の場合は円環状赤青カラーマップを生成する関数を定義
            def color(i):
                return colors.red_blue_circle(i / len(shap_values))
        else:
            # シングル出力の場合のデフォルトは青系 (blue_rgb)
            color = colors.blue_rgb

    # --- 4) features や feature_names の形状チェックと DataFrame → np.array変換 など ---
    idx2cat = None
    if isinstance(features, pd.DataFrame):
        if feature_names is None:
            feature_names = features.columns
        idx2cat = features.dtypes.astype(str).isin(["object", "category"]).tolist()
        features = features.values
    elif isinstance(features, list):
        if feature_names is None:
            feature_names = features
        features = None
    elif (features is not None) and len(features.shape) == 1 and feature_names is None:
        feature_names = features
        features = None

    # シングル出力なら shap_values.shape[1]、マルチ出力なら shap_values[0].shape[1] から特徴量数を取得
    num_features = shap_values[0].shape[1] if multi_class else shap_values.shape[1]

    # shap_values と features の列数(特徴量数)が一致するかチェック
    if features is not None:
        shape_msg = "The shape of the shap_values matrix does not match the shape of the provided data matrix."
        if num_features - 1 == features.shape[1]:
            raise ValueError(
                shape_msg + " Perhaps the extra column in the shap_values matrix is the "
                "constant offset? Of so just pass shap_values[:,:-1]."
            )
        else:
            assert num_features == features.shape[1], shape_msg

    # feature_names が与えられていなければ "Feature 0" などを自動生成
    if feature_names is None:
        feature_names = np.array([labels["FEATURE"] % str(i) for i in range(num_features)])

    # x軸をシンメトリック対数スケールにするオプション
    if use_log_scale:
        pl.xscale("symlog")

    # --- 5) (複雑な)SHAP相互作用値を描画する場合のロジック (len(shap_values.shape)==3)
    #     ここでは "compact_dot" のみサポート例などの入り組んだ処理
    if not multi_class and len(shap_values.shape) == 3:
        if plot_type == "compact_dot":
            # 相互作用を 2次元に reshape
            new_shap_values = shap_values.reshape(shap_values.shape[0], -1)
            # features を同様に reshape
            new_features = np.tile(features, (1, 1, features.shape[1])).reshape(features.shape[0], -1)

            # 特徴量名の組み合わせ "Feature1 * - Feature2" を生成
            new_feature_names = []
            for c1 in feature_names:
                for c2 in feature_names:
                    if c1 == c2:
                        new_feature_names.append(c1)
                    else:
                        new_feature_names.append(c1 + "* - " + c2)

            # ここで再帰的に summary_legacy を呼び出し、"dot" プロットを行う
            # max_display, color, axis_color 等は引き継ぐ
            return summary_legacy(
                new_shap_values,
                new_features,
                new_feature_names,
                max_display=max_display,
                plot_type="dot",
                color=color,
                axis_color=axis_color,
                title=title,
                alpha=alpha,
                show=show,
                sort=sort,
                color_bar=color_bar,
                plot_size=plot_size,
                class_names=class_names,
                color_bar_label="*" + color_bar_label,
            )

        # 相互作用のある場合、 max_display をデフォルト 7 にしたりなどの調整
        if max_display is None:
            max_display = 7
        else:
            max_display = min(len(feature_names), max_display)

        # ここからさらに interaction values を表示するための複雑な可視化ロジック
        sort_inds = np.argsort(-np.abs(shap_values.sum(1)).sum(0))

        # グラフ描画エリアを1行 max_display 列に分割して、各特徴量ごとに subplot
        delta = 1.0 / (shap_values.shape[1] ** 2)
        slow = np.nanpercentile(shap_values, delta)
        shigh = np.nanpercentile(shap_values, 100 - delta)
        v = max(abs(slow), abs(shigh))
        slow = -v
        shigh = v

        # figsize を設定
        pl.figure(figsize=(1.5 * max_display + 1, 0.8 * max_display + 1))

        # 1枚目のサブプロット (sort_inds[0])
        pl.subplot(1, max_display, 1)
        proj_shap_values = shap_values[:, sort_inds[0], sort_inds]
        proj_shap_values[:, 1:] *= 2
        summary_legacy(
            proj_shap_values,
            features[:, sort_inds] if features is not None else None,
            feature_names=np.array(feature_names)[sort_inds].tolist(),
            sort=False,
            show=False,
            color_bar=False,
            plot_size=None,
            max_display=max_display,
        )
        pl.xlim((slow, shigh))
        pl.xlabel("")
        title_length_limit = 11
        pl.title(shorten_text(feature_names[sort_inds[0]], title_length_limit))

        # 2枚目以降 (sort_inds[1] など) を同様に subplot で並べる
        for i in range(1, min(len(sort_inds), max_display)):
            ind = sort_inds[i]
            pl.subplot(1, max_display, i + 1)
            proj_shap_values = shap_values[:, ind, sort_inds]
            proj_shap_values *= 2
            proj_shap_values[:, i] /= 2
            summary_legacy(
                proj_shap_values,
                features[:, sort_inds] if features is not None else None,
                sort=False,
                feature_names=["" for i in range(len(feature_names))],
                show=False,
                color_bar=False,
                plot_size=None,
                max_display=max_display,
            )
            pl.xlim((slow, shigh))
            pl.xlabel("")
            if i == min(len(sort_inds), max_display) // 2:
                pl.xlabel(labels["INTERACTION_VALUE"])
            pl.title(shorten_text(feature_names[ind], title_length_limit))

        # レイアウトや表示設定
        pl.tight_layout(pad=0, w_pad=0, h_pad=0.0)
        pl.subplots_adjust(hspace=0, wspace=0.1)
        if show:
            pl.show()
        return

    # ここで return されなければ、さらに下にある "dot", "bar", "violin", "layered_violin" などの分岐が続く。

    # -------------------------------------------------------------
    # 1) max_display が指定されていなければ 20 にする
    # -------------------------------------------------------------
    if max_display is None:
        max_display = 20

    # sort=True の場合、|SHAP値| の大きい順に特徴量を並べ替える
    if sort:
        # multi_class = True (マルチ出力) なら、
        #   各クラスのSHAPを平均→さらに全クラス平均→絶対値の総和が大きいもの
        if multi_class:
            feature_order = np.argsort(np.sum(np.mean(np.abs(shap_values), axis=1), axis=0))
        else:
            # シングル出力なら、単純に列方向(|SHAP|を合計→ソート)
            feature_order = np.argsort(np.sum(np.abs(shap_values), axis=0))
        # 上位 max_display 個のインデックス (末尾から max_display 個) を得る
        feature_order = feature_order[-min(max_display, len(feature_order)) :]
    else:
        # sort=False の場合は単純に 0..(max_display-1) を反転させた順序を割り当てる
        feature_order = np.flip(np.arange(min(max_display, num_features)), 0)

    # -------------------------------------------------------------
    # 2) プロットサイズの設定
    # -------------------------------------------------------------
    row_height = 0.4
    if plot_size == "auto":
        # 縦の長さを feature_order の数に応じて動的に設定
        pl.gcf().set_size_inches(8, len(feature_order) * row_height + 1.5)
    elif type(plot_size) in (list, tuple):
        # ユーザー指定のサイズ (幅, 高さ)
        pl.gcf().set_size_inches(plot_size[0], plot_size[1])
    elif plot_size is not None:
        # 単一floatの場合、縦の長さを float * 特徴量数 + 1.5
        pl.gcf().set_size_inches(8, len(feature_order) * plot_size + 1.5)

    # x=0 に縦線を引いて、SHAP値がプラス/マイナスの境界線を可視化
    pl.axvline(x=0, color="#999999", zorder=-1)

    # -------------------------------------------------------------
    # 3) plot_type ごとに可視化の仕方を分ける
    # -------------------------------------------------------------

    # (A) plot_type="dot"
    #    -> beeswarmに近いドット表示 (旧式の実装)
    if plot_type == "dot":
        for pos, i in enumerate(feature_order):
            # y=pos の水平線を薄く描画 (特徴量ごとに横線)
            pl.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)

            # i番目特徴量のSHAP値 (全サンプル) を取得
            shaps = shap_values[:, i]
            # 同じく i番目の元特徴量データがあれば取得 (色付け等で使う)
            values = None if features is None else features[:, i]

            # サンプルインデックスをシャッフルして散布のばらつきに利用
            inds = np.arange(len(shaps))
            np.random.shuffle(inds)
            if values is not None:
                values = values[inds]
            shaps = shaps[inds]

            # カテゴリ型フラグ (idx2cat) を確認し、数値型でカラーリング可能か判定
            colored_feature = True
            try:
                if idx2cat is not None and idx2cat[i]:
                    colored_feature = False
                else:
                    # 数値に変換できない場合は例外で弾く
                    values = np.array(values, dtype=np.float64)
            except Exception:
                colored_feature = False

            # ここから点を上下にずらしながら散布 (beeswarm ロジック)
            N = len(shaps)
            nbins = 100
            # SHAP値を [min, max] -> [0, nbins] のビンに割り当て
            quant = np.round(nbins * (shaps - np.min(shaps)) / (np.max(shaps) - np.min(shaps) + 1e-8))
            inds = np.argsort(quant + np.random.randn(N) * 1e-6)
            layer = 0
            last_bin = -1
            ys = np.zeros(N)

            # 同じビンに属する点は layer を増やして y座標をずらす
            for ind in inds:
                if quant[ind] != last_bin:
                    layer = 0
                ys[ind] = np.ceil(layer / 2) * ((layer % 2) * 2 - 1)
                layer += 1
                last_bin = quant[ind]

            # ys を row_height に合わせてスケールし、pos に加算して散布
            ys *= 0.9 * (row_height / np.max(ys + 1))

            # --- 数値特徴量がある場合のカラーリング ---
            if features is not None and colored_feature:
                # vmin,vmaxを [5%,95%] タイルで決定 (外れ値等を抑える)
                vmin = np.nanpercentile(values, 5)
                vmax = np.nanpercentile(values, 95)
                if vmin == vmax:
                    vmin = np.nanpercentile(values, 1)
                    vmax = np.nanpercentile(values, 99)
                    if vmin == vmax:
                        vmin = np.min(values)
                        vmax = np.max(values)
                if vmin > vmax:
                    vmin = vmax

                # features数とSHAP値のサンプル数が合わなければエラー
                assert features.shape[0] == len(shaps), "Feature and SHAP matrices must have the same number of rows!"

                # NaN値のサンプルはグレーで散布
                nan_mask = np.isnan(values)
                pl.scatter(
                    shaps[nan_mask],
                    pos + ys[nan_mask],
                    color="#777777",
                    s=16,
                    alpha=alpha,
                    linewidth=0,
                    zorder=3,
                    rasterized=len(shaps) > 500,
                )

                # NaN でないサンプルは cmap を使って色付け
                cvals = values[~nan_mask].astype(np.float64)
                cvals_imp = cvals.copy()
                cvals_imp[np.isnan(cvals)] = (vmin + vmax) / 2.0
                cvals[cvals_imp > vmax] = vmax
                cvals[cvals_imp < vmin] = vmin
                pl.scatter(
                    shaps[~nan_mask],
                    pos + ys[~nan_mask],
                    cmap=cmap,         # 指定されたカラーマップ
                    vmin=vmin,
                    vmax=vmax,
                    s=16,
                    c=cvals,
                    alpha=alpha,
                    linewidth=0,
                    zorder=3,
                    rasterized=len(shaps) > 500,
                )
            # --- 単色表示の場合 ---
            else:
                pl.scatter(
                    shaps,
                    pos + ys,
                    s=16,
                    alpha=alpha,
                    linewidth=0,
                    zorder=3,
                    color=color if colored_feature else "#777777",
                    rasterized=len(shaps) > 500,
                )

    # (B) plot_type="violin"
    #     -> バイオリンプロット形式でSHAP値の分布を可視化
    elif plot_type == "violin":
        # 各特徴量の行ごとに水平線を描画
        for pos in range(len(feature_order)):
            pl.axhline(y=pos, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)

        # features がある場合は自前のガウシアンカーネル密度推定を使って
        #   各サンプルの特徴量値で色付けしつつバイオリンを描画
        if features is not None:
            global_low = np.nanpercentile(shap_values[:, : len(feature_names)].flatten(), 1)
            global_high = np.nanpercentile(shap_values[:, : len(feature_names)].flatten(), 99)
            ...
            # ここで各特徴量についてガウシアンカーネル密度を求め、
            #   plt.fill_between などを使ってバイオリン形状を描画
            #   カラーリングは feature 値に基づく（vmin, vmax clip）
        else:
            # matplotlib の標準 violinplot を利用
            parts = pl.violinplot(
                shap_values[:, feature_order],
                range(len(feature_order)),
                points=200,
                vert=False,
                widths=0.7,
                showmeans=False,
                showextrema=False,
                showmedians=False,
            )
            # バイオリンの色やアルファ値を設定
            for pc in parts["bodies"]:
                pc.set_facecolor(color)
                pc.set_edgecolor("none")
                pc.set_alpha(alpha)

    # (C) plot_type="layered_violin"
    #     -> bin分割した特徴量の値ごとにKDEを積み重ねる特殊バイオリン
    elif plot_type == "layered_violin":
        ...
        # SHAP値全体の min~max を用いて、分割したbinごとのKDEを連続的に重ねる。
        # それを x軸方向にプロットしていき、y軸を特徴量インデックスに対応させる。

    # (D) plot_type="bar"
    #     -> SHAP値の絶対値平均などを棒グラフ表示
    elif not multi_class and plot_type == "bar":
        # シングル出力の場合
        feature_inds = feature_order[:max_display]
        y_pos = np.arange(len(feature_inds))
        global_shap_values = np.abs(shap_values).mean(0)
        # 横向きバーを描画
        pl.barh(y_pos, global_shap_values[feature_inds], 0.7, align="center", color=color)
        pl.yticks(y_pos, fontsize=13)
        pl.gca().set_yticklabels([feature_names[i] for i in feature_inds])

    elif multi_class and plot_type == "bar":
        # マルチクラスの場合
        if class_names is None:
            class_names = ["Class " + str(i) for i in range(len(shap_values))]
        feature_inds = feature_order[:max_display]
        y_pos = np.arange(len(feature_inds))
        left_pos = np.zeros(len(feature_inds))

        # class_inds の順番でバーを積み重ねる
        if class_inds is None:
            class_inds = np.argsort([-np.abs(shap_values[i]).mean() for i in range(len(shap_values))])
        elif class_inds == "original":
            class_inds = range(len(shap_values))

        # 凡例にクラスごとの平均SHAP値を表示するかどうか (show_values_in_legend)
        if show_values_in_legend:
            # 最小桁数を計算して丸めの幅を決める
            smallest_shap = np.min(np.abs(shap_values).mean((1, 2)))
            if smallest_shap > 1:
                n_decimals = 1
            else:
                n_decimals = int(-np.floor(np.log10(smallest_shap)))

        for i, ind in enumerate(class_inds):
            global_shap_values = np.abs(shap_values[ind]).mean(0)
            if show_values_in_legend:
                label = f"{class_names[ind]} ({np.round(np.mean(global_shap_values),(n_decimals+1))})"
            else:
                label = class_names[ind]

            # 横向きバーを追加し、left_pos をずらしてスタック
            pl.barh(
                y_pos, global_shap_values[feature_inds], 0.7, left=left_pos, align="center", color=color(i), label=label
            )
            left_pos += global_shap_values[feature_inds]
        pl.yticks(y_pos, fontsize=13)
        pl.gca().set_yticklabels([feature_names[i] for i in feature_inds])
        pl.legend(frameon=False, fontsize=12)


    # -------------------------------------------------------------
    # 4) カラーバーを描画 (bar 以外で必要なら)
    # -------------------------------------------------------------
    if (
        color_bar
        and features is not None
        and plot_type != "bar"
        and (plot_type != "layered_violin" or color in pl.colormaps)
    ):
        import matplotlib.cm as cm
        m = cm.ScalarMappable(cmap=cmap if plot_type != "layered_violin" else pl.get_cmap(color))
        m.set_array([0, 1])
        cb = pl.colorbar(m, ax=pl.gca(), ticks=[0, 1], aspect=80)
        cb.set_ticklabels([labels["FEATURE_VALUE_LOW"], labels["FEATURE_VALUE_HIGH"]])
        cb.set_label(color_bar_label, size=12, labelpad=0)
        cb.ax.tick_params(labelsize=11, length=0)
        cb.set_alpha(1)
        cb.outline.set_visible(False)  # type: ignore

    # -------------------------------------------------------------
    # 5) 軸の設定などを行い、plot を整形して終了
    # -------------------------------------------------------------
    pl.gca().xaxis.set_ticks_position("bottom")
    pl.gca().yaxis.set_ticks_position("none")
    pl.gca().spines["right"].set_visible(False)
    pl.gca().spines["top"].set_visible(False)
    pl.gca().spines["left"].set_visible(False)
    pl.gca().tick_params(color=axis_color, labelcolor=axis_color)

    # y軸目盛を feature_order に対応づけたラベルに更新
    pl.yticks(range(len(feature_order)), [feature_names[i] for i in feature_order], fontsize=13)

    # bar の場合は X軸ラベルを「Global Value」に、それ以外は「Value」に
    if plot_type != "bar":
        pl.gca().tick_params("y", length=20, width=0.5, which="major")
        pl.xlabel(labels["VALUE"], fontsize=13)
    else:
        pl.xlabel(labels["GLOBAL_VALUE"], fontsize=13)

    # プロットの余白を整える
    pl.tight_layout()

    # 最後に show=True なら表示
    if show:
        pl.show()