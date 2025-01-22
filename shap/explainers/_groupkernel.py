import copy
import gc
import itertools
import logging
import time
import warnings

import numpy as np
import pandas as pd
import scipy.sparse
import sklearn
from packaging import version
from scipy.special import binom
from sklearn.linear_model import Lasso, LassoLarsIC, lars_path
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm
from typing import Union, List

from .._explanation import Explanation
from ..utils import safe_isinstance
from ..utils._exceptions import DimensionError
from ..utils._legacy import (
    DenseData,
    SparseData,
    convert_to_data,
    convert_to_instance,
    convert_to_instance_with_index,
    convert_to_link,
    convert_to_model,
    match_instance_to_data,
    match_model_to_data,
)
from ._explainer import Explainer

log = logging.getLogger("shap")


class GroupkernelExplainer(Explainer):
    def __init__(self, model, data, feature_names=None, link="identity", **kwargs):
    #   ユーザが与えた model や data をSHAP用の標準形式に変換し、背景データやリンク関数、モデルの期待値(ベースライン)などを初期化する。
    #   また、背景データが多い場合には警告を出す。
    # data = background data
        print("GroupkernelExplainer.__init__")
        if feature_names is not None:
            self.data_feature_names = feature_names
        elif isinstance(data, pd.DataFrame):
            self.data_feature_names = list(data.columns)

        # link, model, data を SHAP の標準化された形式に変換
        self.link = convert_to_link(link)
        self.keep_index = kwargs.get("keep_index", False)
        self.keep_index_ordered = kwargs.get("keep_index_ordered", False)
        self.model = convert_to_model(model, keep_index=self.keep_index)
        self.data = convert_to_data(data, keep_index=self.keep_index)

        # モデルとデータの対応づけを行い、予測値などを取得
        model_null = match_model_to_data(self.model, self.data)

        # 現在サポートしているデータ形式かどうかをチェック (DenseData または SparseData のみ)
        if not isinstance(self.data, (DenseData, SparseData)):
            emsg = "Shap explainer only supports the DenseData and SparseData input currently."
            raise TypeError(emsg)
        # 転置された形式は対応外としてエラーを出す
        if self.data.transposed:
            emsg = "Shap explainer does not support transposed DenseData or SparseData currently."
            raise DimensionError(emsg)

        # バックグラウンドデータが大きすぎる場合に警告 (高速性が失われる可能性がある)
        if len(self.data.weights) > 100:
            log.warning(
                "Using "
                + str(len(self.data.weights))
                + " background data samples could cause "
                + "slower run times. Consider using shap.sample(data, K) or shap.kmeans(data, K) to "
                + "summarize the background as K samples."
            )

        # init our parameters
        self.N = self.data.data.shape[0]
        self.P = self.data.data.shape[1]
        self.linkfv = np.vectorize(self.link.f)
        self.nsamplesAdded = 0
        self.nsamplesRun = 0

        # find E_x[f(x)]
        if isinstance(model_null, (pd.DataFrame, pd.Series)):
            model_null = np.squeeze(model_null.values)
        if safe_isinstance(model_null, "tensorflow.python.framework.ops.EagerTensor"):
            model_null = model_null.numpy()
        elif safe_isinstance(model_null, "tensorflow.python.framework.ops.SymbolicTensor"):
            model_null = self._convert_symbolic_tensor(model_null)
        self.fnull = np.sum((model_null.T * self.data.weights).T, 0)
        self.expected_value = self.linkfv(self.fnull)

        # see if we have a vector output
        self.vector_out = True
        if len(self.fnull.shape) == 0:
            self.vector_out = False
            self.fnull = np.array([self.fnull])
            self.D = 1
            self.expected_value = float(self.expected_value)
        else:
            self.D = self.fnull.shape[0]

    @staticmethod
    # symbolic_tensor が tensorflow の tensor である場合、numpy の ndarray に変換する。
    # tf.Session() (あるいは tf.compat.v1.Session()) を用いてテンソルを実際の値に変換する。
    def _convert_symbolic_tensor(symbolic_tensor) -> np.ndarray:
        import tensorflow as tf

        if tf.__version__ >= "2.0.0":
            with tf.compat.v1.Session() as sess:
                sess.run(tf.compat.v1.global_variables_initializer())
                tensor_as_np_array = sess.run(symbolic_tensor)
        else:
            # this is untested
            with tf.Session() as sess:
                sess.run(tf.global_variables_initializer())
                tensor_as_np_array = sess.run(symbolic_tensor)
        return tensor_as_np_array

    # ユーザが説明を求めるサンプル群 X に対して、実際に shap_values を計算し、Explanation オブジェクトとして返す。
    # どの行(サンプル)も同じ「期待値」を持つようにタイル処理し、出力をまとめる。
    def __call__(self, X, l1_reg="num_features(10)", silent=False):
        print("KernelExplainer.__call__")
        start_time = time.time()

        if isinstance(X, pd.DataFrame):
            feature_names = list(X.columns)
        else:
            feature_names = getattr(self, "data_feature_names", None)

        # shap_values メソッドを呼び出して SHAP 値を計算
        v = self.shap_values(X, l1_reg=l1_reg, silent=silent)
        if isinstance(v, list):
            v = np.stack(v, axis=-1)  # put outputs at the end

        # the explanation object expects an expected value for each row
        if hasattr(self.expected_value, "__len__"):
            ev_tiled = np.tile(self.expected_value, (v.shape[0], 1))
        else:
            ev_tiled = np.tile(self.expected_value, v.shape[0])

        return Explanation(
            v,
            base_values=ev_tiled,
            data=X.to_numpy() if isinstance(X, pd.DataFrame) else X,
            feature_names=feature_names,
            compute_time=time.time() - start_time,
        )

    #   ユーザが指定したサンプル(1件または複数件)に対して、SHAP値を計算するメソッド。
    #   単一サンプルの場合は explain メソッドを1回呼び出して結果を返却。
    #   複数サンプルの場合は各サンプルごとに explain を呼び出し、その結果をまとめて返す。
    #   マルチ出力のときは次元整形を行い、出力形状を揃える。
    def shap_values(
        self,
        X: Union[np.ndarray, pd.DataFrame, scipy.sparse.spmatrix],
        **kwargs
    ) -> Union[np.ndarray, List[np.ndarray]]:
        print(" KernelExplainer.shap_values")

        # convert dataframes
        if isinstance(X, pd.Series):
            X = X.values
        elif isinstance(X, pd.DataFrame):
            if self.keep_index:
                index_value = X.index.values
                index_name = X.index.name
                column_name = list(X.columns)
            X = X.values

        x_type = str(type(X))
        arr_type = "'numpy.ndarray'>"
        # if sparse, convert to lil for performance
        if scipy.sparse.issparse(X) and not scipy.sparse.isspmatrix_lil(X):
            X = X.tolil()
        assert x_type.endswith(arr_type) or scipy.sparse.isspmatrix_lil(X), "Unknown instance type: " + x_type

        # single instance
        if len(X.shape) == 1:
            #   単一サンプルの場合は explain メソッドを1回呼び出して結果を返却。
            data = X.reshape((1, X.shape[0]))
            if self.keep_index:
                data = convert_to_instance_with_index(data, column_name, index_name, index_value)
            explanation = self.explain(data, **kwargs)

            # vector-output
            s = explanation.shape
            out = np.zeros(s)
            out[:] = explanation
            return out

        # explain the whole dataset
        # X.shape[0] = (サンプル数, 特徴量数)
        elif len(X.shape) == 2:
            #   複数サンプルの場合は各サンプルごとに explain を呼び出し、その結果をまとめて返す。
            explanations = []
            # tqdmを使う場合
            # for i in tqdm(range(X.shape[0]), disable=kwargs.get("silent", False)):
            # tqdmを使わない場合
            for i in range(X.shape[0]):
                data = X[i : i + 1, :]
                if self.keep_index:
                    data = convert_to_instance_with_index(data, column_name, index_value[i : i + 1], index_name)
                # SHAP値を計算するメソッド
                # explain呼び出し
                explanations.append(self.explain(data, **kwargs)) #data: 1行分のデータ
                if kwargs.get("gc_collect", False):
                    gc.collect()

            # vector-output
            s = explanations[0].shape
            if len(s) == 2:
                outs = [np.zeros((X.shape[0], s[0])) for j in range(s[1])]
                for i in range(X.shape[0]):
                    for j in range(s[1]):
                        outs[j][i] = explanations[i][:, j]
                outs = np.stack(outs, axis=-1)
                return outs

            # single-output
            else:
                out = np.zeros((X.shape[0], s[0]))
                for i in range(X.shape[0]):
                    out[i] = explanations[i]
                return out

        else:
            emsg = "Instance must have 1 or 2 dimensions!"
            raise DimensionError(emsg)

    #   X.testの回数だけ実行される
    def explain(self, incoming_instance, **kwargs):
    #   指定された単一インスタンス（サンプル）に対して、実際にSHAP値を推定するロジックを内包する。
    #   「変動する特徴量(グループ)」がいくつあるかを調べ、
    #       変動なし: すべて 0
    #       変動が1つのみ: その1つに全寄与を割り当てる
    #       複数: Kernel SHAP のサンプリング手法によりShapley値を近似計算
    #   計算には allocate → addsample → run → solve といった一連のメソッドを呼び出して進める。
        print("     KernelExplainer.explain")

        # インスタンスを SHAP の標準化されたオブジェクトに変換
        # incoming_instance: X.testの1行分のデータ
        instance = convert_to_instance(incoming_instance)
        # 入力インスタンスと背景データの整合性をチェック
        match_instance_to_data(instance, self.data)

        # 0. varying_groups
        # 現在のサンプル x と、背景データの値が異なる（実際に変動し得る）特徴量を特定
        print("     KernelExplainer.explain: varying_groups")
        self.varyingInds = self.varying_groups(instance.x)
        if self.data.groups is None:
            # グループ指定がなければ、そのままインデックスを配列として保持
            # グループ指定箇所: shap/utils/_legacy.py の SparseData クラスの groups
            self.varyingFeatureGroups = np.array([i for i in self.varyingInds])
            self.M = self.varyingFeatureGroups.shape[0]
        else:
            # グループ指定がある場合は、そのグループIDを取得
            print("     KernelExplainer.explain: varyingInds = ", self.varyingInds)
            self.varyingFeatureGroups = [self.data.groups[i] for i in self.varyingInds]
            self.M = len(self.varyingFeatureGroups)
            groups = self.data.groups
            # ジャギー配列でない場合は numpy 配列化して高速化
            if (
                self.varyingFeatureGroups and
                all(len(groups[i]) == len(groups[0]) for i in self.varyingInds)
            ):
                self.varyingFeatureGroups = np.array(self.varyingFeatureGroups)
                # もし各グループが要素1つだけなら、フラットな配列に変換
                if self.varyingFeatureGroups.shape[1] == 1:
                    self.varyingFeatureGroups = self.varyingFeatureGroups.flatten()

        # f(x) を計算 (現在のサンプルをモデルに入力し、出力を取得)
        if self.keep_index:
            model_out = self.model.f(instance.convert_to_df())
        else:
            model_out = self.model.f(instance.x)
        # 出力が DataFrame や Series、あるいは TensorFlow シンボリックテンソルの可能性を考慮
        if isinstance(model_out, (pd.DataFrame, pd.Series)):
            model_out = model_out.values
        elif safe_isinstance(model_out, "tensorflow.python.framework.ops.SymbolicTensor"):
            model_out = self._convert_symbolic_tensor(model_out)
        self.fx = model_out[0]  # 現在のサンプルにおけるモデル出力 (マルチ出力なら先頭要素のみ)

        # 出力がベクトル（マルチ出力）かどうかを確認
        if not self.vector_out:
            self.fx = np.array([self.fx])

        # 変動する特徴量がひとつもない場合 (背景データと同一で違いが無い)
        if self.M == 0:
            phi = np.zeros((self.data.groups_size, self.D))
            phi_var = np.zeros((self.data.groups_size, self.D))

        # 変動する特徴量が1つだけなら、その1つがすべての差分を担う
        elif self.M == 1:
            phi = np.zeros((self.data.groups_size, self.D))
            phi_var = np.zeros((self.data.groups_size, self.D))
            # fx と fnull の差分 (link関数を通して) がそのまま SHAP 値
            diff = self.link.f(self.fx) - self.link.f(self.fnull)
            for d in range(self.D):
                phi[self.varyingInds[0], d] = diff[d]

        # 変動する特徴量が2つ以上ある場合は、サンプリングを使った Kernel SHAP を実行
        # 実計算箇所
        else:
            self.l1_reg = kwargs.get("l1_reg", "num_features(10)")
            # サンプリング数 nsamples が指定されていなければ自動で設定
            self.nsamples = kwargs.get("nsamples", "auto")
            if self.nsamples == "auto":
                self.nsamples = 2 * self.M + 2**11

            # もし M <= 30 なら、2^M-2 が全列挙に必要な最大サンプリング数
            self.max_samples = 2**30
            if self.M <= 30:
                self.max_samples = 2**self.M - 2
                if self.nsamples > self.max_samples:
                    self.nsamples = self.max_samples

        # 1. allocate
            # SHAP の計算に必要な領域を確保
            print("     KernelExplainer.explain: allocate")
            self.allocate()

            # サブセットサイズに応じた重みの計算や、全列挙が可能なサブセットの探索
            num_subset_sizes = int(np.ceil((self.M - 1) / 2.0))
            num_paired_subset_sizes = int(np.floor((self.M - 1) / 2.0))
            weight_vector = np.array([(self.M - 1.0) / (i * (self.M - i)) for i in range(1, num_subset_sizes + 1)])
            weight_vector[:num_paired_subset_sizes] *= 2
            weight_vector /= np.sum(weight_vector)
            log.debug(f"{weight_vector = }")
            log.debug(f"{num_subset_sizes = }")
            log.debug(f"{num_paired_subset_sizes = }")
            log.debug(f"{self.M = }")

            # 重みが対応するサブセットサイズ分を全列挙 (必要なら)
            num_full_subsets = 0
            num_samples_left = self.nsamples
            group_inds = np.arange(self.M, dtype="int64")
            mask = np.zeros(self.M)
            remaining_weight_vector = copy.copy(weight_vector)
            for subset_size in range(1, num_subset_sizes + 1):
                # subset_size の組み合わせ数 (必要なら complement を含めて2倍)
                nsubsets = binom(self.M, subset_size)
                if subset_size <= num_paired_subset_sizes:
                    nsubsets *= 2
                log.debug(f"{subset_size = }")
                log.debug(f"{nsubsets = }")
                log.debug(
                    "self.nsamples*weight_vector[subset_size-1] = "
                    f"{num_samples_left * remaining_weight_vector[subset_size - 1]}"
                )
                log.debug(
                    "self.nsamples*weight_vector[subset_size-1]/nsubsets = "
                    f"{num_samples_left * remaining_weight_vector[subset_size - 1] / nsubsets}"
                )

                # もし十分なサンプリング数があれば、subset_size のすべてを全列挙
                if num_samples_left * remaining_weight_vector[subset_size - 1] / nsubsets >= 1.0 - 1e-8:
                    num_full_subsets += 1
                    num_samples_left -= nsubsets

                    # 使い切った重みに合わせて、残りの重みを正規化
                    if remaining_weight_vector[subset_size - 1] < 1.0:
                        remaining_weight_vector /= 1 - remaining_weight_vector[subset_size - 1]

                    # subset_size の組み合わせをすべて addsample() により追加
        #2. addsample
                    print("     KernelExplainer.explain: addsample")
                    w = weight_vector[subset_size - 1] / binom(self.M, subset_size)
                    if subset_size <= num_paired_subset_sizes:
                        w /= 2.0
                    for inds in itertools.combinations(group_inds, subset_size):
                        mask[:] = 0.0
                        mask[np.array(inds, dtype="int64")] = 1.0
                        self.addsample(instance.x, mask, w)
                        if subset_size <= num_paired_subset_sizes:
                            # subset の補集合も追加
                            mask[:] = np.abs(mask - 1)
                            self.addsample(instance.x, mask, w)
                else:
                    break
            log.info(f"{num_full_subsets = }")

            # 全列挙できなかった部分はランダムにサンプリングしてカバー
            nfixed_samples = self.nsamplesAdded
            samples_left = self.nsamples - self.nsamplesAdded
            log.debug(f"{samples_left = }")
            if num_full_subsets != num_subset_sizes:
                remaining_weight_vector = copy.copy(weight_vector)
                # サブセットと補集合を同時にサンプリングするために半分にする
                remaining_weight_vector[:num_paired_subset_sizes] /= 2
                remaining_weight_vector = remaining_weight_vector[num_full_subsets:]
                remaining_weight_vector /= np.sum(remaining_weight_vector)
                log.info(f"{remaining_weight_vector = }")
                log.info(f"{num_paired_subset_sizes = }")

                # まとめてサブセットサイズを抽選し、その後で各サンプルを生成
                ind_set = np.random.choice(
                    len(remaining_weight_vector),
                    4 * samples_left,
                    p=remaining_weight_vector
                )
                ind_set_pos = 0
                used_masks = {}
                while samples_left > 0 and ind_set_pos < len(ind_set):
                    mask.fill(0.0)
                    ind = ind_set[ind_set_pos]
                    ind_set_pos += 1
                    subset_size = ind + num_full_subsets + 1
                    mask[np.random.permutation(self.M)[:subset_size]] = 1.0

                    # マスクが初めて出てきた組み合わせなら新規登録、重複ならそのサンプルの重みを加算
                    mask_tuple = tuple(mask)
                    new_sample = False
                    if mask_tuple not in used_masks:
                        new_sample = True
                        used_masks[mask_tuple] = self.nsamplesAdded
                        samples_left -= 1
                        self.addsample(instance.x, mask, 1.0)
                    else:
                        self.kernelWeights[used_masks[mask_tuple]] += 1.0

                    # サブセットの補集合についても同様に処理
                    if samples_left > 0 and subset_size <= num_paired_subset_sizes:
                        mask[:] = np.abs(mask - 1)
                        if new_sample:
                            samples_left -= 1
                            self.addsample(instance.x, mask, 1.0)
                        else:
                            # 補集合は同じ順序で次のサンプルになるので +1
                            self.kernelWeights[used_masks[mask_tuple] + 1] += 1.0

                # ランダムサンプリングした部分について kernelWeights を正規化
                weight_left = np.sum(weight_vector[num_full_subsets:])
                log.info(f"{weight_left = }")
                self.kernelWeights[nfixed_samples:] *= weight_left / self.kernelWeights[nfixed_samples:].sum()
        #3. run
            # これまでに生成したサンプルをモデルに入力して予測値を得る
            print("     KernelExplainer.explain: run")
            self.run()

            # 回帰により得た部分結果を、変動しない特徴量も含めたフルサイズ (self.data.groups_size) に拡張
            phi = np.zeros((self.data.groups_size, self.D))
            phi_var = np.zeros((self.data.groups_size, self.D))
        #4. solve
            print("     KernelExplainer.explain: solve")
            for d in range(self.D):
                vphi, vphi_var = self.solve(self.nsamples / self.max_samples, d)
                phi[self.varyingInds, d] = vphi
                phi_var[self.varyingInds, d] = vphi_var

        # single-output の場合は squeeze
        if not self.vector_out:
            phi = np.squeeze(phi, axis=1)
            phi_var = np.squeeze(phi_var, axis=1)

        # SHAP 値( phi )を返す
        return phi

    @staticmethod
    def not_equal(i, j):
    #   2つの要素が「実質的に等しいかどうか」を判定するヘルパー関数。
	#   数値型なら np.isclose で、その他型なら直接 == で比較し、一致しなければ 1、そうでなければ 0 を返す。
        number_types = (int, float, np.number)
        if isinstance(i, number_types) and isinstance(j, number_types):
            return 0 if np.isclose(i, j, equal_nan=True) else 1
        else:
            return 0 if i == j else 1

    #   X.testの回数だけ実行される
    #   x:1行分のデータ
    def varying_groups(self, x):
    #   入力サンプル x と背景データを比較し、「背景データと異なる値を取る特徴量(グループ)」のインデックスを見つける。
    #   背景データが sparse な場合、特徴量が 0 である場合は「変動なし」として 0 を返す。
    #   疎行列・密行列の両方に対応し、不一致があるかどうかで変動する特徴量を特定する。
    #
    # 大きく分けて以下の手順で処理を行う:
    # 1) 密行列の場合:
    #    - グループごとに値を比較し、異なるものが見つかった際にそのグループを変動ありとする。
    # 2) 疎行列の場合:
    #    - 非ゼロの列のみに注目し、背景データと入力サンプルが完全に一致しない列を抽出する。
    #    - 一部だけ異なる列がある場合や、背景データの一部がゼロで入力サンプルが非ゼロの場合も考慮する。

        if not scipy.sparse.issparse(x):
            varying = np.zeros(self.data.groups_size) #varying = [0. 0. ...]
            for i in range(self.data.groups_size): #i番目の特徴量に対して実行
                inds = self.data.groups[i] #グループ指定を取り出し
                x_group = x[0, inds] #xは[[x1,x2,...]]の形式なので、x[0, inds]でグループ指定の特徴量を取り出す
                print("     KernelExplainer.varying_groups: x_group = ", x_group)
                if scipy.sparse.issparse(x_group):
                    if all(j not in x.nonzero()[1] for j in inds):
                        varying[i] = False
                        continue
                    x_group = x_group.todense()
                num_mismatches = np.sum(np.frompyfunc(self.not_equal, 2, 1)(x_group, self.data.data[:, inds]))
                varying[i] = num_mismatches > 0
            varying_indices = np.nonzero(varying)[0]
            return varying_indices
        else:
            varying_indices = []
            # go over all nonzero columns in background and evaluation data
            # if both background and evaluation are zero, the column does not vary
            varying_indices = np.unique(np.union1d(self.data.data.nonzero()[1], x.nonzero()[1]))
            remove_unvarying_indices = []
            for i in range(len(varying_indices)):
                varying_index = varying_indices[i]
                # now verify the nonzero values do vary
                data_rows = self.data.data[:, [varying_index]]
                nonzero_rows = data_rows.nonzero()[0]

                if nonzero_rows.size > 0:
                    background_data_rows = data_rows[nonzero_rows]
                    if scipy.sparse.issparse(background_data_rows):
                        background_data_rows = background_data_rows.toarray()
                    num_mismatches = np.sum(np.abs(background_data_rows - x[0, varying_index]) > 1e-7)
                    # Note: If feature column non-zero but some background zero, can't remove index
                    if num_mismatches == 0 and not (
                        np.abs(x[0, [varying_index]][0, 0]) > 1e-7 and len(nonzero_rows) < data_rows.shape[0]
                    ):
                        remove_unvarying_indices.append(i)
            mask = np.ones(len(varying_indices), dtype=bool)
            mask[remove_unvarying_indices] = False
            varying_indices = varying_indices[mask]
            return varying_indices

    def allocate(self):
    #   サンプリングによって生成する合成データ( synth_data )や、マスク行列( maskMatrix ), カーネル重み( kernelWeights ), モデル出力格納用( y / ey )などを初期化・確保する。
    #   背景データが疎行列の場合と密行列の場合で処理を分け、性能を最適化している。
        if scipy.sparse.issparse(self.data.data):
            # We tile the sparse matrix in csr format but convert it to lil
            # for performance when adding samples
            shape = self.data.data.shape
            nnz = self.data.data.nnz
            data_rows, data_cols = shape
            rows = data_rows * self.nsamples
            shape = rows, data_cols
            if nnz == 0:
                self.synth_data = scipy.sparse.csr_matrix(shape, dtype=self.data.data.dtype).tolil()
            else:
                data = self.data.data.data
                indices = self.data.data.indices
                indptr = self.data.data.indptr
                last_indptr_idx = indptr[len(indptr) - 1]
                indptr_wo_last = indptr[:-1]
                new_indptrs = []
                for i in range(self.nsamples - 1):
                    new_indptrs.append(indptr_wo_last + (i * last_indptr_idx))
                new_indptrs.append(indptr + ((self.nsamples - 1) * last_indptr_idx))
                new_indptr = np.concatenate(new_indptrs)
                new_data = np.tile(data, self.nsamples)
                new_indices = np.tile(indices, self.nsamples)
                self.synth_data = scipy.sparse.csr_matrix((new_data, new_indices, new_indptr), shape=shape).tolil()
        else:
            self.synth_data = np.tile(self.data.data, (self.nsamples, 1))

        self.maskMatrix = np.zeros((self.nsamples, self.M))
        self.kernelWeights = np.zeros(self.nsamples)
        self.y = np.zeros((self.nsamples * self.N, self.D))
        self.ey = np.zeros((self.nsamples, self.D))
        self.lastMask = np.zeros(self.nsamples)
        self.nsamplesAdded = 0
        self.nsamplesRun = 0
        if self.keep_index:
            self.synth_data_index = np.tile(self.data.index_value, self.nsamples)

    def addsample(self, x, m, w):
    #	あるサブセットを表すマスクベクトル m に基づいて合成データを作成する。
    #   x の対応する特徴量だけを背景データ上書きして合成サンプルを生成し、マスク行列やカーネル重みも合わせて保存する。
        #print("     KernelExplainer.addsample")
        offset = self.nsamplesAdded * self.N
        if isinstance(self.varyingFeatureGroups, (list,)):
            for j in range(self.M):
                for k in self.varyingFeatureGroups[j]:
                    if m[j] == 1.0:
                        self.synth_data[offset : offset + self.N, k] = x[0, k]
        else:
            # for non-jagged numpy array we can significantly boost performance
            mask = m == 1.0
            groups = self.varyingFeatureGroups[mask]
            if len(groups.shape) == 2:
                for group in groups:
                    self.synth_data[offset : offset + self.N, group] = x[0, group]
            else:
                # further performance optimization in case each group has a single feature
                evaluation_data = x[0, groups]
                # In edge case where background is all dense but evaluation data
                # is all sparse, make evaluation data dense
                if scipy.sparse.issparse(x) and not scipy.sparse.issparse(self.synth_data):
                    evaluation_data = evaluation_data.toarray()
                self.synth_data[offset : offset + self.N, groups] = evaluation_data
        self.maskMatrix[self.nsamplesAdded, :] = m
        self.kernelWeights[self.nsamplesAdded] = w
        self.nsamplesAdded += 1

    def run(self):
    #   これまでに追加された合成サンプルを実際にモデルに入力し、モデルの出力を得る。
    #   モデル出力を self.y に格納し、さらに各サンプルの期待値(バックグラウンド重み付き平均)を計算して self.ey に保存する。
        num_to_run = self.nsamplesAdded * self.N - self.nsamplesRun * self.N
        data = self.synth_data[self.nsamplesRun * self.N : self.nsamplesAdded * self.N, :]
        if self.keep_index:
            index = self.synth_data_index[self.nsamplesRun * self.N : self.nsamplesAdded * self.N]
            index = pd.DataFrame(index, columns=[self.data.index_name])
            data = pd.DataFrame(data, columns=self.data.group_names)
            data = pd.concat([index, data], axis=1).set_index(self.data.index_name)
            if self.keep_index_ordered:
                data = data.sort_index()
        modelOut = self.model.f(data)
        if isinstance(modelOut, (pd.DataFrame, pd.Series)):
            modelOut = modelOut.values
        elif safe_isinstance(modelOut, "tensorflow.python.framework.ops.SymbolicTensor"):
            modelOut = self._convert_symbolic_tensor(modelOut)

        self.y[self.nsamplesRun * self.N : self.nsamplesAdded * self.N, :] = np.reshape(modelOut, (num_to_run, self.D))

        # find the expected value of each output
        for i in range(self.nsamplesRun, self.nsamplesAdded):
            eyVal = np.zeros(self.D)
            for j in range(self.N):
                eyVal += self.y[i * self.N + j, :] * self.data.weights[j]

            self.ey[i, :] = eyVal
            self.nsamplesRun += 1

    def solve(self, fraction_evaluated, dim):
    #   サンプリングした合成サンプルの結果(y, eyなど)を用いて、重み付き線形回帰を解き、各特徴量(グループ)のShapley値を推定するメソッド。
    #   必要に応じてL1正則化(Lasso)やAIC/BIC基準などの手法で特徴選択を行う。
    #   出力合計が (link.f(self.fx) - link.f(self.fnull)) になるように調整したベクトルをSHAP値( phi )として返す。
        eyAdj = self.linkfv(self.ey[:, dim]) - self.link.f(self.fnull[dim])
        s = np.sum(self.maskMatrix, 1)

        # do feature selection if we have not well enumerated the space
        nonzero_inds = np.arange(self.M)
        log.debug(f"{fraction_evaluated = }")
        if self.l1_reg == "auto":
            warnings.warn("l1_reg='auto' is deprecated and will be removed in a future version.", DeprecationWarning)
        if (self.l1_reg not in ["auto", False, 0]) or (fraction_evaluated < 0.2 and self.l1_reg == "auto"):
            w_aug = np.hstack((self.kernelWeights * (self.M - s), self.kernelWeights * s))
            log.info(f"{np.sum(w_aug) = }")
            log.info(f"{np.sum(self.kernelWeights) = }")
            w_sqrt_aug = np.sqrt(w_aug)
            eyAdj_aug = np.hstack((eyAdj, eyAdj - (self.link.f(self.fx[dim]) - self.link.f(self.fnull[dim]))))
            eyAdj_aug *= w_sqrt_aug
            mask_aug = np.transpose(w_sqrt_aug * np.transpose(np.vstack((self.maskMatrix, self.maskMatrix - 1))))
            # var_norms = np.array([np.linalg.norm(mask_aug[:, i]) for i in range(mask_aug.shape[1])])

            # select a fixed number of top features
            if isinstance(self.l1_reg, str) and self.l1_reg.startswith("num_features("):
                r = int(self.l1_reg[len("num_features(") : -1])
                nonzero_inds = lars_path(mask_aug, eyAdj_aug, max_iter=r)[1]

            # use an adaptive regularization method
            elif self.l1_reg in ("auto", "bic", "aic"):
                c = "aic" if self.l1_reg == "auto" else self.l1_reg

                # "Normalize" parameter of LassoLarsIC was deprecated in sklearn version 1.2
                if version.parse(sklearn.__version__) < version.parse("1.2.0"):
                    kwg = dict(normalize=False)
                else:
                    kwg = {}
                model = make_pipeline(StandardScaler(with_mean=False), LassoLarsIC(criterion=c, **kwg))
                nonzero_inds = np.nonzero(model.fit(mask_aug, eyAdj_aug)[1].coef_)[0]

            # use a fixed regularization coefficient
            else:
                nonzero_inds = np.nonzero(Lasso(alpha=self.l1_reg).fit(mask_aug, eyAdj_aug).coef_)[0]

        if len(nonzero_inds) == 0:
            return np.zeros(self.M), np.ones(self.M)

        # eliminate one variable with the constraint that all features sum to the output
        eyAdj2 = eyAdj - self.maskMatrix[:, nonzero_inds[-1]] * (
            self.link.f(self.fx[dim]) - self.link.f(self.fnull[dim])
        )
        etmp = np.transpose(np.transpose(self.maskMatrix[:, nonzero_inds[:-1]]) - self.maskMatrix[:, nonzero_inds[-1]])
        log.debug(f"{etmp[:4, :] = }")

        # solve a weighted least squares equation to estimate phi
        # least squares:
        #     phi = min_w ||W^(1/2) (y - X w)||^2
        # the corresponding normal equation:
        #     (X' W X) phi = X' W y
        # with
        #     X = etmp
        #     W = np.diag(self.kernelWeights)
        #     y = eyAdj2
        #
        # We could just rely on sciki-learn
        #     from sklearn.linear_model import LinearRegression
        #     lm = LinearRegression(fit_intercept=False).fit(etmp, eyAdj2, sample_weight=self.kernelWeights)
        # Under the hood, as of scikit-learn version 1.3, LinearRegression still uses np.linalg.lstsq and
        # there are more performant options. See https://github.com/scikit-learn/scikit-learn/issues/22855.
        y = np.asarray(eyAdj2)
        X = etmp
        WX = self.kernelWeights[:, None] * X
        try:
            w = np.linalg.solve(X.T @ WX, WX.T @ y)
        except np.linalg.LinAlgError:
            warnings.warn(
                "Linear regression equation is singular, a least squares solutions is used instead.\n"
                "To avoid this situation and get a regular matrix do one of the following:\n"
                "1) turn up the number of samples,\n"
                "2) turn up the L1 regularization with num_features(N) where N is less than the number of samples,\n"
                "3) group features together to reduce the number of inputs that need to be explained."
            )
            # XWX = np.linalg.pinv(X.T @ WX)
            # w = np.dot(XWX, np.dot(np.transpose(WX), y))
            sqrt_W = np.sqrt(self.kernelWeights)
            w = np.linalg.lstsq(sqrt_W[:, None] * X, sqrt_W * y, rcond=None)[0]
        log.debug(f"{np.sum(w) = }")
        log.debug(
            f"self.link(self.fx) - self.link(self.fnull) = {self.link.f(self.fx[dim]) - self.link.f(self.fnull[dim])}"
        )
        log.debug(f"self.fx = {self.fx[dim]}")
        log.debug(f"self.link(self.fx) = {self.link.f(self.fx[dim])}")
        log.debug(f"self.fnull = {self.fnull[dim]}")
        log.debug(f"self.link(self.fnull) = {self.link.f(self.fnull[dim])}")
        phi = np.zeros(self.M)
        phi[nonzero_inds[:-1]] = w
        phi[nonzero_inds[-1]] = (self.link.f(self.fx[dim]) - self.link.f(self.fnull[dim])) - sum(w)
        log.info(f"{phi = }")

        # clean up any rounding errors
        for i in range(self.M):
            if np.abs(phi[i]) < 1e-10:
                phi[i] = 0

        return phi, np.ones(len(phi))
