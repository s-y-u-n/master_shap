import warnings

import numpy as np
import pandas as pd
from scipy.sparse import issparse
from tqdm.auto import tqdm

from .. import links, maskers
from ..utils._exceptions import (
    DimensionError,
    InvalidFeaturePerturbationError,
    InvalidModelError,
)
from ._explainer import Explainer


class LinearExplainer(Explainer):
    """
    Computes SHAP values for a linear model, optionally accounting for inter-feature correlations.

    これは線形モデルの SHAP 値を計算するためのクラス。
    相関のないケース(“interventional”) と、
    特徴量間の相関(“correlation_dependent”) を考慮するケースの両方をサポートする。

    Parameters
    ----------
    model : (coef, intercept) or sklearn.linear_model.*
        ユーザーが与える線形モデル。 (w, b) のタプルか、sklearnの線形モデルオブジェクト。

    masker : function, numpy.array, pandas.DataFrame, tuple(mean, cov), shap.maskers.Masker
        特徴量をマスクする仕組み or 背景データ（平均・共分散を持つ）など。

    data : (mean, cov), numpy.array, pandas.DataFrame, ...
        背景データ。ここから平均や共分散を取得し、(X - mean) により SHAP 値を計算などを行う。

    nsamples : int
        相関付き計算で使うサンプリング回数。デフォルト 1000。

    feature_perturbation : "interventional" or "correlation_dependent" or None
        特徴量の相関を無視する(interventional)か、考慮する(correlation_dependent)かを指定する。
        Noneの場合は内部で "interventional" が選択される。
    """

    def __init__(self, model, masker, link=links.identity, nsamples=1000, feature_perturbation=None, **kwargs):
        # --- 1) feature_dependence -> feature_perturbation への改名対応 ---
        if "feature_dependence" in kwargs:
            emsg = "The option feature_dependence has been renamed to feature_perturbation!"
            raise ValueError(emsg)

        # --- 2) feature_perturbation が指定された場合のWarning (互換性のため) ---
        if feature_perturbation is not None:  # pragma: no cover
            wmsg = (
                "The feature_perturbation option is now deprecated in favor of using the appropriate "
                "masker (maskers.Independent, maskers.Partition or maskers.Impute)."
            )
            warnings.warn(wmsg, FutureWarning)
        else:
            feature_perturbation = "interventional"

        # interventional or correlation_dependent 以外ならエラー
        if feature_perturbation not in ("interventional", "correlation_dependent"):
            emsg = "feature_perturbation must be one of 'interventional' or 'correlation_dependent'"
            raise InvalidFeaturePerturbationError(emsg)
        self.feature_perturbation = feature_perturbation

        # --- 3) masker を適切な maskers.* クラスにラップする (DataFrame → Impute/Independent 等) ---
        if isinstance(masker, pd.DataFrame) or (
            (isinstance(masker, np.ndarray) or issparse(masker)) and len(masker.shape) == 2
        ):
            if self.feature_perturbation == "correlation_dependent":
                masker = maskers.Impute(masker)
            else:
                masker = maskers.Independent(masker)
        elif issubclass(type(masker), tuple) and len(masker) == 2:
            # (mean, cov) タプルの場合
            if self.feature_perturbation == "correlation_dependent":
                masker = maskers.Impute({"mean": masker[0], "cov": masker[1]}, method="linear")
            else:
                masker = maskers.Independent({"mean": masker[0], "cov": masker[1]})

        # --- 4) 親クラスExplainerの __init__ 呼び出し (モデルやmasker設定) ---
        super().__init__(model, masker, link=link, **kwargs)

        self.nsamples = nsamples

        # --- 5) モデルから (coef, intercept) を取り出す ---
        self.coef, self.intercept = LinearExplainer._parse_model(model)

        # --- 6) マスカーの種類を見て feature_perturbation を確定し、背景データを取得する ---
        if issubclass(type(self.masker), (maskers.Independent, maskers.Partition)):
            self.feature_perturbation = "interventional"
        elif issubclass(type(self.masker), maskers.Impute):
            self.feature_perturbation = "correlation_dependent"
        else:
            raise NotImplementedError(
                "The Linear explainer only supports the Independent, Partition, and Impute maskers right now!"
            )
        data = getattr(self.masker, "data", None)

        # pandas.DataFrameなら ndarrayに変換
        if isinstance(data, pd.DataFrame):
            data = data.values

        # --- 7) mean/cov の取得: maskerに mean, cov があるか、data が (mean, cov)か 等を判定 ---
        if getattr(self.masker, "mean", None) is not None:
            self.mean = self.masker.mean
            self.cov = self.masker.cov
        elif isinstance(data, dict) and len(data) == 2:
            self.mean = data["mean"]
            if isinstance(self.mean, pd.Series):
                self.mean = self.mean.values

            self.cov = data["cov"]
            if isinstance(self.cov, pd.DataFrame):
                self.cov = self.cov.values
        elif isinstance(data, tuple) and len(data) == 2:
            self.mean = data[0]
            if isinstance(self.mean, pd.Series):
                self.mean = self.mean.values

            self.cov = data[1]
            if isinstance(self.cov, pd.DataFrame):
                self.cov = self.cov.values
        elif data is None:
            raise ValueError("A background data distribution must be provided!")
        else:
            # 通常の array か sparse か
            if issparse(data):
                self.mean = np.array(np.mean(data, 0))[0]
                if self.feature_perturbation != "interventional":
                    raise NotImplementedError(
                        "Only feature_perturbation = 'interventional' is supported for sparse data"
                    )
            else:
                self.mean = np.array(np.mean(data, 0)).flatten()
                if self.feature_perturbation == "correlation_dependent":
                    self.cov = np.cov(data, rowvar=False)

        # --- 8) mean と coef で expected_value を計算 (f(mean) = coef • mean + intercept) ---
        if issparse(self.mean) or str(type(self.mean)).endswith("matrix'>"):
            # sparseやmatrix型に対応
            self.expected_value = np.dot(self.coef, self.mean) + self.intercept
            if len(self.expected_value) == 1:
                self.expected_value = self.expected_value[0, 0]
            else:
                self.expected_value = np.array(self.expected_value)[0]
        else:
            self.expected_value = np.dot(self.coef, self.mean) + self.intercept

        self.M = len(self.mean)

        # --- 9) correlation_dependent の場合、共分散行列を用いた補正を用意 ---
        if self.feature_perturbation == "correlation_dependent":
            # 対角要素が小さい(=ほぼゼロ分散) の特徴量を除外
            self.valid_inds = np.where(np.diag(self.cov) > 1e-8)[0]
            self.mean = self.mean[self.valid_inds]
            self.cov = self.cov[:, self.valid_inds][self.valid_inds, :]
            self.coef = self.coef[self.valid_inds]

            # group perfectly redundant variables
            self.avg_proj, sum_proj = duplicate_components(self.cov)
            self.cov = np.matmul(np.matmul(self.avg_proj, self.cov), self.avg_proj.T)
            self.mean = np.matmul(self.avg_proj, self.mean)
            self.coef = np.matmul(sum_proj, self.coef)

            # 特異行列になりそうなら正則化 (1e-6 I を足す)
            e, _ = np.linalg.eig(self.cov)
            if e.min() < 1e-7:
                self.cov = self.cov + np.eye(self.cov.shape[0]) * 1e-6

            # サンプリングで 行列変換(mean_transform, x_transform) を推定
            mean_transform, x_transform = self._estimate_transforms(nsamples)
            self.mean_transformed = np.matmul(mean_transform, self.mean)
            self.x_transform = x_transform
        elif self.feature_perturbation == "interventional":
            # nsamples は関与しないので Warning
            if nsamples != 1000:
                warnings.warn("Setting nsamples has no effect when feature_perturbation = 'interventional'!")
        else:
            raise InvalidFeaturePerturbationError(
                "Unknown type of feature_perturbation provided: " + self.feature_perturbation
            )

    def _estimate_transforms(self, nsamples):
        """
        Uses block matrix inversion identities to quickly estimate transforms.

        特徴量間の相関を考慮して SHAP 値を正しく割り当てるために、
        部分的な行列逆や再帰的なアップデートをサンプリングしながら行い、
        行列変換(mean_transform, x_transform) を推定する。
        """
        M = len(self.coef)

        mean_transform = np.zeros((M, M))
        x_transform = np.zeros((M, M))
        inds = np.arange(M, dtype=int)

        for _ in tqdm(range(nsamples), "Estimating transforms"):
            np.random.shuffle(inds)
            cov_inv_SiSi = np.zeros((0, 0))
            cov_Si = np.zeros((M, 0))

            for j in range(M):
                i = inds[j]

                # cov_S, cov_inv_SS は "S" と呼ばれる部分集合に関する共分散やその逆行列
                cov_S = cov_Si
                cov_inv_SS = cov_inv_SiSi

                # 新たに i を追加した S ∪ {i} の共分散行列などを取り出す
                cov_Si = self.cov[:, inds[: j + 1]]

                # ブロック行列の逆行列を更新する手法 (Sherman–Morrison–Woodbury 近似に似た処理)
                d = cov_Si[i, :-1].T
                t = np.matmul(cov_inv_SS, d)
                Z = self.cov[i, i]
                u = Z - np.matmul(t.T, d)
                cov_inv_SiSi = np.zeros((j + 1, j + 1))
                if j > 0:
                    cov_inv_SiSi[:-1, :-1] = cov_inv_SS + np.outer(t, t) / u
                    cov_inv_SiSi[:-1, -1] = cov_inv_SiSi[-1, :-1] = -t / u
                cov_inv_SiSi[-1, -1] = 1 / u

                # mean_transform, x_transform の各要素を更新
                # + coef @ R(Sui) や - coef @ R(S) など、
                # どのようにSが拡張されるかを考慮しつつ寄与を加算
                mean_transform[i, i] += self.coef[i]

                coef_R_Si = np.matmul(self.coef[inds[j + 1 :]], np.matmul(cov_Si, cov_inv_SiSi)[inds[j + 1 :]])
                mean_transform[i, inds[: j + 1]] += coef_R_Si

                coef_R_S = np.matmul(self.coef[inds[j:]], np.matmul(cov_S, cov_inv_SS)[inds[j:]])
                mean_transform[i, inds[:j]] -= coef_R_S

                x_transform[i, i] += self.coef[i]
                x_transform[i, inds[: j + 1]] += coef_R_Si
                x_transform[i, inds[:j]] -= coef_R_S

        mean_transform /= nsamples
        x_transform /= nsamples
        return mean_transform, x_transform

    @staticmethod
    def _parse_model(model):
        """
        モデルobj (sklearnかタプル) から coef, intercept を取り出す。
        """
        if isinstance(model, tuple) and len(model) == 2:
            coef = model[0]
            intercept = model[1]

        elif hasattr(model, "coef_") and hasattr(model, "intercept_"):
            # shape が (1, n_features) みたいな可能性を考慮
            if len(model.coef_.shape) > 1 and model.coef_.shape[0] == 1:
                coef = model.coef_[0]
                try:
                    intercept = model.intercept_[0]
                except TypeError:
                    intercept = model.intercept_
            else:
                coef = model.coef_
                intercept = model.intercept_
        else:
            raise InvalidModelError("An unknown model type was passed: " + str(type(model)))

        return coef, intercept

    @staticmethod
    def supports_model_with_masker(model, masker):
        """
        この explainer でモデルとマスカーの組み合わせを扱えるかどうかのチェック関数。
        """
        if not isinstance(masker, (maskers.Independent, maskers.Partition, maskers.Impute)):
            return False

        try:
            LinearExplainer._parse_model(model)
        except Exception:
            return False
        return True

    def explain_row(self, *row_args, max_evals, main_effects, error_bounds, batch_size, outputs, silent):
        """
        1サンプル(row_args[0])に対する SHAP 値を計算して辞書で返す。
        """
        assert len(row_args) == 1, "Only single-argument functions are supported by the Linear explainer!"

        X = row_args[0]
        if len(X.shape) == 1:
            X = X.reshape(1, -1)

        # DataFrame → ndarray 化
        if isinstance(X, (pd.Series, pd.DataFrame)):
            X = X.values

        if len(X.shape) not in (1, 2):
            raise DimensionError(f"Instance must have 1 or 2 dimensions! Not: {len(X.shape)}")

        # correlation_dependentの場合 => (X - mean) に x_transform行列をかけてSHAP値を割り当て
        if self.feature_perturbation == "correlation_dependent":
            if issparse(X):
                raise InvalidFeaturePerturbationError(
                    "Only feature_perturbation = 'interventional' is supported for sparse data"
                )
            phi = (
                np.matmul(np.matmul(X[:, self.valid_inds], self.avg_proj.T), self.x_transform.T)
                - self.mean_transformed
            )
            phi = np.matmul(phi, self.avg_proj)

            # valid_inds を元にフルの形に戻す
            full_phi = np.zeros((phi.shape[0], self.M))
            full_phi[:, self.valid_inds] = phi
            phi = full_phi

        elif self.feature_perturbation == "interventional":
            # 相関無視 => phi = (X - mean) * coef
            if issparse(X):
                phi = np.array(np.multiply(X - self.mean, self.coef))
            else:
                phi = np.array(X - self.mean) * self.coef

        return {
            "values": phi.T,                 # shape (features, samples)
            "expected_values": self.expected_value,  # baseline
            "mask_shapes": (X.shape[1:],),   # 入力の形状 (特徴量数, ) など
            "main_effects": phi.T,           # 相互作用は考慮していないので main_effects=phi
            "clustering": None,
        }
    
    def shap_values(self, X):
        """
        複数サンプルに対して SHAP 値を一括計算し返す。
        """
        if isinstance(X, (pd.Series, pd.DataFrame)):
            X = X.values

        if len(X.shape) not in (1, 2):
            raise DimensionError(f"Instance must have 1 or 2 dimensions! Not: {len(X.shape)}")

        if self.feature_perturbation == "correlation_dependent":
            if issparse(X):
                raise InvalidFeaturePerturbationError(
                    "Only feature_perturbation = 'interventional' is supported for sparse data"
                )
            phi = (
                np.matmul(np.matmul(X[:, self.valid_inds], self.avg_proj.T), self.x_transform.T)
                - self.mean_transformed
            )
            phi = np.matmul(phi, self.avg_proj)

            full_phi = np.zeros((phi.shape[0], self.M))
            full_phi[:, self.valid_inds] = phi

            return full_phi

        elif self.feature_perturbation == "interventional":
            if issparse(X):
                if len(self.coef.shape) == 1:
                    return np.array(np.multiply(X - self.mean, self.coef))
                else:
                    return np.stack(
                        [np.array(np.multiply(X - self.mean, self.coef[i])) for i in range(self.coef.shape[0])],
                        axis=-1
                    )
            else:
                if len(self.coef.shape) == 1:
                    # シングル出力 => (num_samples, features)
                    return np.array(X - self.mean) * self.coef
                else:
                    # マルチ出力 => stacks
                    return np.stack(
                        [np.array(X - self.mean) * self.coef[i] for i in range(self.coef.shape[0])],
                        axis=-1
                    )

def duplicate_components(C):
    """
    共分散行列C 内でほぼ同一・冗長な成分を検出してグループ化する関数。
    相関が1に近い特徴量同士をまとめるなどして行列を縮小。
    """
    D = np.diag(1 / np.sqrt(np.diag(C)))
    C = np.matmul(np.matmul(D, C), D)
    components = -np.ones(C.shape[0], dtype=int)
    count = -1
    for i in range(C.shape[0]):
        found_group = False
        for j in range(C.shape[0]):
            if components[j] < 0 and np.abs(2 * C[i, j] - C[i, i] - C[j, j]) < 1e-8:
                if not found_group:
                    count += 1
                    found_group = True
                components[j] = count

    proj = np.zeros((len(np.unique(components)), C.shape[0]))
    proj[0, 0] = 1
    for i in range(1, C.shape[0]):
        proj[components[i], i] = 1
    return (proj.T / proj.sum(1)).T, proj