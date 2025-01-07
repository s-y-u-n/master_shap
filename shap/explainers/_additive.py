import numpy as np

from ..utils import MaskedModel, safe_isinstance
from ._explainer import Explainer


class AdditiveExplainer(Explainer):
    """
    Computes SHAP values for generalized additive models.

    This assumes that the model only has first-order effects. Extending this to
    second- and third-order effects is future work (if you apply this to those models right now
    you will get incorrect answers that fail additivity).
    """

    def __init__(self, model, masker, link=None, feature_names=None, linearize_link=True):
        """
        指定されたモデルとマスカーを使って、加法的モデル向けのExplainerを構築する。

        Parameters
        ----------
        model : function
            入力データを与えるとモデルの予測値(またはスコア)を返す関数。
            interpret.glassbox.ExplainableBoostingClassifier のようなクラスでも可。

        masker : function or numpy.array or pandas.DataFrame
            特徴量をマスクするためのオブジェクトまたはデータ。
            Independentマスカーなどを想定。特徴量を個別に無効化(0にする等)する仕組みを提供。

        link : リンク関数 (デフォルト: None)
            SHAP値をどのように変換するか。例：ロジット変換など

        feature_names : 特徴量名のリスト (デフォルト: None)

        linearize_link : bool
            リンク関数を線形化するかどうかを指定するオプション(上位クラスExplainer側に渡す)。
        """
        # 親クラスExplainerのコンストラクタを呼ぶ
        super().__init__(model, masker, feature_names=feature_names, linearize_link=linearize_link)

        # interpret.glassbox.ExplainableBoostingClassifier なら特別処理
        if safe_isinstance(model, "interpret.glassbox.ExplainableBoostingClassifier"):
            # model.decision_function を実際の予測関数として使う
            self.model = model.decision_function

            # マスカーが指定されていない場合は、まだ未実装とみなしてエラー
            if self.masker is None:
                self._expected_value = model.intercept_
                # ここ以下のコードはコメントアウトされているが、
                # 本来は EBMから必要な情報を取得して _zero_offset 等を計算する想定と思われる
                raise NotImplementedError(
                    "Masker not given and we don't yet support pulling the distribution centering directly from the EBM model!"
                )
                return

        # 相互作用がない加法的モデルを前提としているので、Tabular(Independent)なマスカーであることをチェック
        assert safe_isinstance(
            self.masker, "shap.maskers.Independent"
        ), "The Additive explainer only supports the Tabular masker at the moment!"

        # 以下、「ベースライン(_zero_offset)」と「各特徴量の単独オフセット(_input_offsets)」を計算
        # まず MaskedModel を使って、一度に複数パターンのマスクを評価し、その出力差分を利用する
        fm = MaskedModel(
            self.model,              # モデル(EBMなど)
            self.masker,             # Independentマスカー
            self.link,               # リンク関数
            self.linearize_link,     # リンク線形化の有無
            np.zeros(self.masker.shape[1])  # 特徴量数だけ0で埋めた入力
        )

        # masks行列を作る: 行の次元 = (特徴量数 + 1)
        #   最初の1行は 全てTrue (全特徴量ON)
        #   2行目以降は特定の1特徴量だけFalse (OFF)
        masks = np.ones((self.masker.shape[1] + 1, self.masker.shape[1]), dtype=bool)
        for i in range(1, self.masker.shape[1] + 1):
            masks[i, i - 1] = False

        # fm(masks) で、いくつかのマスクパターン(全ON + 各1特徴OFF)を一度にモデル評価
        # outputs.shape は (特徴量数+1,) となる想定
        outputs = fm(masks)

        # 全ON時の出力を _zero_offset とする
        self._zero_offset = outputs[0]

        # 各特徴量の単独オフセットを格納する配列
        self._input_offsets = np.zeros(masker.shape[1])
        for i in range(1, self.masker.shape[1] + 1):
            # 特徴量 i-1 だけ OFF にしたときの出力(outputs[i]) と 全ON時(outputs[0]) との差分
            self._input_offsets[i - 1] = outputs[i] - self._zero_offset

        # expected_value = 全特徴オフセットの合計 + ベースライン
        self._expected_value = self._input_offsets.sum() + self._zero_offset

    def __call__(self, *args, max_evals=None, silent=False):
        """
        モデルへの入力(*args) に対する SHAP 値計算を行うエントリポイント。
        ここでは親クラス(Explainer) の __call__ をそのまま利用し、キーワード引数(**kwargs) を除去しているだけ。
        """
        return super().__call__(*args, max_evals=max_evals, silent=silent)

    @staticmethod
    def supports_model_with_masker(model, masker):
        """
        与えられた model と masker が、この AdditiveExplainer で扱えるかどうかを返す静的メソッド。
        """
        # EBM クラス(ExlainableBoostingClassifier) かつ 相互作用(interactions)が 0 の場合のみ対応
        if safe_isinstance(model, "interpret.glassbox.ExplainableBoostingClassifier"):
            if model.interactions != 0:
                raise NotImplementedError("Need to add support for interaction effects!")
            return True

        return False

    def explain_row(self, *row_args, max_evals, main_effects, error_bounds, batch_size, outputs, silent):
        """
        単一のサンプル(row_args[0])に対して SHAP 値(=各特徴量の寄与)を計算し、辞書形式で返す。
        """
        x = row_args[0]
        # 1. inputs行列を対角に x[i] を配置し、それ以外は0にする。
            #   => 「特定の特徴量 i だけ本来の値を使い、他は 0(=OFF)」という入力を複数行作るイメージ。
            # x = [10, 5, 2]  の場合、以下のような inputs 行列ができる
            #   [[10, 0, 0],
            #    [0, 5, 0],
            #    [0, 0, 2]]
        inputs = np.zeros((len(x), len(x)))
        for i in range(len(x)):
            inputs[i, i] = x[i]

        # モデルにこれらの inputs を通し、_zero_offset や _input_offsets との比較で寄与度(phi)を計算
        phi = self.model(inputs) - self._zero_offset - self._input_offsets

        print(f"phi: {phi}")

        # SHAP では "values" が寄与度、"expected_values" がベースラインに相当
        return {
            "values": phi,
            "expected_values": self._expected_value,
            "mask_shapes": [a.shape for a in row_args],
            "main_effects": phi,  # 今回は相互作用考慮なし => main_effects = phi と同じ
            "clustering": getattr(self.masker, "clustering", None),
        }