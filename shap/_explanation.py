from __future__ import annotations

import copy
import operator
from dataclasses import dataclass, field
from typing import Any, Callable, cast

import numpy as np
import pandas as pd
import scipy.cluster
import scipy.sparse
import scipy.spatial
import sklearn
from slicer import Alias, Obj, Slicer

from .utils._clustering import hclust_ordering
from .utils._exceptions import DimensionError
from .utils._general import OpChain

op_chain_root = OpChain("shap.Explanation")


@dataclass
class OpHistoryItem:
    """An operation that has been applied to an Explanation object."""

    name: str
    prev_shape: tuple[int, ...]
    args: tuple[Any, ...] = ()
    kwargs: dict[str, Any] = field(default_factory=dict)
    collapsed_instances: bool = False


class MetaExplanation(type):
    """This metaclass exposes the Explanation object's class methods for creating template op chains."""

    def __getitem__(cls, item):
        return op_chain_root.__getitem__(item)

    @property
    def abs(cls) -> OpChain:
        """Element-wise absolute value op."""
        return op_chain_root.abs

    @property
    def identity(cls) -> OpChain:
        """A no-op."""
        return op_chain_root.identity

    @property
    def argsort(cls) -> OpChain:
        """Numpy style argsort."""
        return op_chain_root.argsort

    @property
    def flip(cls) -> OpChain:
        """Numpy style flip."""
        return op_chain_root.flip

    @property
    def sum(cls) -> OpChain:
        """Numpy style sum."""
        return op_chain_root.sum

    @property
    def max(cls) -> OpChain:
        """Numpy style max."""
        return op_chain_root.max

    @property
    def min(cls) -> OpChain:
        """Numpy style min."""
        return op_chain_root.min

    @property
    def mean(cls) -> OpChain:
        """Numpy style mean."""
        return op_chain_root.mean

    @property
    def sample(cls) -> OpChain:
        """Numpy style sample."""
        return op_chain_root.sample

    @property
    def hclust(cls) -> OpChain:
        """Hierarchical clustering op."""
        return op_chain_root.hclust


class Explanation(metaclass=MetaExplanation):
    """
    A sliceable set of parallel arrays representing a SHAP explanation.

    Notes
    -----
    The *instance* methods such as `.max()` return new Explanation objects with the
    operation applied.

    The *class* methods such as `Explanation.max` return OpChain objects that represent
    a set of dot chained operations without actually running them.
    """

    def __init__(
        self,
        values,
        base_values=None,
        data=None,
        display_data=None,
        instance_names=None,
        feature_names=None,
        output_names=None,
        output_indexes=None,
        lower_bounds=None,
        upper_bounds=None,
        error_std=None,
        main_effects=None,
        hierarchical_values=None,
        clustering=None,
        compute_time=None,
    ):
        # Explanation オブジェクトの生成時に呼ばれる初期化メソッド。
        # パラメータとして、SHAP 値 (values) のほか、ベースライン, 元の特徴量データ,
        # 表示用データ, 特徴量名, 出力名, クラスタリング情報など、多数の引数を取る。
        #
        # shap.Explanationの役割:
        # SHAP値や関連するメタデータを統合管理し、スライス/演算/可視化を容易にする。

        self.op_history: list[OpHistoryItem] = []
        # op_history: 操作履歴を記録するためのリスト。
        # Explanationオブジェクトに対して行われた各種演算などを保持して、
        # 後から何が行われたか追跡できるようにする仕組み。

        self.compute_time = compute_time
        # compute_time: SHAP値の計算にかかった時間などを保持したい場合に使うオプション。

        # --- 以下、values 引数がすでに Explanation 型だった場合の特別処理 ---
        if isinstance(values, Explanation):
            e = values
            values = e.values
            base_values = e.base_values
            data = e.data
            # もし 'values' に Explanation が渡されていたら、
            # その中身を展開して使う (今作っている Explanation に転写する形)。

        # output_dims を決定するための補助関数を呼び出し (下部で定義)
        self.output_dims = compute_output_dims(values, base_values, data, output_names)

        # 変数 values_shape: values の shape をタプル形式で得る (_compute_shape 参照)
        values_shape = _compute_shape(values)

        # output_names が指定されていなければ、自動生成するケース (単一出力の場合など)
        if output_names is None and len(self.output_dims) == 1:
            num_names = values_shape[self.output_dims[0]]
            assert num_names is not None, "Unexpected shape of values"
            output_names = [f"Output {i}" for i in range(num_names)]
            # 例: ["Output 0", "Output 1", ...] のような名前リスト

        # feature_names が 1次元の配列として与えられている場合、slicer で使うため Alias 化する
        if (
            len(_compute_shape(feature_names)) == 1
        ):  # TODO: should always be an alias once slicer supports per-row aliases
            if len(values_shape) >= 2 and len(feature_names) == values_shape[1]:
                # 特徴量数 == values.shape[1] のとき -> Alias(..., dim=1)
                feature_names = Alias(list(feature_names), 1)
            elif len(values_shape) >= 1 and len(feature_names) == values_shape[0]:
                # 特徴量数 == values.shape[0] のとき -> Alias(..., dim=0)
                feature_names = Alias(list(feature_names), 0)

        # output_names が 1次元配列なら、Alias化しておく (上と同様)
        if (
            len(_compute_shape(output_names)) == 1
        ):  # TODO: should always be an alias once slicer supports per-row aliases
            output_names = Alias(list(output_names), self.output_dims[0])

        # output_names がより高次元の場合 (2Dなど) は、Obj(...) 化して保持
        elif output_names is not None and not isinstance(output_names, Alias):
            output_names_order = len(_compute_shape(output_names))
            if output_names_order == 0:
                pass
            elif output_names_order == 1:
                output_names = Obj(output_names, self.output_dims)
            elif output_names_order == 2:
                output_names = Obj(output_names, [0] + list(self.output_dims))
            else:
                raise ValueError("shap.Explanation does not yet support output_names of order greater than 3!")

        # base_values が配列の場合に、それを Obj(...) 化しておく (出力次元に合わせてセット)
        if not hasattr(base_values, "__len__") or len(base_values) == 0:
            pass
        elif len(_compute_shape(base_values)) == len(self.output_dims):
            base_values = Obj(base_values, list(self.output_dims))
        else:
            base_values = Obj(base_values, [0] + list(self.output_dims))

        # slicer.Slicer オブジェクトを生成し、各配列 (values, data, feature_names 等) を一括管理
        self._s = Slicer(
            values=values,
            base_values=base_values,
            data=list_wrap(data),
            display_data=list_wrap(display_data),
            instance_names=None if instance_names is None else Alias(instance_names, 0),
            feature_names=feature_names,
            output_names=output_names,
            output_indexes=None if output_indexes is None else (self.output_dims, output_indexes),
            lower_bounds=list_wrap(lower_bounds),
            upper_bounds=list_wrap(upper_bounds),
            error_std=list_wrap(error_std),
            main_effects=list_wrap(main_effects),
            hierarchical_values=list_wrap(hierarchical_values),
            clustering=None if clustering is None else Obj(clustering, [0]),
        )
        # 上記で、(values, base_values, data, feature_names, ...) を
        # Slicerにまとめて登録。Slicerは「インデックススライスや形状変換を一括で扱う」仕組み。

    # =================== Slicer passthrough ===================

    @property
    def values(self):
        """Pass-through from the underlying slicer object."""
        # self._s (Slicer) が管理している 'values' (SHAP値本体) を返すプロパティ。
        return self._s.values

    @values.setter
    def values(self, new_values):
        # Slicer内の 'values' を書き換えるセッター。
        self._s.values = new_values

    @property
    def base_values(self):
        """Pass-through from the underlying slicer object."""
        # ベースライン予測値などを保持する配列 (base_values) へアクセスするプロパティ。
        return self._s.base_values

    @base_values.setter
    def base_values(self, new_base_values):
        self._s.base_values = new_base_values

    @property
    def data(self):
        """Pass-through from the underlying slicer object."""
        # 元の特徴量データ (ex: X_test) へのアクセサ。
        return self._s.data

    @data.setter
    def data(self, new_data):
        self._s.data = new_data

    @property
    def display_data(self):
        """Pass-through from the underlying slicer object."""
        # 可視化用に整形したデータ (ときには生データと違う場合がある)。
        return self._s.display_data

    @display_data.setter
    def display_data(self, new_display_data):
        # もし pandas DataFrame なら、 `.values` で配列化して保存
        if isinstance(new_display_data, pd.DataFrame):
            new_display_data = new_display_data.values
        self._s.display_data = new_display_data

    @property
    def instance_names(self):
        """Pass-through from the underlying slicer object."""
        # 各サンプル(インスタンス)の名前 (ex: 行ラベル) へのアクセサ。
        return self._s.instance_names

    @property
    def output_names(self):
        """Pass-through from the underlying slicer object."""
        # 多クラス出力などで、出力次元に対応する名前リストを返す。
        return self._s.output_names

    @output_names.setter
    def output_names(self, new_output_names):
        self._s.output_names = new_output_names

    @property
    def output_indexes(self):
        """Pass-through from the underlying slicer object."""
        # 出力次元のインデックス情報を返す (多クラスなど)。
        return self._s.output_indexes

    @property
    def feature_names(self):
        """Pass-through from the underlying slicer object."""
        # 特徴量の名前一覧を返す。
        return self._s.feature_names

    @feature_names.setter
    def feature_names(self, new_feature_names):
        self._s.feature_names = new_feature_names

    @property
    def lower_bounds(self):
        """Pass-through from the underlying slicer object."""
        # SHAP値の下界 (confidence interval の下限など) を返す (オプション機能)。
        return self._s.lower_bounds

    @property
    def upper_bounds(self):
        """Pass-through from the underlying slicer object."""
        # SHAP値の上界 (confidence interval の上限など) を返す (オプション機能)。
        return self._s.upper_bounds

    @property
    def error_std(self):
        """Pass-through from the underlying slicer object."""
        # SHAP値の標準誤差などを保持できるフィールド。 (オプション機能)
        return self._s.error_std

    @property
    def main_effects(self):
        """Pass-through from the underlying slicer object."""
        # main_effects: SHAPの2次・3次相互作用を考慮するとき、個々の主効果を別途保持するオプション。
        return self._s.main_effects

    @main_effects.setter
    def main_effects(self, new_main_effects):
        self._s.main_effects = new_main_effects

    @property
    def hierarchical_values(self):
        """Pass-through from the underlying slicer object."""
        # 階層構造での SHAP値を保持 (例: 特徴量が階層クラスタリングされている場合など)。
        return self._s.hierarchical_values

    @hierarchical_values.setter
    def hierarchical_values(self, new_hierarchical_values):
        self._s.hierarchical_values = new_hierarchical_values

    @property
    def clustering(self):
        """Pass-through from the underlying slicer object."""
        # 特徴量のクラスタリング情報 (階層的クラスタツリーなど) を返す。
        return self._s.clustering

    @clustering.setter
    def clustering(self, new_clustering):
        self._s.clustering = new_clustering

    # =================== Data model ===================
    def __repr__(self):
        """Display some basic printable info, but not everything."""
        # print() や 変数参照で表示されるときの表現を定義。
        # 主に .values, .base_values, .data の簡単な内容を表示する。
        out = f".values =\n{self.values!r}"
        if self.base_values is not None:
            out += f"\n\n.base_values =\n{self.base_values!r}"
        if self.data is not None:
            out += f"\n\n.data =\n{self.data!r}"
        return out

    def __getitem__(self, item) -> Explanation:
        """This adds support for OpChain indexing."""
        # ex: explanation[0], explanation[:, 5], explanation[..., 'feature_name'] などを可能にする。
        # Slicer を使って複数の配列を一貫してスライスし、新しい Explanation を返す。
        #
        # `OpChain` や Magic string 等を解釈する仕組みが入り混じり、複雑な実装になっている。

        new_self = None

        # item がタプルじゃない場合はタプルに変換 (ex: item = (item,))
        if not isinstance(item, tuple):
            item = (item,)

        # ループ内で必要に応じて OpChain を apply したり、Explanation を展開したり、2Dの feature_names, output_names の特殊処理などを行う。
        pos = -1
        for t in item:
            pos += 1
            if t is Ellipsis:
                # Ellipsis (...) があれば、shape 調整 (ndim が足りないときの埋め合わせ) として扱う。
                pos += len(self.shape) - len(item)
                continue

            orig_t = t
            if isinstance(t, OpChain):
                # OpChain の場合: t = t.apply(self) で実際のインデックスに変換する。
                t = t.apply(self)
                if isinstance(t, (np.int64, np.int32)): 
                    t = int(t)
                elif isinstance(t, np.ndarray):
                    t = [int(v) for v in t]
            elif isinstance(t, Explanation):
                # Explanation がスライスとして来た場合は t.values を用いる。
                t = t.values
            elif isinstance(t, str):
                # 文字列の場合: 多次元 output_names や feature_names の検索などを行うこともある。
                # ここで新しい Explanation(new_self)を生成し、2Dに対応する場合がある。
                # work around for 2D output_names since they are not yet slicer supported
                output_names_dims = []
                if "output_names" in self._s._objects:
                    output_names_dims = self._s._objects["output_names"].dim
                elif "output_names" in self._s._aliases:
                    output_names_dims = self._s._aliases["output_names"].dim
                if pos != 0 and pos in output_names_dims:
                    if len(output_names_dims) == 1:
                        t = np.argwhere(np.array(self.output_names) == t)[0][0]
                    elif len(output_names_dims) == 2:
                        new_values = []
                        new_base_values = []
                        new_data = []
                        new_self = copy.deepcopy(self)
                        for i, v in enumerate(self.values):
                            for j, s in enumerate(self.output_names[i]):
                                if s == t:
                                    new_values.append(np.array(v[:, j]))
                                    new_data.append(np.array(self.data[i]))
                                    new_base_values.append(self.base_values[i][j])

                        new_self = Explanation(
                            np.array(new_values),
                            base_values=np.array(new_base_values),
                            data=np.array(new_data),
                            display_data=self.display_data,
                            instance_names=self.instance_names,
                            feature_names=np.array(new_data),  # FIXME: this is probably a bug
                            output_names=t,
                            output_indexes=self.output_indexes,
                            lower_bounds=self.lower_bounds,
                            upper_bounds=self.upper_bounds,
                            error_std=self.error_std,
                            main_effects=self.main_effects,
                            hierarchical_values=self.hierarchical_values,
                            clustering=self.clustering,
                        )
                        new_self.op_history = copy.copy(self.op_history)
                        # new_self = copy.deepcopy(self)
                        # new_self.values = np.array(new_values)
                        # new_self.base_values = np.array(new_base_values)
                        # new_self.data = np.array(new_data)
                        # new_self.output_names = t
                        # new_self.feature_names = np.array(new_data)
                        # new_self.clustering = None

                # work around for 2D feature_names since they are not yet slicer supported
                feature_names_dims = []
                if "feature_names" in self._s._objects:
                    feature_names_dims = self._s._objects["feature_names"].dim
                if pos != 0 and pos in feature_names_dims and len(feature_names_dims) == 2:
                    new_values = []
                    new_data = []
                    for i, val_i in enumerate(self.values):
                        for s, v, d in zip(self.feature_names[i], val_i, self.data[i]):
                            if s == t:
                                new_values.append(v)
                                new_data.append(d)
                    new_self = copy.deepcopy(self)
                    new_self.values = new_values
                    new_self.data = new_data
                    new_self.feature_names = t
                    new_self.clustering = None
                    # return new_self

            if isinstance(t, (np.int8, np.int16, np.int32, np.int64)):
                t = int(t)

            if t is not orig_t:
                tmp = list(item)
                tmp[pos] = t
                item = tuple(tmp)

        # 上記で整形された item を Slicer による本来のスライスに渡す
        item = tuple(v for v in item)

        if len(item) == 0:
            # item が空になった場合は上記の処理で new_self ができていればそれを返す
            return new_self

        if new_self is None:
            new_self = copy.copy(self)

        new_self._s = new_self._s.__getitem__(item)
        new_self.op_history.append(OpHistoryItem(name="__getitem__", args=(item,), prev_shape=self.shape))
        return new_self

    @property
    def shape(self) -> tuple[int, ...]:
        """Compute the shape over potentially complex data nesting."""
        # .values (SHAP値配列) の shape を _compute_shape() で求め、返す。
        # Explanationの一次元目がサンプル数、二次元目が特徴量数、三次元目が出力次元...といった概念。
        shap_values_shape = _compute_shape(self._s.values)
        return cast(tuple[int, ...], shap_values_shape)

    def __len__(self):
        # len(explanation) で返す値 -> shape[0] (サンプル数) と想定
        return self.shape[0]    
    
    def __copy__(self) -> Explanation:
        # Python の copy.copy() を呼び出したときの処理。
        # ほとんどのフィールドを複製し、別の Slicer を持つ新しい Explanation を返す。
        new_exp = Explanation(
            self.values,
            base_values=self.base_values,
            data=self.data,
            display_data=self.display_data,
            instance_names=self.instance_names,
            feature_names=self.feature_names,
            output_names=self.output_names,
            output_indexes=self.output_indexes,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
            error_std=self.error_std,
            main_effects=self.main_effects,
            hierarchical_values=self.hierarchical_values,
            clustering=self.clustering,
        )
        new_exp.op_history = copy.copy(self.op_history)
        return new_exp

    # =================== Operations ===================

    def _apply_binary_operator(self, other, binary_op, op_name):
        # 二項演算 ( +, -, *, / ) を実装するための共通関数。
        # `binary_op` は operator.add 等が渡される。
        new_exp = self.__copy__()
        new_exp.op_history.append(OpHistoryItem(name=op_name, args=(other,), prev_shape=self.shape))

        # other が Explanation なら、両者の values / data / base_values を演算
        if isinstance(other, Explanation):
            new_exp.values = binary_op(new_exp.values, other.values)
            if new_exp.data is not None:
                new_exp.data = binary_op(new_exp.data, other.data)
            if new_exp.base_values is not None:
                new_exp.base_values = binary_op(new_exp.base_values, other.base_values)
        else:
            # other がスカラー/配列などのケース
            new_exp.values = binary_op(new_exp.values, other)
            if new_exp.data is not None:
                new_exp.data = binary_op(new_exp.data, other)
            if new_exp.base_values is not None:
                new_exp.base_values = binary_op(new_exp.base_values, other)
        return new_exp

    def __add__(self, other):
        return self._apply_binary_operator(other, operator.add, "__add__")

    def __radd__(self, other):
        return self._apply_binary_operator(other, operator.add, "__add__")

    def __sub__(self, other):
        return self._apply_binary_operator(other, operator.sub, "__sub__")

    def __rsub__(self, other):
        return self._apply_binary_operator(other, operator.sub, "__sub__")

    def __mul__(self, other):
        return self._apply_binary_operator(other, operator.mul, "__mul__")

    def __rmul__(self, other):
        return self._apply_binary_operator(other, operator.mul, "__mul__")

    def __truediv__(self, other):
        return self._apply_binary_operator(other, operator.truediv, "__truediv__")

    def _numpy_func(self, fname, **kwargs):
        """
        Apply a numpy-style function to this Explanation.
        例: mean, max, min, sum などを numpy 関数として適用する共通処理。
        """
        new_self = copy.copy(self)
        axis = kwargs.get("axis", None)

        # axis を指定している場合、indexing などで一度スライスすることがある
        if axis in [0, 1, 2]:
            new_self = new_self[axis]
            new_self.op_history = new_self.op_history[:-1]  # 直前の __getitem__ の履歴を削除

        # もし feature_names が 2次元などの場合、軸 = 0 のときに flatten する操作などがある
        if self.feature_names is not None and not is_1d(self.feature_names) and axis == 0:
            new_values = self._flatten_feature_names()
            new_self.feature_names = np.array(list(new_values.keys()))
            new_self.values = np.array([getattr(np, fname)(v, 0) for v in new_values.values()])
            new_self.clustering = None
        else:
        # 実際の値 (values / data / base_values) に対して numpy の fname (ex: np.mean) を適用
            new_self.values = getattr(np, fname)(np.array(self.values), **kwargs)
            if new_self.data is not None:
                try:
                    new_self.data = getattr(np, fname)(np.array(self.data), **kwargs)
                except Exception:
                    new_self.data = None
            # base_values の集約処理 (axis指定時には base_values を collapse するなど)
            if new_self.base_values is not None and isinstance(axis, int) and len(self.base_values.shape) > axis:
                new_self.base_values = getattr(np, fname)(self.base_values, **kwargs)
            elif isinstance(axis, int):
                new_self.base_values = None

        # clustering 情報が 3次元だったりする場合は破棄するなどの処理
        if axis == 0 and self.clustering is not None and len(self.clustering.shape) == 3:
            if self.clustering.std(0).sum() < 1e-8:
                new_self.clustering = self.clustering[0]
            else:
                new_self.clustering = None

        # op_history に記録を追加
        new_self.op_history.append(
            OpHistoryItem(
                name=fname,
                kwargs=kwargs,
                prev_shape=self.shape,
                collapsed_instances=axis == 0,
            ),
        )

        return new_self

    @property
    def abs(self):
        #  Explanation の各要素に対して絶対値を取る操作。
        return self._numpy_func("abs")

    @property
    def identity(self):
        # 何もしない no-op 操作。
        return self

    @property
    def argsort(self):
        # Numpy の argsort と同じような操作を行う。
        return self._numpy_func("argsort")

    @property
    def flip(self):
        # Numpy の flip と同じような操作を行う。
        return self._numpy_func("flip")

    def mean(self, axis: int):
        """Numpy-style mean function."""
        # Numpy の mean と同じような操作を行う。
        return self._numpy_func("mean", axis=axis)

    def max(self, axis: int):
        """Numpy-style max function."""
        # Numpy の max と同じような操作を行う。
        return self._numpy_func("max", axis=axis)

    def min(self, axis: int):
        """Numpy-style min function."""
        # Numpy の min と同じような操作を行う。
        return self._numpy_func("min", axis=axis)

    def sum(self, axis: int | None = None, grouping=None):
        """Numpy-style sum function."""
        # Numpy の sum と同じような操作を行う。
        # grouping が指定されていれば、特徴量をグループ化して合算する (例: 部分的に特徴量をまとめる)
        if grouping is None:
            return self._numpy_func("sum", axis=axis)
        if axis == 1 or len(self.shape) == 1:
            return group_features(self, grouping)
        raise DimensionError("Only axis = 1 is supported for grouping right now...")

    def percentile(self, q, axis=None) -> Explanation:
        # np.percentile と同等の操作を行うメソッド
        new_self = copy.deepcopy(self)
        # 2次元 feature_names を flatten する場合などのロジックあり
        # main部分は new_self.values = np.percentile(new_self.values, q, axis)
        if self.feature_names is not None and not is_1d(self.feature_names) and axis == 0:
            new_values = self._flatten_feature_names()
            new_self.feature_names = np.array(list(new_values.keys()))
            new_self.values = np.array([np.percentile(v, q) for v in new_values.values()])
            new_self.clustering = None
        else:
            new_self.values = np.percentile(new_self.values, q, axis)
            new_self.data = np.percentile(new_self.data, q, axis)
        # new_self.data = None
        new_self.op_history.append(
            OpHistoryItem(
                name="percentile",
                args=(axis,),
                prev_shape=self.shape,
                collapsed_instances=axis == 0,
            ),
        )
        return new_self

    def sample(self, max_samples: int, replace: bool = False, random_state: int = 0) -> Explanation:
        """Randomly samples the instances (rows) of the Explanation object.

        Parameters
        ----------
        max_samples : int
            The number of rows to sample. Note that if ``replace=False``, then
            fewer than max_samples will be drawn if ``len(explanation) < max_samples``.

        replace : bool
            Sample with or without replacement.

        random_state : int
            Random seed to use for sampling, defaults to 0.

        """
        """
        ランダムサンプリング (行方向) で Explanation オブジェクトをスライスし、サブセットを返す。
        """
        rng = np.random.RandomState(random_state)
        length = self.shape[0]
        assert length is not None
        inds = rng.choice(length, size=min(max_samples, length), replace=replace)
        return self[list(inds)]

    def hclust(self, metric: str = "sqeuclidean", axis: int = 0):
        """Computes an optimal leaf ordering sort order using hclustering.

        hclust(metric="sqeuclidean")

        Parameters
        ----------
        metric : str
            A metric supported by scipy clustering. Defaults to "sqeuclidean".

        axis : int
            The axis to cluster along.

        """
        """
        ヒエラルキークラスタリング (hclust) を行い、最適な順序を返す (インデックス順)。
        SHAP値が 2次元である必要あり (shape=(サンプル数, 特徴量数)など)。
        """
        values = self.values

        if len(values.shape) != 2:
            raise DimensionError("The hclust order only supports 2D arrays right now!")

        if axis == 1:
            values = values.T

        return hclust_ordering(X=values, metric=metric)

    # =================== Utilities ===================

    def hstack(self, other: Explanation) -> Explanation:
        """
        2つの Explanation を列方向 (特徴量方向) に結合する。
        values, data, feature_names 等を結合するため、shape[0] (サンプル数) が同じである必要がある。
        """
        assert self.shape[0] == other.shape[0], "Can't hstack explanations with different numbers of rows!"
        if not np.allclose(self.base_values, other.base_values, atol=1e-6):
            raise ValueError("Can't hstack explanations with different base values!")

        new_exp = Explanation(
            values=np.hstack([self.values, other.values]),
            base_values=self.base_values,
            data=self.data,
            display_data=self.display_data,
            instance_names=self.instance_names,
            feature_names=self.feature_names,
            output_names=self.output_names,
            output_indexes=self.output_indexes,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
            error_std=self.error_std,
            main_effects=self.main_effects,
            hierarchical_values=self.hierarchical_values,
            clustering=self.clustering,
        )
        return new_exp

    def cohorts(self, cohorts: int | list[int] | tuple[int] | np.ndarray) -> Cohorts:
        """
        Explanation オブジェクトを複数の cohort (グループ) に分割し、Cohorts オブジェクトを返す。
        cohorts が int の場合、決定木を使って自動的にサンプルをグループ分けする。
        cohorts が array の場合、その要素ごとにグループを分ける。
        """
        if self.values.ndim > 2:
            raise ValueError(
                "Cohorts cannot be calculated on multiple outputs at once. ..."
            )
        if isinstance(cohorts, int):
            return _auto_cohorts(self, max_cohorts=cohorts)
        if isinstance(cohorts, (list, tuple, np.ndarray)):
            cohorts = np.array(cohorts)
            return Cohorts(**{name: self[cohorts == name] for name in np.unique(cohorts)})
        raise TypeError("The given set of cohort indicators is not recognized! Please give an array or int.")

    def _flatten_feature_names(self) -> dict:
        """
        feature_names が 2D の場合に、行方向にある重複をまとめたりするための内部メソッド。
        """
        new_values: dict[Any, Any] = {}
        for i in range(len(self.values)):
            for s, v in zip(self.feature_names[i], self.values[i]):
                if s not in new_values:
                    new_values[s] = []
                new_values[s].append(v)
        return new_values

    def _use_data_as_feature_names(self):
        """
        data を feature_names の代わりに使う場合の内部メソッド (2D対応)。
        """
        new_values: dict[Any, Any] = {}
        for i in range(len(self.values)):
            for s, v in zip(self.data[i], self.values[i]):
                if s not in new_values:
                    new_values[s] = []
                new_values[s].append(v)
        return new_values

# クラスの終わり


def group_features(shap_values, feature_map) -> Explanation:
    # TODO: support and deal with clusterings
    reverse_map: dict[Any, list[Any]] = {}
    for name in feature_map:
        reverse_map[feature_map[name]] = reverse_map.get(feature_map[name], []) + [name]

    curr_names = shap_values.feature_names
    sv_new = copy.deepcopy(shap_values)
    found = {}
    i = 0
    rank1 = len(shap_values.shape) == 1
    for name in curr_names:
        new_name = feature_map.get(name, name)
        if new_name in found:
            continue
        found[new_name] = True

        new_name = feature_map.get(name, name)
        cols_to_sum = reverse_map.get(new_name, [new_name])
        old_inds = [curr_names.index(v) for v in cols_to_sum]

        if rank1:
            sv_new.values[i] = shap_values.values[old_inds].sum()
            sv_new.data[i] = shap_values.data[old_inds].sum()
        else:
            sv_new.values[:, i] = shap_values.values[:, old_inds].sum(1)
            sv_new.data[:, i] = shap_values.data[:, old_inds].sum(1)
        sv_new.feature_names[i] = new_name
        i += 1

    return Explanation(
        sv_new.values[:i] if rank1 else sv_new.values[:, :i],
        base_values=sv_new.base_values,
        data=sv_new.data[:i] if rank1 else sv_new.data[:, :i],
        display_data=None
        if sv_new.display_data is None
        else (sv_new.display_data[:, :i] if rank1 else sv_new.display_data[:, :i]),
        instance_names=None,
        feature_names=None if sv_new.feature_names is None else sv_new.feature_names[:i],
        output_names=None,
        output_indexes=None,
        lower_bounds=None,
        upper_bounds=None,
        error_std=None,
        main_effects=None,
        hierarchical_values=None,
        clustering=None,
    )


def compute_output_dims(values, base_values, data, output_names) -> tuple[int, ...]:
    """Uses the passed data to infer which dimensions correspond to the model's output."""
    values_shape = _compute_shape(values)

    # input shape matches the data shape
    if data is not None:
        data_shape = _compute_shape(data)

    # if we are not given any data we assume it would be the same shape as the given values
    else:
        data_shape = values_shape

    # output shape is known from the base values or output names
    if output_names is not None:
        output_shape = _compute_shape(output_names)

        # if our output_names are per sample then we need to drop the sample dimension here
        if (
            values_shape[-len(output_shape) :] != output_shape
            and values_shape[-len(output_shape) + 1 :] == output_shape[1:]
            and values_shape[0] == output_shape[0]
        ):
            output_shape = output_shape[1:]

    elif base_values is not None:
        output_shape = _compute_shape(base_values)[1:]
    else:
        output_shape = tuple()

    interaction_order = len(values_shape) - len(data_shape) - len(output_shape)
    output_dims = range(len(data_shape) + interaction_order, len(values_shape))
    return tuple(output_dims)


def is_1d(val):
    return not (isinstance(val[0], (list, np.ndarray)))


def _compute_shape(x) -> tuple[int | None, ...]:
    """Computes the shape of a generic object ``x``."""

    def _first_item(iterable):
        for item in iterable:
            return item
        return None

    if not hasattr(x, "__len__") or isinstance(x, str):
        return tuple()
    if not scipy.sparse.issparse(x) and len(x) > 0 and isinstance(_first_item(x), str):
        return (None,)
    if isinstance(x, dict):
        return (len(x),) + _compute_shape(x[next(iter(x))])

    # 2D arrays: we just take their shape as-is
    if len(getattr(x, "shape", tuple())) > 1:
        return x.shape

    # 1D arrays: we need to look inside
    if len(x) == 0:
        return (0,)
    if len(x) == 1:
        return (1,) + _compute_shape(_first_item(x))
    first_shape = _compute_shape(_first_item(x))
    if first_shape == tuple():
        return (len(x),)

    # Else we have an array of arrays...
    matches = np.ones(len(first_shape), dtype=bool)
    for i in range(1, len(x)):
        shape = _compute_shape(x[i])
        assert len(shape) == len(first_shape), "Arrays in Explanation objects must have consistent inner dimensions!"
        for j in range(len(shape)):
            matches[j] &= shape[j] == first_shape[j]
    return (len(x),) + tuple(first_shape[j] if match else None for j, match in enumerate(matches))


class Cohorts:
    """A collection of :class:`.Explanation` objects, typically each explaining a cluster of similar samples.

    Examples
    --------
    A ``Cohorts`` object can be initialized in a variety of ways.

    By explicitly specifying the cohorts:

    >>> exp = Explanation(
    ...     values=np.random.uniform(low=-1, high=1, size=(500, 5)),
    ...     data=np.random.normal(loc=1, scale=3, size=(500, 5)),
    ...     feature_names=list("abcde"),
    ... )
    >>> cohorts = Cohorts(
    ...     col_a_neg=exp[exp[:, "a"].data < 0],
    ...     col_a_pos=exp[exp[:, "a"].data >= 0],
    ... )
    >>> cohorts
    <shap._explanation.Cohorts object with 2 cohorts of sizes: [(198, 5), (302, 5)]>

    Or using the :meth:`.Explanation.cohorts` method:

    >>> cohorts2 = exp.cohorts(3)
    >>> cohorts2
    <shap._explanation.Cohorts object with 3 cohorts of sizes: [(182, 5), (12, 5), (306, 5)]>

    Most of the :class:`.Explanation` interface is also exposed in ``Cohorts``. For example, to retrieve the
    SHAP values corresponding to column 'a' across all cohorts, you can use:

    >>> cohorts[..., 'a'].values
    <shap._explanation.Cohorts object with 2 cohorts of sizes: [(198,), (302,)]>

    To actually retrieve the values of a particular :class:`.Explanation`, you'll need to access it via the
    :meth:`.Cohorts.cohorts` property:

    >>> cohorts.cohorts["col_a_neg"][..., 'a'].values
    array([...])  # truncated

    """

    def __init__(self, **kwargs: Explanation) -> None:
        self.cohorts = kwargs
        self._callables: dict[str, Callable] = {}

    @property
    def cohorts(self) -> dict[str, Explanation]:
        """Internal collection of cohorts, stored as a dictionary."""
        return self._cohorts

    @cohorts.setter
    def cohorts(self, cval):
        if not isinstance(cval, dict):
            emsg = "self.cohorts must be a dictionary!"
            raise TypeError(emsg)
        for exp in cval.values():
            if not isinstance(exp, Explanation):
                emsg = f"Arguments to a Cohorts set must be Explanation objects, but found {type(exp)}"
                raise TypeError(emsg)

        self._cohorts: dict[str, Explanation] = cval

    def __getitem__(self, item) -> Cohorts:
        new_cohorts = {}
        for k in self._cohorts:
            new_cohorts[k] = self._cohorts[k].__getitem__(item)
        return Cohorts(**new_cohorts)

    def __getattr__(self, name: str) -> Cohorts:
        new_cohorts = Cohorts()
        for k in self._cohorts:
            result = getattr(self._cohorts[k], name)
            if callable(result):
                new_cohorts._callables[k] = result  # bound methods like .mean, .sample
            else:
                new_cohorts._cohorts[k] = result
        return new_cohorts

    def __call__(self, *args, **kwargs) -> Cohorts:
        """Call the bound methods on the Explanation objects retrieved during attribute access.

        For example,
        ``Cohorts(...).mean(axis=0)`` would first run ``__getattr__("mean")`` and return a bound method
        ``Explanation.mean`` for all the :class:`Explanation` objects inside the ``Cohorts``, returned as a
        new ``Cohorts`` object. Then the ``(axis=0)`` call would be executed on that returned ``Cohorts``
        object, which is why we need ``__call__`` defined.
        """
        if not self._callables:
            emsg = "No methods to __call__!"
            raise ValueError(emsg)

        new_cohorts = {}
        for k, bound_method in self._callables.items():
            new_cohorts[k] = bound_method(*args, **kwargs)
        return Cohorts(**new_cohorts)

    def __repr__(self):
        return f"<shap._explanation.Cohorts object with {len(self._cohorts)} cohorts of sizes: {[v.shape for v in self._cohorts.values()]}>"


def _auto_cohorts(shap_values, max_cohorts) -> Cohorts:
    """This uses a DecisionTreeRegressor to build a group of cohorts with similar SHAP values."""
    # fit a decision tree that well separates the SHAP values
    m = sklearn.tree.DecisionTreeRegressor(max_leaf_nodes=max_cohorts)
    m.fit(shap_values.data, shap_values.values)

    # group instances by their decision paths
    paths = m.decision_path(shap_values.data).toarray()
    path_names = []

    # mark each instance with a path name
    for i in range(shap_values.shape[0]):
        name = ""
        for j in range(len(paths[i])):
            if paths[i, j] > 0:
                feature = m.tree_.feature[j]
                threshold = m.tree_.threshold[j]
                val = shap_values.data[i, feature]
                if feature >= 0:
                    name += str(shap_values.feature_names[feature])
                    if val < threshold:
                        name += " < "
                    else:
                        name += " >= "
                    name += str(threshold) + " & "
        path_names.append(name[:-3])  # the -3 strips off the last unneeded ' & '
    path_names_arr = np.array(path_names)

    # split the instances into cohorts by their path names
    cohorts = {}
    for name in np.unique(path_names_arr):
        cohorts[name] = shap_values[path_names_arr == name]

    return Cohorts(**cohorts)


def list_wrap(x):
    """A helper to patch things since slicer doesn't handle arrays of arrays (it does handle lists of arrays)"""
    if isinstance(x, np.ndarray) and len(x.shape) == 1 and isinstance(x[0], np.ndarray):
        return [v for v in x]
    else:
        return x
