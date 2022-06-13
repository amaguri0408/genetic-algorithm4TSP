import random
import os 
import shutil
import time
from itertools import combinations
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from multiprocessing import Pool

class GeneticAlgorithm():
    """
    遺伝的アルゴリズムクラス

    Attributes
    ----------
    locus : list of list of list of int
        染色体。(max_gen * num_sample * locus_length)次元のlist
    locus_length : int
        染色体の長さ
    num_par : int
        親の数
    num_sample : int
        世代あたりの染色体の数
    evaluation : list of list of int
        評価値。(max_gen * num_sample)次元のlist
    evaluation_ranking : list of dict
        評価値のランキング。長さはmax_gen。dictは(key : 順位, value : 染色体のインデックス)
    eval_func : function
        評価値を計算する関数。引数に染色体、返り値を評価値(小さくする)とする
    generation : int
        処理中の世代
    max_gen : int
        処理する最大の世代
    mutation_log : list of list of list of int
        突然変異の記録、出力用。入れ替えたindexを格納。
        (max_gen * num_sample * 2)次元のlist
    """
    
    def __init__(self, locus_length, eval_func, num_par:int=3, num_sample:int=9, max_gen:int=100) -> None:
        """
        初期化

        Parameters
        ----------
        locus_length : int
            染色体の長さ
        eval_func : function
            評価値を計算する関数。引数に染色体、返り値を評価値(小さくする)とする
        num_par : int, default = 3
            親の数
        num_sample : int, default = 9
            世代あたりの染色体の数
        max_gen : int, default = 100
            処理する最大の世代
        """
        self.locus_length = locus_length
        self.eval_func = eval_func
        self.num_par = num_par
        self.num_sample = num_sample
        self.max_gen = max_gen
        self.locus = [[[-1] * self.locus_length for _ in range(self.num_sample)] for _ in range(self.max_gen)]
        self.evaluation = [[0] * self.num_sample for _ in range(self.max_gen)]
        self.evaluation_ranking = [dict() for _ in range(self.max_gen)]
        self.mutation_log = [[[-1] * 2 for _ in range(self.num_sample)] for _ in range(self.max_gen)]
        # 初期集団作成
        for i in range(self.num_sample):
            self.locus[0][i] = list(range(self.locus_length))
            random.shuffle(self.locus[0][i])

    def run_ga(self) -> None:
        """
        遺伝的アルゴリズムの学習を行う関数
        """
        for i in range(1, self.max_gen):
            self.generation = i
            self.func_evaluation()
            self.func_selection()
            self.func_crossover()
            self.func_mutation()

    def func_evaluation(self) -> None:
        """
        評価を行う関数
        先代の染色体の評価値を計算してランキングする
        """
        # 評価値を計算
        for i in range(self.num_sample):
            self.evaluation[self.generation - 1][i] = self.eval_func(self.locus[self.generation - 1][i])
        # 評価値のランキング
        eval_idx_list = self.evaluation[self.generation - 1].copy()
        for i in range(self.num_sample): eval_idx_list[i] = (eval_idx_list[i], i)
        eval_idx_list.sort()
        for idx, value in enumerate(eval_idx_list):
            self.evaluation_ranking[self.generation - 1][idx] = value[1]

    def func_selection(self) -> None:
        """
        先代の個体から評価値の高い順にself.num_par個だけ選択して現代の個体の親とする
        """
        for i in range(self.num_par):
            self.locus[self.generation][i] = self.locus[self.generation - 1][self.evaluation_ranking[self.generation-1][i]]

    def func_crossover(self) -> None:
        """
        親を選んで交叉させて子とする
        交叉する親はitertools.permutationsで選ぶ
        """
        for i, v in enumerate(combinations(range(self.num_par), 2)):
            cross = self.func_simple_crossover(self.locus[self.generation][v[0]], self.locus[self.generation][v[1]])
            if 2 * i + self.num_par >= self.num_sample:
                break
            elif 2 * i + self.num_par + 1 >= self.num_sample:
                self.locus[self.generation][2 * i + self.num_par] = cross[0]
            else:
                for j in range(2):
                    self.locus[self.generation][2 * i + self.num_par + j] = cross[j]

    def func_simple_crossover(self, locus1: list, locus2: list) -> tuple:
        """
        2つの個体を受け取り交叉させて2つの個体を返す

        Parameters
        ----------
        locus1 : list of int
            交叉させる個体1つ目
        locus2 : list of int
            交叉させる個体2つ目

        Returns
        -------
        tuple of list of int
            交叉させてできた個体2つ
        """
        # child1 = locus1[:self.locus_length//2]
        # child2 = locus2[:self.locus_length//2]
        child1 = locus1[:random.randint(0, self.locus_length)]
        child2 = locus2[:random.randint(0, self.locus_length)]
        child1_set = set(child1)
        child2_set = set(child2)
        for i in range(self.locus_length):
            if not locus2[i] in child1_set:
                child1.append(locus2[i])
            if not locus1[i] in child2_set:
                child2.append(locus1[i])
        return child1, child2        

    def func_mutation(self) -> None:
        """
        突然変異を起こす
        """
        for i in range(1, self.num_sample):
            pre_locus = self.locus[self.generation][i].copy()
            if i >= self.num_par:
                self.locus[self.generation][i] = self.func_simple_mutiation(pre_locus)
                if random.random() > 0.5: self.locus[self.generation][i] = self.func_simple_mutiation(self.locus[self.generation][i])
            # 記録する
            log = []
            for j, a in enumerate(zip(pre_locus, self.locus[self.generation][i])):
                if a[0] == a[1]: continue
                log.append(j)
            self.mutation_log[self.generation][i] = log

    def func_simple_mutiation(self, locus: list) -> list:
        """
        1つの個体を受け取り突然変異させて返す

        Parameters
        ----------
        locus : list of int
            突然変異させる個体

        Returns
        -------
        list of int
            突然変異させた個体
        """
        locus = locus.copy()
        idx1, idx2 = random.sample(range(self.locus_length), 2)
        locus[idx1], locus[idx2] = locus[idx2], locus[idx1]
        return locus

class TSPGeneticAlgorithm(GeneticAlgorithm):
    """
    遺伝的アルゴリズムで巡回セールスマン問題を解く

    Attributes
    ----------
    vertex : list of tuple of float
        訪問する頂点の座標(x, y)
    eval_memo : dict
        個体と評価値を一対一で保存する
    """

    def __init__(self, num_vertex:int=16, num_par:int=3, num_sample:int=9, max_gen:int=100):
        """
        初期化

        superで初期化、eval_funcに巡回セールスマン問題の評価関数を渡す

        Parameters
        ----------
        num_vertex : int
            頂点数
        num_par : int
        num_sampe : int
        max_gen : int
        """
        self.num_vertex = num_vertex
        # self.vertex = [[random.uniform(1, 9) for _ in range(2)] for _ in range(self.num_vertex)]
        self.vertex = [[random.uniform(1, 9) for _ in range(2)]]
        for _ in range(self.num_vertex-1):
            while True:
                nx, ny = [random.uniform(1, 9) for _ in range(2)]
                flag = True
                for x, y in self.vertex:
                    if (nx - x) ** 2 + (ny - y) ** 2 < 1 ** 2:
                        flag = False
                        break
                if flag:
                    self.vertex.append([nx, ny])
                    break
        self.vertex.sort()
        self.eval_memo = dict()
        super(TSPGeneticAlgorithm, self).__init__(
            locus_length=self.num_vertex, 
            eval_func=self.eval_func, 
            num_par=num_par, 
            num_sample=num_sample, 
            max_gen=max_gen
        )
        # super(TSPGeneticAlgorithm, self).__init__(
        #     locus_length=self.num_vertex - 1, 
        #     eval_func=self.eval_func, 
        #     num_par=num_par, 
        #     num_sample=num_sample, 
        #     max_gen=max_gen
        # )

    def eval_func(self, locus: list) -> float:
        """
        評価値を計算する関数
        経路の移動距離の合計を返す

        Parameters
        ----------
        locus : list of int
            染色体のlist

        Returns
        -------
        int
            評価値
        """
        # メモされてたら返す
        memo_key = ''.join(map(str, locus))
        if memo_key in self.eval_memo: 
            return self.eval_memo[memo_key]
        res = 0
        search_order = locus.copy()
        # search_order.append(self.num_vertex - 1)   # 固定の変数を加える
        for i in range(self.num_vertex):
            res += ((self.vertex[search_order[i-1]][0] - self.vertex[search_order[i]][0]) ** 2 
                    + (self.vertex[search_order[i-1]][1] - self.vertex[search_order[i]][1]) ** 2) ** 0.5
        self.eval_memo[memo_key] = res
        return self.eval_memo[memo_key]

    def out_data(self, save_dir: str="tmp"):
        """
        結果をまとめた画像を出力する関数
        
        Parameters
        ----------
        save_dir : str
            保存するディレクトリ、./data/の相対パス
        """
        if os.path.exists(save_dir):
            shutil.rmtree(save_dir)
        os.mkdir(save_dir)
        detail_log_dir = os.path.join(save_dir, 'detail_log')
        simple_log_dir = os.path.join(save_dir, 'simple_log')
        os.mkdir(detail_log_dir)
        os.mkdir(simple_log_dir)
        # 日本語フォント対応
        plt.rcParams['font.family'] = "Meiryo"
        # 2行5列
        num_row = 3
        num_col = 5
        # GridSpecで区切る
        gs_master = GridSpec(num_row, 1)
        gs = [0] * num_row
        for i in range(num_row):
            gs[i] = GridSpecFromSubplotSpec(2, num_col, height_ratios=[5, 1], subplot_spec=gs_master[i])
        # poolのための関数
        def save_fig(gen):
            fig = plt.figure(figsize=(5 * num_col / 2, 6 * num_row / 2), dpi = 100)
            # figタイトル
            fig.text(0.1, 0.98, f"第{gen+1}世代", size="xx-large", va="top")
            ax = [[0] * num_col for _ in range(num_row * 2)]
            # 経路、スコア等書き込み
            for i in range(num_row):
                for j in range(num_col):
                    sample_idx = i * num_col + j
                    if sample_idx >= self.num_sample: break
                    tmp_fig = fig.add_subplot(gs[i][0, j])
                    if sample_idx < self.num_par and gen != 0:
                        tmp_fig.set(facecolor='lightgreen')
                    # 軸ラベル消去
                    tmp_fig.set_xticks([])
                    tmp_fig.set_yticks([])
                    # 軸範囲
                    tmp_fig.set_xlim(0, 10)     
                    tmp_fig.set_ylim(0, 10)
                    # 経路順に座標を格納
                    x_list = []
                    y_list = []
                    for k in range(-1, self.num_vertex):
                        # if k in [self.num_vertex - 1, -1]:
                        #     x_list.append(self.vertex[k][0])
                        #     y_list.append(self.vertex[k][1])
                        # else:
                        #     x_list.append(self.vertex[self.locus[gen][sample_idx][k]][0])
                        #     y_list.append(self.vertex[self.locus[gen][sample_idx][k]][1])
                        x_list.append(self.vertex[self.locus[gen][sample_idx][k]][0])
                        y_list.append(self.vertex[self.locus[gen][sample_idx][k]][1])
                    # プロット
                    tmp_fig.plot(x_list, y_list, marker="o", linestyle="-", label="path")
                    # # 頂点名をプロット
                    # for k in range(self.num_vertex):
                    #     tmp_fig.text(self.vertex[k][0], self.vertex[k][1], k)
                    # タイトル
                    if gen == 0:
                        title = f"初{sample_idx+1}"
                    elif sample_idx < self.num_par:
                        title = f"親{sample_idx+1}"
                    else:
                        title = f"子{sample_idx - self.num_par + 1}"
                    title += f" score : {self.evaluation[gen][sample_idx]:.2f}"
                    rank = list(filter(lambda x: x[1] == sample_idx, self.evaluation_ranking[gen].items()))[0][0] + 1
                    title += f" ({rank})"
                    if rank <= self.num_par:
                        tmp_fig.set_title(title, color='red')
                    else:
                        tmp_fig.set_title(title)
                    ax[i * 2][j] = tmp_fig
            
            # 遺伝子
            for i in range(num_row):
                for j in range(num_col):
                    sample_idx = i * num_col + j
                    color = [["white"] * self.num_vertex]
                    if sample_idx < self.num_sample and gen != 0:
                        for k in range(self.locus_length):
                            if self.locus[gen][sample_idx][k] in self.mutation_log[gen][sample_idx]:
                                color[0][k] = "red"
                    # print(gen, sample_idx, color)
                    if sample_idx >= self.num_sample: break
                    table = fig.add_subplot(gs[i][1, j])
                    # table.table([self.locus[gen][sample_idx] + [self.num_vertex - 1]], loc="center", cellLoc="center", cellColours=color)
                    table.table([self.locus[gen][sample_idx]], loc="center", cellLoc="center", cellColours=color)
                    table.axis('tight')
                    table.axis('off')
                    ax[i * 2 + 1][j] = table
            fig.savefig(os.path.join(detail_log_dir, f"{gen}.png"))
            plt.clf()
            plt.close()

            # シンプルログ出力
            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(111)
            ax.set_xticks([])
            ax.set_yticks([])
            x_list = []
            y_list = []
            for i in range(-1, self.num_vertex):
                # if i in [self.num_vertex - 1, -1]:
                #     x_list.append(self.vertex[i][0])
                #     y_list.append(self.vertex[i][1])
                # else:
                #     x_list.append(self.vertex[self.locus[gen][0][i]][0])
                #     y_list.append(self.vertex[self.locus[gen][0][i]][1])
                x_list.append(self.vertex[self.locus[gen][0][i]][0])
                y_list.append(self.vertex[self.locus[gen][0][i]][1])
            # プロット
            ax.plot(x_list, y_list, marker="o", linestyle="-", label="path")
            ax.set_title(f'第{gen+1}世代 score : {self.evaluation[gen][0]:.2f}')
            fig.savefig(os.path.join(simple_log_dir, f"{gen}.png"))
            plt.clf()
            plt.close()

        # メイン処理
        for gen in range(self.max_gen-1):
            fig = plt.figure(figsize=(5 * num_col / 2, 6 * num_row / 2), dpi = 100)
            # figタイトル
            fig.text(0.1, 0.98, f"第{gen+1}世代", size="xx-large", va="top")
            ax = [[0] * num_col for _ in range(num_row * 2)]
            # 経路、スコア等書き込み
            for i in range(num_row):
                for j in range(num_col):
                    sample_idx = i * num_col + j
                    if sample_idx >= self.num_sample: break
                    tmp_fig = fig.add_subplot(gs[i][0, j])
                    if sample_idx < self.num_par and gen != 0:
                        tmp_fig.set(facecolor='lightgreen')
                    # 軸ラベル消去
                    tmp_fig.set_xticks([])
                    tmp_fig.set_yticks([])
                    # 軸範囲
                    tmp_fig.set_xlim(0, 10)     
                    tmp_fig.set_ylim(0, 10)
                    # 経路順に座標を格納
                    x_list = []
                    y_list = []
                    for k in range(-1, self.num_vertex):
                        # if k in [self.num_vertex - 1, -1]:
                        #     x_list.append(self.vertex[k][0])
                        #     y_list.append(self.vertex[k][1])
                        # else:
                        #     x_list.append(self.vertex[self.locus[gen][sample_idx][k]][0])
                        #     y_list.append(self.vertex[self.locus[gen][sample_idx][k]][1])
                        x_list.append(self.vertex[self.locus[gen][sample_idx][k]][0])
                        y_list.append(self.vertex[self.locus[gen][sample_idx][k]][1])
                    # プロット
                    tmp_fig.plot(x_list, y_list, marker="o", linestyle="-", label="path")
                    # # 頂点名をプロット
                    # for k in range(self.num_vertex):
                    #     tmp_fig.text(self.vertex[k][0], self.vertex[k][1], k)
                    # タイトル
                    if gen == 0:
                        title = f"初{sample_idx+1}"
                    elif sample_idx < self.num_par:
                        title = f"親{sample_idx+1}"
                    else:
                        title = f"子{sample_idx - self.num_par + 1}"
                    title += f" score : {self.evaluation[gen][sample_idx]:.2f}"
                    rank = list(filter(lambda x: x[1] == sample_idx, self.evaluation_ranking[gen].items()))[0][0] + 1
                    title += f" ({rank})"
                    if rank <= self.num_par:
                        tmp_fig.set_title(title, color='red')
                    else:
                        tmp_fig.set_title(title)
                    ax[i * 2][j] = tmp_fig
            
            # 遺伝子
            for i in range(num_row):
                for j in range(num_col):
                    sample_idx = i * num_col + j
                    color = [["white"] * self.num_vertex]
                    if sample_idx < self.num_sample and gen != 0:
                        for k in range(self.locus_length):
                            if self.locus[gen][sample_idx][k] in self.mutation_log[gen][sample_idx]:
                                color[0][k] = "red"
                    # print(gen, sample_idx, color)
                    if sample_idx >= self.num_sample: break
                    table = fig.add_subplot(gs[i][1, j])
                    # table.table([self.locus[gen][sample_idx] + [self.num_vertex - 1]], loc="center", cellLoc="center", cellColours=color)
                    table.table([self.locus[gen][sample_idx]], loc="center", cellLoc="center", cellColours=color)
                    table.axis('tight')
                    table.axis('off')
                    ax[i * 2 + 1][j] = table
            fig.savefig(os.path.join(detail_log_dir, f"{gen}.png"))
            plt.clf()
            plt.close()

            # シンプルログ出力
            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(111)
            ax.set_xticks([])
            ax.set_yticks([])
            x_list = []
            y_list = []
            for i in range(-1, self.num_vertex):
                # if i in [self.num_vertex - 1, -1]:
                #     x_list.append(self.vertex[i][0])
                #     y_list.append(self.vertex[i][1])
                # else:
                #     x_list.append(self.vertex[self.locus[gen][0][i]][0])
                #     y_list.append(self.vertex[self.locus[gen][0][i]][1])
                x_list.append(self.vertex[self.locus[gen][0][i]][0])
                y_list.append(self.vertex[self.locus[gen][0][i]][1])
            # プロット
            ax.plot(x_list, y_list, marker="o", linestyle="-", label="path")
            ax.set_title(f'第{gen+1}世代 score : {self.evaluation[gen][0]:.2f}')
            fig.savefig(os.path.join(simple_log_dir, f"{gen}.png"))
            plt.clf()
            plt.close()

            print("\r", f"{(gen + 1)/self.max_gen * 100:.1f}%", end="")

# ga = GeneticAlgorithm(locus_length=16, eval_func=eval_func, num_par=3, num_sample=9, max_gen=100)
t1 = time.time()
ga = TSPGeneticAlgorithm(num_vertex=25, num_par=4, num_sample=15, max_gen=401)
ga.run_ga()
# print(len(ga.mutation_log), len(ga.mutation_log[0]), len(ga.mutation_log[0][0]), type(ga.mutation_log[0][0][0]))
# for i in range(ga.max_gen):
#     print(ga.mutation_log[i])
# print("学習", time.time() - t1)
t1 = time.time()
ga.out_data('./data/result29')
# for i in range(ga.max_gen-1):
#     print(ga.evaluation[i][ga.evaluation_ranking[i][0]])
#     # print(ga.evaluation_ranking[i])
print("\n出力", time.time() - t1)