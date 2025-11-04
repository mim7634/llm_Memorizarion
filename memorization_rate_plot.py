import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import numpy as np

class MEMORIZATION_RATE_PLOT:
    def __init__(self):
        self.colors = [
            '#1f77b4',  # 1. 青 (Matplotlib標準)
            '#ff7f0e',  # 2. オレンジ (Matplotlib標準)
            '#2ca02c',  # 3. 緑 (Matplotlib標準)
            '#d62728',  # 4. 赤 (Matplotlib標準)
            '#9467bd',  # 5. 紫 (Matplotlib標準)
            '#8c564b',  # 6. 茶色
            '#e377c2',  # 7. ピンク
            '#7f7f7f',  # 8. 灰色
            '#bcbd22',  # 9. オリーブ
            '#17becf',  # 10. シアン
            '#006400',  # 11. 濃い緑 (Dark Green)
            '#ff4500',  # 12. オレンジ赤 (Orange Red)
            '#4682b4',  # 13. スティールブルー (Steel Blue)
            '#a52a2a',  # 14. 栗色 (Brown)
            '#ffd700',  # 15. 金色 (Gold)

            # 追加の17色
            '#7fff00',  # 16. シャルトルーズグリーン (Chartreuse Green)
            '#00ced1',  # 17. ダークターコイズ (Dark Turquoise)
            '#ff1493',  # 18. ディープピンク (Deep Pink)
            '#8a2be2',  # 19. ブルーバイオレット (Blue Violet)
            '#b8860b',  # 20. ダークゴールデンロッド (Dark Goldenrod)
            '#00bfff',  # 21. ディープスカイブルー (Deep Sky Blue)
            '#adff2f',  # 22. グリーンイエロー (Green Yellow)
            '#fa8072',  # 23. サーモン (Salmon)
            '#008080',  # 24. ティール (Teal)
            '#ffb6c1',  # 25. ライトピンク (Light Pink)
            '#483d8b',  # 26. ダークスレートブルー (Dark Slate Blue)
            '#ff8c00',  # 27. ダークオレンジ (Dark Orange)
            '#3cb371',  # 28. ミディアムシーグリーン (Medium Sea Green)
            '#9370db',  # 29. ミディアムパープル (Medium Purple)
            '#c71585',  # 30. ミディアムバイオレットレッド (Medium Violet Red)
            '#00ff7f',  # 31. スプリンググリーン (Spring Green)
            '#da70d6'   # 32. オーキッド (Orchid)
        ]

    def do_plot(self, csv_filename='analyze/epc_memorization_data(default).csv',
            x_min=None, x_max=None, x_width=None, # x_width を目盛りステップにも利用
            y_min=None, y_max=None, y_width=None):
        """
        エポック数と暗記率のデータを読み込み、プロットを作成して保存する。
        x_widthは、範囲設定に使われない場合、X軸の目盛りステップとして機能する。
        """
        
        # 0. 日本語フォントの設定 (省略)
        try:
            jp_font_name = 'Meiryo' 
            plt.rcParams['font.family'] = jp_font_name
            plt.rcParams['axes.unicode_minus'] = False 
            print(f"日本語フォント '{jp_font_name}' を設定しました。")
        except Exception as e:
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Meiryo', 'Yu Gothic', 'sans-serif']
            print(f"注意: フォント設定中にエラーが発生したため、一般的な代替フォント設定を使用します。")

        # 1. データの読み込み
        try:
            df_sorted = pd.read_csv(csv_filename)
            print(f"データファイル {csv_filename} の読み込みに成功しました。")
        except FileNotFoundError:
            print(f"エラー: データファイル {csv_filename} が見つかりません。")
            return

        # 2. プロットの作成
        plt.figure(figsize=(10, 6))
        plt.plot(df_sorted['epc(回)'], df_sorted['暗記率(%)'], 
                marker='o', linestyle='-', color='blue', linewidth=2, markersize=4)

        # 3. 軸の表示範囲の設定 (前回のロジック - x_min/x_max/x_width で範囲を決定)
        
        x_set = False
        x_min_final = df_sorted['epc(回)'].min() # 初期値
        x_max_final = df_sorted['epc(回)'].max() # 初期値
        
        # X軸範囲決定ロジック
        if x_min is not None and x_max is not None:
            # 1. x_min と x_max が両方指定されている場合 (最優先)
            x_min_final, x_max_final = x_min, x_max
            x_set = True
            log_msg = f"X軸の表示範囲を [下限:{x_min}, 上限:{x_max}] に設定しました。"
            
        elif x_width is not None:
            # 2. x_width が指定されている場合、これを範囲の幅として使用
            if x_min is not None:
                # a) x_min と x_width が設定されている場合
                x_min_final = x_min
                x_max_final = x_min + x_width
                x_set = True
                log_msg = f"X軸の表示範囲を [下限:{x_min}, 幅:{x_width}] に設定しました。"
            else:
                # b) x_width のみが設定されている場合 (下限はデータ最小値)
                x_min_final = df_sorted['epc(回)'].min()
                x_max_final = x_min_final + x_width
                x_set = True
                log_msg = f"X軸の範囲を [データ最小値:{x_min_final}, 幅:{x_width}] に設定しました。"

        if x_set:
            plt.xlim(x_min_final, x_max_final)
            print(log_msg)

        # Y軸の範囲決定ロジック (省略。X軸と同様に y_min_final, y_max_final が決定されるものとする)
        y_set = False
        y_min_final = df_sorted['暗記率(%)'].min()
        y_max_final = df_sorted['暗記率(%)'].max()

        if y_min is not None and y_max is not None:
            y_min_final, y_max_final = y_min, y_max
            y_set = True
        elif y_width is not None:
            if y_min is not None:
                y_min_final = y_min
                y_max_final = y_min + y_width
                y_set = True
            else:
                y_min_final = df_sorted['暗記率(%)'].min()
                y_max_final = y_min_final + y_width
                y_set = True
        
        if y_set:
            plt.ylim(y_min_final, y_max_final)
            # print(y_log_msg)

        # 4. タイトルとラベル、その他の設定
        plt.title('エポック数と暗記率の推移', fontsize=16)
        plt.xlabel('エポック数 (epc)', fontsize=14)
        plt.ylabel('訓練データの暗記率 (%)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # ★目盛り設定ロジック (x_width を目盛りステップとして再利用)
        # x_width が設定されている **かつ** 範囲設定のロジックで幅として使用されなかった場合、または目盛りステップとして上書きしたい場合
        # ここでは、x_maxが設定されたケースで x_width を目盛りステップとして利用できるようにする
        
        # 呼び出しで x_width と x_max が両方設定された場合に目盛りとして使用
        if x_width is not None and x_max is not None:
            x_tick_step_value = x_width
            
            # np.arange を使用して目盛りの位置を計算
            # 最小値と最大値の計算が完了しているため、それを利用する
            ticks = np.arange(0, x_max_final + x_tick_step_value, x_tick_step_value)
            
            # plt.xticks() で目盛りを強制設定
            plt.xticks(ticks, rotation=45, ha='right')
            print(f"X軸の目盛り間隔を x_width の値 ({x_tick_step_value}) に設定しました。")
        
        # 例外: x_width が範囲設定に使われず、目盛りステップとして利用されなかった場合は、Matplotlibの自動設定に従う
        
        plt.tight_layout()

        # 5. 画像として保存
        png_filename = csv_filename.replace('.csv', '.png')
        plot_filename_fix = png_filename
        plt.savefig(plot_filename_fix)
        plt.close()

        print(f"\n日本語対応のグラフを '{plot_filename_fix}' に作成し保存しました。")

    def do_plot_mutipul(self, legend_label_list=[], csv_path_list=[], filename='analyze/do_plot_mutipul(default).png',
                    x_min=None, x_max=None, x_width=None, # X軸の範囲と目盛りステップ
                    y_min=None, y_max=None, y_width=None): # Y軸の範囲
        """
        複数のCSVファイルを読み込み、重ねてプロットする。
        
        引数:
            legend_label_list (list): 凡例のラベルリスト。
            csv_path_list (list): 読み込むCSVファイルのパスリスト。
            filename (str): 保存するグラフのファイル名。
            x_min, x_max, x_width, y_min, y_max, y_width: 軸の範囲と目盛りステップ設定用。
        """
        
        # 0. 日本語フォントの設定
        try:
            jp_font_name = 'Meiryo' 
            plt.rcParams['font.family'] = jp_font_name
            plt.rcParams['axes.unicode_minus'] = False 
            print(f"日本語フォント '{jp_font_name}' を設定しました。")
        except Exception as e:
            plt.rcParams['font.family'] = 'sans-serif'
            plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Meiryo', 'Yu Gothic', 'sans-serif']
            print(f"注意: フォント設定中にエラーが発生したため、一般的な代替フォント設定を使用します。")

        # データの読み込みとプロットをループ内で行う
        plt.figure(figsize=(10, 6))
        
        # プロットに必要なデータを格納するリスト（軸範囲と目盛り計算に使用）
        all_epc = []
        
        for i, file in enumerate(csv_path_list):
            # データの読み込み
            try:
                df_sorted = pd.read_csv(file)
                print(f"データファイル {file} の読み込みに成功しました。")
            except FileNotFoundError:
                print(f"エラー: ファイル {file} が見つかりませんでした。スキップします。")
                continue
                
            # 全エポック数を結合（軸のデフォルト範囲計算用）
            all_epc.extend(df_sorted['epc(回)'].tolist())

            # プロット
            # self.colors[i] は、このクラスの外部で定義されていることを前提とします。
            plt.plot(df_sorted['epc(回)'], df_sorted['暗記率(%)'],
                marker='o', linestyle='-', color=self.colors[i], linewidth=1, markersize=2,
                label=legend_label_list[i]
            )

        # 全データの最小エポックと最大エポックを取得
        if all_epc:
            epc_min = min(all_epc)
            epc_max = max(all_epc)
        else:
            # データが一つも読み込めなかった場合
            print("エラー: 有効なデータファイルが一つも読み込めませんでした。")
            return

        # --- 3. 軸の表示範囲の設定 ---
        
        # ★修正点: x_set を初期化
        x_set = False 
        
        # X軸の最終範囲計算の初期値
        x_min_final, x_max_final = epc_min, epc_max
        
        if x_min is not None and x_max is not None:
            x_min_final, x_max_final = x_min, x_max
            x_set = True # 範囲が明示的に設定された
            print(f"X軸の表示範囲を [下限:{x_min}, 上限:{x_max}] に設定しました。")
        elif x_width is not None and x_min is not None:
            x_min_final, x_max_final = x_min, x_min + x_width
            x_set = True # 範囲が明示的に設定された
            print(f"X軸の表示範囲を [下限:{x_min}, 幅:{x_width}] に設定しました。")
        elif x_width is not None:
            x_min_final, x_max_final = epc_min, epc_min + x_width
            x_set = True # 範囲が明示的に設定された
            print(f"X軸の表示範囲を [データ最小値:{epc_min}, 幅:{x_width}] に設定しました。")

        plt.xlim(x_min_final, x_max_final)
        
        # Y軸の最終範囲計算 (暗記率は通常0-100%なので、デフォルトでこの範囲を考慮する)
        y_data_min = 0 # グラフの性質上0%をデフォルトの最小値とする
        y_data_max = 100 # グラフの性質上100%をデフォルトの最大値とする

        y_min_final, y_max_final = y_data_min, y_data_max
        if y_min is not None and y_max is not None:
            y_min_final, y_max_final = y_min, y_max
            print(f"Y軸の表示範囲を [下限:{y_min}, 上限:{y_max}] に設定しました。")
        elif y_width is not None and y_min is not None:
            y_min_final, y_max_final = y_min, y_min + y_width
            print(f"Y軸の表示範囲を [下限:{y_min}, 幅:{y_width}] に設定しました。")
        elif y_width is not None:
            y_min_final, y_max_final = y_data_min, y_data_min + y_width
            print(f"Y軸の表示範囲を [データ最小値:{y_data_min}, 幅:{y_width}] に設定しました。")

        plt.ylim(y_min_final, y_max_final)

        # --- 4. 凡例とグラフの装飾 ---
        legend_obj = plt.legend(loc='best', fontsize=12)
        exp_ext = legend_obj.get_bbox_to_anchor()
        legend_obj.get_frame().set_alpha(1.0)
        legend_obj.get_frame().set_facecolor('white')
        legend_filename = filename.replace('.png', '_legend.png')
        plt.savefig(
            legend_filename, 
            bbox_inches=exp_ext.transformed(plt.gcf().dpi_scale_trans.inverted()),
            transparent=True,
            dpi=300 # 高解像度で保存
        )
        legend_obj.remove()
        
        plt.title('n_embdパラメータと暗記容量の飽和点の関係', fontsize=16)
        plt.xlabel('エポック数 (epc)', fontsize=14)
        plt.ylabel('訓練データの暗記率 (%)', fontsize=14)
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # X軸の目盛り設定 (x_width を目盛りステップとして再利用)
        # x_maxが設定されている、または範囲がx_min/x_widthで設定された場合
        if x_width is not None and (x_set or x_max is not None): 
            x_tick_step_value = x_width
            
            # 目盛りは 0 から X軸の最終上限まで、x_width刻みで生成
            ticks = np.arange(0, x_max_final + x_tick_step_value, x_tick_step_value)
            
            # plt.xticks() で目盛りを強制設定 (回転は90度を維持)
            plt.xticks(ticks, rotation=0, ha='center')
            print(f"X軸の目盛り間隔を x_width の値 ({x_tick_step_value}) に設定しました。")
        else:
            # x_width が目盛り設定に使われなかった場合、自動設定に任せる
            pass

        plt.tight_layout()
        
        plt.savefig(filename)
        plt.close()

        print(f"\n日本語対応の複数プロットグラフを '{filename}' に作成し保存しました。")


if __name__ == '__main__':
    obj = MEMORIZATION_RATE_PLOT()
    csv_path_list = ['analyze\\models\\embd_8_withSEP_epc50000\\data\\embd_8_withSEP_epc50000.csv']
    obj.do_plot_mutipul(x_min=0, x_width=5000, y_min=0, y_width=100, x_max=51000, y_max=105, legend_label_list=['embd_8_withSEP'], csv_path_list=csv_path_list, filename='analyze\\models\\embd_8_withSEP_epc50000\\data\\embd_8_withSEP_epc50000.png')