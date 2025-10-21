import os
import csv
from natsort import natsorted
from verify_memorization import VERIFY_MEMORYZATION # VERIFY_MEMORYZATIONクラスを参照
from memorization_rate_plot import MEMORIZATION_RATE_PLOT


def verify(folder_path='model', filename='analyze/epc_memorization_data.csv', train_data_size=100):
    FOLDER_PATH = folder_path
    FILE_NAME = filename

    # モデルフォルダのリストを取得し、連番順にソート
    file_list_with_dirs = os.listdir(FOLDER_PATH)
    file_list_with_dirs_sorted = natsorted([f for f in file_list_with_dirs if os.path.isdir(os.path.join(FOLDER_PATH, f))])

    # CSV出力ディレクトリの作成
    output_dir = os.path.dirname(FILE_NAME)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    csv_data = [
        ['epc(回)', '暗記率(%)']
    ]

    print("---------------------------------------------------")
    print("🚀 モデル解析を開始 (CSVファイルに結果を追記)")
    print("---------------------------------------------------")

    for file in file_list_with_dirs_sorted:
        print(f"解析中: {file}")
        
        # フォルダであることの確認（natsortでソート済みだが、安全のため）
        if os.path.isdir(os.path.join(FOLDER_PATH, file)):
            
            # 1. 検証オブジェクトのインスタンス化 (モデルのロードが自動で走る)
            try:
                obj = VERIFY_MEMORYZATION(model_base_dir=folder_path, model_folder_input=file, train_data_size=train_data_size)
            except Exception as e:
                print(f"  ⚠️ ロードエラー: {e}. スキップします。")
                continue

            # 2. 訓練データの検証を実行 (メソッド名を変更)
            # 戻り値は (model_folder_input, memorization_rate, avg_ppl)
            model_folder_input, memorization_rate, avg_ppl = obj.verify_train_data() 
            
            # 3. エポック数 ('epc_XXXX') の抽出
            parts = file.split('_')
            try:
                epoch_index = parts.index('epc')
                epoch = parts[epoch_index + 1] # 'epc_'の次の要素が回数
            except ValueError:
                epoch = 'N/A'
            
            # 4. 結果の表示とデータへの追加
            print(f'  -> 暗記率: {memorization_rate:.2f}% (Epc: {epoch})')
            csv_data.append([epoch, f'{memorization_rate:.2f}'])

            # 5. CSVファイルへの書き込み (毎回上書きするロジックを維持)
            with open(FILE_NAME, 'w', newline='', encoding='utf-8') as csvfile:
                csv_writer = csv.writer(csvfile)
                csv_writer.writerows(csv_data)

            print("  CSVファイルへの書き込みが完了しました。")

    print("\n---------------------------------------------------")
    print("✅ 全てのモデルの解析が完了しました。")
    print("---------------------------------------------------")

if __name__ == '__main__':
    plot_obj = MEMORIZATION_RATE_PLOT()
    
    folder_list = ['01_20251021_epc_500_ppl_34-3723_llm', '01_20251021_epc_1000_ppl_8-3803_llm', '01_20251021_epc_1500_ppl_3-8728_llm', '01_20251021_epc_2000_ppl_2-4029_llm', '01_20251021_epc_2500_ppl_1-7462_llm', '01_20251021_epc_3000_ppl_1-3877_llm', '01_20251021_epc_3500_ppl_1-1829_llm', '01_20251021_epc_4000_ppl_1-0777_llm', '01_20251021_epc_4500_ppl_1-0295_llm', '01_20251021_epc_5000_ppl_1-0112_llm', '01_20251021_epc_5500_ppl_1-0051_llm', '01_20251021_epc_6000_ppl_1-0025_llm', '01_20251021_epc_6500_ppl_1-0014_llm', '01_20251021_epc_7000_ppl_1-0008_llm', '01_20251021_epc_7500_ppl_1-0005_llm', '01_20251021_epc_8000_ppl_1-0003_llm', '01_20251021_epc_8500_ppl_1-0002_llm', '01_20251021_epc_9000_ppl_1-0004_llm', '01_20251021_epc_9500_ppl_1-0001_llm', '01_20251021_epc_10000_ppl_1-009_llm']

    #for i, folder in enumerate(folder_list):
    #    filename='analyze/incremental_learning_embd_80(epc_'+str(i*500)+').csv'
    #    folder_path='model/'+folder
    #    verify(folder_path=folder_path, filename=filename, train_data_size=100)
    #    
    #    plot_obj.do_plot(x_min=0, x_width=100, y_min=0, y_width=100, x_max=3050, y_max=105, csv_filename=filename)

    filename='analyze/models/incremental_learning_embd_72/data/epc_to_10000.csv'
    #folder_path='model/'
    verify(filename=filename, train_data_size=100)
    
    plot_obj.do_plot(x_min=0, x_width=500, y_min=0, y_width=100, x_max=10050, y_max=105, csv_filename=filename)

    