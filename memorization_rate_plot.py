import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# 日本語フォントの設定
# 1. 適切な日本語フォントのパスを取得
# (例として、IPAex Gothicを検索。環境により適宜変更してください。)
try:
    # IPAex Gothicを探す（多くの環境で利用可能）
    font_path = fm.findfont(fm.FontProperties(family='IPAexGothic'))
    # もし見つからなければ、他の一般的なフォントを試す
    if not font_path:
        font_path = fm.findfont(fm.FontProperties(family='Meiryo'))
    if not font_path:
        # 最後の手段として、sans-serifのデフォルト設定に任せる
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Meiryo', 'Yu Gothic', 'sans-serif']
        print("注意: IPAexGothicが見つからなかったため、代替フォントを使用します。")
    else:
        # 見つかったフォントを指定
        jp_font = fm.FontProperties(fname=font_path)
        plt.rcParams['font.family'] = jp_font.get_name()
    
    # マイナス記号が文字化けしないように設定
    plt.rcParams['axes.unicode_minus'] = False 

except:
    # 外部ライブラリがない、または検索に失敗した場合の一般的な設定
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Meiryo', 'Yu Gothic', 'sans-serif']
    print("注意: フォント検索に失敗したため、一般的な代替フォント設定を使用します。")


# 1. データの読み込み (前の手順と同じ)
df_sorted = pd.read_csv('analyze_data/epc_memorization_data.csv')
print("データファイル 'epc_memorization_data.csv' の読み込みに成功しました。")

# 2. プロットの作成
plt.figure(figsize=(10, 6))
plt.plot(df_sorted['epc'], df_sorted['暗記率 (%)'], 
         marker='o', linestyle='-', color='blue', linewidth=2, markersize=8)

# 日本語フォントが適用されるようにタイトルとラベルを設定
plt.title('エポック数と暗記率の推移', fontsize=16)
plt.xlabel('エポック数 (epc)', fontsize=14)
plt.ylabel('訓練データの暗記率 (%)', fontsize=14)
plt.grid(True, linestyle='--', alpha=0.6)
plt.xticks(df_sorted['epc'], rotation=45, ha='right')
plt.tight_layout()

# 画像として保存
plot_filename_fix = 'epc_vs_memorization_rate_japanese_fix.png'
plt.savefig(plot_filename_fix)
plt.close()

print(f"\n日本語対応のグラフを '{plot_filename_fix}' に再作成し保存しました。")