# chat_with_model.py - PPL計算機能追加済み

import torch
from transformers import AutoTokenizer, GPT2LMHeadModel
import os
import sys
import numpy as np # PPL計算に必要

class CHAT_WITH_MODEL:
    def __init__(self, model_base_dir='model'):
        # --- 設定 ---
        self.MODEL_BASE_DIR = model_base_dir
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.MAX_NEW_TOKENS = 64 # 生成するトークンの最大長

    def calculate_ppl(self, model, tokenizer, text, device):
        """
        与えられたテキストのPerplexity (PPL)を計算する。
        """
        # テキスト全体をトークン化
        input_ids = tokenizer.encode(text, return_tensors='pt', truncation=True).to(device)
        
        # 損失計算のために、入力自身をラベルとして使用
        labels = input_ids.clone()
        
        # モデルの順伝播を実行し、損失を取得
        # attention_maskも渡すことで、パディングの影響を防ぐ（通常、encodeで生成されないが明示的に渡すのが安全）
        with torch.no_grad():
            # labels=labelsとすることで、内部でクロスエントロピー損失が計算される
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss.item()

        # Perplexity (PPL) = exp(loss)
        ppl = np.exp(loss)
        return ppl

    def load_and_chat(self):
        # 1. モデルフォルダ名の入力
        model_folder_input = input("チャットしたいモデルフォルダ名を入力してください: ")
        MODEL_DIR = os.path.join(self.MODEL_BASE_DIR, model_folder_input)

        # 2. モデルとトークナイザーのロード
        try:
            model = GPT2LMHeadModel.from_pretrained(MODEL_DIR).to(self.DEVICE).eval()
            tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
            print(f"\n✅ モデル '{model_folder_input}' のロードに成功しました。")
            
            # モデルが新しい [PAD] トークンに対応していることを確認
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
        except Exception as e:
            print(f"\n❌ エラー: モデルのロードに失敗しました。パスを確認してください: '{MODEL_DIR}'")
            print(f"詳細: {e}")
            sys.exit(1)

        print("\n--- チャットモードを開始 ---")
        print("モデルに文字列を入力してください ('exit'で終了)。")

        # 3. チャットループ
        while True:
            # ユーザー入力の取得
            user_input = input("You: ")
            
            if user_input.lower() == 'exit':
                print("チャットを終了します。")
                break

            # 4. トークン化と生成
            with torch.no_grad():
                # プロンプトをトークン化し、GPUへ送る
                input_ids = tokenizer.encode(user_input, return_tensors='pt', truncation=True).to(self.DEVICE)
                
                # Greedy Decoding で生成
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=self.MAX_NEW_TOKENS,
                    do_sample=False, 
                    num_beams=1,
                    pad_token_id=tokenizer.pad_token_id
                )

                # 5. デコードと表示
                generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
                
                # 元の入力部分を削除して、モデルが生成した続きのみを表示
                response = generated_text[len(user_input):].strip()
                
                # ★★★ PPLの計算 ★★★
                # PPLは、ユーザー入力 + モデル生成部分の「全体」に対して計算します。
                ppl_value = self.calculate_ppl(model, tokenizer, generated_text, self.DEVICE)
                
                # 6. 結果の表示 (PPLをカッコ書きで追加)
                print(f"Model: {response} ({ppl_value:.4f})")

if __name__ == '__main__':
    obj = CHAT_WITH_MODEL(model_base_dir='model/')
    obj.load_and_chat()