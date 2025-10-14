import os
from verify_memorization import VERIFY_MEMORYZATION

folder_path = 'model'
file_list_with_dirs = os.listdir(folder_path)

print("---------------------------------------------------")

for file in file_list_with_dirs:
    print(file)
    if os.path.isdir(os.path.join(folder_path, file)):
        obj = VERIFY_MEMORYZATION(model_folder_input=file)
        model_folder_input, memorization_rate, avg_ppl = obj.verify()

        print(f"モデル: {model_folder_input}")
        print(f"暗記率: {memorization_rate:.2f}%")
        print(f"平均PPL: {avg_ppl:.4f}")
        print("---------------------------------------------------")