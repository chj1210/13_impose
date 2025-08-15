import pandas as pd
import os
import glob
from typing import List, Tuple

def load_and_preprocess_data(data_dir: str) -> Tuple[pd.DataFrame, List[str]]:
    """
    載入並預處理所有案件資料。

    Args:
        data_dir: 包含案件 Excel 檔案的目錄路徑。

    Returns:
        一個包含預處理後資料的 pandas DataFrame，以及一個不重複的調查人員名單。
    """
    all_files = glob.glob(os.path.join(data_dir, "*.xlsx"))
    if not all_files:
        raise FileNotFoundError(f"在 '{data_dir}' 目錄中找不到任何 .xlsx 檔案。")

    df_list = []
    for f in all_files:
        try:
            df = pd.read_excel(f)
            df_list.append(df)
        except Exception as e:
            print(f"讀取檔案 {f} 時發生錯誤: {e}")
            continue
    
    if not df_list:
        raise ValueError("沒有成功讀取任何 Excel 檔案。")

    cases_df = pd.concat(df_list, ignore_index=True)

    # 檢查必要欄位是否存在
    required_columns = ['案由', '調查意見', '派查日期', '報告日期', '調查委員', '協查人員']
    for col in required_columns:
        if col not in cases_df.columns:
            raise ValueError(f"資料中缺少必要欄位: {col}")

    # 2. 合併文本欄位
    cases_df['案由'] = cases_df['案由'].fillna('')
    cases_df['調查意見'] = cases_df['調查意見'].fillna('')
    cases_df['case_text'] = cases_df['案由'] + " " + cases_df['調查意見']

    # 3. 處理日期欄位 (民國年轉西元年)
    def convert_roc_date(date_str):
        if pd.isna(date_str) or not isinstance(date_str, str):
            return pd.NaT
        try:
            parts = date_str.split('/')
            year = int(parts[0]) + 1911
            return pd.to_datetime(f"{year}/{parts[1]}/{parts[2]}", errors='coerce')
        except (ValueError, IndexError):
            return pd.NaT

    cases_df['派查日期'] = cases_df['派查日期'].apply(convert_roc_date)
    cases_df['報告日期'] = cases_df['報告日期'].apply(convert_roc_date)
    
    # 丟棄日期轉換失敗的行
    cases_df.dropna(subset=['派查日期', '報告日期'], inplace=True)
    cases_df = cases_df.reset_index(drop=True)

    # 4. 提取調查人員名單
    def extract_names(series):
        names = set()
        series = series.dropna()
        for item in series:
            if isinstance(item, str):
                item = item.replace('、', ',').replace(' ', '')
                for name in item.split(','):
                    if name:
                        # 移除括號內的文字 (例如: (學習))
                        if '︵' in name:
                            name = name.split('︵')[0]
                        if '(' in name:
                            name = name.split('(')[0]
                        names.add(name.strip())
        return names

    investigators = set()
    investigators.update(extract_names(cases_df['調查委員']))
    investigators.update(extract_names(cases_df['協查人員']))

    return cases_df, sorted(list(investigators))

if __name__ == '__main__':
    # 假設 person_excel 在上一層目錄
    data_path = 'person_excel'
    try:
        processed_df, investigator_list = load_and_preprocess_data(data_path)
        print("資料預處理完成。")
        print(f"總案件數量: {len(processed_df)}")
        print(f"總調查人員數量: {len(investigator_list)}")
        print("="*20)
        print("前5筆資料預覽:")
        print(processed_df.head())
        print("="*20)
        print("\n日期欄位格式:")
        print(processed_df[['派查日期', '報告日期']].info())
        print("="*20)
        print("\n調查人員名單 (前20位):")
        print(investigator_list[:20])
    except (FileNotFoundError, ValueError) as e:
        print(f"錯誤: {e}")