import pandas as pd
import numpy as np
from datetime import datetime

class WorkloadAnalyzer:
    """
    分析調查人員的工作負荷。
    """
    def __init__(self, cases_df: pd.DataFrame):
        """
        初始化分析器。

        Args:
            cases_df: 包含所有案件資訊的 DataFrame。
                      需要 '派查日期', '報告日期', '調查委員', '協查人員' 欄位。
        """
        if cases_df is None or cases_df.empty:
            raise ValueError("案件 DataFrame 不能為空。")
        self.cases_df = cases_df
        self._prepare_data()

    def _prepare_data(self):
        """
        準備資料以進行分析，例如確保日期格式正確。
        """
        self.cases_df['派查日期'] = pd.to_datetime(self.cases_df['派查日期'])
        self.cases_df['報告日期'] = pd.to_datetime(self.cases_df['報告日期'])

    def _get_person_cases(self, investigator_name: str) -> pd.DataFrame:
        """
        獲取指定人員參與的所有案件。
        """
        return self.cases_df[
            self.cases_df['調查委員'].str.contains(investigator_name, na=False) |
            self.cases_df['協查人員'].str.contains(investigator_name, na=False)
        ]

    def calculate_current_load(self, investigator_name: str, current_date: datetime) -> int:
        """
        計算指定人員在特定日期下，正在處理中的案件數量。
        """
        person_cases = self._get_person_cases(investigator_name)
        if person_cases.empty:
            return 0
        
        current_date = pd.to_datetime(current_date)
        
        # 案件正在處理中，代表 current_date 介於 派查日期 和 報告日期 之間
        in_progress = person_cases[
            (person_cases['派查日期'] <= current_date) & 
            (person_cases['報告日期'] >= current_date)
        ]
        return len(in_progress)

    def calculate_avg_duration(self, investigator_name: str) -> float:
        """
        計算指定人員所有已完成案件的平均辦理時長（天）。
        """
        person_cases = self._get_person_cases(investigator_name)
        if person_cases.empty:
            return 0.0
        
        # 計算每個案件的時長
        durations = (person_cases['報告日期'] - person_cases['派查日期']).dt.days
        return durations.mean()

    def get_normalized_score(self, investigator_name: str, current_date: datetime) -> float:
        """
        計算標準化的工作負荷分數 (0-1)。分數越高代表越繁忙。
        這裡使用一個簡單的正規化方法：基於當前案件數。
        未來可以擴展以包含更多指標。
        """
        current_load = self.calculate_current_load(investigator_name, current_date)
        
        # 簡單的正規化：假設同時處理超過5件案子就算滿載
        # 這是一個可以調整的參數
        max_load_threshold = 5.0
        
        normalized_score = min(current_load / max_load_threshold, 1.0)
        return normalized_score

if __name__ == '__main__':
    from data_loader import load_and_preprocess_data

    try:
        data_path = 'person_excel'
        cases_df, investigator_list = load_and_preprocess_data(data_path)
        
        analyzer = WorkloadAnalyzer(cases_df)
        
        # 測試一位調查人員
        test_investigator = investigator_list[10] # 隨機選一位
        today = datetime.now()
        
        current_load = analyzer.calculate_current_load(test_investigator, today)
        avg_duration = analyzer.calculate_avg_duration(test_investigator)
        normalized_score = analyzer.get_normalized_score(test_investigator, today)
        
        print(f"測試調查人員: {test_investigator}")
        print(f"當前日期: {today.strftime('%Y-%m-%d')}")
        print(f"進行中的案件數量: {current_load}")
        print(f"歷史平均辦案時長 (天): {avg_duration:.2f}")
        print(f"標準化工作負荷分數: {normalized_score:.2f}")

    except (FileNotFoundError, ValueError) as e:
        print(f"錯誤: {e}")