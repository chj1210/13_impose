import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from typing import List, Dict, Any

from .expertise_engine import ExpertiseEngine
from .workload_analyzer import WorkloadAnalyzer

class RecommendationEngine:
    """
    整合專業畫像和工作負荷，提供推薦人選。
    """
    def __init__(self, expertise_engine: ExpertiseEngine, workload_analyzer: WorkloadAnalyzer):
        """
        初始化推薦引擎。

        Args:
            expertise_engine: 已初始化的 ExpertiseEngine 實例。
            workload_analyzer: 已初始化的 WorkloadAnalyzer 實例。
        """
        self.expertise_engine = expertise_engine
        self.workload_analyzer = workload_analyzer
        
        if self.expertise_engine.investigator_profiles is None:
            raise ValueError("ExpertiseEngine 尚未建立專業畫像。請先執行 build_profiles()。")

    def _calculate_expertise_scores(self, new_case_embedding: np.ndarray) -> Dict[str, float]:
        """
        計算新案件與每位調查人員的專業匹配度分數。
        """
        profiles = self.expertise_engine.investigator_profiles
        investigator_names = list(profiles.keys())
        profile_vectors = np.array(list(profiles.values()))
        
        # 計算餘弦相似度
        similarities = cosine_similarity(new_case_embedding.reshape(1, -1), profile_vectors)
        
        scores = {name: float(score) for name, score in zip(investigator_names, similarities[0])}
        return scores

    def recommend(self, new_case_text: str, w_exp: float, w_load: float, top_n: int) -> List[Dict[str, Any]]:
        """
        根據新案件文本，生成推薦人選列表。

        Args:
            new_case_text: 新案件的案由文本。
            w_exp: 專業匹配度權重。
            w_load: 工作負荷權重。
            top_n: 要返回的推薦人選數量。

        Returns:
            一個排序後的推薦人選列表。
        """
        # 1. 新案件向量化
        new_case_embedding = self.expertise_engine.get_embedding(new_case_text)
        
        # 2. 計算專業匹配度分數
        expertise_scores = self._calculate_expertise_scores(new_case_embedding)
        
        # 3. 計算工作負荷分數並合併
        recommendations = []
        current_date = datetime.now()
        
        for name, exp_score in expertise_scores.items():
            workload_score = self.workload_analyzer.get_normalized_score(name, current_date)
            
            # 4. 計算最終分數
            final_score = (w_exp * exp_score) + (w_load * (1 - workload_score))
            
            recommendations.append({
                "name": name,
                "final_score": float(final_score),
                "expertise_score": float(exp_score),
                "workload_score": float(workload_score)
            })
            
        # 5. 排序
        sorted_recommendations = sorted(recommendations, key=lambda x: x['final_score'], reverse=True)
        
        # 6. 添加排名和推薦理由
        final_list = []
        for i, rec in enumerate(sorted_recommendations[:top_n]):
            rec['rank'] = i + 1
            rec['explanation'] = (
                f"專業領域契合度: {rec['expertise_score']:.2%}，"
                f"當前工作負荷: {'低' if rec['workload_score'] < 0.5 else '高'} ({rec['workload_score']:.2%})。"
            )
            final_list.append(rec)
            
        return final_list

if __name__ == '__main__':
    from data_loader import load_and_preprocess_data

    try:
        print("正在初始化所有引擎...")
        data_path = 'person_excel'
        
        # 初始化 ExpertiseEngine 並建立畫像
        expertise_engine = ExpertiseEngine()
        expertise_engine.build_profiles(data_path)
        
        # 初始化 WorkloadAnalyzer
        workload_analyzer = WorkloadAnalyzer(expertise_engine.cases_df)
        
        # 初始化 RecommendationEngine
        recommendation_engine = RecommendationEngine(expertise_engine, workload_analyzer)
        
        print("\n引擎初始化完成，開始進行推薦測試。")
        
        # 測試推薦
        new_case = "一件關於上市公司內部交易和財務報表不實的經濟犯罪案件。"
        recommendations = recommendation_engine.recommend(
            new_case_text=new_case,
            w_exp=0.7,
            w_load=0.3,
            top_n=3
        )
        
        print(f"\n對於案件: '{new_case}'")
        print("推薦人選如下:")
        for rec in recommendations:
            print(
                f"  Rank {rec['rank']}: {rec['name']} "
                f"(綜合分數: {rec['final_score']:.4f}, "
                f"專業分數: {rec['expertise_score']:.4f}, "
                f"負荷分數: {rec['workload_score']:.4f})"
            )
            print(f"    推薦理由: {rec['explanation']}")

    except Exception as e:
        print(f"發生錯誤: {e}")