from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Any
import os

from .expertise_engine import ExpertiseEngine
from .workload_analyzer import WorkloadAnalyzer
from .recommendation_engine import RecommendationEngine

# --- 全域變數 ---
# 在應用程式啟動時，我們會初始化這些引擎。
# 這樣可以避免每次請求都重新載入模型。
recommendation_engine: RecommendationEngine = None

# --- FastAPI 應用程式實例 ---
app = FastAPI(
    title="智慧派案推薦系統 API",
    description="此 API 根據案件內容和調查人員的工作負荷，提供派案推薦。",
    version="1.0.0",
)

# --- 生命週期事件 ---
@app.on_event("startup")
def startup_event():
    """
    應用程式啟動時執行的事件。
    初始化所有必要的引擎和分析器。
    """
    global recommendation_engine
    print("伺服器啟動中，開始初始化引擎...")
    
    # 假設 person_excel 資料夾在專案根目錄
    data_path = 'person_excel'
    if not os.path.exists(data_path):
        raise RuntimeError(f"找不到資料目錄: {data_path}。請確保資料夾位於專案根目錄。")

    try:
        artifacts_path = 'artifacts'
        if not all(os.path.exists(os.path.join(artifacts_path, f)) for f in ['cases_df.pkl', 'investigator_profiles.pkl']):
             raise RuntimeError(f"找不到必要的 artifacts。請先執行 'python build_artifacts.py' 來生成 artifacts。")

        expertise_engine = ExpertiseEngine()
        expertise_engine.load_artifacts(artifacts_path)
        
        workload_analyzer = WorkloadAnalyzer(expertise_engine.cases_df)
        
        recommendation_engine = RecommendationEngine(expertise_engine, workload_analyzer)
        print("引擎初始化完成，伺服器準備就緒。")
    except Exception as e:
        print(f"引擎初始化失敗: {e}")
        # 在這種情況下，伺服器仍然會啟動，但 /recommend 端點會返回錯誤。
        recommendation_engine = None


# --- Pydantic 模型 (用於請求和回應) ---
class RecommendRequest(BaseModel):
    case_text: str = Field(..., min_length=10, description="新案件的案由描述。")
    expertise_weight: float = Field(0.7, ge=0, le=1, description="專業匹配度權重。")
    workload_weight: float = Field(0.3, ge=0, le=1, description="工作負荷權重。")
    top_n: int = Field(3, ge=1, le=20, description="要返回的推薦人選數量。")

class Recommendation(BaseModel):
    rank: int
    name: str
    final_score: float
    expertise_score: float
    workload_score: float
    explanation: str

# --- API 端點 ---
@app.post("/recommend", response_model=List[Recommendation])
def get_recommendations(request: RecommendRequest):
    """
    接收案件內容，返回推薦的調查人員列表。
    """
    if recommendation_engine is None:
        raise HTTPException(
            status_code=503, 
            detail="伺服器正在初始化或初始化失敗，請稍後再試。"
        )
    
    # 權重加總應接近 1
    if not (0.99 < request.expertise_weight + request.workload_weight < 1.01):
        raise HTTPException(
            status_code=400,
            detail="權重 (expertise_weight + workload_weight) 總和應為 1。"
        )

    try:
        recommendations = recommendation_engine.recommend(
            new_case_text=request.case_text,
            w_exp=request.expertise_weight,
            w_load=request.workload_weight,
            top_n=request.top_n
        )
        return recommendations
    except Exception as e:
        # 捕捉推薦過程中可能發生的任何錯誤
        raise HTTPException(status_code=500, detail=f"推薦過程中發生錯誤: {str(e)}")

@app.get("/")
def read_root():
    return {"message": "歡迎使用智慧派案推薦系統 API。請訪問 /docs 來查看 API 文件。"}