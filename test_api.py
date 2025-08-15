import requests
import json

# API 的 URL
API_URL = "http://127.0.0.1:8000/recommend"

# 請求的 Body
request_data = {
  "case_text": "一件關於捷運工程採購弊案，可能涉及官商勾結與綁標。",
  "expertise_weight": 0.7,
  "workload_weight": 0.3,
  "top_n": 3
}

def test_recommendation_api():
    """
    測試 /recommend API 端點。
    """
    print(f"正在向 {API_URL} 發送請求...")
    print("請求內容:")
    print(json.dumps(request_data, indent=2, ensure_ascii=False))
    
    try:
        response = requests.post(API_URL, json=request_data)
        
        # 檢查回應狀態碼
        if response.status_code == 200:
            print("\n請求成功！")
            print("API 回應:")
            recommendations = response.json()
            print(json.dumps(recommendations, indent=2, ensure_ascii=False))
        else:
            print(f"\n請求失敗，狀態碼: {response.status_code}")
            print("錯誤訊息:")
            print(response.text)
            
    except requests.exceptions.ConnectionError as e:
        print(f"\n無法連接到伺服器: {e}")
        print("請確認您已經在另一個終端機中執行了以下命令來啟動伺服器:")
        print(".venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8000")

if __name__ == "__main__":
    test_recommendation_api()