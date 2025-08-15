import pandas as pd
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
from typing import List, Dict
from bertopic import BERTopic
from .data_loader import load_and_preprocess_data

class ExpertiseEngine:
    """
    負責處理與調查人員專業畫像相關的所有計算。
    """
    def __init__(self, model_name: str = 'bert-base-chinese'):
        """
        初始化引擎，載入預訓練的 BERT 模型和 Tokenizer。

        Args:
            model_name: 要使用的預訓練模型名稱。
        """
        print("正在初始化 ExpertiseEngine...")
        self.device = torch.device("cpu")
        print(f"使用設備: {self.device}")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name).to(self.device)
        self.model.eval()  # 設置為評估模式
        print("模型載入完成。")
        
        self.cases_df = None
        self.investigator_list = None
        self.case_embeddings = None
        self.topic_model = None
        self.investigator_profiles = None

    def load_artifacts(self, artifacts_dir: str = 'artifacts'):
        """
        從 artifacts 資料夾載入預先計算好的物件。
        """
        print(f"正在從 '{artifacts_dir}' 載入 artifacts...")
        self.cases_df = pd.read_pickle(os.path.join(artifacts_dir, 'cases_df.pkl'))
        with open(os.path.join(artifacts_dir, 'investigator_list.pkl'), 'rb') as f:
            self.investigator_list = pickle.load(f)
        self.case_embeddings = np.load(os.path.join(artifacts_dir, 'case_embeddings.npy'))
        with open(os.path.join(artifacts_dir, 'investigator_profiles.pkl'), 'rb') as f:
            self.investigator_profiles = pickle.load(f)
        print("Artifacts 載入完成。")

    def get_embedding(self, text: str) -> np.ndarray:
        """
        為單一文本生成語義嵌入向量。

        Args:
            text: 輸入的文本。

        Returns:
            一個768維的 numpy 陣列。
        """
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512, padding=True).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        # 使用 [CLS] token 的輸出來代表整個句子的語義
        embedding = outputs.last_hidden_state[:, 0, :].cpu().numpy().flatten()
        return embedding

    def vectorize_all_cases(self, case_texts: List[str]) -> np.ndarray:
        """
        將所有案件文本轉換為語義嵌入向量。

        Args:
            case_texts: 包含所有案件文本的列表。

        Returns:
            一個 (案件數量 x 768) 的 numpy 陣列。
        """
        print(f"正在向量化 {len(case_texts)} 筆案件...")
        embeddings = [self.get_embedding(text) for text in case_texts]
        print("所有案件向量化完成。")
        return np.array(embeddings)

    def discover_topics(self, case_embeddings: np.ndarray, case_texts: List[str]) -> BERTopic:
        """
        使用 BERTopic 進行主題模型分析。

        Args:
            case_embeddings: 所有案件的語義嵌入向量。
            case_texts: 所有案件的原始文本。

        Returns:
            訓練好的 BERTopic 模型。
        """
        print("開始進行主題模型分析...")
        # 為了加速，可以考慮設定 min_topic_size 或 nr_topics
        topic_model = BERTopic(verbose=True, min_topic_size=5, embedding_model=self.model)
        topics, _ = topic_model.fit_transform(case_texts, case_embeddings)
        print("主題模型分析完成。")
        print(f"共發現 {len(topic_model.get_topic_info())} 個主題。")
        return topic_model

    def generate_investigator_profiles(self) -> Dict[str, np.ndarray]:
        """
        為每位調查人員生成專業畫像向量。

        Returns:
            一個字典，鍵為調查人員姓名，值為其專業畫像向量。
        """
        if self.cases_df is None or self.investigator_list is None or self.case_embeddings is None:
            raise ValueError("請先執行 build_profiles() 來載入和處理資料。")

        print("正在生成調查人員專業畫像...")
        profiles = {}
        for name in self.investigator_list:
            # 找出該人員參與過的所有案件的索引
            indices = self.cases_df[
                self.cases_df['調查委員'].str.contains(name, na=False) |
                self.cases_df['協查人員'].str.contains(name, na=False)
            ].index
            
            if len(indices) > 0:
                # 提取對應的案件向量並計算平均值
                person_embeddings = self.case_embeddings[indices]
                profile_vector = np.mean(person_embeddings, axis=0)
                profiles[name] = profile_vector
        
        print("專業畫像生成完成。")
        return profiles

    def build_profiles(self, data_dir: str):
        """
        執行完整的專業畫像建立流程。
        """
        # 1. 載入資料
        print("步驟 1/4: 載入並預處理資料...")
        self.cases_df, self.investigator_list = load_and_preprocess_data(data_dir)
        
        # 2. 向量化所有案件
        print("步驟 2/4: 向量化所有案件...")
        self.case_embeddings = self.vectorize_all_cases(self.cases_df['case_text'].tolist())
        
        # 3. 探索專業領域 (主題模型)
        print("步驟 3/4: 探索專業領域...")
        self.topic_model = self.discover_topics(self.case_embeddings, self.cases_df['case_text'].tolist())
        
        # 4. 生成調查人員專業畫像
        print("步驟 4/4: 生成專業畫像...")
        self.investigator_profiles = self.generate_investigator_profiles()
        
        print("\n專業畫像引擎建置完成！")


if __name__ == '__main__':
    engine = ExpertiseEngine()
    
    try:
        data_path = 'person_excel'
        engine.build_profiles(data_path)
        
        print("\n建置結果預覽:")
        print(f"總案件數量: {len(engine.cases_df)}")
        print(f"總調查人員數量: {len(engine.investigator_list)}")
        print(f"主題數量: {len(engine.topic_model.get_topic_info())}")
        
        print("\n主題關鍵詞 (前5個主題):")
        for i in range(min(5, len(engine.topic_model.get_topic_info()))):
            print(f"主題 {i-1}: {engine.topic_model.get_topic(i-1)}")

        print("\n專業畫像向量預覽 (隨機一位):")
        random_investigator = np.random.choice(list(engine.investigator_profiles.keys()))
        print(f"調查人員: {random_investigator}")
        print(f"畫像向量 (前10個值): {engine.investigator_profiles[random_investigator][:10]}")

    except (FileNotFoundError, ValueError) as e:
        print(f"錯誤: {e}")