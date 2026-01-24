import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

# ==========================================
# 第一部分：全局变量与配置
# ==========================================
# 运算符映射表
d = {"and": "&", "AND": "&",
     "or": "|", "OR": "|",
     "not": "1 -", "NOT": "1 -",
     "(": "(", ")": ")"}

# 这些变量将在 main 函数中被填充，设为全局以便 rewrite_token 访问
documents = []
t2i = {}
sparse_td_matrix = None

# ==========================================
# 第二部分：数据加载 (基于 Task 1)
# ==========================================
def get_week1_documents():
    # 确保文件名和你截图里的完全一致
    file_path = "week1ylenews_2026-01-18.csv" 
    
    try:
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        
        # 数据清洗：处理空值，防止拼接时报错
        df["Time"] = df["Time"].fillna("")
        df["Category"] = df["Category"].fillna("")
        df["Headline"] = df["Headline"].fillna("")
        
        # 将三列合并成一个字符串作为“文档”
        df["content"] = "[" + df["Time"] + "] " + df["Category"] + ": " + df["Headline"]
        
        return df["content"].tolist()

    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return []
    except Exception as e:
        print(f"An error occurred while loading data: {e}")
        return []

# ==========================================
# 第三部分：搜索引擎核心逻辑 (基于 Task 2 & 3)
# ==========================================
def rewrite_token(t):
    # 1. 如果是操作符 (AND, OR, NOT, 括号)，直接返回对应的 Python 符号
    if t in d:
        return d[t]
    
    # 2. (Requirement 3) 处理未知词汇：如果词不在字典里
    if t not in t2i:
        # 返回一个全零向量（表示在该词在任何文档都没出现）
        # 我们生成一段代码字符串，让 eval() 去执行生成这个零向量
        return f'np.zeros((1, {len(documents)}), dtype=int)'
    
    # 3. 正常词汇：去矩阵里查
    return f'sparse_td_matrix[t2i["{t}"]].todense()'

def rewrite_query(query):
    # 将查询语句拆分，逐个单词进行重写
    return " ".join(rewrite_token(t) for t in query.split())

# ==========================================
# 第四部分：用户交互主程序 (你的 Task 4)
# ==========================================
def main():
    global documents, t2i, sparse_td_matrix # 声明使用全局变量

    # 1. 加载数据
    documents = get_week1_documents()
    if not documents:
        print("No documents loaded. Exiting.")
        return

    # 2. 创建索引 (Vectorization)
    print(f"Indexing {len(documents)} documents...")
    try:
        # token_pattern=r"(?u)\b\w+\b" 可以匹配包含数字的词 (Requirement 4)
        cv = CountVectorizer(lowercase=True, binary=True, token_pattern=r"(?u)\b\w+\b")
        sparse_matrix = cv.fit_transform(documents)
        
        # 获取词汇表 (t2i) 和 转置矩阵 (TD Matrix)
        t2i = cv.vocabulary_
        sparse_td_matrix = sparse_matrix.T.tocsr()
        print(f"Success! Vocabulary size: {len(t2i)} unique words.")
        
    except Exception as e:
        print(f"Indexing failed: {e}")
        return

    # 3. 启动交互循环 (Requirement 1)
    print("\n" + "="*60)
    print("Welcome to the Week 2 Search Engine!")
    print("Example queries: 'foreign', 'foreign AND policy', 'NOT education'")
    print("Type 'quit' or just press Enter to exit.")
    print("="*60)

    while True:
        user_input = input("\nEnter query > ").strip()

        # 退出条件
        if user_input == "" or user_input.lower() == 'quit':
            print("Goodbye!")
            break

        try:
            # 核心搜索步骤
            rewritten_command = rewrite_query(user_input)
            hits_matrix = eval(rewritten_command) # 执行布尔运算
            
            # 将结果矩阵转换为文档索引列表
            # hits_matrix 通常是 (1, N) 的矩阵，我们需要非零元素的列索引
            hits_list = list(np.array(hits_matrix)[0].nonzero()[0]) 
            # 注意：这里根据矩阵格式可能需要调整，如果报错可以用 hits_matrix.nonzero()[1]
            if isinstance(hits_matrix, np.matrix):
                 hits_list = list(np.array(hits_matrix).flatten().nonzero()[0])
            else:
                 hits_list = list(np.array(hits_matrix).flatten().nonzero()[0])

            total_hits = len(hits_list)
            print(f"Found {total_hits} matching documents.")

            # (Requirement 2) 打印结果，限制前5条
            if total_hits > 0:
                print(f"Showing top {min(total_hits, 5)} results:")
                for i, idx in enumerate(hits_list[:5]):
                    content = documents[idx]
                    # 截断长文本
                    snippet = content[:100] + "..." if len(content) > 100 else content
                    print(f"  {i+1}. {snippet}")
            else:
                print("No matches found.")

        except SyntaxError:
            print("Query Error: Please check your syntax (balanced brackets, valid operators).")
        except Exception as e:
            # 这里会捕获其他所有意外错误，保证程序不崩
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()