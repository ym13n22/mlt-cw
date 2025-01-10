import nltk, re, pprint
from nltk import word_tokenize
from collections import defaultdict, Counter
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import scipy
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from nltk.stem import PorterStemmer
from sklearn.decomposition import NMF
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler
from scipy import stats
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import Normalizer

import seaborn as sns
from sklearn.decomposition import TruncatedSVD
nltk.download('punkt')
nltk.download('punkt_tab') 
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger_eng')


num_words = {
    'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven', 'twelve', 'zero', # Cardinal numbers
    'first', 'second', 'third', 'fourth', 'fifth', 'sixth', 'seventh', 'eighth', 'ninth', 'tenth',  # Ordinal numbers
    'meter', 'kilometer', 'liter', 'gram', 'kilogram', 'second', 'minute', 'hour', 'day', 'year',  # Units of measurement
    'and','or','but','because','also','so', 'for', 'yet', 'although', 'because', 'since', 'unless', 'until', 'while', 'if', 'though', 'once', 'when','fine','also'
}
# 常见单位
units = {'million', 'billion', 'kg', 'g', 'lb', 'm', 'cm', 'km', 'hour', 'minute', 'second', 'percent'}

with open("text8/text8","r")as f:
    data =f.readline()

type(data)
print(type(data))
print(len(data))
print(data[:75])
tokens=word_tokenize(data)
type(tokens)
print("type(tokens) " + str(type(tokens)))
print("len(tokens) "+str(len(tokens)))
print("tokens[:10] "+str(tokens[:10]))
text=nltk.Text(tokens)
print("type(text) "+str(type(text)))
print("len(text) "+str(len(text)))
print("text[:10]"+str(text[:10]))

#build a vocabulary
words=[w.lower() for w in tokens]
vocab=set(set(words))
print("vocab "+str(len(vocab)))

cleaned_words=[w for w in words if 20>len(w)>=3]
print("cleaned words "+str(len(cleaned_words)))
cleaned_words1 = [w for w in cleaned_words if not re.search(r'[aeiou]{3,}', w)]
print("cleaned words1 "+str(len(cleaned_words1)))

stop_words = set(stopwords.words("english"))

filtered_words = [w for w in cleaned_words1 if w not in stop_words]

print("filtered word "+str(len(filtered_words)))

nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()

# 逐一处理 `filtered_words` 的词性和词形还原
tagged_words = pos_tag(filtered_words)

lemmatized_words = []

for word, tag in tagged_words:
    if tag.startswith('J'):
        wordnet_pos = wordnet.ADJ
    elif tag.startswith('V'):
        wordnet_pos = wordnet.VERB
    elif tag.startswith('N'):
        wordnet_pos = wordnet.NOUN
    elif tag.startswith('R'):
        wordnet_pos = wordnet.ADV
    else:
        wordnet_pos = None

    if wordnet_pos:
        lemmatized_word = lemmatizer.lemmatize(word, wordnet_pos)
    else:
        lemmatized_word = lemmatizer.lemmatize(word)

    lemmatized_words.append(lemmatized_word)

print(f"Lemmatized words count: {len(lemmatized_words)}")

#stemmer = PorterStemmer()
#tagged_words = pos_tag(filtered_words)
#nouns = [word for word, tag in tagged_words if tag.startswith('NN')]
#print("nouns "+str(len(nouns)))
#stems = [stemmer.stem(noun) for noun in filtered_words]

print("stems "+str(len(lemmatized_words)))

cleaned_tokens = []

for token in lemmatized_words:
    # 去掉纯数字
    if token.isdigit():
        continue
    # 去掉数字文字形式（如one, two, three等）
    elif token.lower() in num_words:
        continue
    # 去掉单位
    elif token.lower() in units:
        continue
    # 其他有效词保留
    else:
        cleaned_tokens.append(token)


print("cleaned_tokens "+str(len(cleaned_tokens)))

final_nouns = []

for token, tag in nltk.pos_tag(cleaned_tokens):  # 使用 POS 标注器对清理后的 token 进行标注
    #if tag.startswith('N'):
    if tag.startswith('N') or tag.startswith('V') or tag.startswith('J'):
        final_nouns.append(token)

print("Final nouns: ", str(len(final_nouns)))
word_freq = Counter(final_nouns)
# 将词频按从小到大排序
sorted_word_freq = sorted(word_freq.items(), key=lambda x: x[1])

# 计算中间段的起始和结束位置
n = len(sorted_word_freq)
start = n*916 // 1000  # 从四分之一位置开始
end = n * 999999 // 1000000  # 到四分之三位置

# 取中间段
middle_segment = sorted_word_freq[start:end]

# 打印中间段的第一个和最后一个词及其词频
if middle_segment:  # 确保中间段不为空
    first_word, first_freq = middle_segment[0]
    last_word, last_freq = middle_segment[-1]
    print(f"First: {first_word}: {first_freq}")
    print(f"Last: {last_word}: {last_freq}")
else:
    print("Middle segment is empty.")

middle_words = set(word for word, _ in middle_segment)
print("middle_words "+str(len(middle_words)))
filtered_tokens = [word for word in final_nouns if word in middle_words]
print("filtered_tokens "+str(len(filtered_tokens)))
min_freq = 30  # 最小词频
max_freq = 10000  # 最大词频
'''
important_words = [word for word in cleaned_tokens if min_freq < word_freq[word]]
print("important word "+str(len(important_words)))
print("important word set "+str(len(set(important_words))))
print("print(important_words[:10]) "+str(important_words[:10]))

important_words_text = " ".join(important_words)

# 创建一个 TF-IDF 向量化器
tfidf_vectorizer = TfidfVectorizer()

# 将 important_words_text 转换为一个文档列表（只有一个文档）
tfidf_matrix = tfidf_vectorizer.fit_transform([important_words_text])

# 获取词汇表中的所有特征名称（单词）
feature_names = tfidf_vectorizer.get_feature_names_out()

# 将 TF-IDF 矩阵转换为稠密格式，方便查看每个词的值
dense_tfidf = tfidf_matrix.todense()

# 获取第一个（也是唯一一个）文档的 TF-IDF 值
tfidf_values = dense_tfidf.tolist()[0]

# 创建一个词汇与其对应 TF-IDF 值的字典
word_tfidf = dict(zip(feature_names, tfidf_values))

# 输出每个重要单词的 TF-IDF 值
#print("Important words and their TF-IDF values:")
#for word, tfidf_val in word_tfidf.items():
 #   print(f"{word}: {tfidf_val}")

# 设置一个阈值
threshold = 0.0015  # 可以根据需要调整这个值，较高的值会留下重要的单词

# 筛选出 TF-IDF 值高于阈值的单词
important_tfidf_words = [word for word, tfidf_val in word_tfidf.items() if tfidf_val > threshold]

# 输出筛选后的重要单词
print(f"重要单词数: {len(important_tfidf_words)}")
print(f"重要单词: {important_tfidf_words[:10]}")  # 查看前10个重要单词

# 输出筛选出的重要单词

important_words = [word for word in important_words if word in important_tfidf_words]
print("Important words after TF-IDF filtering:", str(len(important_words)))

'''







#Let's build cooccurrence counts
#window_size = 25 #How many words in sequence to consider to be in the window
# Create a list of co-occurring word pairs
co_occurrences = defaultdict(Counter)
window_size = max(5, min(25, len(filtered_tokens) // 2))
for i, word in enumerate(filtered_tokens):
    for j in range(max(0, i - window_size), min(len(filtered_tokens), i + window_size + 1)):
        if i != j:
            distance = abs(i - j)
            weight = 1 / (distance + 1)  # 使用距离的反比作为权重
            co_occurrences[word][filtered_tokens[j]] += weight

# Create a list of unique words
unique_words = list(set(filtered_tokens))
# Initialize the co-occurrence matrix
co_matrix = np.zeros((len(unique_words), len(unique_words)), dtype=int)

# Populate the co-occurrence matrix
word_index = {word: idx for idx, word in enumerate(unique_words)}
for word, neighbors in co_occurrences.items():
    for neighbor, count in neighbors.items():
        co_matrix[word_index[word]][word_index[neighbor]] = count


co_matrix_df = pd.DataFrame(co_matrix, index=unique_words, columns=unique_words)

print(co_matrix_df)

#Convert the above matrix to sparse representation, saves memory
print(scipy.sparse.csr_matrix(co_matrix_df))



non_zero_count = np.count_nonzero(co_matrix)
print(f"Non-zero elements in the matrix: {non_zero_count}")

# Convert DataFrame to matrix
mat = co_matrix_df
mat_array = mat.values


inverse_cos_sim = 1 - cosine_similarity(mat_array)

print("inverse_cos_sim :", inverse_cos_sim)

# 标准化操作
mean_val = inverse_cos_sim.mean()
std_val = inverse_cos_sim.std()

# 标准化：每个元素减去均值并除以标准差
standardized_inverse_cos_sim = (inverse_cos_sim - mean_val) / std_val

# 归一化
min_val = standardized_inverse_cos_sim.min()
max_val = standardized_inverse_cos_sim.max()

# 归一化：缩放到 [0, 1] 范围
normalized_inverse_cos_sim = (standardized_inverse_cos_sim - min_val) / (max_val - min_val)

#cosine_sim_normalized = (cosine_sim + 1)/2
#print("cosine_sim normalized:", cosine_sim_normalized)
#threshold = 0.2
#upper_limit=0.8
#cosine_sim_normalized[cosine_sim_normalized < threshold] = 0
#cosine_sim_normalized[cosine_sim_normalized > upper_limit] = upper_limit

#scaler = MinMaxScaler(feature_range=(0, 1))
#cosine_sim_normalized = scaler.fit_transform(cosine_sim_normalized)



#scaler = StandardScaler()
#standardized_array = scaler.fit_transform(cosine_sim)





#nmf = NMF(n_components=50, max_iter=200, random_state=42)
#nmf_result = nmf.fit_transform(mat_array)
#print("nmf_result.shape",str(mat_array.shape))
#tsne = TSNE(n_components=2, random_state=42)
#mat_array_tsne = tsne.fit_transform(mat_array)
svd = TruncatedSVD(n_components=100, random_state=42)
mat_array_svd = svd.fit_transform(normalized_inverse_cos_sim)
print("mat_array_svd.shape",str(mat_array_svd.shape))
tsne = TSNE(n_components=2, random_state=42)
mat_array_tsne = tsne.fit_transform(mat_array_svd)
# Using sklearn
#km = KMeans(n_clusters=5)
#km.fit(mat)
# Get cluster assignment labels
#labels = km.labels_
#print(len(labels))





# 绘制散点图
plt.figure(figsize=(12, 10))


plt.scatter(mat_array_tsne[:, 0], mat_array_tsne[:, 1], c='blue', s=10,alpha=0.6)




# 设置图标题和标签
plt.title("Co-occurrence Matrix Scatter Plot", fontsize=16)
plt.xlabel("Words (Feature1)", fontsize=14)
plt.ylabel("Words (Feature2)", fontsize=14)

plt.savefig(f'co-occurrence.png')

# 显示图形
plt.show()

# 定义候选的 k 值范围
k_values = range(2, 26)  # 尝试不同的 k 值
kf = KFold(n_splits=10, shuffle=True, random_state=42)  # 5折交叉验证
inertias = []  # 存储惯性
silhouette_scores_cv = []  # 存储轮廓系数
tsne = TSNE(n_components=2, random_state=42, perplexity=30, n_iter=1000)  # t-SNE 参数

# 预先将高维数据降维到2D
X_2d = tsne.fit_transform(normalized_inverse_cos_sim)

# 遍历 k 值进行聚类分析
for k in k_values:
    print("start with K: "+str(k))
    km_inertias = []  # 储存当前 k 的所有折叠的惯性
    km_silhouette_scores = []  # 储存当前 k 的所有折叠的轮廓系数

    for train_idx, test_idx in kf.split(normalized_inverse_cos_sim):
        # 按索引划分训练集和验证集
        X_train, X_test = normalized_inverse_cos_sim[train_idx],normalized_inverse_cos_sim[test_idx]

        # KMeans 聚类
        km = KMeans(n_clusters=k,tol=1e-4,max_iter=1000, random_state=42)
        km.fit(X_train)
        labels = km.predict(X_test)  # 对验证集进行预测

        if len(np.unique(labels)) <= 1:
            print(f"Skipping silhouette score calculation for k={k}, single cluster detected.")
            continue  # 跳过单簇情况

        # 计算验证集的惯性和轮廓系数
        inertia = km.inertia_
        silhouette_avg = silhouette_score(X_test, labels)
        print(f"current silhouette score with {k} is : ", silhouette_avg)

        km_inertias.append(inertia)
        km_silhouette_scores.append(silhouette_avg)

    # 计算当前 k 值的平均惯性和平均轮廓系数
    inertias.append(np.mean(km_inertias))
    silhouette_scores_cv.append(np.mean(km_silhouette_scores))
    print(f"average silhouette score with {k} is : ", silhouette_scores_cv)

    # 在二维空间中可视化聚类
    labels_full_data = KMeans(n_clusters=k, random_state=42).fit_predict(normalized_inverse_cos_sim)
    plt.figure(figsize=(10, 8))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=labels_full_data, cmap='viridis', s=10, alpha=0.8)
    plt.colorbar()
    plt.title(f"t-SNE Visualization of Clustering (k={k})")
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.savefig(f"tsne_clustering_k_{k}.png")  # 保存图像
    plt.show()

# 找到轮廓系数最大时的 k 值
k_optimal_silhouette = k_values[np.argmax(silhouette_scores_cv)]
terms = unique_words

# 绘制肘部法则图和轮廓系数曲线
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(k_values, inertias, marker='o')
plt.title("Elbow Method with Cross-Validation")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Average Inertia")

plt.subplot(1, 2, 2)
plt.plot(k_values, silhouette_scores_cv, marker='o')
plt.title("Silhouette Score with Cross-Validation")
plt.xlabel("Number of clusters (k)")
plt.ylabel("Average Silhouette Score")

plt.tight_layout()
plt.show()

# 输出最优结果
print(f"The best k for silhouette score is: {k_optimal_silhouette}, with silhouette score: {max(silhouette_scores_cv)}")
kmeans_final = KMeans(n_clusters=k_optimal_silhouette, random_state=42)
labels_final = kmeans_final.fit_predict(normalized_inverse_cos_sim)

# 构建簇中包含词条的信息
cluster_words = {i: [] for i in range(k_optimal_silhouette)}

# 假设 `terms` 是词汇表的列表或数组
for i, label in enumerate(labels_final):
    cluster_words[label].append(terms[i])

# 输出每个簇中词的数量和前 10 个词
for cluster_id, words in cluster_words.items():
    print(f"Cluster {cluster_id}:")
    print(f"  Number of words: {len(words)}")
    print(f"  Top 10 words: {words[:10]}")