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
import seaborn as sns
from sklearn.decomposition import TruncatedSVD

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

cleaned_words=[w for w in words if len(w)>=3]
print("cleaned words "+str(len(cleaned_words)))
cleaned_words1 = [w for w in cleaned_words if not re.search(r'[aeiou]{3,}', w)]
print("cleaned words1 "+str(len(cleaned_words1)))

stop_words = set(stopwords.words("english"))

filtered_words = [w for w in cleaned_words1 if w not in stop_words]

print("filtered word "+str(len(filtered_words)))

word_freq = Counter(filtered_words)
threshold = 5  # 高频词阈值
#important_words = {word for word, freq in word_freq.items() if freq > threshold}
important_words = [word for word in filtered_words if word_freq[word] > 25]
print("important word "+str(len(important_words)))
print("print(important_words[:10]) "+str(important_words[:10]))





important_words_list = list(important_words)
window_size = 5 #How many words in sequence to consider to be in the window
# Create a list of co-occurring word pairs
co_occurrences = defaultdict(Counter)
for i, word in enumerate(important_words):
    for j in range(max(0, i - window_size), min(len(important_words), i + window_size + 1)):
        if i != j and word != important_words[j]:
            co_occurrences[word][important_words[j]] += 1

# Create a list of unique words
unique_words = list(set(important_words))
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

tsne = TSNE(n_components=2, random_state=42)
mat_array_tsne = tsne.fit_transform(mat_array)
# Using sklearn
km = KMeans(n_clusters=5)
km.fit(mat)
# Get cluster assignment labels
labels = km.labels_
print(len(labels))





# 绘制散点图
plt.figure(figsize=(12, 10))


plt.scatter(mat_array_tsne[:, 0], mat_array_tsne[:, 1], c='blue', s=10)




# 设置图标题和标签
plt.title("Co-occurrence Matrix Scatter Plot", fontsize=16)
plt.xlabel("Words (Feature1)", fontsize=14)
plt.ylabel("Words (Feature2)", fontsize=14)

plt.savefig(f'co-occurrence.png')

# 显示图形
plt.show()



# Format results as a DataFrame
k_values = range(2, 10)  # 从 2 到 10 尝试不同的簇数


inertias = []


# 计算每个 K 值的惯性
for k in k_values:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(mat_array_tsne)  # 使用共现矩阵
    inertias.append(km.inertia_)

inertia_diff = np.diff(inertias)  # 求出相邻两个 Inertia 之间的差异
inertia_diff2 = np.diff(inertia_diff)  # 求出差异的差异（加速/减速）
elbow_point = np.argmax(inertia_diff2) + 1

# 绘制肘部法则图
plt.plot(k_values, inertias, marker='o')
plt.title('Elbow Method for Optimal K')
plt.xlabel('Number of clusters (K)')
plt.ylabel('Inertia')
plt.axvline(x=k_values[elbow_point], color='red', linestyle='--', label=f'Elbow at k={k_values[elbow_point]}')
plt.savefig('elbow_method_plot3.png')
plt.show()
print(f"the best k from Elbow Method is: {k_values[elbow_point]}")


#print("print(mat_array.shape) "+str(mat_array.shape))
silhouette_scores = []


for k in k_values:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(mat_array_tsne)  # 用 co-occurrence matrix 进行聚类
    labels = km.labels_

    plt.figure(figsize=(8, 6))

    # 为每个簇绘制不同颜色的点
    for i in range(k):
        plt.scatter(mat_array[labels == i, 0], mat_array[labels == i, 1], label=f'Cluster {i + 1}', s=50)

    # 绘制簇的中心
    centers = km.cluster_centers_
    plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200, marker='*', label='Centroids')

    # 添加标题和标签
    plt.title(f'K-means Clustering (k={k})', fontsize=14)
    plt.xlabel('Feature 1', fontsize=12)
    plt.ylabel('Feature 2', fontsize=12)
    plt.legend()

    # 显示图形

    plt.savefig(f'k_means_clustering_k_{k}.png')  # 保存为PNG格式，文件名包括k的值
    plt.show()
    plt.close()  # 关闭当前图形，避免显示多张图

    silhouette_avg = silhouette_score(mat_array, labels)
    silhouette_scores.append(silhouette_avg)
    print(f"For k={k}, silhouette score: {silhouette_avg}")

# 找到轮廓系数最大时的 k 值
k_optimal_silhouette = k_values[silhouette_scores.index(max(silhouette_scores))]
print(f"the best k for silhouette: {k_optimal_silhouette} ,with {max(silhouette_scores)}")

