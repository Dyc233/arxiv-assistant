import os
import sys
import sqlite3
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, split, regexp_extract, hour, to_timestamp, to_date, lower, regexp_replace, concat_ws, array_union, lit
from pyspark.ml.feature import Tokenizer, StopWordsRemover, NGram


#初始化环境
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

def process_arxiv_data():
    print("1. 正在初始化 Spark 集群...")
    spark = SparkSession.builder \
        .appName("ArxivDataProcess") \
        .master("local[*]") \
        .config("spark.executor.memory", "4g") \
        .config("spark.driver.memory", "4g") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .config("spark.sql.execution.arrow.pyspark.fallback.enabled", "true") \
        .config("spark.python.worker.memory", "2g") \
        .config("spark.sql.shuffle.partitions", "50") \
        .getOrCreate()

    print("2. 正在从 SQLite 读取原始数据...")
    conn = sqlite3.connect('D:/CODING/BS/data/arxiv_papers.db') 
    pdf = pd.read_sql_query("SELECT * FROM papers", conn)
    conn.close()

    print("3. 将数据转换为 Spark DataFrame 并立即重分区...")
    # 通过 Arrow 优化转换，并立即打散数据，避免单个 Task 过大
    df = spark.createDataFrame(pdf).repartition(50) 

    print("4. 开始执行核心清洗与特征工程...")
    
    df_clean = df.dropDuplicates(["id"])
    df_clean = df_clean.withColumn("published_ts", to_timestamp(col("published"))) \
                 .withColumn("publish_date", to_date(col("published_ts"))) \
                 .withColumn("publish_hour", hour(col("published_ts")))

    conference_regex = r"(ACL|EMNLP|NAACL|EACL|AACL|COLING|CVPR|ICLR|NeurIPS|ICML|IJCNLP|AAAI|IJCAI|SIGIR|WWW)\s*\d{4}"
    
    df_clean = df_clean.withColumn("top_conference", regexp_extract(col("comment"), conference_regex, 0)) \
                 .withColumn("author_list", split(col("authors"), r",\s*")) \
                 .withColumn("category_list", split(col("categories"), r",\s*")) \
                 .filter(col("summary").isNotNull() & col("title").isNotNull()) \
                 .withColumn("content_to_embed", concat_ws("\n", lit("Title:"), col("title"), lit("Summary:"), col("summary"))) # 拼接为 RAG 标准格式

# ================= N-Gram 词频与词组特征工程 =================
    # 3.1 提取纯净文本：去掉标点符号，全小写
    df_clean = df_clean.withColumn("pure_text", regexp_replace(lower(concat_ws(" ", col("title"), col("summary"))), "[^a-z\\s]", ""))

    # 3.2 基础分词
    tokenizer = Tokenizer(inputCol="pure_text", outputCol="raw_tokens")
    df_clean = tokenizer.transform(df_clean)

    # 3.3 去除停用词 (必须在 NGram 前做，否则会提取出 "of the" 这种垃圾词组)
    remover = StopWordsRemover(inputCol="raw_tokens", outputCol="filtered_words")
    df_clean = remover.transform(df_clean)

    # 3.4 生成 2-Gram (如: large language) 和 3-Gram (如: large language model)
    ngram2 = NGram(n=2, inputCol="filtered_words", outputCol="bigrams")
    df_clean = ngram2.transform(df_clean)

    ngram3 = NGram(n=3, inputCol="filtered_words", outputCol="trigrams")
    df_clean = ngram3.transform(df_clean)

    # 3.5 合并单词、双词和三词，生成最终的 keywords 列
    df_clean = df_clean.withColumn("keywords", array_union(col("filtered_words"), col("bigrams")))
    df_clean = df_clean.withColumn("keywords", array_union(col("keywords"), col("trigrams")))

    # 3.6 清理中间运算产生的临时列，保持 Parquet 干净小巧
    df_clean = df_clean.drop("pure_text", "raw_tokens", "filtered_words", "bigrams", "trigrams")

    print("5. 清洗完成，预览数据...")
    df_clean.select("id", "top_conference", "keywords").show(5, truncate=50)

    print("6. 正在执行持久化保存...")
    # 如果仍然崩溃，可以尝试先转回 Pandas 再保存，或者减小分区写入压力
    try:
        # 显式降低写入时的并发，防止 Windows 文件系统句柄用尽
        df_clean.coalesce(10).write.mode("overwrite").parquet("cleaned_papers.parquet")
        print("处理完毕！文件已生成：cleaned_papers.parquet")
    except Exception as e:
        print(f"写入失败，尝试备选方案: {e}")
        # 备选方案：如果本地 Spark 权限实在搞不定，可以用 pandas 写 parquet
        # pdf_final = df_clean.toPandas()
        # pdf_final.to_parquet("cleaned_papers.parquet")

    spark.stop()

if __name__ == "__main__":
    process_arxiv_data()