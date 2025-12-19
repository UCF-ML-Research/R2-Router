import json
import re
import ast
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Optional
import wandb
import pandas as pd
import pickle
from sklearn.metrics import confusion_matrix




def ultimate_json_score_extractor(data: Union[str, dict, None], 
                                 score_key: str = 'correctness_score') -> Optional[float]:
    """
    最完整版的JSON分数提取函数
    
    Args:
        data: 输入数据，可以是字符串、字典或None
        score_key: 要提取的键名，默认为'correctness_score'
    
    Returns:
        float: 提取的分数值，失败返回None
    """
    
    # === 第1层：基础验证 ===
    if data is None:
        return None
    
    # 如果已经是字典，直接提取
    if isinstance(data, dict):
        value = data.get(score_key)
        return float(value) if value is not None else None
    
    # 如果不是字符串，尝试转换
    if not isinstance(data, str):
        try:
            data = str(data)
        except:
            return None
    
    # === 第2层：字符串预处理 ===
    # 移除BOM和控制字符
    data = data.strip()
    if data.startswith('\ufeff'):  # UTF-8 BOM
        data = data[1:]
    if data.startswith('\ufffe') or data.startswith('\xef\xbb\xbf'):  # 其他BOM
        data = data[3:]
    
    # 移除多余的空白字符
    data = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', data)
    
    # === 第3层：标准JSON解析 ===
    try:
        parsed = json.loads(data)
        if isinstance(parsed, dict) and score_key in parsed:
            value = parsed[score_key]
            return float(value) if value is not None else None
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    
    # === 第3.5层：处理JSON后有额外内容的情况 ===
    # 尝试找到第一个完整的JSON对象
    if '{' in data and '}' in data:
        try:
            # 找到第一个{的位置
            start = data.find('{')
            # 从该位置开始，找到匹配的}
            brace_count = 0
            end = start
            for i, char in enumerate(data[start:], start):
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        end = i + 1
                        break
            
            if end > start:
                json_part = data[start:end]
                parsed = json.loads(json_part)
                if isinstance(parsed, dict) and score_key in parsed:
                    value = parsed[score_key]
                    return float(value) if value is not None else None
        except:
            pass
    
    # === 第4层：修复常见JSON问题 ===
    fixes = [
        # 修复单引号
        lambda x: x.replace("'", '"'),
        # 修复转义字符
        lambda x: x.replace('\\', '\\\\'),
        # 修复未转义的换行符
        lambda x: x.replace('\n', '\\n').replace('\r', '\\r').replace('\t', '\\t'),
        # 修复尾部逗号
        lambda x: re.sub(r',\s*}', '}', x),
        lambda x: re.sub(r',\s*]', ']', x),
        # 修复缺少引号的键
        lambda x: re.sub(r'([{,]\s*)([a-zA-Z_][a-zA-Z0-9_]*)\s*:', r'\\1"\\2":', x),
    ]
    
    for fix in fixes:
        try:
            fixed_data = fix(data)
            parsed = json.loads(fixed_data)
            if isinstance(parsed, dict) and score_key in parsed:
                value = parsed[score_key]
                return float(value) if value is not None else None
        except:
            continue
    
    # === 第5层：AST解析（处理Python字典格式）===
    try:
        # 尝试将类似Python字典的字符串转换
        parsed = ast.literal_eval(data)
        if isinstance(parsed, dict) and score_key in parsed:
            value = parsed[score_key]
            return float(value) if value is not None else None
    except:
        pass
    
    # === 第6层：正则表达式直接提取 ===
    patterns = [
        # 标准格式: "correctness_score": 0.5
        rf'"{score_key}":\s*([0-9]*\.?[0-9]+)',
        # 单引号格式: 'correctness_score': 0.5
        rf"'{score_key}':\s*([0-9]*\.?[0-9]+)",
        # 无引号格式: correctness_score: 0.5
        rf'{score_key}:\s*([0-9]*\.?[0-9]+)',
        # 带空格格式: "correctness_score" : 0.5
        rf'"{score_key}"\s*:\s*([0-9]*\.?[0-9]+)',
    ]
    
    for pattern in patterns:
        try:
            matches = re.findall(pattern, data, re.IGNORECASE)
            if matches:
                return float(matches[0])
        except:
            continue
    
    # === 第7层：模糊匹配 ===
    # 查找任何包含score_key的行
    lines = data.split('\n')
    for line in lines:
        if score_key.lower() in line.lower():
            # 提取该行中的数字
            numbers = re.findall(r'([0-9]*\.?[0-9]+)', line)
            if numbers:
                try:
                    return float(numbers[0])
                except:
                    continue
    
    # === 第8层：最后尝试 - 查找任何浮点数 ===
    # 如果JSON看起来只包含一个分数值
    if len(data.strip()) < 100:  # 短字符串
        numbers = re.findall(r'\b([0-1]\.?[0-9]*)\b', data)
        if len(numbers) == 1:
            try:
                return float(numbers[0])
            except:
                pass
    
    # === 完全失败 ===
    return None



def create_prediction_plot(y_true, y_pred, epoch, metrics, model_type="Model"):
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    
    ax.scatter(y_true, y_pred, alpha=0.6, s=20)
    xlabel = 'Actual Score'
    ylabel = 'Predicted Score'
    
    ax.plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    
    title = f'{model_type} - Epoch {epoch}\n'
    title += f'R² = {metrics.get("r2", 0):.4f}, Pearson = {metrics.get("pearson", 0):.4f}'
    ax.set_title(title)
    
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    ax.set_xlim(-0.1, 1.1)
    ax.set_ylim(-0.1, 1.1)
    ax.set_xticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    ax.set_aspect('equal')
    
    plt.tight_layout()
    
    img = wandb.Image(fig, caption=f"{model_type} Predictions - Epoch {epoch}")
    plt.close(fig)
    
    return img


def create_confusion_matrix_plot(y_true, y_pred, bucket_names, epoch, target_length, model_type="Model"):
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=bucket_names, yticklabels=bucket_names)
    
    ax.set_xlabel('Predicted Bucket')
    ax.set_ylabel('True Bucket')
    ax.set_title(f'{model_type} - Epoch {epoch} - Length {target_length} - Confusion Matrix')
    
    plt.tight_layout()
    
    img = wandb.Image(fig, caption=f"{model_type} Confusion Matrix - Epoch {epoch} - Length {target_length}")
    plt.close(fig)
    
    return img


def plot_bucket_distribution(y_train, y_test, bucket_names, target_length=None, model_type="Model", save_path=None):
    """
    绘制桶分布图（用于分类任务）
    
    Args:
        y_train: 训练集标签数据
        y_test: 测试集标签数据
        bucket_names: 桶名称列表
        target_length: 目标长度
        model_type: 模型类型名称
        save_path: 保存路径（可选）
    
    Returns:
        wandb.Image: 用于wandb记录的图像对象
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 计算训练集每个桶的样本数
    train_unique, train_counts = np.unique(y_train, return_counts=True)
    train_counts_dict = dict(zip(train_unique, train_counts))
    train_counts_full = [train_counts_dict.get(i, 0) for i in range(len(bucket_names))]
    
    # 计算测试集每个桶的样本数
    test_unique, test_counts = np.unique(y_test, return_counts=True)
    test_counts_dict = dict(zip(test_unique, test_counts))
    test_counts_full = [test_counts_dict.get(i, 0) for i in range(len(bucket_names))]
    
    # 绘制训练集分布
    ax1.bar(range(len(bucket_names)), train_counts_full, alpha=0.7, color='skyblue')
    ax1.set_xlabel('Bucket')
    ax1.set_ylabel('Count')
    ax1.set_title(f'Train Set Distribution')
    ax1.set_xticks(range(len(bucket_names)))
    ax1.set_xticklabels(bucket_names, rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # 绘制测试集分布
    ax2.bar(range(len(bucket_names)), test_counts_full, alpha=0.7, color='lightcoral')
    ax2.set_xlabel('Bucket')
    ax2.set_ylabel('Count')
    ax2.set_title(f'Test Set Distribution')
    ax2.set_xticks(range(len(bucket_names)))
    ax2.set_xticklabels(bucket_names, rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # 设置总标题
    title = f'{model_type} - Length {target_length}: Bucket Distribution' if target_length else f'{model_type}: Bucket Distribution'
    fig.suptitle(title, fontsize=14)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Distribution plot saved to {save_path}")
    
    # 创建wandb Image
    img = wandb.Image(fig, caption=f"{model_type} Bucket Distribution - Length {target_length}")
    plt.close(fig)  # 关闭图形以节省内存
    
    return img


def plot_comprehensive_results(y_test, y_pred, metrics, target_length, model_type="Model", 
                             bucket_names=None, save_path=None):
    """
    绘制综合结果图（包含混淆矩阵、训练历史等）
    
    Args:
        y_test: 测试集真实值
        y_pred: 测试集预测值
        metrics: 评估指标字典
        target_length: 目标长度
        model_type: 模型类型名称
        bucket_names: 桶名称列表（用于分类任务）
        save_path: 保存路径（可选）
    """
    is_classification = bucket_names is not None
    
    if is_classification:
        # 分类任务：2x2布局
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # 混淆矩阵
        from sklearn.metrics import confusion_matrix
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                    xticklabels=bucket_names, yticklabels=bucket_names)
        ax1.set_xlabel('Predicted Bucket')
        ax1.set_ylabel('True Bucket')
        ax1.set_title(f'{model_type} - Length {target_length}: Confusion Matrix')
        
        # 训练历史（损失）
        ax2.plot(metrics['train_losses'], label='Train Loss')
        ax2.plot(metrics['test_losses'], label='Test Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('CrossEntropy Loss')
        ax2.set_title(f'{model_type} - Length {target_length}: Training History')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 训练历史（准确率）
        ax3.plot(metrics['train_accuracies'], label='Train Accuracy')
        ax3.plot(metrics['test_accuracies'], label='Test Accuracy')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Accuracy')
        ax3.set_title(f'{model_type} - Length {target_length}: Accuracy History')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 桶分布
        unique, counts = np.unique(y_test, return_counts=True)
        ax4.bar(range(len(unique)), counts, alpha=0.7)
        ax4.set_xlabel('Bucket')
        ax4.set_ylabel('Count')
        ax4.set_title(f'{model_type} - Length {target_length}: Test Set Distribution')
        ax4.set_xticks(range(len(bucket_names)))
        ax4.set_xticklabels(bucket_names, rotation=45)
        
    else:
        # 回归任务：1x2布局
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # 预测散点图
        ax1.scatter(y_test, y_pred, alpha=0.6)
        ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
        ax1.set_xlabel('Actual Score')
        ax1.set_ylabel('Predicted Score')
        ax1.set_title(f'{model_type} - Length {target_length}: Actual vs Predicted\n'
                     f'R² = {metrics.get("test_r2", 0):.4f}, MSE = {metrics.get("test_mse", 0):.6f}')
        ax1.grid(True, alpha=0.3)
        
        # 训练历史
        ax2.plot(metrics['train_losses'], label='Train Loss')
        ax2.plot(metrics['test_losses'], label='Test Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('MSE Loss')
        ax2.set_title(f'{model_type} - Length {target_length}: Training History')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Comprehensive results plot saved to {save_path}")
    
    plt.show()


def load_data(csv_path: str, embeddings_path: str):
    df = pd.read_csv(csv_path)
    with open(embeddings_path, 'rb') as f:
        emb_data = pickle.load(f)

    if isinstance(emb_data, dict):
        embeddings = emb_data['embeddings']
    else:
        embeddings = emb_data

    # 取分数列：全是数字的列
    score_columns = [col for col in df.columns if str(col).isdigit()]

    if len(embeddings) != len(df):
        raise ValueError(f"❌ 行数不一致: embeddings={len(embeddings)}, csv={len(df)}")

    aligned_data = []
    for i, row in df.iterrows():
        row_data = {
            "row_id": i,
            "embedding": embeddings[i],
            "embedding_idx": i,
        }
        for col in score_columns:
            if col in row and pd.notna(row[col]):
                row_data[col] = float(row[col])
        aligned_data.append(row_data)

    return aligned_data, score_columns


def prepare_dataset(aligned_data, target_length):
    """Prepare X, y with fixed 10-bin buckets over [0,1].""" 
    X = []
    y_scores = []
    
    for i, sample in enumerate(aligned_data):
        X.append(sample['embedding'])
        y_scores.append(sample[target_length])

    X = np.array(X)
    y_scores = np.array(y_scores)
        
    print(f"Length {target_length}: {len(X)} valid samples")
    print(f"Score range: {y_scores.min():.4f} - {y_scores.max():.4f}")
    print(f"Score mean: {y_scores.mean():.4f}, std: {y_scores.std():.4f}")
    
    return X, y_scores