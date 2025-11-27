from typing import Dict, List, Tuple, Optional, Any
import re
import os
from pathlib import Path


class BPETokenizer:
    """
    BPE分词器实现 - 基于字节级别的BPE
    
    参数:
        vocab: 词汇表，映射 token_id -> bytes
        merges: 合并规则列表，每个元素是 (bytes1, bytes2) 的元组
        special_tokens: 特殊标记列表（如 ["<|endoftext|>"]）
    """
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: Optional[List[str]] = None):
        self.vocab = vocab  # 词汇表，id -> bytes
        self.merges = merges  # 合并规则列表（字节格式）
        self.special_tokens = special_tokens or []  # 特殊标记列表
        
        # 创建反向词汇表 bytes -> id，用于快速查找
        self.vocab_rev = {v: k for k, v in vocab.items()}
        
        # 构建特殊标记的字节集合，用于判断某个bytes是否是特殊标记
        self.special_token_bytes = {token.encode('utf-8'): token for token in self.special_tokens}
        
        # 如果有特殊标记，构建正则表达式模式用于匹配
        if self.special_tokens:
            # 按长度降序排序特殊标记，确保优先匹配更长的标记
            sorted_tokens = sorted(self.special_tokens, key=len, reverse=True)
            # 转义特殊字符，防止正则表达式解析错误
            escaped_tokens = [re.escape(token) for token in sorted_tokens]
            # 编译正则表达式，用 | 连接所有特殊标记（表示"或"）
            self.special_token_pattern = re.compile('|'.join(escaped_tokens))
        else:
            self.special_token_pattern = None
    
    def encode(self, text: str) -> List[int]:
        """
        将文本编码为token IDs列表
        
        处理流程:
        1. 如果文本包含特殊标记，先分离出特殊标记
        2. 对非特殊标记部分按GPT-2预分词规则分词
        3. 对每个预分词后的token应用BPE编码
        4. 返回最终的token ID列表
        """
        if not text:
            return []
        
        # 如果有特殊标记，需要特殊处理
        if self.special_token_pattern:
            # 使用正则表达式找到所有特殊标记的匹配
            special_matches = list(self.special_token_pattern.finditer(text))
            
            if special_matches:
                result = []
                last_end = 0  # 上一个匹配结束的位置
                
                for match in special_matches:
                    # 编码特殊标记之前的普通文本
                    if match.start() > last_end:
                        result.extend(self._encode_text(text[last_end:match.start()]))
                    
                    # 添加特殊标记的ID
                    special_token = match.group(0)
                    special_token_bytes = special_token.encode('utf-8')
                    
                    if special_token_bytes in self.vocab_rev:
                        result.append(self.vocab_rev[special_token_bytes])
                    else:
                        # 如果特殊标记不在词汇表中，按普通文本编码
                        result.extend(self._encode_text(special_token))
                    
                    last_end = match.end()
                
                # 编码最后一个特殊标记之后的文本
                if last_end < len(text):
                    result.extend(self._encode_text(text[last_end:]))
                
                return result
        
        # 没有特殊标记，直接编码整个文本
        return self._encode_text(text)
    
    def _encode_text(self, text: str) -> List[int]:
        """
        对普通文本应用GPT-2风格的预分词和BPE编码
        
        参数:
            text: 要编码的普通文本（不包含特殊标记）
        
        返回:
            token ID列表
        """
        if not text:
            return []
        
        # 使用GPT-2的预分词模式
        try:
            import regex as re
            # GPT-2的完整预分词模式
            pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        except ImportError:
            # 如果regex库不可用，使用简化版本
            import re
            pattern = re.compile(r" ?[^\s]+")
        
        # 预分词
        tokens = pattern.findall(text)
        
        result = []
        for token in tokens:
            # 将token转换为字节序列
            token_bytes = token.encode('utf-8')
            # 应用BPE编码
            result.extend(self._bpe_encode(token_bytes))
        
        return result
    
    def _bpe_encode(self, token_bytes: bytes) -> List[int]:
        """
        对单个token的字节序列应用BPE编码
        
        参数:
            token_bytes: token的字节序列
        
        返回:
            token ID列表
        """
        # 快速路径：如果整个token已经在词汇表中
        if token_bytes in self.vocab_rev:
            return [self.vocab_rev[token_bytes]]
        
        # 将字节序列分解为单个字节，每个字节作为一个token
        # 使用元组以便可以作为字典键
        word = tuple(bytes([b]) for b in token_bytes)
        
        # 如果没有合并规则，直接返回单字节ID
        if not self.merges:
            return [self.vocab_rev[bytes([b])] for b in token_bytes]
        
        # 应用BPE合并规则
        # 构建合并规则的优先级字典：先出现的合并规则优先级更高
        merge_ranks = {pair: i for i, pair in enumerate(self.merges)}
        
        while len(word) > 1:
            # 找到当前word中优先级最高的字节对
            pairs = [(word[i], word[i+1]) for i in range(len(word)-1)]
            
            # 找到在merge_ranks中优先级最高（rank最小）的pair
            bigram = min(pairs, key=lambda pair: merge_ranks.get(pair, float('inf')))
            
            # 如果这个pair不在合并规则中，停止合并
            if bigram not in merge_ranks:
                break
            
            # 应用这个合并规则
            first, second = bigram
            new_word = []
            i = 0
            while i < len(word):
                # 寻找当前pair的出现
                if i < len(word) - 1 and word[i] == first and word[i+1] == second:
                    # 合并这两个字节
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            
            word = tuple(new_word)
        
        # 将最终的word转换为token IDs
        result = []
        for part in word:
            if part in self.vocab_rev:
                result.append(self.vocab_rev[part])
            else:
                # 如果这个部分不在词汇表中，分解为单字节
                for b in part:
                    result.append(self.vocab_rev[bytes([b])])
        
        return result
    
    def encode_iterable(self, iterable):
        """
        逐行编码可迭代对象（如文件对象），返回生成器
        这个方法是内存高效的，因为它使用yield逐个返回token ID
        
        参数:
            iterable: 可迭代对象，每个元素是一个字符串
        
        yield:
            每次yield一个token ID
        """
        # 将可迭代对象连接成完整文本
        text = ''.join(iterable)
        # 编码文本并逐个yield token ID
        for token_id in self.encode(text):
            yield token_id
    
    def decode(self, ids: List[int]) -> str:
        """
        将token ID列表解码为文本字符串
        
        参数:
            ids: token ID列表
        
        返回:
            解码后的文本字符串
        """
        # 先收集所有的字节，然后一次性解码
        # 这样可以正确处理跨越多个token的Unicode字符
        result_bytes = b''
        
        for token_id in ids:
            if token_id in self.vocab:
                result_bytes += self.vocab[token_id]
        
        # 将收集到的字节序列一次性解码为UTF-8字符串
        try:
            return result_bytes.decode('utf-8')
        except UnicodeDecodeError:
            # 如果解码失败，使用errors='replace'来替换无效字节
            return result_bytes.decode('utf-8', errors='replace')


def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: List[str],
    **kwargs
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    """
    训练字节级BPE分词器（GPT-2风格）
    
    参数:
        input_path: 训练数据文件路径
        vocab_size: 目标词汇表大小
        special_tokens: 特殊标记列表（如 ["<|endoftext|>"]）
    
    返回:
        (vocab, merges) 元组:
            - vocab: 词汇表字典 {token_id: bytes}
            - merges: 合并规则列表 [(bytes1, bytes2), ...]
    """
    # 将路径转换为字符串
    if isinstance(input_path, Path):
        input_path = str(input_path)
    
    # 读取输入文件内容
    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()
    
    # 使用regex库实现GPT-2风格的预分词
    # GPT-2的分词模式：区分缩写、字母、数字、标点符号等
    try:
        import regex as re
        # GPT-2的完整预分词模式
        # 匹配：缩写 | 可选空格+字母序列 | 可选空格+数字序列 | 可选空格+非空白非字母数字 | 空白
        pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
    except ImportError:
        # 如果regex库不可用，使用简化版本
        import re
        pattern = re.compile(r" ?[^\s]+")
    
    # 先用特殊标记分割文本，这样特殊标记就不会参与BPE训练
    # 这确保特殊标记的内容不会被分解成字节对
    text_parts = [text]
    for special_token in special_tokens:
        new_parts = []
        for part in text_parts:
            # 用特殊标记分割每个部分
            segments = part.split(special_token)
            for i, seg in enumerate(segments):
                if i > 0:
                    # 在分割点插入特殊标记占位符（我们不训练它，只是为了统计）
                    pass  # 特殊标记不参与BPE训练，所以跳过
                new_parts.append(seg)
        text_parts = new_parts
    
    # 对所有非特殊标记的文本部分进行预分词
    words = []
    for part in text_parts:
        if part:  # 跳过空字符串
            words.extend(pattern.findall(part))
    
    # 将每个预分词后的word转换为字节列表
    word_tokens = []
    for word in words:
        byte_sequence = word.encode("utf-8")
        # 将字节序列转换为字节列表，每个字节作为一个独立的token
        word_tokens.append(list(byte_sequence))
    
    # 初始化词汇表：包含所有可能的字节（0-255）
    vocab = {i: bytes([i]) for i in range(256)}
    
    # 添加特殊标记到词汇表
    special_token_ids = set()  # 存储特殊标记的ID
    special_token_bytes_set = set()  # 存储完整的特殊标记
    
    for token in special_tokens:
        token_bytes = token.encode("utf-8")
        token_id = len(vocab)
        vocab[token_id] = token_bytes
        special_token_ids.add(token_id)
        special_token_bytes_set.add(token_bytes)
    
    # 统计每个word出现的频率
    from collections import Counter
    word_counts = Counter(tuple(word) for word in word_tokens)
    
    # 初始化合并规则列表
    merges = []
    
    # 主训练循环
    print(f"Starting BPE training. Target vocab size: {vocab_size}")
    print(f"Initial vocabulary size: {len(vocab)}")
    
    while len(vocab) < vocab_size:
        # 统计所有相邻字节对的频率
        pair_freqs = {}
        
        for word_tuple, count in word_counts.items():
            word_list = list(word_tuple)
            if len(word_list) < 2:
                continue
            
            # 遍历word中的所有相邻字节对
            for i in range(len(word_list) - 1):
                pair = (word_list[i], word_list[i + 1])
                pair_freqs[pair] = pair_freqs.get(pair, 0) + count
        
        # 如果没有可合并的字节对，退出循环
        if not pair_freqs:
            break
        
        # 找到频率最高的字节对
        # 当频率相同时，按字典序排序以保证结果的确定性
        # 排序规则：(1)频率降序 (2)第一个token字典序降序 (3)第二个token字典序降序
        # 使用max而不是min，因为我们要最大化所有维度
        best_pair = max(pair_freqs.items(), key=lambda x: (x[1], vocab[x[0][0]], vocab[x[0][1]]))[0]
        
        # 创建新token（合并两个字节）
        # 需要确保不会创建出包含特殊标记片段的token
        new_token_id = len(vocab)
        new_token_bytes = vocab[best_pair[0]] + vocab[best_pair[1]]
        
        # 检查新token是否包含特殊标记的特殊前缀（如 b'<|'）
        # 注意：只需要检查特殊标记的独特部分，其他常见片段允许合并
        def contains_special_fragment(token_bytes):
            """检查token是否包含特殊标记的独特片段（如 <| ）"""
            # 如果是完整的特殊标记，允许
            if token_bytes in special_token_bytes_set:
                return False
            # 对于 GPT-2 风格的特殊标记 <|endoftext|>，只需检查是否包含 b'<|'
            # 这是特殊标记的独特前缀，不会出现在正常文本中
            if b'<|' in token_bytes:
                return True
            return False
        
        # 如果包含特殊标记的片段，跳过这个合并
        while contains_special_fragment(new_token_bytes):
            # 从 pair_freqs 中删除这个 pair，选择下一个
            del pair_freqs[best_pair]
            if not pair_freqs:
                break
            best_pair = max(pair_freqs.items(), key=lambda x: (x[1], vocab[x[0][0]], vocab[x[0][1]]))[0]
            new_token_bytes = vocab[best_pair[0]] + vocab[best_pair[1]]
        
        # 如果所有pair都被跳过了，退出循环
        if not pair_freqs:
            break
        
        vocab[new_token_id] = new_token_bytes
        
        # 记录合并规则
        merges.append((vocab[best_pair[0]], vocab[best_pair[1]]))
        
        # 更新word_counts：将所有包含该字节对的word进行合并
        new_word_counts = {}
        for word_tuple, count in word_counts.items():
            word_list = list(word_tuple)
            i = 0
            new_word = []
            
            while i < len(word_list):
                # 检查是否可以合并当前字节对
                if (i < len(word_list) - 1 and 
                    word_list[i] == best_pair[0] and 
                    word_list[i + 1] == best_pair[1]):
                    new_word.append(new_token_id)
                    i += 2
                else:
                    new_word.append(word_list[i])
                    i += 1
            
            new_word_tuple = tuple(new_word)
            new_word_counts[new_word_tuple] = new_word_counts.get(new_word_tuple, 0) + count
        
        word_counts = new_word_counts
        
        # 定期打印进度
        if len(vocab) % 50 == 0:
            print(f"Vocabulary size: {len(vocab)}")
    
    print(f"BPE training completed. Vocabulary size: {len(vocab)}")
    return vocab, merges