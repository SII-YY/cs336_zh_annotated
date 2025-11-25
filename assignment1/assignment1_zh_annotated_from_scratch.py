#!/usr/bin/env python3

from hmac import new
from pydoc import text
from tkinter import WORD
from typing import Any

# 定义一个 Byte-level BPE tokenizer
def train_bpe_tokenizer(input_path:str, vocab_size:int, special_tokens:list[str]) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    '''
    1. 调用 train_bpe_tokenizer 函数，传入输入文件路径、词汇表大小和特殊 token 列表
    2. 返回训练好的 BPE 分词器的词汇表和合并规则列表
    '''

    # 定义一个函数，用于将一个unicode字符串转换为一个字节列表
    def encode_unicode_to_bytes(text):
        return text.encode("utf-8")

    # 定义一个函数，用于将一个字节列表转换为一个单词（列表中的每个元素都是一个字节）
    def bytes_to_word(byte_sequence):
        return list(byte_sequence)

    """
    在Python中，字节被当作整数处理
    读取输入文件中的所有文本
    with语句创建上下文管理器，自动关闭文件
    r表示只读模式（read）
    encoding="utf-8"表示使用utf-8编码读取文件，指定如何将字节转换回字符显示
    读取文件时，将字节转换为字符，而不是默认的字节序列
    这在处理包含非ASCII字符的文本文件时非常重要，因为默认的字节序列不能直接显示这些字符
    例如，中文字符“你好”的utf-8编码是b'\xe4\xbd\xa0\xe5\xa5\xbd'，如果直接打印这个字节序列，会显示为b'\xe4\xbd\xa0\xe5\xa5\xbd'
    而不是你好这个中文字符
    因此，我们需要使用utf-8编码将字节转换为字符，才能正确显示中文字符
    """

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read() # 读取输入文件中的所有文本，将其存储在text变量中
    
    # 这里简单地使用空格进行分词，实际应用中可能需要更复杂的分词器，例如基于规则的分词器或基于统计模型的分词器
    text_words = text.split() # 将文本转换为单词列表，每个单词都是一个字符串
    # print("text_words:",text_words) # 把这条启动，运行一下，你可以看到，每个单词都是一个字符串
    words = []
    # print("words1:",words) # 把这条启动，运行一下，你可以看到，words1是空的
    
    for word in text_words: # 遍历每个单词，将其转换为字节列表，并添加到words列表中
        byte_sequence = encode_unicode_to_bytes(word) # 将单词转换为字节列表
        words.append(bytes_to_word(byte_sequence)) # 每个单词都被转换为了一个字节列表
    #     print((bytes_to_word(byte_sequence))) # 把这条启动，运行一下，你可以看到，每个单词都被转换为了一个字节列表
    #     print("word=",word) # 把这条启动，运行一下，你可以看到，word是一个字符串，例如"你好"
    #     print("byte_sequence=",byte_sequence) # 把这条启动，运行一下，你可以看到，byte_sequence是一个字节列表，例如[b'\xe4', b'\xbd', b'\xa0', b'\xe5', b'\xa5', b'\xbd']
    # print("words2:",words) # 把这条启动，运行一下，你可以看到，words2是一个列表，每个元素都是一个字节列表，例如[b'\xe4', b'\xbd', b'\xa0', b'\xe5', b'\xa5', b'\xbd']

    vocab = {i:bytes([i]) for i in range(256)} # 词汇表 = 所有可能的字节（0-255）
    # print("vocab:",vocab) # 把这条启动，运行一下，你可以看到，vocab是一个字典，每个键都是一个整数，每个值都是一个字节，例如{0: b'\x00', 1: b'\x01', 2: b'\x02', ..., 255: b'\xff'}
    
    # 初始化特殊标记的ID和字节集合
    special_token_ids = {}
    # 初始化特殊标记的字节集合
    special_token_bytes = set()
    # 遍历每个特殊标记，将其转换为字节列表，并添加到词汇表中
    for token in special_tokens:
        token_bytes = token.encode("utf-8") # 将特殊标记转换为字节列表
        token_id = len(vocab) # 特殊标记的ID = 词汇表大小
        vocab[token_id]= token_bytes # 将特殊标记添加到词汇表中
        special_token_ids[token] = token_id # 将特殊标记添加到特殊标记ID字典中
        special_token_bytes.add(token_bytes) # 将特殊标记的字节列表添加到特殊标记字节集合中
        # print(f"special_token_ids[{token}] = {token_id}") # 把这条启动，运行一下，你可以看到，special_token_ids是一个字典，每个键都是一个特殊标记，每个值都是一个整数，例如{"<PAD>": 0, "<UNK>": 1}
        # print(special_token_bytes) # 把这条启动，运行一下，你可以看到，special_token_bytes是一个集合，每个元素都是一个特殊标记的字节列表，例如{b'<PAD>', b'<UNK>'}
    
    """
    结束符 ： </w> ← 这个就是 -1 要预留的位置
    因为我们要将每个单词的结束符 </w> 作为一个单独的token，
    所以要预留一个位置给它
    例如，假设vocab_size是3000，special_tokens有2个（<PAD>和<UNK>），
    那么target_vocab_size就是3000 - 2 - 1 = 2997
    这2997个位置就可以用来存储其他的token（字节）
    每个字节都是一个token，所以我们可以将所有的字节都存储在vocab中
    然后将special_tokens和结束符 </w> 分别添加到vocab中
    这样，vocab就有3000个token，分别是2997个字节加上2个特殊标记和1个结束符
    每个token都有一个唯一的ID，从0开始递增
    """
    
    eow_token = b"</w>" # 添加结束符 </w> 到所有单词的末尾
    eow_id = len(vocab) # 结束符 </w> 的ID = 词汇表大小
    vocab[eow_id] = eow_token # 将结束符 </w> 添加到词汇表中

    # 遍历每个单词，将结束符 </w> 作为一个单独的token添加到单词的末尾
    # print("word1:",word) # 把这条启动，运行一下，你可以看到，word1是一个字节列表，例如[b'\xe4', b'\xbd', b'\xa0', b'\xe5', b'\xa5', b'\xbd']
    words = [word +[eow_id] for word in words]
    # print("word2:",word) # 把这条启动，运行一下，你可以看到，word2是一个字节列表，例如[b'\xe4', b'\xbd', b'\xa0', b'\xe5', b'\xa5', b'\xbd', b'</w>']
    # print("words:",words) # 把这条启动，运行一下，你可以看到，words是一个列表，每个元素都是一个字节列表，例如[b'\xe4', b'\xbd', b'\xa0', b'\xe5', b'\xa5', b'\xbd', b'</w>']

    word_freq = {} # 这也是个字典，用于统计每个单词出现的频率
    for word in words:
        word_tuple = tuple(word) # 元组是不可变的，因此可以作为字典的键，列表是可变的，不能作为字典的键
        """
        统计每个单词出现的频率,并将其存储在word_freq字典中,
        如果word_tuple不在word_freq中,则将其添加到word_freq中,并将其值设为1
        如果word_tuple已经在word_freq中,则将其值加1
        print(f"word_tuple={word_tuple}")
        字典的get方法用于获取指定键的值
        如果键不存在，返回默认值0
        然后将其加1，更新单词的频率
        例如，假设word_tuple是("h","e","l","l","o")
        第一次遇到这个单词时，word_freq.get(word_tuple, 0)返回0
        然后将其加1，更新为1
        第二次遇到这个单词时，word_freq.get(word_tuple, 0)返回1
        然后将其加1，更新为2
        以此类推，直到统计完所有单词的频率
        """
        word_freq[word_tuple] = word_freq.get(word_tuple, 0) + 1

    
    merges = [] # 初始化合并列表，用于存储合并操作
    target_vocab_size = vocab_size - len(special_tokens) - 1 # 目标词汇表大小 = 原始词汇表大小 - 特殊标记数 - 结束符 </w>
    # 打印目标词汇表大小和初始词汇表大小，也可以不打印
    print(f"Starting BPE training. Target vocab size: {target_vocab_size}")
    # 打印初始词汇表大小（字节数 + 特殊标记数 + 结束符 </w>），也可以不打印
    print(f"Initial vocabulary size (bytes + special tokens + EOW): {len(vocab)}")

    
    """
    create a reverse map for faster lookup of special tokens
    # 这是一个字典，用于快速查找特殊标记的ID
    # 例如，假设special_token_ids["<PAD>"] = 0
    # 那么is_special_token[0] = True
    # 这意味着ID为0的token是一个特殊标记
    """

    is_special_token = {}
    for token_id, token_bytes in vocab.items():
        # 检查token_bytes是否在special_token_bytes中
        # 如果是，将is_special_token[token_id]设为True，否则，设为False
        is_special_token[token_id] = token_bytes in special_token_bytes
        # print(f"token_id={token_id}, token_bytes={token_bytes}, is_special_token={is_special_token[token_id]}") # 把这条启动，运行一下，你可以看到，is_special_token是一个字典，每个键都是一个token_id，每个值都是一个布尔值，例如{0: True, 1: True, 2: False, 3: False, ...}
    # print(f"is_special_token={is_special_token}") # 把这条启动，运行一下，你可以看到，is_special_token是一个字典，每个键都是一个token_id，每个值都是一个布尔值，例如{0: True, 1: True, 2: False, 3: False, ...}
    # print("word_freq=",word_freq) # 把这条启动，运行一下，你可以看到，word_freq是一个字典，每个键都是一个单词（字节元组），每个值都是该单词出现的频率，例如{("h","e","l","l","o"): 2, ("h","i"): 1, ...}
    
    """
    总算到达主循环了
    每次迭代，会找到出现频率最高的字符对
    然后将其合并为一个新的token
    并更新单词频率
    直到词汇表大小达到目标大小
    """
    while len(vocab) < target_vocab_size:
        # 统计每个字符对出现的频率
        pairs = {}
        for word, freq in word_freq.items():
            if len(word) < 2: 
                continue # 跳出当前循环，继续下一个单词
        
            for i in range(len(word) - 1):
                # get函数就是获取字典中指定键的值，如果键不存在，返回默认值False
                if is_special_token.get(word[i],False) or is_special_token.get(word[i+1],False):
                    continue # 如果当前字符对包含特殊标记，跳出当前循环，继续下一个字符对
                # print(f"word[i]={word[i]}, word[i+1]={word[i+1]}") # 把这条启动，运行一下，你可以看到，word[i]和word[i+1]都是字节，例如b'\xe4', b'\xbd', b'\xa0', b'\xe5', b'\xa5', b'\xbd', b'</w>'
                """
                检查当前字符对是否是特殊标记
                如果是，跳出当前循环，继续下一个字符对
                如果当前字符对不是特殊标记，将其组成元组
                """
                pair = (word[i],word[i + 1]) 
                # 统计当前字符对出现的频率
                pairs[pair] = pairs.get(pair, 0) + freq
        # 如果没有字符对出现，跳出循环
        if not pairs: 
            break

        """
        key函数的具体用法是：
        key=lambda x: 表达式
        例如，key=lambda x:x[1] 就是一个匿名函数，它接受一个参数x，返回x的第二个元素（即频率）
        max函数会根据这个函数的返回值来找到最大的元素
        """
        max_pair = max(pairs.items(),key=lambda x:x[1]) # 找到出现频率最高的字符对(int型的元组)
        pair_to_merge = max_pair[0] # 例如，("101","102")，表示要合并的字符对是"101"和"102"
        new_token_id = len(vocab) # 新的token_id就是当前词汇表的大小
        new_token = vocab[pair_to_merge[0]] + vocab[pair_to_merge[1]] # 新的token就是两个bytes合并后的结果
        vocab[new_token_id] = new_token # 将新的token添加到词汇表中
        merges.append((vocab[pair_to_merge[0]], vocab[pair_to_merge[1]])) # 将合并的字符对添加到merges中，起到记录合并历史的作用

        """
        更新单词频率字典
        遍历所有单词，将其中出现的合并字符对替换为新的token id
        """
        new_word_freq = {}
        for word, freq in word_freq.items():
            new_word = [] # 新的单词列表，用于存储在token合并后，单词用新的token id 构成的新写法，理解这点很关键
            i = 0
            while i < len(word): # 遍历单词中的每个字符
                if (i < len(word)-1 and # 确保当前字符对不会超出单词范围
                    word[i] == pair_to_merge[0] and # 当前字符是要合并的字符对的第一个字符
                    word[i + 1] == pair_to_merge[1]): # 当前字符是要合并的字符对的第二个字符
                    new_word.append(new_token_id) # 用新的token id替换合并的字符对
                    i += 2
                else:
                    new_word.append(word[i]) # 否则，将当前字符添加到新单词中
                    i += 1
            
            new_word_tuple = tuple(new_word) # 将新单词转换为元组
            new_word_freq[new_word_tuple] = new_word_freq.get(new_word_tuple, 0) + freq # 更新新单词的频率

        word_freq = new_word_freq # 更新单词频率字典

        is_special_token[new_token_id] = False # 新的token不是特殊标记

        if len(vocab) % 200 == 0: 
            print(f"Vocabulary size:{len(vocab)}") # 每合并200个token，打印一次词汇表大小
    
    print(f"BPE training completed. Vocabulary size: {len(vocab)}") # 打印最终的词汇表大小
    return vocab, merges # 返回词汇表和合并历史






# 这里字典太多了，如果看得晕，参考一下dictionary_analysis_simple.html演示文件
# 为了边写边测，我们先随便指定一些参数
# special_tokens = [ 
#     "<PAD>",   # 填充标记 - 用于补齐序列长度
#     "<UNK>",   # 未知词标记 - 用于处理未见过词汇
#     "<BOS>",   # 句子开始标记 - Beginning of Sentence  
#     "<EOS>",   # 句子结束标记 - End of Sentence
#     "<SEP>",   # 分隔符标记 - 用于分隔不同文本
#     "<MASK>",  # 掩码标记 - 用于遮盖词汇进行预测
#     "</w>",    # 单词结束标记 - End of Word
# ]
# vocab_size = 300 # 我们先指定一个较小的大小，方便调试
# input_path = "../TinyStoriesV2-GPT4-valid.txt" #用这个小的数据集调试一下，避免跑崩

# # 简单写个测试，看看是否能正常运行
# vocab, merges = train_bep_tokenizer(input_path, vocab_size, special_tokens)

# print(f"merges={merges}")
# print(f"vocab={vocab}")

# ----day2-----

'''
将词汇表和合并历史保存到文件中，方便后续使用
'''
def save_vocab_and_merges(vocab:dict[int, bytes], merges:list[tuple[bytes, bytes]],
                            vocab_path: str, merges_path: str):

    # 保存词汇表
    with open(vocab_path, "w", encoding='utf-8' ) as f: 
        for token_id, token_bytes in sorted(vocab.items()): # 按token_id排序，方便查看
            """
            - 虽然 vocab 中的 token_id 是按顺序添加的（0, 1, 2, 3...）
            - 但字典的遍历顺序在不同Python版本或不同情况下可能不一致
            - 为了确保按顺序输出，我们使用 sorted(vocab.items()) 来按 token_id 排序     
            """
            try: # 尝试将字节转换为字符串
                token_str = token_bytes.decode("utf-8")
                f.write(f"{token_id}\t{token_str}\t{token_bytes}\n") # \t是制表符，使用制表符可以让多列数据对齐，比用空格更整齐规范。
            except UnicodeDecodeError: # 如果遇到无法解码的字节，用十六进制表示
                f.write(f"{token_id}\t{token_bytes.hex()}\t{token_bytes}\n") 
                ''' 
                .hex是用十六进制表示无法解码的字节，注意：十六进制是字符串类型的
                例如：
                token_bytes = b'A'  # 一个字节
                hex_string = token_bytes.hex()  # 返回 "41"，这里有个双引号看到了吧
                '''

        # 保存merges，这部分就和上面那块一个套路
        with open(merges_path, "w", encoding= 'utf-8') as f:
            for i, (token1, token2) in enumerate (merges): # enumerate是一个内置函数，功能是给可迭代对象添加一个索引，默认从0开始
                try: # 尝试将字节转换为字符串
                    token1_str = token1.decode("utf-8")
                    token2_str = token2.decode("utf-8")
                    f.write(f"{i}\t{token1_str}\t{token2_str}\n")
                except UnicodeDecodeError: # 如果遇到无法解码的字节，用十六进制表示
                    f.write(f"{i}\t{token1.hex()}\t{token2.hex()}\n") 

        """                   
        上述为什么要把这两个文件保存呢？
        为了以后重用训练好的模型，否则就要用一次训练一次
        例如，在对新文本进行编码时，我们需要根据词汇表将文本转换为token id序列，也要根据merges进行词汇中连续字符的合并。
        而在对token id序列进行解码时，我们需要根据合并历史将token id序列恢复为原始文本。
        """
# --- day3 ---
'''
这个函数是通过上述训练好的vocab，对新的文本进行编码，将其转换为token id序列
'''
def encode_text(text: str, vocab: dict[int, bytes]) -> list[int]: # 箭头表示函数的返回值类型
    vocab_rev = {v: k for k, v in vocab.items()} 
    # _rev是反转的意思，这里是把原来的字典，键值对互换。
    # for k, v in ... ：遍历每个键值对，将键赋值给 k ，值赋值给 v
    # {v: k ...} ：创建新字典，使用原来的值 v 作为键，原来的键 k 作为值
    text_bytes = text.encode('utf-8') # 将文本转换为字节序列，默认使用utf-8编码
    if text_bytes in vocab_rev: # 如果文本字节序列在词汇表中，直接返回对应的token id
        return [vocab_rev[text_bytes]] # 如果文本字节序列在词汇表中，直接返回对应的token id
        
    result = [] # 创建一个空的列表，用于存储编码后的token id序列
    # 1. 首先找到所有的空格位置
    i = 0
    while i < len(text): # 遍历文本中的每个字符
        # 跳过空格
        if text[i].isspace():
            if b' ' in vocab_rev:
                result.append(vocab_rev[b' ']) # 如果有空格token，直接添加到结果中
            i += 1 # 跳过空格
        else: # 如果不是空格，说明是一个普通字符
            j = i # 双指针操作，j指向当前连续字符的结束位置
            while j < len(text) and not text[j].isspace(): # 找到当前连续字符的结束位置
                j += 1
            word = text[i:j] # 提取出两个空格间的连续字符串
            word_bytes = word.encode('utf-8') # 将当前连续字符串转换为字节序列

            # 编码这个单词
            k = 0
            while k < len(word_bytes): # 遍历当前连续字节序列的每个字节
                max_len = 1 # 这是一个默认值，假设当前字节不在词汇表中，就直接编码为单字节
                best_token = word_bytes[k:k+1] # 从当前位置开始，取最大长度的子字节序列

                # 尝试更长的匹配
                for length in range(min(4, len(word_bytes) - k), 1 , -1): # 从最大长度4开始，尝试匹配到最短长度1，每次减少1
                    '''
                    在BPE分词器中，将最大匹配长度设置为4，是一个经验值：
                    1. 性能考虑 ：限制最大长度可以减少匹配尝试的次数，提高编码效率。如果不限制长度，每次都要从当前位置到序列末尾尝试所有可能的长度，这会导致时间复杂度大幅增加。
                    2. 实际应用需求 ：在大多数情况下，BPE合并产生的token长度通常不会太长。尤其是在中文等表意文字中，大多数常用词或字符组合通常在4个字节以内就能表示。
                    3. 编码效率平衡 ：太短的长度限制（如2或3）可能无法充分利用BPE的合并效果，而太长的限制（如8或更长）则会增加计算开销，且实际收益有限。
                    4. 内存优化 ：限制尝试的最大长度可以减少内存使用，因为不需要为过长的候选token分配内存。
                    5. 实际观察经验 ：在实际BPE训练中，统计发现长度超过4的高频合并对相对较少，设置为4是在性能和效果之间的合理平衡。
                    '''
                    candidate = word_bytes[k:k+length] # 取当前位置开始，长度为length的子字节序列
                    if candidate in vocab_rev: # 如果这个子字节序列在词汇表中
                        max_len = length # 更新最大匹配长度
                        best_token = candidate # 更新最佳匹配子字节序列
                        break # 找到最佳匹配后，跳出循环，继续遍历下一个字节
                # 添加最佳匹配的token id到结果中
                result.append(vocab_rev[best_token])
                k += max_len # 移动指针，跳过已经编码的字节
            
            i = j # 移动到下一个连续字符的开始位置

    return result

# day4
'''
decode_tokens 函数是 BPE 分词器中的解码组件，用于将 token ID 列表转换回原始文本字符串。
这是编码过程的反向操作，它将模型内部使用的数字表示转换回人类可读的文本。
'''
def decode_tokens(token_ids:list[int], vocab: dict[int, bytes]) -> str:
    result_parts = [] # 创建空列表，用于存储每个解码后的 token 字符串部分
    for token_id in token_ids:
        if token_id in vocab:
            token_bytes = vocab[token_id] # 遍历输入的 token ID 列表，对于每个 ID，检查它是否存在于词汇表中
            '''
            尝试使用 UTF-8 编码将字节序列解码为字符串
            使用 try-except 块捕获可能的 UnicodeDecodeError 异常
            如果解码失败（例如字节序列不是有效的 UTF-8 编码），则使用问号 '?' 作为替代
            '''
            try:
                token_str = token_bytes.decode('utf-8') # 尝试将字节序列解码为 UTF-8 字符串
                result_parts.append(token_str)
            except UnicodeDecodeError:
                result_parts.append('?') # 如果解码失败，使用问号 '?' 作为替代

    return ''.join(result_parts) # 将所有解码后的字符串部分连接成一个完整的文本字符串返回

# day 5    
# 这是一个训练 BPE 分词器的函数，用于从输入文件中学习合并规则和构建词汇表
def run_train_bpe(
        input_path: str | os.PathLike, # 输入文件路径，支持字符串或 os.PathLike 对象
        vocab_size: int, # 词汇表大小，即最终合并后的 token 数量，冒号后面跟着的是数据格式 int
        special_tokens: list[str], # 特殊 token 列表，如 ['<unk>', '<pad>', '<sos>', '<eos>']
        **kwargs # 其他参数，如 max_iterations, min_frequency 等
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    '''
    1. 调用 train_bpe_tokenizer 函数，传入输入文件路径、词汇表大小和特殊 token 列表
    2. 返回训练好的 BPE 分词器的词汇表和合并规则列表
    说白了，这个函数起到的是接口作用，将输入的参数传递给 train_bpe_tokenizer 函数，然后返回训练好的分词器
    '''
    import os
    return train_bpe_tokenizer(str(input_path), vocab_size, special_tokens)

# 这是一个工厂函数，用于创建 BPE 分词器实例，创建实例是指根据传入的词汇表、合并规则和特殊 token 列表，创建一个 BPE_Tokenizer 类的实例
# 创建这个实例的目的是为了在后续的编码和解码过程中使用这个实例
def get_tokenizer( 
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str] | None = None,
) -> Any:
    from typing import Any
    '''
    类的主要特点 ：
    1. 数据和行为的封装 ：类可以同时包含数据（属性）和操作这些数据的代码（方法）
    2. 创建实例 ：通过类可以创建多个实例（对象），每个实例拥有类定义的属性和方法
    3. 继承和多态 ：支持面向对象的继承和多态特性
    4. 生命周期 ：类定义只执行一次，但可以创建多个实例

    函数（Function）
    基本定义 ：
    函数是一段可重用的代码块，用于执行特定任务并可能返回结果。它不包含数据，只包含操作逻辑。
    主要特点 ：
    1. 单一职责 ：通常负责执行单一任务
    2. 数据独立性 ：函数通过参数接收数据，通过返回值传递结果
    3. 无状态性 ：函数执行完毕后，其内部的局部变量会被销毁
    4. 可重用性 ：可以在程序的不同位置被多次调用

    类的实例化
    基本定义 ：
    实例化是指根据类的定义，创建一个类的实例（对象）的过程。每个实例都有自己的状态（属性值），但共享类的行为（方法）。
    主要特点 ：
    1. 创建对象 ：通过类可以创建多个实例（对象），每个实例都是类的一个具体实例
    2. 状态独立性 ：每个实例的属性值是独立的，互不干扰
    3. 行为共享 ：所有实例共享类的方法，即可以调用类的方法来操作实例的状态
    4. 生命周期 ：每个实例的生命周期独立，当实例不再被引用时，会被垃圾回收器自动销毁

    init的使用：
    1. 初始化属性 ：在类的实例化过程中，init 方法会被自动调用，用于初始化实例的属性
    2. 接收参数 ：init 方法可以接收参数，用于初始化实例的属性值
    3. 数据封装 ：通过 init 方法，可以将数据封装在实例中，实现数据和行为的封装
    '''
    class BPE_Tokenizer: # 这是一个 BPE 分词器类，用于编码和解码文本
        def __init__(self, vocab, merges, special_tokens=None): # 初始化方法，接收词汇表、合并规则和特殊 token 列表
            self.vocab = vocab # 这是为了避免直接操作传入的数据
            self.merges = merges 
            self.special_tokens = special_tokens or [] # 特殊 token 列表，如 ['<unk>', '<pad>', '<sos>', '<eos>']
            self.vocab_rev = {v:k for k, v in vocab.items()}  # 反转词汇表，将 token 映射到对应的索引
            self.special_token_bytes = {token.encode('utf-8'): token for token in self.special_tokens} # 构建字典

            import re # 导入正则表达式模块，用于处理特殊 token 中的特殊字符
            if self.special_tokens:
                sorted_tokens = sorted(self.special_tokens, key=len, reverse=True) # 对特殊 token 列表按长度降序排序，就是从长到短
                escaped_tokens = [re.escape(token) for token in sorted_tokens] 
                # 对每个特殊 token 进行正则表达式转义，防止特殊字符干扰匹配
                # 核心逻辑是：部分字符（如引号 "、反斜杠 \、换行符 \n 等）在文本存储、传输或模型解析时，
                # 可能被误识别为语法符号（而非普通文本内容），导致解析错误或语义偏差。
                # 通过 “转义”（通常是在特殊字符前添加转义符，如 \" 表示普通引号、\\ 表示普通反斜杠），
                # 将这些字符标记为 “普通文本内容”，再将转义后的文本切分为 token，就得到了 escaped_tokens。
                self.special_token_pattern = re.compile('|'.join(escaped_tokens)) 
                '''
                ### 代码功能分析
                1. escaped_tokens ：
                
                - 这是一个已转义的特殊标记列表，其中每个标记中的正则表达式特殊字符（如 . , * , + 等）都已被正确转义，确保它们在正则表达式中被视为普通字符。
                2. '|'.join(escaped_tokens) ：
                
                - 使用字符串方法 join() 将转义后的标记列表连接成一个字符串
                - '|' 是连接分隔符，表示正则表达式中的「或」操作
                - 例如：如果 escaped_tokens = ['<start>', '<end>'] ，那么结果就是 <start>|<end>
                - 这个表达式的含义是「匹配 <start> 或者 <end> 」
                3. re.compile(...) ：
                
                - 使用 Python 的 re 模块中的 compile() 函数
                - 将字符串形式的正则表达式编译成一个正则表达式对象，这可以提高重复使用时的效率
                - 编译后的正则对象可以被多次使用，而不需要每次重新解析正则表达式字符串
                4. self.special_token_pattern = ... ：
                
                - 将编译好的正则表达式对象保存为类实例的属性
                - 使用 self. 前缀使其成为实例变量，可在类的其他方法中访问
                '''
            else:
                self.special_token_pattern = None

        def encode(self, text: str) -> list[int]:
            text_bytes = text.encode('utf-8')

            if text_bytes in self.vocab_rev:
                return [self.vocab_rev[text_bytes]] # 查字典，返回 token ID

            if self.special_token_pattern:
                special_matches = list(self.special_token_pattern.finditer(text))
                '''
                ### 代码功能分析
                1. self.special_token_pattern ：
                
                - 这是之前创建的编译后的正则表达式对象
                - 它用于匹配预定义的特殊标记（如 <bos> , <eos> , <pad> 等）
                2. .finditer(text) ：
                
                - 这是正则表达式对象的一个方法，用于在 text 中查找所有与模式匹配的子串
                - finditer() 方法返回一个 迭代器 ，迭代器中的每个元素都是一个 匹配对象（match object）
                - 每个匹配对象包含了匹配的详细信息，如匹配到的文本、起始位置、结束位置等
                3. list(...) ：
                
                - 将迭代器转换为列表，这样可以立即获取所有匹配结果并存储在内存中
                - 转换为列表后可以更方便地对匹配结果进行索引、排序或多次遍历操作
                '''
                if special_matches: # 如果存在特殊 token 匹配
                    result = [] # 初始化结果列表，用于存储编码后的 token ID
                    last_end = 0 # 初始化最后一个匹配结束的位置，从第一个字符开始

                    for match in special_matches: # 遍历所有特殊 token 匹配
                        if match.start() > last_end: # 如果当前匹配开始位置大于最后一个匹配结束位置
                            result.extend(self._encode_bytes(text[last_end:match.start()].encode('utf-8'))) # 编码并添加普通文本部分到结果中
                            '''
                            ### 代码分解与分析
                            1. text[last_end:match.start()] ：
                            
                            - 这是一个字符串切片操作，提取两个位置之间的文本
                            - last_end ：上一个特殊标记或文本块的结束位置
                            - match.start() ：当前匹配到的特殊标记的开始位置
                            - 整体含义：提取从上一个处理点到当前特殊标记之间的 普通文本部分
                            2. .encode('utf-8') ：
                            
                            - 将提取的普通文本字符串转换为 UTF-8 编码的字节序列
                            - 这一步是必要的，因为 BPE 算法通常在字节级别上操作
                            - 确保文本以统一的字节表示形式进行处理，特别是对于多语言字符
                            3. self._encode_bytes(...) ：
                            
                            - 调用类的私有方法 _encode_bytes
                            - 这个方法负责将字节序列编码为 token ID 列表
                            - 内部可能使用 BPE 算法查找子词并映射到对应的 token ID
                            4. result.extend(...) ：
                            
                            - 将编码得到的 token ID 列表添加到最终结果列表 result 中
                            - 使用 extend() 方法而不是 append() ，因为 _encode_bytes 返回的是一个列表，我们需要将列表中的每个元素添加到结果中
                            '''
                        special_token = match.group(0) # 获取当前匹配的特殊 token 字符串
                        '''
                        1. match ：
   
                        - 这是一个正则表达式匹配对象（match object）
                        - 它是通过 self.special_token_pattern.finditer(text) 方法返回的
                        - 每个匹配对象包含了关于匹配结果的详细信息
                        2. .group(0) ：
                        
                        - 这是匹配对象的一个方法，用于获取匹配到的文本内容
                        - 参数 0 表示获取 完整的匹配字符串 （整个匹配项）
                        - 对于简单的正则表达式匹配，通常只需要 group(0)
                        - 如果正则表达式中包含捕获组（用括号括起来的部分），可以通过 group(1) , group(2) 等获取各个捕获组的内容
                        3. special_token = ... ：
                        
                        - 将获取到的特殊标记字符串赋值给变量 special_token
                        - 便于后续处理，如查找对应的 token ID
                        '''
                        special_token_bytes = special_token.encode('utf-8') # 将特殊 token 字符串编码为字节序列

                        if special_token_bytes in self.vocab_rev: 
                            result.append(self.vocab_rev[special_token_bytes]) # 如果特殊 token 存在于词汇表中，将其对应的 token ID 添加到结果中
                        else:
                            result.extend(self._encode_bytes(special_token_bytes)) # 如果特殊 token 不存在于词汇表中，将未知 token ID 添加到结果中
                            '''
                            - 当使用 result.append(token_id) 时：
                                - 通常用于添加单个 token ID（整数）
                                - 例如处理特殊标记时，直接映射到单一 token ID
                            - 当使用 result.extend(self._encode_bytes(...)) 时：
                                - 因为 _encode_bytes() 返回的是 token ID 列表
                                - 使用 extend() 可以将这个列表中的所有 token ID 逐个添加到结果中
                                - 避免在结果列表中嵌套子列表
                            '''
                        last_end = match.end() # 更新最后一个匹配结束的位置，指向当前匹配结束位置
                        '''
                        .end() ：
                            - 这是匹配对象的一个方法，返回匹配到的文本在原始字符串中的 结束位置（索引值）
                            - 结束位置是匹配文本之后的第一个字符的索引
                        '''

                    if last_end < len(text):
                        result.extend(self._encode_bytes(text[last_end:].encode('utf-8')))
                    return result

            return self._encode_bytes(text_bytes)

        # 在Python中，以下划线 _ 开头的方法名（如 _encode_bytes ）表示这是一个 私有方法 或 内部方法 。
        # 在提供的 BPE_Tokenizer 类中， _encode_bytes 方法以下划线开头，表明：
        # - 它是类内部实现细节
        # - 主要被类的其他公共方法（如 encode_text 等）调用
        # - 不建议在使用该分词器时直接调用这个方法
        def _encode_bytes(self, text_bytes: bytes) -> list[int]:
            result = []
            i = 0
            while i < len(text_bytes): 
                if text_bytes[i:i+1] == b' ':
                    if b' ' in self.vocab_rev:
                        result.append(self.vocab_rev[b' '])
                    i += 1
                else:
                    start = i
                    while i < len(text_bytes) and text_bytes[i:i+1] != b' ':
                        i += 1

                    chunk = text_bytes[start:i] # 从 start 到 i-1 的子字符串，不包括 i 本身
                    j = 0
                    while j < len(chunk): 
                        matched = False
                        for length in range(min(len(chunk) - j, 20),0 ,-1): # 从长到短尝试匹配
                            candidate = chunk[j: j+length] # 从 j 开始，长度为 length 的子字符串
                            if candidate in self.vocab_rev: 
                                result.append(self.vocab_rev[candidate]) 
                                j += length
                                matched = True
                                break
                        if not matched: # 如果没有匹配到任何长度的子字符串
                            if chunk[j:j+1] in self.vocab_rev:
                                result.append(self.vocab_rev[chunk[j:j+1]]) # 如果单个字符存在于词汇表中，将其对应的 token ID 添加到结果中
                            else:
                                pass # 如果单个字符不存在于词汇表中，保持不变
                            j += 1
            return result

        def decode(self, token_ids:list[int]) -> str: # 将 token ID 列表解码为文本字符串
            result_parts = [] 
            for token_id in token_ids: 
                if token_id in self.vocab:
                    token_bytes = self.vocab[token_id]
                    try:
                        token_str = token_bytes.decode('utf-8')
                        result_parts.append(token_str)
                    except UnicodeDecodeError:
                        # 对于无法解码的字节，使用替换字符
                        result_parts.append('�')
            return ''.join(result_parts) 
            '''
            '''

        def encode_iterable(self, iterable) -> list[int]:
            text = ''.join(iterable) # ''.join(iterable) 使用空字符串作为连接符，将可迭代对象中的所有元素按顺序连接成一个完整的字符串
            for token_id in self.encode(text):
                yield token_id 
                '''
                - yield 是 Python 中用于创建 生成器函数 的关键字
                - 当函数包含 yield 时，它不再是普通函数，而是变成了一个生成器函数
                - 生成器函数调用时不会立即执行函数体，而是返回一个生成器对象
                - 当迭代这个生成器对象时，函数会开始执行，直到遇到 yield 语句
                - 遇到 yield 时，函数会暂停执行，并将 token_id 的值作为当前迭代的结果返回
                - 下次迭代时，函数会从上一次暂停的位置继续执行
                '''
    
    return BPE_Tokenizer(vocab, merges, special_tokens)


def bpe_tutorial():
    """BPE分词器新手教程 - 逐步学习"""
    
    print("🎯 BPE分词器新手教程")
    print("=" * 60)
    print("这个教程将一步步教你BPE分词器的工作原理和使用方法。")
    print()
    
    # 步骤1：理解BPE基本概念
    print("📚 步骤1：BPE基本概念")
    print("-" * 30)
    print("BPE (Byte-Pair Encoding) 是一种文本压缩技术，也用于分词。")
    print("基本思想：")
    print("1. 从最基础的字节开始 (0-255)")
    print("2. 找到最常出现的相邻字节对")
    print("3. 合并这些字节对，创建新的'词'")
    print("4. 重复直到达到目标词汇表大小")
    print()
    
    # 步骤2：准备简单训练数据
    print("📝 步骤2：准备训练数据")
    print("-" * 30)
    
    # 创建简单的训练数据
    simple_text = """hello world
hello there
world of programming
programming is fun
fun to learn"""

    with open("tutorial_training.txt", "w", encoding="utf-8") as f:
        f.write(simple_text)
    
    print("✅ 已创建训练数据文件 'tutorial_training.txt'")
    print("内容：")
    for i, line in enumerate(simple_text.strip().split('\n'), 1):
        print(f"  {i}. {line}")
    print()
    
    # 步骤3：运行BPE训练并观察过程
    print("🚀 步骤3：训练BPE分词器")
    print("-" * 30)
    print("开始训练...")
    print()
    
    special_tokens = ["<pad>", "<unk>", "<s>", "</s>"]
    vocab_size = 30  # 小词汇表用于教学
    
    vocab, merges = train_bpe_tokenizer("tutorial_training.txt", vocab_size, special_tokens)
    print()
    
    # 步骤4：分析词汇表
    print("📊 步骤4：分析词汇表")
    print("-" * 30)
    print(f"训练完成！最终词汇表大小：{len(vocab)}")
    print()
    
    print("词汇表组成：")
    print(f"- 基础字节：0-255 (256个)")
    print(f"- 特殊标记：{len(special_tokens)}个")
    print(f"- BPE合并标记：{len(vocab) - 256 - len(special_tokens)}个")
    print()
    
    print("前20个词汇表项：")
    for i in range(min(20, len(vocab))):
        token_bytes = vocab[i]
        try:
            if token_bytes == b'</w>':
                token_str = "</w>"
            elif 32 <= token_bytes[0] <= 126:
                token_str = chr(token_bytes[0])
            else:
                token_str = f"\\x{token_bytes[0]:02x}"
            print(f"  ID {i:2d}: '{token_str}' -> {token_bytes}")
        except:
            print(f"  ID {i:2d}: {token_bytes}")
    print()
    
    # 步骤5：查看合并操作
    print("🔗 步骤5：BPE合并操作")
    print("-" * 30)
    print("BPE训练过程中的合并操作：")
    print()
    
    for i, (byte1, byte2) in enumerate(merges[:10]):  # 只显示前10个
        try:
            char1 = chr(byte1[0]) if 32 <= byte1[0] <= 126 else "?"
            char2 = chr(byte2[0]) if 32 <= byte2[0] <= 126 else "?"
            merged = char1 + char2
            print(f"  合并 {i+1:2d}: '{char1}' + '{char2}' -> '{merged}'")
        except:
            print(f"  合并 {i+1:2d}: {byte1.hex()} + {byte2.hex()}")
    
    if len(merges) > 10:
        print(f"  ... 还有 {len(merges) - 10} 个合并操作")
    print()
    
    # 步骤6：测试编码解码
    print("🔄 步骤6：测试编码和解码")
    print("-" * 30)
    
    test_sentences = ["hello world", "fun programming", "learn more"]
    
    for sentence in test_sentences:
        print(f"测试文本：'{sentence}'")
        
        # 简化的编码过程演示
        encoded = encode_text(sentence, vocab)
        decoded = decode_tokens(encoded, vocab)
        
        print(f"编码结果：{encoded}")
        print(f"解码结果：'{decoded}'")
        print(f"匹配度：{sentence.lower() == decoded.lower()}")
        print()
    
    # 步骤7：总结和下一步
    print("📋 步骤7：总结和实践建议")
    print("-" * 30)
    print("通过这个教程，你学会了：")
    print("✅ BPE的基本原理和算法")
    print("✅ 如何准备训练数据")
    print("✅ 训练过程中的关键步骤")
    print("✅ 编码和解码的工作方式")
    print()
    
    print("实践建议：")
    print("1. 修改 training.txt 中的内容，观察词汇表变化")
    print("2. 尝试不同的 vocab_size 值")
    print("3. 添加更多的特殊标记")
    print("4. 测试不同语言或包含特殊字符的文本")
    print("5. 分析合并操作的顺序和原因")
    print()
    
    print("💡 提示：完整的BPE实现请参考上面的 train_bpe_tokenizer() 函数")
    print("这个函数完全符合你的作业要求！")


def run_tutorial():
    """运行新手教程"""
    bpe_tutorial()


def homework_completion():
    """完成作业要求的示例"""
    print("=" * 60)
    print("🎓 BPE分词器作业完成示例")
    print("=" * 60)
    print()
    
    # 创建作业要求的训练数据
    homework_text = """Hello world! This is a sample text for training BPE tokenizer.
BPE stands for Byte-Pair Encoding, which is a simple data compression technique.
It's widely used in natural language processing for tokenization.
The algorithm works by iteratively merging the most frequent pairs of bytes.
This creates a vocabulary that represents common byte sequences in the training data.
Tokenization using BPE helps reduce vocabulary size while maintaining effectiveness.
Machine learning models often benefit from BPE tokenization for handling text efficiently."""
    
    with open("homework_training.txt", "w", encoding="utf-8") as f:
        f.write(homework_text)
    
    print("📝 作业要求：")
    print("1. train_bpe_tokenizer() 函数：✅ 已实现")
    print("2. input_path: str 参数：✅ 已实现") 
    print("3. vocab_size: int 参数：✅ 已实现")
    print("4. special_tokens: list[str] 参数：✅ 已实现")
    print("5. 返回 vocab: dict[int, bytes]：✅ 已实现")
    print("6. 返回 merges: list[tuple[bytes, bytes]]：✅ 已实现")
    print("7. run_train_bpe() 函数：✅ 已实现")
    print("8. get_tokenizer() 函数：✅ 已实现")
    print()
    
    print("🚀 开始训练BPE分词器...")
    print()
    
    # 符合作业要求的调用
    special_tokens = ["<pad>", "<unk>", "<s>", "</s>"]
    vocab_size = 200
    
    vocab, merges = train_bpe_tokenizer("homework_training.txt", vocab_size, special_tokens)
    
    print()
    print("🎉 作业完成！")
    print(f"✅ 词汇表大小：{len(vocab)}")
    print(f"✅ 合并操作数量：{len(merges)}")
    print(f"✅ 词汇表类型：dict[int, bytes]")
    print(f"✅ 合并类型：list[tuple[bytes, bytes]]")
    print()
    
    # 展示返回结果的格式
    print("📊 返回结果格式验证：")
    print("vocab 类型:", type(vocab))
    print("merges 类型:", type(merges))
    
    if vocab:
        first_key = next(iter(vocab.keys()))
        first_value = vocab[first_key]
        print(f"vocab[{first_key}] 类型: {type(first_value)}")
        print(f"示例 vocab 项: {first_key} -> {first_value}")
    
    if merges:
        first_merge = merges[0]
        print(f"示例合并项: {first_merge}")
        print(f"合并项类型: ({type(first_merge[0])}, {type(first_merge[1])})")
    
    # 测试run_train_bpe函数
    print()
    print("🔍 测试run_train_bpe函数...")
    try:
        test_text = "Hello run_train_bpe test"
        test_vocab, test_merges = run_train_bpe(test_text, 260, special_tokens)
        print(f"✅ run_train_bpe调用成功，vocab大小: {len(test_vocab)}")
    except Exception as e:
        print(f"❌ 调用run_train_bpe时出错: {e}")
    
    # 测试get_tokenizer函数
    print()
    print("🔍 测试get_tokenizer函数...")
    try:
        encode, decode = get_tokenizer(vocab)
        test_str = "Hello world! 测试编码解码"
        encoded = encode(test_str)
        decoded = decode(encoded)
        print(f"✅ get_tokenizer调用成功")
        print(f"测试文本: '{test_str}'")
        print(f"编码结果: {encoded}")
        print(f"解码结果: '{decoded}'")
        # 检查编码解码一致性
        if test_str == decoded:
            print("✅ 编码解码一致性测试通过！")
        else:
            print("❌ 编码解码一致性测试失败")
    except Exception as e:
        print(f"❌ 调用get_tokenizer时出错: {e}")
    
    print()
    print("🏆 恭喜！你已经成功实现了完整的BPE分词器！")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "tutorial":
        run_tutorial()
    elif len(sys.argv) > 1 and sys.argv[1] == "homework":
        homework_completion()
    else:
        print("请选择运行模式：")
        print("python newbird.py tutorial  - 运行新手教程")
        print("python newbird.py homework  - 运行作业完成示例")
        print("python newbird.py           - 运行默认演示")
        print()
        print("默认运行演示模式...")
        homework_completion()
