#!/usr/bin/env python3

from hmac import new
from pydoc import text
from tkinter import WORD
from typing import Any

# 定义一个 Byte-level BPE tokenizer
def train_bep_tokenizer(input_path:str, vocab_size:int, special_tokens:list[str]):

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

    



