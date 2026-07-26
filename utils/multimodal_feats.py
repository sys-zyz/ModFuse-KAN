class SafeSMILESTokenizer:
    def __init__(self, max_len=50):
        self.max_len = max_len
        # 扩展SMILES常用字符（包含特殊符号）
        self.vocab = list('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789()[]=-+#$%/:.,@\\~πσλθ')
        self.char2idx = {c: i + 1 for i, c in enumerate(self.vocab)}  # 0=padding
        self.idx2char = {i + 1: c for i, c in enumerate(self.vocab)}

    def encode(self, s):
        ids = [self.char2idx.get(c, 0) for c in s]  # 保留原始大小写
        ids = ids[:self.max_len]
        ids += [0] * (self.max_len - len(ids))
        return ids

    def decode(self, ids):
        """新增解码函数，方便调试"""
        chars = [self.idx2char.get(i, '') for i in ids if i != 0]
        return ''.join(chars)

