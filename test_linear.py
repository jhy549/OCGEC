import torch 

# print(576//128)
def window(ip_tensor,ip_size,out_size):
    compressed_tensor = torch.empty(out_size)
    window_size = ip_size//out_size + 1
    stride = ip_size//out_size + 1
    for i in range(0,ip_size , stride):
    # 提取滑窗区域
        window = ip_tensor[i:i+window_size]
    
    # 计算滑窗区域的平均值
        average = torch.mean(window)
    
    # 将平均值添加到压缩张量中
        compressed_tensor[i//stride] = average
    return compressed_tensor

# 最终得到的压缩张量大小为[128]
input_tensor = torch.tensor([1,2,3,4,5,6],dtype=float)
print(window(input_tensor,6,3))