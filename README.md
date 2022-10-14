
参数classes 默认2 前景或背景

输入
image和mask尺寸相同
assert img.size == mask.size,  f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

输入尺寸（3channel）
torch.Size([1, 3, 640, 959])

if img_ndarray.ndim == 2:
    img_ndarray = img_ndarray[np.newaxis, ...]
else:
    img_ndarray = img_ndarray.transpose((2, 0, 1))
3通道数据增加一维batch



DoubleConv两次卷积通用模块

F.pad 扩充
https://blog.csdn.net/jorg_zhao/article/details/105295686

逆卷积ConvTranspose2d
https://blog.csdn.net/qq_27261889/article/details/86304061



损失两部分
https://blog.csdn.net/smallworldxyl/article/details/121568778
loss = criterion(masks_pred, true_masks) + dice_loss(F.softmax(masks_pred, dim=1).float(),
F.one_hot(true_masks, net.n_classes).permute(0, 3, 1, 2).float(),multiclass=True)
1、类别损失，前景或背景
2、集合相似度损失


multiclass_dice_coeff计算每个channel集合损失
    dice_coeff计算每个样本的集合损失




f.softmax(dim)
tf.nn.functional.softmax(x,dim = -1)中的参数dim是指维度的意思
要注意的是当dim=0时， 是对每一维度相同位置的数值进行softmax运算
要注意的是当dim=1时， 是对某一维度的列进行softmax运算
要注意的是当dim=2时， 是对某一维度的行进行softmax运算
要注意的是当dim=-1时， 是对某一维度的行进行softmax运算
dim=-1和dim=2的结果是一样的


f.onehot
x =  tensor([1, 1, 1, 3, 3, 4, 8, 5])
x_shape =  torch.Size([8])
y2 =  tensor([[0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0]])
y2_shape =  torch.Size([8, 10])




evaluate

