import numpy as np

def postprocess_batch(batch):
    batch_new = np.zeros_like(batch)
    for i in range(batch.shape[0]):
        batch_new[i] = post_process(batch[i]) 
    return batch_new

def post_process(img):
    size = 11
    dc = dominant_class(img)
    img_bin = np.where(img == dc, 1, 0)
    img_bin = close(img_bin, size)
    img = np.where(img_bin == 1, dc, img)

    
    img_bin = np.where(img == 7, 1, 0)
    img_bin = close(img_bin, size)
    img = np.where(img_bin == 1, 7, img)
    
    img_bin = np.where(img == 7, 1, 0)
    img_bin = close(img_bin, size)
    img = np.where(img_bin == 1, 7, img)
    
    img_bin = np.where(img == dc, 1, 0)
    img_bin = close(img_bin, size)
    img = np.where(img_bin == 1, dc, img)
    
    #img_bin = close(img_bin, 9)

    
    return img

def close(img, size = 7):
    out = np.zeros((1, img.shape[1] + size - 1, img.shape[2] + size - 1))

    for i in range(size):
        for j in range(size):
            if (i - size // 2)**2 + (j -size // 2) ** 2 > (size // 2) ** 2:
                continue
            out = out + np.pad(img, ((0,0), (i, size - i - 1), (j, size - j - 1)))
    out = np.where(out > 0, 1, 0)
    out = out[:, size // 2:-size // 2 + 1, size // 2:-size // 2 + 1]
    return out

def dominant_class(img):
    a = np.zeros(7)
    for i in range(7):
        a[i] = np.sum(np.where(img == i, 1, 0))

    return np.argmax(a)
