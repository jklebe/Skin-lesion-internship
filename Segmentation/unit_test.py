import numpy as np
import torch
from train_model_seg import IaU, JaccardAccuracy, IoU, pixel_accuracy 
from eval_model_seg import ratio_backgroundVSlesion, acc_of_nobackground

def create_data():
    a_0 = np.zeros((7,7,20,30))
    a_1 = np.ones((7,1,20,30)) * 0.1

    a = np.concatenate([a_0, a_1], axis = 1)
    b = np.ones((7,1,20,30)) * 7

    for i in range(7):
        for j in range(15):
            for k in range(10):
                a[i,i,k,j] = 1
                b[i,0,k + i, j + i] = i

    aten = torch.from_numpy(a)
    bten = torch.from_numpy(b)

    return aten, bten


def test_iou(aten, bten):
    expection = [ 150, 126, 104,  84,  66,  50,  36,
                  150, 174, 196, 216, 234, 250, 264]

    print('IaU')
    print('expectation: {}'.format(expection))
    arr = IaU(mask = bten, prediction = aten)
    print('result: {}'.format(arr))
    
    print('\nIoU:')
    arr_new = np.zeros((7,))
    for i in range(7): 
        print(i, arr[i] / arr[i + 7])
        arr_new[i] = arr[i] / arr[i + 7]

    print('\nexpectations:')
    print('Jaccard: {}'.format((1 + 126 / 174) / 7))
    print('IoU: {}'.format(arr_new.mean()))
    
    print('\results:')
    print('jacc: {}'.format(JaccardAccuracy(arr)))
    print('iou: {}'.format(IoU(arr)))


def test_pixel_accuracy(pred, mask):
   

    for i in range(11):
        a = pixel_accuracy(mask, pred, 0.1*i)
        print(i * 0.1, a[0] / a[1])

def test_rbvl(mask):
    a = ratio_backgroundVSlesion(mask)
    print(a[0] / a[1])

def test_acc_of_no_background(pred, mask):

    print('test_acc_of_no...')
    for i in range(11):
        print(0.1 * i, acc_of_nobackground(mask, pred, i * 0.1))

if __name__ == '__main__':
    pred, mask = create_data()

    test_iou(pred, mask)
    test_pixel_accuracy(pred, mask)
    test_rbvl(mask)
    test_acc_of_no_background(pred, mask)
