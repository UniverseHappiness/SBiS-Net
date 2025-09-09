
import argparse
import os
import shutil

import h5py
import nibabel as nib
import numpy as np
import SimpleITK as sitk
import torch
from medpy import metric
from scipy.ndimage import zoom
from tqdm import tqdm
from model.unet import BiDSA_UNet
import torch.nn.functional as F

import csv

FLAGS = {
    'exp': 'unimatch_sce3_reconstruction_new',
    'labeled_num': '7',
    'subfolder':'',
    'model': 'unet',
    'epoch':'92'
}

FLAGS = argparse.Namespace(**FLAGS)

def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    dice = metric.binary.dc(pred, gt)
    iou = metric.binary.jc(pred, gt)
    asd = metric.binary.asd(pred, gt)
    hd95 = metric.binary.hd95(pred, gt)
    return dice,iou, hd95, asd

def test_single_volume(case, net, test_save_path, csv_writer):
    print(test_save_path)
    h5f = h5py.File("dataset/datasets/ACDC/{}.h5".format(case), 'r')
    image = h5f['image'][:]
    label = h5f['label'][:]
    prediction = np.zeros_like(label)
    # print(case)
    # print(image.shape)
    # print(label.shape)
    for ind in range(image.shape[0]):
        slice = image[ind, :, :]
        # print("original slice:",slice.shape)
        x, y = slice.shape[0], slice.shape[1]
        # slice = zoom(slice, (256 / x, 256 / y), order=0) # 0
        # 假设 slice 是形状为 (C, H, W) 或 (1, C, H, W) 的 numpy array
       
        input = torch.from_numpy(slice).float()  # 转换为 float 类型的 tensor
        input = input.unsqueeze(0).unsqueeze(0).float().cuda()
        input = F.interpolate(input, (256, 256), mode='bilinear', align_corners=False)

        # print("zoom original slice:", slice.shape)
        # input = torch.from_numpy(slice).unsqueeze(
        #     0).unsqueeze(0).float().cuda()
        # input = slice_tensor.unsqueeze(
        #     0).unsqueeze(0).float().cuda()
        # print("input:", input.shape)
        net.eval()
        with torch.no_grad():
            if FLAGS.model == "unet_urds":
                out_main, _, _, _ = net(input)
            else:
                out_main = net(input)
                # print("out_main",out_main.shape)
            out = torch.argmax(torch.softmax(
                out_main, dim=1), dim=1).squeeze(0)
            
            # print("out",out.shape)
            # out = out.cpu().detach().numpy()
            # pred = zoom(out, (x / 256, y / 256), order=0) # 0
            pred = F.interpolate(out_main, (x ,y), mode='bilinear', align_corners=False)
            pred = pred.argmax(dim=1).unsqueeze(0)
            pred = pred.cpu().detach().numpy()
            # print("pred:",pred.shape)
            prediction[ind] = pred

    # print(prediction.shape)
    # 计算每个类别的指标
    first_metric = calculate_metric_percase(prediction == 1, label == 1)
    second_metric = calculate_metric_percase(prediction == 2, label == 2)
    third_metric = calculate_metric_percase(prediction == 3, label == 3)

    # 将结果写入 CSV 文件
    csv_writer.writerow([
        case,  # 样本名称
        *first_metric,  # 类别 1 的指标 (Dice, IoU, HD95, ASD)
        *second_metric,  # 类别 2 的指标
        *third_metric  # 类别 3 的指标
    ])

    # 保存预测结果为 NIfTI 文件
    img_itk = sitk.GetImageFromArray(image.astype(np.float32))
    img_itk.SetSpacing((1, 1, 10))
    prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
    prd_itk.SetSpacing((1, 1, 10))
    lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
    lab_itk.SetSpacing((1, 1, 10))
    sitk.WriteImage(prd_itk, test_save_path + case + "_pred.nii.gz")
    sitk.WriteImage(img_itk, test_save_path + case + "_img.nii.gz")
    sitk.WriteImage(lab_itk, test_save_path + case + "_gt.nii.gz")

    return first_metric, second_metric, third_metric

def Inference():
    with open('splits/acdc/test.txt', 'r') as f:
        image_list = f.readlines()
    image_list = sorted([item.replace('\n', '').split(".")[0]
                         for item in image_list])
    snapshot_path = snapshot_path = f'exp/acdc/{FLAGS.exp}/{FLAGS.model}/{FLAGS.labeled_num}/{FLAGS.subfolder}'
    test_save_path = f"exp/acdc/{FLAGS.exp}/{FLAGS.model}/{FLAGS.labeled_num}/{FLAGS.subfolder}/predictions_test_{FLAGS.epoch}/"
    if os.path.exists(test_save_path):
        shutil.rmtree(test_save_path)
    os.makedirs(test_save_path)
    os.makedirs(test_save_path + 'data')

    # 创建 CSV 文件并写入表头
    csv_path = os.path.join(test_save_path, 'results.csv')
    with open(csv_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow([
            'Case',  # 样本名称
            'Dice_1', 'IoU_1', 'HD95_1', 'ASD_1',  # 类别 1 的指标
            'Dice_2', 'IoU_2', 'HD95_2', 'ASD_2',  # 类别 2 的指标
            'Dice_3', 'IoU_3', 'HD95_3', 'ASD_3'  # 类别 3 的指标
        ])
        
        # 加载模型
        net = BiDSA_UNet(in_chns=1, class_num=4).cuda()
        save_mode_path = os.path.join(
            snapshot_path, 'epoch' + str(FLAGS.epoch) + '.pth')
        
        # save_mode_path = os.path.join(
        #     snapshot_path, 'best.pth')
        temp_dict = torch.load(save_mode_path)['model']
        temp_dict = {k.replace('module.',''):v for k,v in temp_dict.items()}
        net.load_state_dict(temp_dict, strict=False)
        print("init weight from {}".format(save_mode_path))
        net.eval()

        # 测试每个样本并记录结果
        first_total = 0.0
        second_total = 0.0
        third_total = 0.0
        for case in tqdm(image_list):
            first_metric, second_metric, third_metric = test_single_volume(
                case, net, test_save_path, csv_writer)
            first_total += np.asarray(first_metric)
            second_total += np.asarray(second_metric)
            third_total += np.asarray(third_metric)

        # 计算平均指标
        avg_metric = [first_total / len(image_list), second_total /
                      len(image_list), third_total / len(image_list)]
        return avg_metric
    

    
if __name__ == '__main__':
    metric = Inference()
    print((metric[0]+metric[1]+metric[2])/3)