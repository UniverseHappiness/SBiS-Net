import argparse
import logging
import os
import pprint

import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import yaml

from dataset.acdc import ACDCDataset
from model.unet import UNet_contrastive
from util.classes import CLASSES
from util.utils import AverageMeter, count_params, init_log, DiceLoss
from util.dist_helper import setup_distributed


parser = argparse.ArgumentParser(description='Revisiting Weak-to-Strong Consistency in Semi-Supervised Semantic Segmentation')
parser.add_argument('--config', type=str, required=True)
parser.add_argument('--labeled-id-path', type=str, required=True)
parser.add_argument('--unlabeled-id-path', type=str, required=True)
parser.add_argument('--save-path', type=str, required=True)
parser.add_argument('--local-rank', default=0, type=int)
parser.add_argument('--port', default=None, type=int)


# def elastic_contrastive_loss(feature1, feature2, epsilon=0.05, temperature=1):
#     """
#     计算两组特征的对比损失，并增加扰动，提高 Elasticity
#     feature1, feature2: 经过不同增强的特征
#     epsilon: 扰动的强度
#     """
#     feature1 = F.normalize(feature1, dim=1)
#     feature2 = F.normalize(feature2, dim=1)

#     # 加入小扰动，模拟弹性变化
#     # noise = epsilon * torch.randn_like(feature1)
#     # perturbed_feature1 = F.normalize(feature1 + noise, dim=1)
 
#     # 计算相似度
#     similarity = torch.sum(feature1  * feature2, dim=1)

#     # 计算对比损失
#     loss = -torch.mean(similarity / temperature)
#     return loss


def elastic_contrastive_loss(feature1, feature2, epsilon=0.05, temperature=1):
    """
    计算两组特征的对比损失，并增加扰动，提高 Elasticity
    feature1, feature2: 经过不同增强的特征
    epsilon: 扰动的强度
    temperature: 温度参数，控制对比损失的尺度
    """
    feature1 = F.normalize(feature1, dim=1)
    feature2 = F.normalize(feature2, dim=1)

    # 计算相似度（取负数转换为损失）
    similarity = torch.sum(feature1 * feature2, dim=1)

    # 转换成正的损失 (1 - similarity) / temperature
    loss = torch.mean((1 - similarity) / temperature)

    return loss


# def info_nce_loss(feature1, feature2, temperature=0.1):
#     """
#     计算 InfoNCE 对比损失，适用于对比学习
#     feature1, feature2: 经过不同增强的特征，shape=(B, C)
#     temperature: 控制 softmax 分布的平滑度
#     """
#     # 归一化特征
#     feature1 = F.normalize(feature1, dim=1)  # (B, C)
#     feature2 = F.normalize(feature2, dim=1)  # (B, C)

#     # 计算相似度矩阵 (B, B)，即 feature1 的每个样本与 feature2 所有样本的相似度
#     similarity_matrix = torch.matmul(feature1, feature2.T) / temperature

#     # 生成正例（对角线元素是正例）
#     batch_size = feature1.shape[0]
#     labels = torch.arange(batch_size).to(feature1.device)

#     # 计算 InfoNCE 损失
#     loss = F.cross_entropy(similarity_matrix, labels)
#     return loss


def channel_group_pooling(feature, groups=2):
    """
    使用 Group Pooling 进行通道降维，将 feature 按 groups 分组，并在每个组内取均值
    feature: 输入特征，形状为 [B, C, H, W]
    groups: 分组数，C 必须能被 groups 整除
    返回降维后的特征，形状为 [B, C // groups, H, W]
    """
    B, C, H, W = feature.shape
    assert C % groups == 0, f"通道数 {C} 不能被 groups={groups} 整除"
    
    feature = feature.view(B, groups, C // groups, H, W)  # 变成 [B, groups, C//groups, H, W]
    pooled_feature = feature.mean(dim=1)  # 在组维度上取均值，变成 [B, C//groups, H, W]
    
    return pooled_feature



def main():
    args = parser.parse_args()

    cfg = yaml.load(open(args.config, "r"), Loader=yaml.Loader)

    logger = init_log('global', logging.INFO)
    logger.propagate = 0

    rank, world_size = setup_distributed(port=args.port)

    if rank == 0:
        all_args = {**cfg, **vars(args), 'ngpus': world_size}
        logger.info('{}\n'.format(pprint.pformat(all_args)))
        
        writer = SummaryWriter(args.save_path)
        
        os.makedirs(args.save_path, exist_ok=True)
    
    cudnn.enabled = True
    cudnn.benchmark = True

    model = UNet_contrastive(in_chns=1, class_num=cfg['nclass'])    
    if rank == 0:
        logger.info('Total params: {:.1f}M\n'.format(count_params(model)))
        
    optimizer = SGD(model.parameters(), cfg['lr'], momentum=0.9, weight_decay=0.0001)

    local_rank = int(os.environ["LOCAL_RANK"])
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model.cuda()

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], broadcast_buffers=False,
                                                      output_device=local_rank, find_unused_parameters=False)

    criterion_ce = nn.CrossEntropyLoss()
    criterion_dice = DiceLoss(n_classes=cfg['nclass'])
    
    trainset_u = ACDCDataset(cfg['dataset'], cfg['data_root'], 'train_u',
                             cfg['crop_size'], args.unlabeled_id_path)
    trainset_l = ACDCDataset(cfg['dataset'], cfg['data_root'], 'train_l',
                             cfg['crop_size'], args.labeled_id_path, nsample=len(trainset_u.ids))
    valset = ACDCDataset(cfg['dataset'], cfg['data_root'], 'val')
    testset = ACDCDataset(cfg['dataset'], cfg['data_root'], 'test')

    trainsampler_l = torch.utils.data.distributed.DistributedSampler(trainset_l)
    trainloader_l = DataLoader(trainset_l, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=8, drop_last=True, sampler=trainsampler_l)
    trainsampler_u = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                               pin_memory=True, num_workers=8, drop_last=True, sampler=trainsampler_u)
    trainsampler_u_mix = torch.utils.data.distributed.DistributedSampler(trainset_u)
    trainloader_u_mix = DataLoader(trainset_u, batch_size=cfg['batch_size'],
                                   pin_memory=True, num_workers=8, drop_last=True, sampler=trainsampler_u_mix)
    valsampler = torch.utils.data.distributed.DistributedSampler(valset)
    valloader = DataLoader(valset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=valsampler)
    
    testsampler = torch.utils.data.distributed.DistributedSampler(testset)
    testloader = DataLoader(testset, batch_size=1, pin_memory=True, num_workers=1,
                           drop_last=False, sampler=testsampler)


    total_iters = len(trainloader_u) * cfg['epochs']
    previous_best = 0.0
    epoch = -1
    
    if os.path.exists(os.path.join(args.save_path, 'latest.pth')):
        checkpoint = torch.load(os.path.join(args.save_path, 'latest.pth'))
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        epoch = checkpoint['epoch']
        previous_best = checkpoint['previous_best']
        
        if rank == 0:
            logger.info('************ Load from checkpoint at epoch %i\n' % epoch)
    
    for epoch in range(epoch + 1, cfg['epochs']):
        if rank == 0:
            logger.info('===========> Epoch: {:}, LR: {:.5f}, Previous best: {:.2f}'.format(
                epoch, optimizer.param_groups[0]['lr'], previous_best))

        total_loss = AverageMeter()
        total_loss_x = AverageMeter()
        total_loss_s = AverageMeter()
        total_loss_w_fp = AverageMeter()
        total_loss_w_elastic_0 = AverageMeter()
        total_loss_w_elastic_1 = AverageMeter()
        total_loss_w_elastic_2 = AverageMeter()
        total_loss_w_elastic_3 = AverageMeter()
        total_mask_ratio = AverageMeter()

        trainloader_l.sampler.set_epoch(epoch)
        trainloader_u.sampler.set_epoch(epoch)
        trainloader_u_mix.sampler.set_epoch(epoch + cfg['epochs'])
        
        loader = zip(trainloader_l, trainloader_u, trainloader_u_mix)

        for i, ((img_x, mask_x),
                (img_u_w, img_u_s1, img_u_s2, cutmix_box1, cutmix_box2),
                (img_u_w_mix, img_u_s1_mix, img_u_s2_mix, _, _)) in enumerate(loader):
            
            img_x, mask_x = img_x.cuda(), mask_x.cuda()
            img_u_w = img_u_w.cuda()
            img_u_s1, img_u_s2 = img_u_s1.cuda(), img_u_s2.cuda()
            cutmix_box1, cutmix_box2 = cutmix_box1.cuda(), cutmix_box2.cuda()
            img_u_w_mix = img_u_w_mix.cuda()
            img_u_s1_mix, img_u_s2_mix = img_u_s1_mix.cuda(), img_u_s2_mix.cuda()

            with torch.no_grad():
                model.eval()

                pred_u_w_mix = model(img_u_w_mix).detach()
                conf_u_w_mix = pred_u_w_mix.softmax(dim=1).max(dim=1)[0]
                mask_u_w_mix = pred_u_w_mix.argmax(dim=1)

            img_u_s1[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1] = \
                img_u_s1_mix[cutmix_box1.unsqueeze(1).expand(img_u_s1.shape) == 1]
            img_u_s2[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1] = \
                img_u_s2_mix[cutmix_box2.unsqueeze(1).expand(img_u_s2.shape) == 1]

            model.train()

            num_lb, num_ulb = img_x.shape[0], img_u_w.shape[0]

            # low_feature_x, high_feature_x = model(img_x, contrastive_feature=True)
            # low_feature_u_w, high_feature_u_w = model(img_u_w, contrastive_feature=True)
            low_feature_u_s1, high_feature_u_s1 = model(img_u_s1, contrastive_feature=True)
            low_feature_u_s2, high_feature_u_s2 = model(img_u_s2, contrastive_feature=True)



            # print(low_feature[0].shape, low_feature[1].shape, low_feature[2].shape, low_feature[3].shape)
            # print(high_feature[0].shape, high_feature[1].shape, high_feature[2].shape, high_feature[3].shape)
            # feature_s1 = model(img_u_s1_mix, contrastive_feature=True)
            # feature_s2 = model(img_u_s2_mix, contrastive_feature=True)

            # # 计算不同层的 Group Pooling 结果
            # feature1_reduced = channel_group_pooling(feature[1], groups=2)
            # feature2_reduced = channel_group_pooling(feature[2], groups=2)
            # feature3_reduced = channel_group_pooling(feature[3], groups=2)  # 128 -> 64
            # feature4_reduced = channel_group_pooling(feature[4], groups=2)  # 256 -> 128

            # 进行空间尺寸对齐
            # feature1_resized = F.interpolate(feature1_reduced, size=feature[0].shape[2:], mode='nearest')
            # feature2_resized = F.interpolate(feature2_reduced, size=feature[1].shape[2:], mode='nearest')
            # feature3_resized = F.interpolate(feature3_reduced, size=feature[2].shape[2:], mode='nearest')
            # feature4_resized = F.interpolate(feature4_reduced, size=feature[3].shape[2:], mode='nearest')

            # 计算 Elasticity 对比损失

            preds, preds_fp = model(torch.cat((img_x, img_u_w)), need_fp=True)
            pred_x, pred_u_w = preds.split([num_lb, num_ulb])
            pred_u_w_fp = preds_fp[num_lb:]

            pred_u_s1, pred_u_s2 = model(torch.cat((img_u_s1, img_u_s2))).chunk(2)

            pred_u_w = pred_u_w.detach()
            conf_u_w = pred_u_w.softmax(dim=1).max(dim=1)[0]
            mask_u_w = pred_u_w.argmax(dim=1)

            mask_u_w_cutmixed1, conf_u_w_cutmixed1 = mask_u_w.clone(), conf_u_w.clone()
            mask_u_w_cutmixed2, conf_u_w_cutmixed2 = mask_u_w.clone(), conf_u_w.clone()

            mask_u_w_cutmixed1[cutmix_box1 == 1] = mask_u_w_mix[cutmix_box1 == 1]
            conf_u_w_cutmixed1[cutmix_box1 == 1] = conf_u_w_mix[cutmix_box1 == 1]

            mask_u_w_cutmixed2[cutmix_box2 == 1] = mask_u_w_mix[cutmix_box2 == 1]
            conf_u_w_cutmixed2[cutmix_box2 == 1] = conf_u_w_mix[cutmix_box2 == 1]

            loss_x = (criterion_ce(pred_x, mask_x) + criterion_dice(pred_x.softmax(dim=1), mask_x.unsqueeze(1).float())) / 2.0

            loss_u_s1 = criterion_dice(pred_u_s1.softmax(dim=1), mask_u_w_cutmixed1.unsqueeze(1).float(),
                                       ignore=(conf_u_w_cutmixed1 < cfg['conf_thresh']).float())
            
            loss_u_s2 = criterion_dice(pred_u_s2.softmax(dim=1), mask_u_w_cutmixed2.unsqueeze(1).float(),
                                       ignore=(conf_u_w_cutmixed2 < cfg['conf_thresh']).float())
            
            loss_u_w_fp = criterion_dice(pred_u_w_fp.softmax(dim=1), mask_u_w.unsqueeze(1).float(),
                                         ignore=(conf_u_w < cfg['conf_thresh']).float())
            
            
            loss_elastic_0 = elastic_contrastive_loss(low_feature_u_s1[0], high_feature_u_s2[0].detach())
            loss_elastic_1 = elastic_contrastive_loss(low_feature_u_s2[0], high_feature_u_s1[0].detach())
            # loss_elastic_1 = elastic_contrastive_loss(low_feature_u_s1[1], high_feature_u_s2[1])
            # loss_elastic_2 = elastic_contrastive_loss(low_feature_u_s1[2], high_feature_u_s2[2])
            # loss_elastic_3 = elastic_contrastive_loss(low_feature_u_s1[3], high_feature_u_s2[3])
            # loss_elastic = 0.5* loss_elastic_2_3 + 0.5*loss_elastic_3_4
            loss = (loss_x + loss_u_s1 * 0.25 + loss_u_s2 * 0.25 + loss_u_w_fp * 0.5 + loss_elastic_0 * 0.25 + loss_elastic_1 * 0.25) / 2.5
          

            torch.distributed.barrier()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss.update(loss.item())
            total_loss_x.update(loss_x.item())
            total_loss_s.update((loss_u_s1.item() + loss_u_s2.item()) / 2.0)
            total_loss_w_fp.update(loss_u_w_fp.item())
            total_loss_w_elastic_0.update(loss_elastic_0.item())
            total_loss_w_elastic_1.update(loss_elastic_1.item())
            # total_loss_w_elastic_2.update(loss_elastic_2.item())
            # total_loss_w_elastic_3.update(loss_elastic_3.item())

            
            mask_ratio = (conf_u_w >= cfg['conf_thresh']).sum() / conf_u_w.numel()
            total_mask_ratio.update(mask_ratio.item())
            
            iters = epoch * len(trainloader_u) + i
            lr = cfg['lr'] * (1 - iters / total_iters) ** 0.9
            optimizer.param_groups[0]["lr"] = lr
            
            if rank == 0:
                writer.add_scalar('train/loss_all', loss.item(), iters)
                writer.add_scalar('train/loss_x', loss_x.item(), iters)
                writer.add_scalar('train/loss_s', (loss_u_s1.item() + loss_u_s2.item()) / 2.0, iters)
                writer.add_scalar('train/loss_w_fp', loss_u_w_fp.item(), iters)
                writer.add_scalar('train/loss_elastic_0',loss_elastic_0.item(),iters)
                writer.add_scalar('train/loss_elastic_1',loss_elastic_1.item(),iters)
                # writer.add_scalar('train/loss_elastic_2',loss_elastic_2.item(),iters)
                # writer.add_scalar('train/loss_elastic_3',loss_elastic_3.item(),iters)
                writer.add_scalar('train/mask_ratio', mask_ratio, iters)
                
            if (i % (len(trainloader_u) // 8) == 0) and (rank == 0):
                logger.info('Iters: {:}, Total loss: {:.3f}, Loss x: {:.3f}, Loss s: {:.3f}, Loss w_fp: {:.3f}, Loss_w_elastic_0: {:.3f}, Loss_w_elastic_1: {:.3f}, Loss_w_elastic_2_3: {:.3f}, Loss_w_elastic_3_4: {:.3f}, Mask ratio: '
                            '{:.3f}'.format(i, total_loss.avg, total_loss_x.avg, total_loss_s.avg, 
                                            total_loss_w_fp.avg, total_loss_w_elastic_0.avg, total_loss_w_elastic_1.avg, total_loss_w_elastic_2.avg, total_loss_w_elastic_3.avg,total_mask_ratio.avg))
        model.eval()
        dice_class = [0] * 3
        dice_class_t = [0] * 3
        with torch.no_grad():
            for img, mask in valloader:
                img, mask = img.cuda(), mask.cuda()

                h, w = img.shape[-2:]
                img = F.interpolate(img, (cfg['crop_size'], cfg['crop_size']), mode='bilinear', align_corners=False)

                img = img.permute(1, 0, 2, 3)
                
                pred = model(img)
                
                pred = F.interpolate(pred, (h, w), mode='bilinear', align_corners=False)
                pred = pred.argmax(dim=1).unsqueeze(0)

                for cls in range(1, cfg['nclass']):
                    inter = ((pred == cls) * (mask == cls)).sum().item()
                    union = (pred == cls).sum().item() + (mask == cls).sum().item()
                    dice_class[cls-1] += 2.0 * inter / union

            for img_t, mask_t in testloader:
                img_t, mask_t = img_t.cuda(), mask_t.cuda()

                h_t, w_t = img_t.shape[-2:]
                img_t = F.interpolate(img_t, (cfg['crop_size'], cfg['crop_size']), mode='bilinear', align_corners=False)

                img_t = img_t.permute(1, 0, 2, 3)
                
                pred_t = model(img_t)
                
                pred_t = F.interpolate(pred_t, (h_t, w_t), mode='bilinear', align_corners=False)
                pred_t = pred_t.argmax(dim=1).unsqueeze(0)

                for cls in range(1, cfg['nclass']):
                    inter_t = ((pred_t == cls) * (mask_t == cls)).sum().item()
                    union_t = (pred_t == cls).sum().item() + (mask_t == cls).sum().item()
                    dice_class_t[cls-1] += 2.0 * inter_t / union_t

        dice_class = [dice * 100.0 / len(valloader) for dice in dice_class]
        dice_class_t = [dice_t * 100.0 / len(testloader) for dice_t in dice_class_t]
        mean_dice = sum(dice_class) / len(dice_class)
        mean_dice_t = sum(dice_class_t) / len(dice_class_t)
        
        if rank == 0:
            for (cls_idx, dice) in enumerate(dice_class):
                logger.info('***** Evaluation ***** >>>> Class [{:} {:}] Dice: '
                            '{:.2f}'.format(cls_idx, CLASSES[cfg['dataset']][cls_idx], dice))
            logger.info('***** Evaluation ***** >>>> MeanDice: {:.2f}\n'.format(mean_dice))

            for (cls_idx_t, dice_t) in enumerate(dice_class_t):
                logger.info('***** Test Evaluation ***** >>>> Class [{:} {:}] Dice: '
                            '{:.2f}'.format(cls_idx_t, CLASSES[cfg['dataset']][cls_idx_t], dice_t))
            logger.info('*****Test Evaluation ***** >>>> MeanDice: {:.2f}\n'.format(mean_dice_t))
            
            writer.add_scalar('eval/MeanDice', mean_dice, epoch)
            writer.add_scalar('test/MeanDice', mean_dice_t, epoch)
            for i, dice in enumerate(dice_class):
                writer.add_scalar('eval/%s_dice' % (CLASSES[cfg['dataset']][i]), dice, epoch)
            for i, dice_t in enumerate(dice_class_t):
                writer.add_scalar('eval/%s_dice' % (CLASSES[cfg['dataset']][i]), dice_t, epoch)

        is_best = mean_dice > previous_best
        previous_best = max(mean_dice, previous_best)
        if rank == 0:
            checkpoint = {
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'previous_best': previous_best,
            }
            torch.save(checkpoint, os.path.join(args.save_path, 'latest.pth'))
            if is_best:
                torch.save(checkpoint, os.path.join(args.save_path, 'best.pth'))


if __name__ == '__main__':
    main()
