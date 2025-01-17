from __future__ import  absolute_import

import os
import resource

import fire
import ipdb
import matplotlib
from tqdm import tqdm
from torch.utils import data as data_

from dataset.dataset import Dataset, TestDataset, inverse_normalize
from model import FasterRCNNVGG16
from trainer import FasterRCNNTrainer
from utils import array_tool as at
from utils.config import opt
from utils.vis_tool import visdom_bbox
from utils.eval_tool import eval_detection_voc

# Pytorch 관련 리소스 제한 설정 (파일 개수 제한 증가)
max_open_files = 20480
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (max_open_files, rlimit[1]))

matplotlib.use('agg')


# 모델을 평가하는 함수 (VOC 평가 방식 사용)
def eval(dataloader, faster_rcnn, test_num=10000):
    pred_bboxes, pred_labels, pred_scores = [], [], []
    gt_bboxes, gt_labels, gt_difficults = [], [], []
    
    for ii, (imgs, sizes, gt_bboxes_, gt_labels_, gt_difficults_) in tqdm(enumerate(dataloader)):
        sizes = [sizes[0][0].item(), sizes[1][0].item()]
        pred_bboxes_, pred_labels_, pred_scores_ = faster_rcnn.predict(imgs, [sizes])
        
        # Ground Truth와 예측값 저장
        gt_bboxes += list(gt_bboxes_.numpy())
        gt_labels += list(gt_labels_.numpy())
        gt_difficults += list(gt_difficults_.numpy())
        pred_bboxes += pred_bboxes_
        pred_labels += pred_labels_
        pred_scores += pred_scores_
        
        if ii == test_num:
            break

    result = eval_detection_voc(
        pred_bboxes, pred_labels, pred_scores,
        gt_bboxes, gt_labels, gt_difficults,
        use_07_metric=True)
    
    return result


# Faster R-CNN 모델 훈련 함수
def train(**kwargs):
    opt._parse(kwargs)

    dataset = Dataset(opt)
    print('데이터를 로드 중입니다...')
    
    dataloader = data_.DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=opt.num_workers
    )
    testset = TestDataset(opt)
    test_dataloader = data_.DataLoader(
        testset,
        batch_size=1,
        num_workers=opt.test_num_workers,
        shuffle=False,
        pin_memory=True
    )
    
    faster_rcnn = FasterRCNNVGG16()
    print('모델 생성 완료!')
    
    trainer = FasterRCNNTrainer(faster_rcnn).cuda()
    if opt.load_path:
        trainer.load(opt.load_path)
        print(f'사전 학습된 모델을 불러왔습니다: {opt.load_path}')
        
    trainer.vis.text(dataset.db.label_names, win='labels')
    
    best_map = 0
    best_path = None
    lr_ = opt.lr
    
    for epoch in range(opt.epoch):
        trainer.reset_meters()
        print(f'현재 Epoch: {epoch + 1}/{opt.epoch}')

        for ii, (img, bbox_, label_, scale) in tqdm(enumerate(dataloader)):
            scale = at.scalar(scale)
            img, bbox, label = img.cuda().float(), bbox_.cuda(), label_.cuda()
            trainer.train_step(img, bbox, label, scale)

            if (ii + 1) % opt.plot_every == 0:
                if os.path.exists(opt.debug_file):
                    ipdb.set_trace()

                # 손실 그래프 시각화
                trainer.vis.plot_many(trainer.get_meter_data())

                # Ground Truth Bounding-box 시각화
                ori_img_ = inverse_normalize(at.tonumpy(img[0]))
                gt_img = visdom_bbox(ori_img_,
                                    at.tonumpy(bbox_[0]),
                                    at.tonumpy(label_[0]))
                trainer.vis.img('gt_img', gt_img)

                # 예측된 Bounding-box 시각화
                _bboxes, _labels, _scores = trainer.faster_rcnn.predict([ori_img_], visualize=True)
                pred_img = visdom_bbox(ori_img_,
                                        at.tonumpy(_bboxes[0]),
                                        at.tonumpy(_labels[0]).reshape(-1),
                                        at.tonumpy(_scores[0]))
                trainer.vis.img('pred_img', pred_img)

                # RPN 및 RoI Confusion Matrix 시각화
                trainer.vis.text(str(trainer.rpn_cm.value().tolist()), win='rpn_cm')
                trainer.vis.img('roi_cm', at.totensor(trainer.roi_cm.conf, False).float())
        
        # 모델 평가
        eval_result = eval(test_dataloader, faster_rcnn, test_num=opt.test_num)
        trainer.vis.plot('test_map', eval_result['map'])
        lr_ = trainer.faster_rcnn.optimizer.param_groups[0]['lr']
        log_info = f"학습률: {lr_}, mAP: {eval_result['map']}, 손실: {trainer.get_meter_data()}"
        trainer.vis.log(log_info)

        # 모델 저장
        if eval_result['map'] > best_map:
            best_map = eval_result['map']
            best_path = trainer.save(best_map=best_map)
            
        # 9 epoch에서 최적 모델을 불러와 학습률 감소 적용
        if epoch == 9 and bath_path is not None:
            trainer.load(best_path)
            trainer.faster_rcnn.scale_lr(opt.lr_decay)
            lr_ = lr_ * opt.lr_decay

        # 13 epoch에서 학습 종료
        if epoch == 13: 
            break


if __name__ == '__main__':
    fire.Fire()