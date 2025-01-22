import time

import numpy as np
import torch
import visdom

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


#####################################
#  예시 라벨명 (VOC_BBOX_LABEL_NAMES)
#####################################
VOC_BBOX_LABEL_NAMES = (
    'fly',
    'bike',
    'bird',
    'boat',
    'pin',
    'bus',
    'c',
    'cat',
    'chair',
    'cow',
    'table',
    'dog',
    'horse',
    'moto',
    'p',
    'plant',
    'shep',
    'sofa',
    'train',
    'tv',
)


def vis_image(img, ax=None):
    """
    (3, H, W) 형태의 RGB 이미지 시각화 (값 범위: [0, 255])
    ax가 None이면 새로운 plt.figure 생성
    """
    if ax is None:
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

    # (C,H,W) -> (H,W,C)
    img = img.transpose((1, 2, 0))
    ax.imshow(img.astype(np.uint8))
    return ax


def vis_bbox(img, bbox, label=None, score=None, ax=None):
    """
    이미지 위에 Bounding Box를 시각화
    bbox: (R,4), (y_min, x_min, y_max, x_max)
    label: (R,) / score: (R,) - 없으면 None
    """
    label_names = list(VOC_BBOX_LABEL_NAMES) + ['bg']

    if label is not None and len(label) != len(bbox):
        raise ValueError('bbox와 label의 길이가 다릅니다')
    if score is not None and len(score) != len(bbox):
        raise ValueError('bbox와 score의 길이가 다릅니다')

    fig, ax = plt.subplots()
    ax = vis_image(img, ax=ax)

    if len(bbox) == 0:
        return fig, ax

    for i, bb in enumerate(bbox):
        xy = (bb[1], bb[0])
        height = bb[2] - bb[0]
        width = bb[3] - bb[1]
        ax.add_patch(
            plt.Rectangle(
                xy, width, height,
                fill=False, edgecolor='red', linewidth=2
            )
        )

        # Caption 만들기
        caption = []
        if label is not None:
            lb = label[i]
            if not (-1 <= lb < len(label_names)):
                raise ValueError('라벨 인덱스 범위 오류')
            caption.append(label_names[lb])

        if score is not None:
            caption.append('{:.2f}'.format(score[i]))

        if caption:
            ax.text(
                bb[1], bb[0], ': '.join(caption),
                style='italic',
                bbox={'facecolor': 'white', 'alpha': 0.5, 'pad': 0}
            )
    return fig, ax


def fig2data(fig):
    """
    Matplotlib Figure -> RGBA 형태 (H, W, 4)의 numpy 배열로 변환
    """
    if isinstance(fig, plt.Axes):  # Axes 객체일 경우 Figure로 변환
        fig = fig.figure
    
    fig.canvas.draw()
    w, h = fig.canvas.get_width_height()

    # ARGB 바이트 스트링 -> np.uint8 배열
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    # (W, H, 4) -> (H, W, 4) 로 맞춤
    buf.shape = (h, w, 4)

    # ARGB -> RGBA
    buf = np.roll(buf, 3, axis=2)
    return buf.reshape(h, w, 4)


def fig4vis(fig):
    """
    Matplotlib Figure -> (C,H,W) 형태로 변환, [0,1] 범위
    """
    img_data = fig2data(fig).astype(np.float32)  # float으로 변경하여 나눗셈
    plt.close(fig)
    # (H,W,C) -> (C,H,W)
    return img_data[:, :, :3].transpose((2, 0, 1)) / 255.


def visdom_bbox(*args, **kwargs):
    """
    vis_bbox 결과를 (C,H,W) ndarray로 변환해 반환
    """
    fig, ax = vis_bbox(*args, **kwargs)
    data = fig4vis(fig)
    return data


class Visualizer:
    """
    Visdom을 간편히 다루기 위한 래퍼 클래스
    plot/line/img 등의 여러 기능을 직관적으로 사용 가능
    """

    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(
            server='localhost',
            env=env,
            use_incoming_socket=False,
            **kwargs
        )
        self._vis_kw = kwargs
        self.index = {}
        self.log_text = ''

    def reinit(self, env='default', **kwargs):
        """
        Visdom 설정 재지정
        """
        self.vis = visdom.Visdom(env=env, **kwargs)
        return self

    def plot_many(self, d):
        """
        여러 (name, value) 쌍을 plot
        예: {'loss':0.1, 'lr':0.001}
        """
        for k, v in d.items():
            if v is not None:
                self.plot(k, v)

    def img_many(self, d):
        """
        여러 (name, 이미지 텐서) 쌍을 이미지로 표시
        """
        for k, v in d.items():
            self.img(k, v)

    def plot(self, name, y, **kwargs):
        """
        특정 name에 대해 scalar 값을 plot (line 그래프)
        예: self.plot('loss',1.2)
        """
        x = self.index.get(name, 0)
        self.vis.line(
            Y=np.array([y]),
            X=np.array([x]),
            win=name,
            opts=dict(title=name),
            update=None if x == 0 else 'append',
            **kwargs
        )
        self.index[name] = x + 1

    def img(self, name, img_, **kwargs):
        """
        Visdom에 이미지 표시
        (C,H,W) 또는 (N,C,H,W) 텐서 가능
        """
        self.vis.images(
            torch.Tensor(img_).cpu().numpy(),
            win=name,
            opts=dict(title=name),
            **kwargs
        )

    def log(self, info, win='log_text'):
        """
        문자열 로그를 누적하여 표시
        """
        self.log_text += '[{time}] {info} <br>'.format(
            time=time.strftime('%m%d_%H%M%S'),
            info=info
        )
        self.vis.text(self.log_text, win=win)

    def __getattr__(self, name):
        """
        self.vis 내 함수에 직접 접근 가능하게 해줌
        """
        return getattr(self.vis, name)

    def state_dict(self):
        """
        현재 상태(인덱스, 로그 등)를 dict로 반환
        """
        return {
            'index': self.index,
            'vis_kw': self._vis_kw,
            'log_text': self.log_text,
            'env': self.vis.env
        }

    def load_state_dict(self, d):
        """
        저장된 state_dict로 복원
        """
        self.vis = visdom.Visdom(
            env=d.get('env', self.vis.env),
            **(d.get('vis_kw', {}))
        )
        self.log_text = d.get('log_text', '')
        self.index = d.get('index', {})
        return self
