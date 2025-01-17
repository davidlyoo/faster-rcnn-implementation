from pprint import pprint


class Config:
    # 데이터셋 및 이미지 크기 설정
    voc_data_dir = '/dataset/PASCAL2007/VOC2007'
    min_size = 600
    max_size = 1000

    # 데이터 로딩 관련 설정
    num_workers = 8
    test_num_workers = 8

    # 모델 학습 관련 파라미터
    rpn_sigma = 3.
    roi_sigma = 1.
    weight_decay = 0.0005
    lr_decay = 0.1 # 1e-3 -> 1e-4
    lr = 1e-3
    epoch = 14

    # 환경 및 시각화 설정
    env = 'faster-rcnn'
    port = 8097
    plot_every = 40

    # 모델 및 데이터 설정
    data = 'voc'
    pretrained_model = 'vgg16'

    # Optimizer 및 기타 설정
    use_adam = False
    use_chainer = False
    use_drop = False

    # 디버깅 및 테스트 관련 설정
    debug_file = '/tmp/debugf'
    test_num = 10000
    load_path = None

    def _parse(self, kwargs):
        """
        사용자가 제공한 설정값으로 Config 클래스의 속성값을 변경
        """
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict():
                raise ValueError('Unknown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config======')
        pprint(self._state_dict())
        print('==========end==========')

    def _state_dict(self):
        """
        _로 시작하는 내부 속성은 제외하고, Config 클래스 속성값을 딕셔너리 형태로 반환
        """
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}

opt = Config()