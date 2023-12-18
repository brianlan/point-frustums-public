from .base_models import Detection3DModel
from .base_runtime import Detection3DRuntime
from .backbones import PointFrustumsBackbone
from .necks import PointFrustumsNeck
from .heads import PointFrustumsHead


class PointFrustumsModel(Detection3DModel):
    def __init__(self, backbone: PointFrustumsBackbone, neck: PointFrustumsNeck, head: PointFrustumsHead):
        super().__init__()
        self.backbone = backbone
        self.neck = neck
        self.head = head


class PointFrustums(Detection3DRuntime):  # pylint: disable=too-many-ancestors
    def __init__(self, model: PointFrustumsModel):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        pass

    def training_step(self, *args, **kwargs):
        pass

    def validation_step(self, *args, **kwargs):
        pass

    def test_step(self, *args, **kwargs):
        pass

    def configure_optimizers(self):
        pass
