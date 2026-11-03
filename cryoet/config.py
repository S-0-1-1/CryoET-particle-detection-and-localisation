CLASSES = ['apo-ferritin', 'beta-amylase', 'beta-galactosidase', 'ribosome', 'thyroglobulin', 'virus-like-particle']
CLASS_ID = {c: i for i, c in enumerate(CLASSES)}
RADIUS = {'apo-ferritin': 60, 'beta-amylase': 65, 'beta-galactosidase': 90, 'ribosome': 150, 'thyroglobulin': 130, 'virus-like-particle': 135}
WEIGHT = {'apo-ferritin': 1, 'beta-amylase': 0, 'beta-galactosidase': 2, 'ribosome': 1, 'thyroglobulin': 2, 'virus-like-particle': 1}
CLASS_LOSS_W = [1.0, 0.5, 2.0, 1.0, 2.0, 1.0]
VERSIONS = ['denoised', 'ctfdeconvolved', 'isonetcorrected', 'wbp']

VOXEL = 10.0
OUT_STRIDE = 2
WIDTHS = (32, 64, 128, 256)

CROP = 96
MARGIN = 8
FG_PROB = 0.7
SIGMA_RATIO = 0.5

TILE = (192, 224, 224)
TILE_STRIDE = (192, 192, 192)
TTA_FLIPS = [(), (3,), (4,), (3, 4)]
NMS_RATIO = 0.8
MIN_CONF = 0.05
