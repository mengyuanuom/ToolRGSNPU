import os
from typing import List, Union
import json
import cv2
import lmdb
import numpy as np
import pyarrow as pa
import torch
from torch.utils.data import Dataset
from .OCID_sub_class_dict import cnames, colors, subnames, sub_to_class
from .simple_tokenizer import SimpleTokenizer as _Tokenizer
import matplotlib.pyplot as plt
from skimage.draw import polygon
from skimage.filters import gaussian

CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
CLIP_STD  = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

def _to_np(x):
    return x.detach().cpu().numpy() if isinstance(x, torch.Tensor) else x

def _denorm_rgb_chw_to_bgr(img_chw, mean=CLIP_MEAN, std=CLIP_STD):
    x = _to_np(img_chw)
    if x.ndim == 3 and x.shape[0] == 3:
        x = x.transpose(1, 2, 0)
    x = x * std[None, None, :] + mean[None, None, :]
    x = np.clip(x, 0, 1)
    x = (x * 255.0).astype(np.uint8)
    return x[:, :, ::-1]  # RGB->BGR

def _to_heatmap(arr, cmap=cv2.COLORMAP_JET, rng=None):
    a = arr.astype(np.float32)
    if rng is not None:
        lo, hi = rng
        a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1)
    else:
        amin, amax = float(np.nanmin(a)), float(np.nanmax(a))
        a = np.zeros_like(a) if amax - amin < 1e-6 else (a - amin) / (amax - amin)
    a8 = (a * 255).astype(np.uint8)
    return cv2.applyColorMap(a8, cmap)

def _put_title(img_bgr, text):
    out = img_bgr.copy()
    cv2.rectangle(out, (0,0), (out.shape[1], 24), (0,0,0), -1)
    cv2.putText(out, text, (6,17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1, cv2.LINE_AA)
    return out

def debug_montage_one_image(
    sample, out_path,
    stride=8, min_off_w=0.1,
    resize_to_width=None):
    """
    将 RGB / qua / sin / cos / wid / off_w / |off| / quiver 拼成一张图并保存。
    返回保存路径。
    """
    img = sample["img"]
    gm = sample["grasp_masks"]
    qua = _to_np(gm["qua"])
    sinm = _to_np(gm["sin"])
    cosm = _to_np(gm["cos"])
    wid = _to_np(gm["wid"])
    off = _to_np(gm.get("off", np.zeros((2, *qua.shape), dtype=np.float32)))
    off_w = _to_np(gm.get("off_w", np.zeros((1, *qua.shape), dtype=np.float32)))
    if off_w.ndim == 3 and off_w.shape[0] == 1: off_w = off_w[0]
    ins_mask = _to_np(sample.get("mask", np.ones_like(qua, dtype=np.float32)))
    if ins_mask.max() > 1.0: ins_mask = ins_mask / 255.0
    bgr = _denorm_rgb_chw_to_bgr(img)

    H, W = qua.shape
    dx, dy = off[0], off[1]
    mag = np.sqrt(dx*dx + dy*dy)

    # 各个 tile（都转成 BGR）
    tile_rgb   = _put_title(bgr, "RGB")
    tile_qua   = _put_title(_to_heatmap(qua, cmap=cv2.COLORMAP_JET, rng=(0,1)), "qua [0,1]")
    tile_cos   = _put_title(_to_heatmap((cosm+1)*0.5, cmap=cv2.COLORMAP_HSV, rng=(0,1)), "cos [-1,1]")
    tile_sin   = _put_title(_to_heatmap((sinm+1)*0.5, cmap=cv2.COLORMAP_HSV, rng=(0,1)), "sin [-1,1]")
    tile_wid   = _put_title(_to_heatmap(wid, cmap=cv2.COLORMAP_JET, rng=(0,1)), "wid [0,1]")
    tile_offw  = _put_title(_to_heatmap(off_w, cmap=cv2.COLORMAP_BONE), "off_w [0,1]")
    tile_mag   = _put_title(_to_heatmap(np.clip(mag,0,1), cmap=cv2.COLORMAP_MAGMA, rng=(0,1)), "|off| [0,1]")

    # quiver 叠加
    quiver = bgr.copy()
    step = max(1, int(stride))
    for y in range(0, H, step):
        for x in range(0, W, step):
            if off_w[y, x] >= min_off_w and ins_mask[y, x] > 0.5:
                u = float(dx[y, x]) * step * 0.8
                v = float(dy[y, x]) * step * 0.8
                x1, y1 = int(x), int(y)
                x2, y2 = int(x + u), int(y + v)
                cv2.arrowedLine(quiver, (x1, y1), (x2, y2), (0, 255, 255), 1, tipLength=0.4)
    tile_quiver = _put_title(quiver, f"off quiver (stride={step}, thr={min_off_w})")

    # 拼成 2x4 网格（你也可以改 3x3）
    row1 = cv2.hconcat([tile_rgb, tile_qua, tile_cos, tile_sin])
    row2 = cv2.hconcat([tile_wid, tile_offw, tile_mag, tile_quiver])
    grid = cv2.vconcat([row1, row2])

    # 可选整体缩放到指定宽度
    if resize_to_width is not None and grid.shape[1] != resize_to_width:
        scale = resize_to_width / grid.shape[1]
        grid = cv2.resize(grid, (resize_to_width, int(grid.shape[0]*scale)), interpolation=cv2.INTER_AREA)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cv2.imwrite(out_path, grid)
    return out_path

def as_xy_points(arr) -> np.ndarray:
    """
    把各种抓取表示统一为 (N,2) 的点坐标 (x,y)：
    - (N,4,2) : 四点 -> 取中心
    - (4,2)   : 单四点 -> 中心
    - (N,6)   : 参数 (cx,cy,...) -> 取 [:, :2]
    - (6,)    : 单参数 -> 取 [:2] -> (1,2)
    - (N,2)   : 已是点
    - (2,)    : 单点 -> (1,2)
    - (2N,)   : 扁平 -> reshape(-1,2)
    """
    a = arr.detach().cpu().numpy() if isinstance(arr, torch.Tensor) else np.asarray(arr)

    if a.size == 0:
        return a.reshape(0, 2).astype(np.float32)

    if a.ndim == 3 and a.shape[1:] == (4, 2):
        return a.mean(axis=1, dtype=np.float32)                    # (N,2)

    if a.ndim == 2:
        if a.shape == (4, 2):
            return a.mean(axis=0, keepdims=True).astype(np.float32)  # (1,2)
        if a.shape[1] == 6:
            return a[:, :2].astype(np.float32)                     # (N,2)
        if a.shape[1] == 2:
            return a.astype(np.float32)                             # (N,2)
        # 其它二维形状不接受
        raise ValueError(f"Unexpected 2D shape for points/grasps: {a.shape}")

    if a.ndim == 1:
        if a.shape[0] == 6:
            return a[:2].reshape(1, 2).astype(np.float32)           # (6,) -> (1,2)
        if a.shape[0] == 2:
            return a.reshape(1, 2).astype(np.float32)               # (2,) -> (1,2)
        if a.size % 2 == 0:
            return a.reshape(-1, 2).astype(np.float32)              # (2N,) -> (N,2)
        raise ValueError(f"Flat length not even: {a.size}")

    raise ValueError(f"Unexpected ndim={a.ndim}, shape={a.shape}")

def apply_affine_to_points(points_xy: np.ndarray, mat_2x3: np.ndarray) -> np.ndarray:

    pts = np.asarray(points_xy, dtype=np.float32)

    if pts.size == 0:
        return pts.reshape(0, 2)

    if pts.ndim == 1:
        if pts.shape[0] != 2:
            raise ValueError(f"points_xy must be (2,) or (N,2); got 1D length={pts.shape[0]}")
        pts = pts.reshape(1, 2)
    elif pts.ndim == 2:
        if pts.shape[1] != 2:
            raise ValueError(f"points_xy must have shape (N,2); got {pts.shape}")
    else:
        raise ValueError(f"points_xy must be 1D or 2D; got ndim={pts.ndim}, shape={pts.shape}")

    M = np.asarray(mat_2x3, dtype=np.float32)
    if M.shape != (2, 3):
        raise ValueError(f"mat_2x3 must be shape (2,3); got {M.shape}")

    ones = np.ones((pts.shape[0], 1), dtype=np.float32)
    homo = np.concatenate([pts, ones], axis=1)     # (N,3)
    out  = (homo @ M.T).astype(np.float32)         # (N,2)
    return out

def apply_affine_to_grasp6(rects6, M, idx=None):
    """
    rects6: (N,6)  例如 [x,y,theta,w,h,score] 或 [x,y,w,h,theta,score]
    idx: 字段索引映射，默认按 A 型:
         {'x':0, 'y':1, 'theta':2, 'w':3, 'h':4, 'score':5}
         如果你是 B 型，就设:
         {'x':0, 'y':1, 'w':2, 'h':3, 'theta':4, 'score':5}
    返回: 变换后的同形状 (N,6)
    """
    R = np.asarray(rects6, dtype=np.float32).copy()
    if R.ndim == 1: R = R[None, :]
    n = R.shape[0]
    if idx is None:
        idx = {'x':0, 'y':1, 'theta':2, 'w':3, 'h':4, 'score':5}

    # 取字段
    x = R[:, idx['x']]
    y = R[:, idx['y']]
    th = R[:, idx['theta']]
    w  = R[:, idx['w']]
    h  = R[:, idx['h']]

    # 线性部分 A
    M = np.asarray(M, dtype=np.float32)
    A = M[:2, :2]  # (2,2)

    # 1) 中心点
    xy_new = apply_affine_to_points(np.stack([x,y], axis=1), M)  # (N,2)

    # 2) 朝向：单位向量经 A 后再 atan2
    u = np.stack([np.cos(th), np.sin(th)], axis=1)   # (N,2)
    v = u @ A.T
    th_new = np.arctan2(v[:,1], v[:,0])

    # 3) 尺度：沿主方向/副方向的伸缩
    scale_w = np.linalg.norm(v, axis=1)
    u_perp  = np.stack([-np.sin(th), np.cos(th)], axis=1)
    v_perp  = u_perp @ A.T
    scale_h = np.linalg.norm(v_perp, axis=1)
    w_new = w * scale_w
    h_new = h * scale_h


    R[:, idx['x']]     = xy_new[:,0]
    R[:, idx['y']]     = xy_new[:,1]
    R[:, idx['theta']] = th_new
    R[:, idx['w']]     = w_new
    R[:, idx['h']]     = h_new
    # score 不变
    return R


def grasp6_centers(rects6, M=None, idx_xy=(0,1)):
    """
    返回增强后中心 (N,2)。若 M 为 None，则直接取 (x,y)。
    idx_xy: (x_idx, y_idx)
    """
    if isinstance(rects6, torch.Tensor):
        rects6 = rects6.detach().cpu().numpy()
    if rects6 is None or len(rects6) == 0:
        return np.zeros((0,2), np.float32)
    rects6 = np.asarray(rects6, dtype=np.float32)
    if rects6.ndim == 1: rects6 = rects6[None, :]
    centers = rects6[:, [idx_xy[0], idx_xy[1]]]
    if M is not None:
        centers = apply_affine_to_points(centers, M)
    return centers.astype(np.float32)


def make_dense_offset_with_radius_np(
    centers_xy: np.ndarray,
    img_size_hw,
    r_pix: float,
    use_gaussian: bool = True,
    sigma: float = None,
):
    """
    生成稠密 offset 与权重图（单张图）。
    - centers_xy : (N,2) 的 (x,y) 中心，坐标在【当前输入分辨率】下
    - img_size_hw: (H, W)
    - r_pix      : 正样本半径（像素）。同时作为 offset 归一化半径
    - use_gaussian: True 用高斯权，False 用二值圆盘
    - sigma      : 高斯标准差（像素）。默认 0.5*r_pix

    返回:
      off   : (2, H, W) float32，归一化位移 (dx,dy)/r_pix（圆内）
      off_w : (1, H, W) float32，权重（圆内>0；高斯/二值）
    说明:
      - 若多个中心重叠，取“距离最近”的那个中心的 offset/权重（nearest-center 逻辑）。
      - centers 为空时返回全零。
    """
    H, W = int(img_size_hw[0]), int(img_size_hw[1])
    off   = np.zeros((2, H, W), dtype=np.float32)
    off_w = np.zeros((1, H, W), dtype=np.float32)
    if centers_xy is None:
        return off, off_w

    pts = np.asarray(centers_xy, dtype=np.float32).reshape(-1, 2)
    if pts.size == 0:
        return off, off_w

    r = float(max(1.0, r_pix))
    if sigma is None:
        sigma = 0.5 * r  # 经验值，半径的一半

    # 记录“当前像素到已分配中心的最小距离平方”，用于处理重叠
    dist2_map = np.full((H, W), np.inf, dtype=np.float32)

    for cx, cy in pts:
        # 只在圆的包围框里计算，减少开销
        x0 = max(0, int(np.floor(cx - r)))
        x1 = min(W, int(np.ceil (cx + r)) + 1)
        y0 = max(0, int(np.floor(cy - r)))
        y1 = min(H, int(np.ceil (cy + r)) + 1)
        if x0 >= x1 or y0 >= y1:
            continue

        # 局部网格
        xs = np.arange(x0, x1, dtype=np.float32)
        ys = np.arange(y0, y1, dtype=np.float32)
        XX, YY = np.meshgrid(xs, ys)  # (h_box, w_box)

        dx = cx - XX  # 注意这里定义为“指向中心的位移”
        dy = cy - YY
        dist2 = dx * dx + dy * dy
        inside = dist2 <= (r * r)
        if not np.any(inside):
            continue

        # 归一化位移（/ r）
        dx_n = (dx / r)
        dy_n = (dy / r)

        # 权重
        if use_gaussian:
            # 高斯随距离衰减；中心处 ~1
            w_loc = np.exp(-dist2 / (2.0 * sigma * sigma))
        else:
            # 二值
            w_loc = np.ones_like(dist2, dtype=np.float32)
        w_loc = w_loc * inside.astype(np.float32)

        # 与已有分配比较，只保留“更近的中心”
        dist2_crop = dist2_map[y0:y1, x0:x1]
        take = (inside) & (dist2 < dist2_crop)
        if not np.any(take):
            continue

        # 更新最小距离
        dist2_crop[take] = dist2[take]
        dist2_map[y0:y1, x0:x1] = dist2_crop

        # 写入 off 与权重（只写更近处）
        off[0, y0:y1, x0:x1][take] = dx_n[take]
        off[1, y0:y1, x0:x1][take] = dy_n[take]
        off_w[0, y0:y1, x0:x1][take] = w_loc[take]

    return off, off_w
    
info = {
    'refcoco': {
        'train': 42404,
        'val': 3811,
        'val-test': 3811,
        'testA': 1975,
        'testB': 1810
    },
    'refcoco+': {
        'train': 42278,
        'val': 3805,
        'val-test': 3805,
        'testA': 1975,
        'testB': 1798
    },
    'refcocog_u': {
        'train': 42226,
        'val': 2573,
        'val-test': 2573,
        'test': 5023
    },
    'refcocog_g': {
        'train': 44822,
        'val': 5000,
        'val-test': 5000
    },
    'refcoco_mixed': {
        'train': 126908, # 42404+42278+42226=126908
        'val': 10189, # 3811+3805+2573=10189
    }
}
_tokenizer = _Tokenizer()


def tokenize(texts: Union[str, List[str]],
             context_length: int = 77,
             truncate: bool = False) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)

    Parameters
    ----------
    texts : Union[str, List[str]]
        An input string or a list of input strings to tokenize

    context_length : int
        The context length to use; all CLIP models use 77 as the context length

    truncate: bool
        Whether to truncate the text in case its encoding is longer than the context length

    Returns
    -------
    A two-dimensional tensor containing the resulting tokens, shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token]
                  for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(
                    f"Input {texts[i]} is too long for context length {context_length}"
                )
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)


class RefDataset(Dataset):
    def __init__(self, lmdb_dir, mask_dir, dataset, split, mode, input_size,
                 word_length):
        super(RefDataset, self).__init__()
        self.lmdb_dir = lmdb_dir
        self.mask_dir = mask_dir
        self.dataset = dataset
        self.split = split
        self.mode = mode
        self.input_size = (input_size, input_size)
        self.word_length = word_length
        self.mean = torch.tensor([0.48145466, 0.4578275,
                                  0.40821073]).reshape(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258,
                                 0.27577711]).reshape(3, 1, 1)
        self.length = info[dataset][split]
        self.env = None

    def _init_db(self):
        self.env = lmdb.open(self.lmdb_dir,
                             subdir=os.path.isdir(self.lmdb_dir),
                             readonly=True,
                             lock=False,
                             readahead=False,
                             meminit=False)
        with self.env.begin(write=False) as txn:
            self.length = loads_pyarrow(txn.get(b'__len__'))
            self.keys = loads_pyarrow(txn.get(b'__keys__'))

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        # Delay loading LMDB data until after initialization: https://github.com/chainer/chainermn/issues/129
        if self.env is None:
            self._init_db()
        env = self.env
        with env.begin(write=False) as txn:
            byteflow = txn.get(self.keys[index])
        ref = loads_pyarrow(byteflow)
        # img
        ori_img = cv2.imdecode(np.frombuffer(ref['img'], np.uint8),
                               cv2.IMREAD_COLOR)
        img = cv2.cvtColor(ori_img, cv2.COLOR_BGR2RGB)
        img_size = img.shape[:2]
        # mask
        seg_id = ref['seg_id']
        mask_dir = os.path.join(self.mask_dir, str(seg_id) + '.png')
        # sentences
        idx = np.random.choice(ref['num_sents'])
        sents = ref['sents']
        # transform
        mat, mat_inv = self.getTransformMat(img_size, True)
        img = cv2.warpAffine(
            img,
            mat,
            self.input_size,
            flags=cv2.INTER_CUBIC,
            borderValue=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255])
        if self.mode == 'train':
            # mask transform
            mask = cv2.imdecode(np.frombuffer(ref['mask'], np.uint8),
                                cv2.IMREAD_GRAYSCALE)
            mask = cv2.warpAffine(mask,
                                  mat,
                                  self.input_size,
                                  flags=cv2.INTER_LINEAR,
                                  borderValue=0.)
            mask = mask / 255.
            # sentence -> vector
            sent = sents[idx]
            word_vec = tokenize(sent, self.word_length, True).squeeze(0)
            img, mask = self.convert(img, mask)
            return img, word_vec, mask
        elif self.mode == 'val':
            # sentence -> vector
            sent = sents[0]
            word_vec = tokenize(sent, self.word_length, True).squeeze(0)
            img = self.convert(img)[0]
            params = {
                'mask_dir': mask_dir,
                'inverse': mat_inv,
                'ori_size': np.array(img_size)
            }
            return img, word_vec, params
        else:
            # sentence -> vector
            img = self.convert(img)[0]
            params = {
                'ori_img': ori_img,
                'seg_id': seg_id,
                'mask_dir': mask_dir,
                'inverse': mat_inv,
                'ori_size': np.array(img_size),
                'sents': sents
            }
            return img, params

    def getTransformMat(self, img_size, inverse=False):
        ori_h, ori_w = img_size
        inp_h, inp_w = self.input_size
        scale = min(inp_h / ori_h, inp_w / ori_w)
        new_h, new_w = ori_h * scale, ori_w * scale
        bias_x, bias_y = (inp_w - new_w) / 2., (inp_h - new_h) / 2.

        src = np.array([[0, 0], [ori_w, 0], [0, ori_h]], np.float32)
        dst = np.array([[bias_x, bias_y], [new_w + bias_x, bias_y],
                        [bias_x, new_h + bias_y]], np.float32)

        mat = cv2.getAffineTransform(src, dst)
        if inverse:
            mat_inv = cv2.getAffineTransform(dst, src)
            return mat, mat_inv
        return mat, None

    def convert(self, img, mask=None):
        # Image ToTensor & Normalize
        img = torch.from_numpy(img.transpose((2, 0, 1)))
        if not isinstance(img, torch.FloatTensor):
            img = img.float()
        img.div_(255.).sub_(self.mean).div_(self.std)
        # Mask ToTensor
        if mask is not None:
            mask = torch.from_numpy(mask)
            if not isinstance(mask, torch.FloatTensor):
                mask = mask.float()
        return img, mask

    def __repr__(self):
        return self.__class__.__name__ + "(" + \
            f"db_path={self.lmdb_dir}, " + \
            f"dataset={self.dataset}, " + \
            f"split={self.split}, " + \
            f"mode={self.mode}, " + \
            f"input_size={self.input_size}, " + \
            f"word_length={self.word_length}"

    # def get_length(self):
    #     return self.length

    # def get_sample(self, idx):
    #     return self.__getitem__(idx)


class GraspTransforms:
    # Class for converting cv2-like rectangle formats and generate grasp-quality-angle-width masks

    def __init__(self, width_factor=400, width=416, height=416):
        self.width_factor = width_factor
        self.width = width 
        self.height = height

    def __call__(self, grasp_rectangles, target):
        # grasp_rectangles: (M, 4, 2)
        M = grasp_rectangles.shape[0]
        p1, p2, p3, p4 = np.split(grasp_rectangles, 4, axis=1)
        
        center_x = (p1[..., 0] + p3[..., 0]) / 2
        center_y = (p1[..., 1] + p3[..., 1]) / 2
        
        width  = np.sqrt((p1[..., 0] - p4[..., 0]) * (p1[..., 0] - p4[..., 0]) + (p1[..., 1] - p4[..., 1]) * (p1[..., 1] - p4[..., 1]))
        height = np.sqrt((p1[..., 0] - p2[..., 0]) * (p1[..., 0] - p2[..., 0]) + (p1[..., 1] - p2[..., 1]) * (p1[..., 1] - p2[..., 1]))
        
        theta = np.arctan2(p4[..., 0] - p1[..., 0], p4[..., 1] - p1[..., 1]) * 180 / np.pi
        theta = np.where(theta > 0, theta - 90, theta + 90)

        target = np.tile(np.array([[target]]), (M,1))

        return np.concatenate([center_x, center_y, width, height, theta, target], axis=1)

    def inverse(self, grasp_rectangles):
        boxes = []
        for rect in grasp_rectangles:
            center_x, center_y, width, height, theta = rect[:5]
            box = ((center_x, center_y), (width, height), -(theta+180))
            box = cv2.boxPoints(box)
            box = np.intp(box)
            boxes.append(box)
        return boxes

    def generate_masks(self, grasp_rectangles):
        pos_out = np.zeros((self.height, self.width))
        ang_out = np.zeros((self.height, self.width))
        wid_out = np.zeros((self.height, self.width))
        for rect in grasp_rectangles:
            center_x, center_y, w_rect, h_rect, theta = rect[:5]
            
            # Get 4 corners of rotated rect
            # Convert from our angle represent to opencv's
            r_rect = ((center_x, center_y), (w_rect/2, h_rect), -(theta+180))
            box = cv2.boxPoints(r_rect)
            box = np.intp(box)
            rr, cc = polygon(box[:, 0], box[:,1])
            valid = (
                (rr >= 0) & (rr < self.width) &
                (cc >= 0) & (cc < self.height)
            )
            rr, cc = rr[valid], cc[valid]
            pos_out[cc, rr] = 1.0
            if theta < 0:
                ang_out[cc, rr] = int(theta + 180)
            else:
                ang_out[cc, rr] = int(theta)
            # Adopt width normalize accoding to class 
            wid_out[cc, rr] = np.clip(w_rect, 0.0, self.width_factor) / self.width_factor
        
        qua_out = (gaussian(pos_out, 3, preserve_range=True) * 255).astype(np.uint8)
        pos_out = (pos_out * 255).astype(np.uint8)
        ang_out = ang_out.astype(np.uint8)
        wid_out = (gaussian(wid_out, 3, preserve_range=True) * 255).astype(np.uint8)
        
        
        return {'pos': pos_out, 
                'qua': qua_out, 
                'ang': ang_out, 
                'wid': wid_out}


class OCIDVLGDataset(Dataset):
    """
    OCID-Vision-Language-Grasping dataset with referring expressions and grasps
    """

    def __init__(
        self,
        root_dir,
        split,
        transform_img=None,
        transform_grasp=GraspTransforms(),
        input_size=416,
        word_length=20,
        with_depth=True,
        with_segm_mask=True,
        with_grasp_masks=True,
        version="multiple",
        # ---- 新增：dense offset 相关开关与参数 ----
        with_grasp_offset: bool = False,
        offset_r_norm: float = 20.0,   # 归一化半径（像素）；None 则用 min(H,W)/20 启发式
        offset_gauss_sigma: float = None,  # 高斯 sigma（像素）；None 用 0.5*r，False/0 走二值权
    ):
        super(OCIDVLGDataset, self).__init__()
        self.root_dir = root_dir
        self.split_dir = os.path.join(root_dir, "data_split")
        self.split_map = {
            "train": "train_expressions.json",
            "val": "val_expressions.json",
            "test": "test_expressions.json",
        }
        self.split = split
        self.refer_dir = os.path.join(root_dir, "refer", version)

        self.transform_img = transform_img
        self.transform_grasp = transform_grasp
        self.with_depth = with_depth
        self.with_segm_mask = with_segm_mask
        self.with_grasp_masks = with_grasp_masks

        self.input_size = (input_size, input_size)
        self.word_length = word_length
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).reshape(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).reshape(3, 1, 1)

        # offset 相关
        self.with_grasp_offset = with_grasp_offset
        self.offset_r_norm = offset_r_norm
        self.offset_gauss_sigma = offset_gauss_sigma

        self._load_dicts()
        self._load_split()

    def _load_dicts(self):
        cwd = os.getcwd()
        os.chdir(self.root_dir)

        cnames_inv = {int(v): k for k, v in cnames.items()}
        subnames_inv = {v: k for k, v in subnames.items()}
        self.class_names = cnames
        self.idx_to_class = cnames_inv
        self.class_instance_names = subnames
        self.idx_to_class_instance = subnames_inv
        self.instance_idx_to_class_idx = sub_to_class
        os.chdir(cwd)

    def _load_split(self):
        refer_data = json.load(open(os.path.join(self.refer_dir, self.split_map[self.split])))
        self.seq_paths, self.img_names, self.scene_ids = [], [], []
        self.bboxes, self.grasps = [], []
        self.sent_to_index, self.sent_indices = {}, []
        self.rgb_paths, self.depth_paths, self.mask_paths = [], [], []
        self.targets, self.sentences, self.semantics, self.objIDs = [], [], [], []
        n = 0
        for item in refer_data["data"]:
            seq_path, im_name = item["image_filename"].split(",")
            self.seq_paths.append(seq_path)
            self.img_names.append(im_name)
            self.scene_ids.append(item["image_filename"])
            self.bboxes.append(item["box"])
            self.grasps.append(item["grasps"])
            self.objIDs.append(item["answer"])
            self.targets.append(item["target"])
            self.sentences.append(item["question"])
            self.semantics.append(item["program"])
            self.rgb_paths.append(os.path.join(seq_path, "rgb", im_name))
            self.depth_paths.append(os.path.join(seq_path, "depth", im_name))
            self.mask_paths.append(os.path.join(seq_path, "seg_mask_instances_combi", im_name))
            self.sent_indices.append(item["question_index"])
            self.sent_to_index[item["question_index"]] = n
            n += 1

    def get_index_from_sent(self, sent_id):
        return self.sent_to_index[sent_id]

    def get_sent_from_index(self, n):
        return self.sent_indices[n]

    def _load_sent(self, sent_id):
        n = self.get_index_from_sent(sent_id)

        scene_id = self.scene_ids[n]

        img_path = os.path.join(self.root_dir, self.rgb_paths[n])
        img = self.get_image_from_path(img_path)

        x, y, w, h = self.bboxes[n]
        bbox = np.asarray([x, y, x + w, y + h])

        sent = self.sentences[n]

        target = self.targets[n]
        target_idx = self.class_instance_names[target]
        objID = self.objIDs[n]

        grasps = np.asarray(self.grasps[n])

        result = {
            "img": self.transform_img(img) if self.transform_img else img,
            "grasps": self.transform_grasp(grasps, target_idx) if self.transform_grasp else None,
            "grasp_rects": self.transform_grasp(grasps, target_idx) if self.transform_grasp else None,
            "sentence": sent,
            "target": target,
            "objID": objID,
            "bbox": bbox,
            "target_idx": target_idx,
            "sent_id": sent_id,
            "scene_id": scene_id,
            "img_path": img_path,
        }

        if self.with_depth:
            depth_path = os.path.join(self.root_dir, self.depth_paths[n])
            depth = self.get_depth_from_path(depth_path)
            result = {**result, "depth": torch.from_numpy(depth) if self.transform_img else depth}

        if self.with_segm_mask:
            mask_path = os.path.join(self.root_dir, self.mask_paths[n])
            msk_full = self.get_mask_from_path(mask_path)
            msk = np.where(msk_full == objID, True, False)
            result = {**result, "mask": torch.from_numpy(msk) if self.transform_img else msk}

        if self.with_grasp_masks:
            grasp_masks = self.transform_grasp.generate_masks(result["grasp_rects"])
            result = {**result, "grasp_masks": grasp_masks}

        result = self.preprocess(result)
        return result

    def get_transform_mat(self, img_size, inverse=False):
        ori_h, ori_w = img_size
        inp_h, inp_w = self.input_size
        scale = min(inp_h / ori_h, inp_w / ori_w)
        new_h, new_w = ori_h * scale, ori_w * scale
        bias_x, bias_y = (inp_w - new_w) / 2.0, (inp_h - new_h) / 2.0

        src = np.array([[0, 0], [ori_w, 0], [0, ori_h]], np.float32)
        dst = np.array(
            [[bias_x, bias_y], [new_w + bias_x, bias_y], [bias_x, new_h + bias_y]], np.float32
        )

        mat = cv2.getAffineTransform(src, dst)
        if inverse:
            mat_inv = cv2.getAffineTransform(dst, src)
            return mat, mat_inv
        return mat, None

    def preprocess(self, data):
        img = data["img"]
        sent = data["sentence"]

        # mask 统一为 uint8（0/255）
        if np.max(data["mask"]) <= 1.0:
            ins_mask = (data["mask"] * 255).astype(np.uint8)
        else:
            ins_mask = data["mask"]

        grasp_qua_mask = data["grasp_masks"]["qua"]
        grasp_ang_mask = data["grasp_masks"]["ang"]
        grasp_wid_mask = data["grasp_masks"]["wid"]

        img_size = img.shape[:2]  # (H, W)

        mat, mat_inv = self.get_transform_mat(img_size, True)

        img = cv2.warpAffine(
            img,
            mat,
            self.input_size,
            flags=cv2.INTER_CUBIC,
            borderValue=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255],
        )
        img = torch.from_numpy(img.transpose((2, 0, 1))).float()
        img.div_(255.0).sub_(self.mean).div_(self.std)

        ins_mask = cv2.warpAffine(ins_mask, mat, self.input_size, flags=cv2.INTER_LINEAR, borderValue=0.0)
        grasp_qua_mask = cv2.warpAffine(grasp_qua_mask, mat, self.input_size, flags=cv2.INTER_LINEAR, borderValue=0.0)
        grasp_ang_mask = cv2.warpAffine(grasp_ang_mask, mat, self.input_size, flags=cv2.INTER_LINEAR, borderValue=0.0)
        grasp_wid_mask = cv2.warpAffine(grasp_wid_mask, mat, self.input_size, flags=cv2.INTER_LINEAR, borderValue=0.0)

        # ---- 以 grasp_rects 的“中心点”为 offset 正样本中心 ----
        rects = data["grasp_rects"]
        if isinstance(rects, torch.Tensor):
            rects = rects.detach().cpu().numpy()

        if rects is None or len(rects) == 0:
            centers = np.zeros((0, 2), np.float32)
        else:
            # 统一转成 (N,2) 的 (x,y)；支持 (N,6)/(N,4,2)/(N,2)/(6,) 等
            xy = as_xy_points(rects)                  # 原图坐标
            centers = apply_affine_to_points(xy, mat) # 输入分辨率坐标

        # ---- 可选：生成 dense offset GT ----
        if self.with_grasp_offset:
            # 半径：优先用用户指定，否则用 min(H,W)/20 启发式
            r_pix = (
                float(self.offset_r_norm)
                if (self.offset_r_norm is not None)
                else min(*self.input_size) / 20.0
            )
            # 权重：默认高斯；若你想用二值，把 use_gaussian=False 传给函数
            use_gauss = True
            sigma = self.offset_gauss_sigma  # None -> 用 0.5*r

            off_np, offw_np = make_dense_offset_with_radius_np(
                centers_xy=centers,
                img_size_hw=self.input_size,
                r_pix=r_pix,
                use_gaussian=use_gauss,
                sigma=sigma,
            )

            data["grasp_masks"]["off"] = off_np          # (2,H,W) float32
            data["grasp_masks"]["off_w"] = offw_np       # (1,H,W) float32

        # ---- 归一化/角度展开 ----
        ins_mask = ins_mask / 255.0
        grasp_qua_mask = grasp_qua_mask / 255.0
        grasp_ang_mask = grasp_ang_mask * np.pi / 180.0
        grasp_wid_mask = grasp_wid_mask / 255.0
        grasp_sin_mask = np.sin(2 * grasp_ang_mask)
        grasp_cos_mask = np.cos(2 * grasp_ang_mask)

        word_vec = tokenize(sent, self.word_length, True).squeeze(0)

        # ---- 回写 ----
        data["img"] = img
        data["mask"] = ins_mask
        data["grasp_masks"]["qua"] = grasp_qua_mask
        data["grasp_masks"]["ang"] = grasp_ang_mask
        data["grasp_masks"]["wid"] = grasp_wid_mask
        data["grasp_masks"]["sin"] = grasp_sin_mask
        data["grasp_masks"]["cos"] = grasp_cos_mask
        data["word_vec"] = word_vec
        data["inverse"] = mat_inv
        data["ori_size"] = np.array(img_size)

        return data

    def __len__(self):
        return len(self.sent_indices)

    def __getitem__(self, n):
        sent_id = self.get_sent_from_index(n)
        data = self._load_sent(sent_id)
        # path = debug_montage_one_image(data, out_path=f"./debug_vis/one_image_{n}.png",
        #                        stride=10, min_off_w=0.2, resize_to_width=1600)
        # print("saved:", path)
        return data

    @staticmethod
    def transform_grasp_inv(grasp_pt):
        pass

    def get_image_from_path(self, path):
        img_bgr = cv2.imread(path)
        if img_bgr is None:
            print(os.path.exists(path))
            print(path)
        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img

    def get_mask_from_path(self, path):
        return cv2.imread(path, cv2.IMREAD_UNCHANGED)

    def get_depth_from_path(self, path):
        return cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0  # mm -> m

    def get_image(self, n):
        img_path = os.path.join(self.root_dir, self.imgs[n])
        return self.get_image_from_path(img_path)

    def get_annotated_image(self, n, text=True):
        sample = self.__getitem__(n)

        img, sent, grasps, bbox = sample["img"], sample["sentence"], sample["grasp_rects"], sample["bbox"]
        if isinstance(img, torch.FloatTensor):
            img = img.permute(1, 2, 0)
            img = (img.cpu().numpy() * 255).astype(np.uint8)
        if self.transform_img:
            # 如果你使用的是 torchvision transforms，这里按需处理
            try:
                import torchvision.transforms.functional as tfn
                img = np.asarray(tfn.to_pil_image(img))
            except Exception:
                pass
        if self.transform_grasp:
            grasps = self.transform_grasp.inverse(grasps)

        tmp = img.copy()
        for entry in grasps:
            ptA, ptB, ptC, ptD = [list(map(int, pt.tolist())) for pt in entry]
            tmp = cv2.line(tmp, ptA, ptB, (0, 0, 0xFF), 2)
            tmp = cv2.line(tmp, ptD, ptC, (0, 0, 0xFF), 2)
            tmp = cv2.line(tmp, ptB, ptC, (0xFF, 0, 0), 2)
            tmp = cv2.line(tmp, ptA, ptD, (0xFF, 0, 0), 2)

        tmp = cv2.rectangle(tmp, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        if text:
            tmp = cv2.putText(tmp, sent, (0, 10), cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 0), 2, cv2.LINE_AA)
        return tmp

    def visualization(self, n, save_path):
        s = self.__getitem__(n)

        rgb = s["img"]
        if isinstance(rgb, torch.FloatTensor):
            rgb = rgb.permute(1, 2, 0)
            rgb = (rgb.cpu().numpy() * 255).astype(np.uint8)
        depth = (0xFF * s["depth"] / 3).astype(np.uint8)
        ii = self.get_annotated_image(n, text=False)
        sentence = s["sentence"]
        msk = s["mask"].astype(np.uint8) / 255

        fig = plt.figure(figsize=(25, 10))

        ax = fig.add_subplot(2, 4, 1)
        ax.imshow(rgb)
        ax.set_title("RGB")
        ax.axis("off")

        ax = fig.add_subplot(2, 4, 2)
        ax.imshow(depth, cmap="gray")
        ax.set_title("Depth")
        ax.axis("off")

        ax = fig.add_subplot(2, 4, 3)
        ax.imshow(msk)
        ax.set_title("Segm Mask")
        ax.axis("off")

        ax = fig.add_subplot(2, 4, 4)
        ax.imshow(ii)
        ax.set_title("Box & Grasp")
        ax.axis("off")

        ax = fig.add_subplot(2, 4, 5)
        plot = ax.imshow(s["grasp_masks"]["qua"], cmap="jet", vmin=0, vmax=1)
        ax.set_title("Grasp quality")
        ax.axis("off")
        plt.colorbar(plot)

        ax = fig.add_subplot(2, 4, 6)
        plot = ax.imshow(s["grasp_masks"]["sin"], cmap="rainbow", vmin=-1, vmax=1)
        ax.set_title("Angle-cosine")
        ax.axis("off")
        plt.colorbar(plot)

        ax = fig.add_subplot(2, 4, 7)
        plot = ax.imshow(s["grasp_masks"]["cos"], cmap="rainbow", vmin=-1, vmax=1)
        ax.set_title("Angle-sine")
        ax.axis("off")
        plt.colorbar(plot)

        ax = fig.add_subplot(2, 4, 8)
        plot = ax.imshow(s["grasp_masks"]["wid"], cmap="jet", vmin=0, vmax=1)
        ax.set_title("Width")
        ax.axis("off")
        plt.colorbar(plot)

        plt.suptitle(f"{sentence}", fontsize=20)
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, f"sample_{n}.png"))

    @staticmethod
    def collate_fn(batch):

        def stack_if_present(key, convert=True):
            if all(key in x["grasp_masks"] for x in batch):
                arrs = [x["grasp_masks"][key] for x in batch]

                if convert and isinstance(arrs[0], np.ndarray):
                    if key == "off_w":
                        # off_w 是 (1,H,W)，堆叠前先转为相同 dtype
                        arrs = [torch.from_numpy(a).float() for a in arrs]
                    else:
                        arrs = [torch.from_numpy(a).float() for a in arrs]
                return torch.stack(arrs)
            return None

        gm_qua = torch.stack([torch.from_numpy(x["grasp_masks"]["qua"]).float() for x in batch])
        gm_sin = torch.stack([torch.from_numpy(x["grasp_masks"]["sin"]).float() for x in batch])
        gm_cos = torch.stack([torch.from_numpy(x["grasp_masks"]["cos"]).float() for x in batch])
        gm_wid = torch.stack([torch.from_numpy(x["grasp_masks"]["wid"]).float() for x in batch])

        gm_off = stack_if_present("off")      # [B,2,H,W] or None
        gm_offw = stack_if_present("off_w")   # [B,1,H,W] or None

        grasp_masks = {"qua": gm_qua, "sin": gm_sin, "cos": gm_cos, "wid": gm_wid}
        if gm_off is not None:
            grasp_masks["off"] = gm_off
        if gm_offw is not None:
            grasp_masks["off_w"] = gm_offw

        return {
            "img": torch.stack([x["img"] for x in batch]),
            "depth": torch.stack([torch.from_numpy(x["depth"]) for x in batch]),
            "mask": torch.stack([torch.from_numpy(x["mask"]).float() for x in batch]),
            "grasp_masks": grasp_masks,
            "word_vec": torch.stack([x["word_vec"].long() for x in batch]),
            "grasps": [x["grasps"] for x in batch],
            "target": [x["target"] for x in batch],
            "sentence": [x["sentence"] for x in batch],
            "bbox": [x["bbox"] for x in batch],
            "target_idx": [x["target_idx"] for x in batch],
            "sent_id": [x["sent_id"] for x in batch],
            "scene_id": [x["scene_id"] for x in batch],
            "inverse": [x["inverse"] for x in batch],
            "ori_size": [x["ori_size"] for x in batch],
            "img_path": [x["img_path"] for x in batch],
        }

class GraspToolTransforms:
    # Class for converting cv2-like rectangle formats and generate grasp-quality-angle-width masks

    def __init__(self, width_factor=400, width=416, height=416):
        self.width_factor = width_factor
        self.width = width 
        self.height = height

    def __call__(self, grasp_rectangles, target):


        M = grasp_rectangles.shape[0]
        result = []

        for rect in grasp_rectangles:
         
            (cx, cy), (w, h), angle = cv2.minAreaRect(np.array(rect, dtype=np.float32))

            if w < h:
                w, h = h, w
                angle += 90.0

            # 角度归一化到 [-90, 90)
            while angle >= 90.0:
                angle -= 180.0
            while angle < -90.0:
                angle += 180.0

            result.append([cx, cy, w, h, angle, target])

        return np.array(result, dtype=np.float32)

    def inverse(self, grasp_rectangles):
        boxes = []
        for rect in grasp_rectangles:
            center_x, center_y, width, height, theta = rect[:5]
            box = ((center_x, center_y), (width, height), -(theta+180))
            box = cv2.boxPoints(box)
            box = np.intp(box)
            boxes.append(box)
        return boxes

    def generate_masks(self, grasp_rectangles):
        pos_out = np.zeros((self.height, self.width))
        ang_out = np.zeros((self.height, self.width))
        wid_out = np.zeros((self.height, self.width))
        for rect in grasp_rectangles:
            center_x, center_y, w_rect, h_rect, theta = rect[:5]
            
            # Get 4 corners of rotated rect
            # Convert from our angle represent to opencv's
            r_rect = ((center_x, center_y), (w_rect/2, h_rect), -(theta+180))
            box = cv2.boxPoints(r_rect)
            box = np.intp(box)

            rr, cc = polygon(box[:, 0], box[:,1])

            mask_rr = rr < self.width
            rr = rr[mask_rr]
            cc = cc[mask_rr]

            mask_cc = cc < self.height
            cc = cc[mask_cc]
            rr = rr[mask_cc]
            pos_out[cc, rr] = 1.0
            if theta < 0:
                ang_out[cc, rr] = int(theta + 180)
            else:
                ang_out[cc, rr] = int(theta)
            # Adopt width normalize accoding to class 
            wid_out[cc, rr] = np.clip(w_rect, 0.0, self.width_factor) / self.width_factor
        
        qua_out = (gaussian(pos_out, 3, preserve_range=True) * 255).astype(np.uint8)
        pos_out = (pos_out * 255).astype(np.uint8)
        ang_out = ang_out.astype(np.uint8)
        wid_out = (gaussian(wid_out, 3, preserve_range=True) * 255).astype(np.uint8)
        
        
        return {'pos': pos_out, 
                'qua': qua_out, 
                'ang': ang_out, 
                'wid': wid_out}

class GraspToolDataset(Dataset):

    def __init__(self, root_dir, input_size=416, split='train', word_length=17,
                 with_offset=False, offset_radius=20.0, offset_sigma=None):
        self.root_dir = root_dir
        self.input_size = (input_size, input_size)
        self.word_length = word_length
        self.samples = []
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).reshape(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).reshape(3, 1, 1)
        self.grasp_transform = GraspToolTransforms(width_factor=100, width=input_size, height=input_size)
        self.with_offset = bool(with_offset)
        self.offset_radius = float(offset_radius)
        self.offset_sigma = offset_sigma

        
        split_dir = os.path.join(root_dir, split)
        for fname in os.listdir(split_dir):
            if fname.endswith('.json'):
                img_name = fname.replace('.json', '.png')
                img_path = os.path.join(split_dir, img_name)
                json_path = os.path.join(split_dir, fname)
                if os.path.exists(img_path):
                    self.samples.append((img_path, json_path))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, json_path = self.samples[idx]

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ori_h, ori_w = image.shape[:2]

        with open(json_path, 'r') as f:
            data = json.load(f)

        obj = data["objects"][0]

        lang = obj["language"]
        # lang = obj["category"]
        polygon = np.array(obj["mask"], dtype=np.int32)
        grasps = [np.array(g, dtype=np.float32) for g in obj.get("grasps", [])]
        
        mask_img = np.zeros((ori_h, ori_w), dtype=np.uint8)
        cv2.drawContours(mask_img, [polygon], contourIdx=-1, color=1, thickness=-1)

        mat, mat_inv = self.get_transform_mat((ori_h, ori_w), self.input_size)
        # image = cv2.warpAffine(image, mat, self.input_size, flags=cv2.INTER_CUBIC,
        #                        borderValue=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255])
        image = cv2.warpAffine(image, mat, self.input_size, flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        image = torch.from_numpy(image.transpose((2, 0, 1))).float()
        image.div_(255.).sub_(self.mean).div_(self.std)

        mask_resized = cv2.warpAffine(mask_img, mat, self.input_size, flags=cv2.INTER_NEAREST,
                        borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        if len(grasps) > 0:
            grasps = np.stack(grasps, axis=0)
            ones = np.ones((grasps.shape[0], grasps.shape[1], 1))
            pts = np.concatenate([grasps, ones], axis=-1)
            pts_trans = np.matmul(mat[None, ...], pts.transpose(0, 2, 1)).transpose(0, 2, 1)
            grasps_trans = pts_trans.astype(np.float32)
        else:
            grasps = np.zeros((0, 4, 2), dtype=np.float32)
            grasps_trans = np.zeros((0, 4, 2), dtype=np.float32)
        
        grasp_target = self.grasp_transform(grasps, target=0)

        grasp_rect_format = self.grasp_transform(grasps_trans, target=0)

        grasp_masks_raw = self.grasp_transform.generate_masks(grasp_rect_format)

        # 处理为标准格式：qua, ang, wid, sin, cos
        qua = grasp_masks_raw["qua"] / 255.
        ang = grasp_masks_raw["ang"] * np.pi / 180.
        wid = grasp_masks_raw["wid"] / 255.
        sin = np.sin(2 * ang)
        cos = np.cos(2 * ang)

        grasp_masks = {
            "qua": torch.from_numpy(qua).float(),
            "sin": torch.from_numpy(sin).float(),
            "cos": torch.from_numpy(cos).float(),
            "wid": torch.from_numpy(wid).float(),
        }
        if self.with_offset:
            centers = (
                grasp_rect_format[:, :2]
                if len(grasp_rect_format)
                else np.zeros((0, 2), dtype=np.float32)
            )
            off, off_w = make_dense_offset_with_radius_np(
                centers_xy=centers,
                img_size_hw=self.input_size,
                r_pix=self.offset_radius,
                use_gaussian=True,
                sigma=self.offset_sigma,
            )
            grasp_masks["off"] = torch.from_numpy(off).float()
            grasp_masks["off_w"] = torch.from_numpy(off_w).float()

        # cv2.imwrite("./debug_vis/mask_img.png", mask_img * 255)
        # cv2.imwrite("./debug_vis/mask_resized.png", mask_resized * 255)
        # save_grasp_maps_with_mask(qua, sin, cos, wid, mask=mask_resized, prefix="sample1")

        word_vec = tokenize(lang, self.word_length, True).squeeze(0)
        if word_vec.shape[0] < self.word_length:
            pad_len = self.word_length - word_vec.shape[0]
            pad = torch.zeros((pad_len,), dtype=torch.long)
            word_vec = torch.cat([word_vec, pad], dim=0)
        elif word_vec.shape[0] > self.word_length:
            word_vec = word_vec[:self.word_length]
        ori_size = np.array([ori_h, ori_w])


        return {
            "img": image,
            "depth": torch.zeros(1, *self.input_size),  
            "mask": torch.from_numpy(mask_resized).float(),
            "grasp_masks": grasp_masks,
            "word_vec": word_vec,
            "grasps": grasp_target,
            "target": obj["category"],
            "sentence": lang,
            "bbox": None,
            "target_idx": 0,  # 无类别映射时可默认 0
            "sent_id": os.path.basename(json_path),
            "scene_id": os.path.basename(json_path),
            "inverse": mat_inv,
            "ori_size": np.array([ori_h, ori_w]),
            "img_path": img_path
        }

    def get_transform_mat(self, img_size, input_size):
        ori_h, ori_w = img_size
        inp_h, inp_w = input_size
        scale = min(inp_h / ori_h, inp_w / ori_w)
        new_h, new_w = ori_h * scale, ori_w * scale
        bias_x, bias_y = (inp_w - new_w) / 2., (inp_h - new_h) / 2.

        src = np.array([[0, 0], [ori_w, 0], [0, ori_h]], np.float32)
        dst = np.array([[bias_x, bias_y], [new_w + bias_x, bias_y], [bias_x, new_h + bias_y]], np.float32)

        mat = cv2.getAffineTransform(src, dst)
        mat_inv = cv2.getAffineTransform(dst, src)
        return mat, mat_inv

    @staticmethod
    def collate_fn(batch):
        grasp_masks = {
            key: torch.stack([x["grasp_masks"][key] for x in batch])
            for key in ("qua", "sin", "cos", "wid")
        }
        for key in ("off", "off_w"):
            if all(key in x["grasp_masks"] for x in batch):
                grasp_masks[key] = torch.stack(
                    [x["grasp_masks"][key] for x in batch]
                )
        return {
            "img": torch.stack([x["img"] for x in batch]),
            "depth": torch.stack([x["depth"] for x in batch]),
            "mask": torch.stack([x["mask"] for x in batch]),
            "grasp_masks": grasp_masks,
            "word_vec": torch.stack([x["word_vec"] for x in batch]),
            "grasps": [x["grasps"] for x in batch],
            "target": [x["target"] for x in batch],
            "sentence": [x["sentence"] for x in batch],
            "bbox": [x["bbox"] for x in batch],
            "target_idx": [x["target_idx"] for x in batch],
            "sent_id": [x["sent_id"] for x in batch],
            "scene_id": [x["scene_id"] for x in batch],
            "inverse": [x["inverse"] for x in batch],
            "ori_size": [x["ori_size"] for x in batch],
            "img_path": [x["img_path"] for x in batch]
        }
