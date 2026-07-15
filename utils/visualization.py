import os, cv2, torch
import numpy as np
from PIL import Image

# --------- helpers ---------
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
IMAGENET_STD  = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)

def _to_uint8_img(img_3chw, mean=None, std=None, try_denorm=True):
    t = torch.as_tensor(img_3chw).detach().cpu().float()
    assert t.ndim == 3 and t.shape[0] == 3, f"image must be [3,H,W], got {tuple(t.shape)}"
    if not try_denorm:
        if t.max() <= 1.0 + 1e-6: t = t * 255.0
        t = t.clamp(0,255)
        return t.byte().permute(1,2,0).numpy()

    if mean is not None and std is not None:
        mean = torch.as_tensor(mean, dtype=t.dtype).view(3,1,1)
        std  = torch.as_tensor(std,  dtype=t.dtype).view(3,1,1)
        t = t * std + mean              # to [0,1]
        t = (t * 255.0).clamp(0,255)
    else:
        mn, mx = float(t.min()), float(t.max())
        if -1e-3 <= mn and mx <= 1.0 + 1e-6:
            t = (t * 255.0).clamp(0,255)
        elif mn < -0.1 or (mx > 1.5 and mx < 10.0):
            t = t * IMAGENET_STD + IMAGENET_MEAN
            t = (t * 255.0).clamp(0,255)
        elif 0.0 <= mn <= 255.0 and 0.0 <= mx <= 255.0:
            t = t.clamp(0,255)
        else:
            t = (t - t.min()) / (t.max() - t.min() + 1e-12)
            t = (t * 255.0).clamp(0,255)
    return t.byte().permute(1,2,0).numpy()

def _to_hw(t):
    t = torch.as_tensor(t)
    while t.ndim > 2 and t.shape[0] == 1:
        t = t.squeeze(0)
    if t.ndim == 4 and t.shape[1] == 1:
        t = t.squeeze(1)
    assert t.ndim == 2, f"expect 2D (H,W), got {tuple(t.shape)}"
    return t

def _pick_topk_from_quality(qmap, k=5, thresh=0.5, nms_radius=8):
    H, W = qmap.shape
    flat = qmap.reshape(-1)
    idx = torch.nonzero(flat >= thresh, as_tuple=False).squeeze(1)
    if idx.numel() == 0:
        val, idm = flat.max(dim=0)
        y = (idm // W).item()
        x = (idm % W).item()
        return [(y, x, val.item())]

    vals = flat[idx]
    order = torch.argsort(vals, descending=True)
    yy = (idx // W); xx = (idx % W)
    yy_ord = yy[order].float(); xx_ord = xx[order].float(); val_ord = vals[order]
    taken = torch.zeros(order.numel(), dtype=torch.bool, device=order.device)
    r2 = float(nms_radius * nms_radius)
    pts = []
    for t in range(order.numel()):
        if taken[t]: continue
        y = int(yy_ord[t].item()); x = int(xx_ord[t].item()); s = float(val_ord[t].item())
        pts.append((y, x, s))
        if nms_radius > 0:
            dy = yy_ord - y; dx = xx_ord - x
            mask = (dy*dy + dx*dx) <= r2
            taken |= mask
        if len(pts) >= k: break
    return pts

# --------- final drawing ---------
def draw_gt_grasps_rectangles(
    image_3chw, qua_1hw, sin_1hw, cos_1hw, wid_1hw,
    save_path,
    k=5, thresh=0.5, jaw=20,
    mean=None, std=None, qua_is_logits=False,
    theta_is_height_dir=False,        # True: θ 表示“高度/法向(短边)方向”；False: θ 表示宽度方向
    width_factor=None,               # 若 wid 是相对值，给一个像素尺度，比如图像宽或对角线
    color_bgr=(0,255,0), thickness=2
):
    """
    从 (qua, sin, cos, wid) 画 grasp 矩形。
    - θ=atan2(s, c)
    - 若 theta_is_height_dir=True：θ 为夹爪高度/法向方向；否则 θ 为宽度方向
    - wid: 若为像素宽度，width_factor=None；若为相对宽度，传入 width_factor（像素）
    """
    # 1) 准备底图
    img = _to_uint8_img(image_3chw, mean=mean, std=std, try_denorm=True)  # HWC, RGB uint8
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    H, W = img.shape[:2]

    # 2) map 提取
    q = _to_hw(qua_1hw[0].detach().float())
    s = _to_hw(sin_1hw[0].detach().float())
    c = _to_hw(cos_1hw[0].detach().float())
    w = _to_hw(wid_1hw[0].detach().float()).clamp(min=0)

    if qua_is_logits:
        q = torch.sigmoid(q)

    # 3) 角度与方向
    # ---- 解角：双角还原 + 单位化以防数值偏 ----
    mag = torch.clamp(torch.sqrt(s*s + c*c), min=1e-6)
    s = s / mag
    c = c / mag
    theta = 0.5 * torch.atan2(s, c)   # ← 关键：双角还原
    # -------------------------------------------

# wid 为相对值 → 像素
    if width_factor is not None:
        w = w * float(width_factor)
    half_w = 0.5 * w
    half_jaw = 0.5 * float(jaw)

    # 单位方向：
    if theta_is_height_dir:
        # 高度方向 u = (cosθ, sinθ); 宽度方向 v = (-sinθ, cosθ)
        ux, uy = torch.cos(theta), torch.sin(theta)
        vx, vy = -torch.sin(theta), torch.cos(theta)
    else:
        # 宽度方向直接用 (cosθ, sinθ)；高度为其法线
        vx, vy = torch.cos(theta), torch.sin(theta)
        ux, uy = -torch.sin(theta), torch.cos(theta)

    # 4) 选 top-k
    pts = _pick_topk_from_quality(q, k=k, thresh=thresh, nms_radius=8)

    # 5) 逐点画框
    for (y, x, score) in pts:
        vx_xy = (float(vx[y, x].item()), float(vy[y, x].item()))
        ux_xy = (float(ux[y, x].item()), float(uy[y, x].item()))
        hw = float(half_w[y, x].item())

        # 宽度两端点
        x1 = x - hw * vx_xy[0]; y1 = y - hw * vx_xy[1]
        x2 = x + hw * vx_xy[0]; y2 = y + hw * vx_xy[1]

        # 沿高度扩展 jaw
        c1 = (x1 + half_jaw * ux_xy[0], y1 + half_jaw * ux_xy[1])
        c2 = (x2 + half_jaw * ux_xy[0], y2 + half_jaw * ux_xy[1])
        c3 = (x2 - half_jaw * ux_xy[0], y2 - half_jaw * ux_xy[1])
        c4 = (x1 - half_jaw * ux_xy[0], y1 - half_jaw * ux_xy[1])

        poly = np.round(np.array([c1, c2, c3, c4])).astype(np.int32)
        poly[:, 0] = np.clip(poly[:, 0], 0, W-1)
        poly[:, 1] = np.clip(poly[:, 1], 0, H-1)

        cv2.polylines(img, [poly], isClosed=True, color=color_bgr, thickness=thickness)
        cv2.drawMarker(img, (int(x), int(y)), color_bgr,
                       markerType=cv2.MARKER_TILTED_CROSS, markerSize=8, thickness=thickness)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).save(save_path)
    return save_path
