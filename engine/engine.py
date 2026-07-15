import os
import time
from tqdm import tqdm

import cv2
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
import wandb
from loguru import logger
from utils.dataset import tokenize
from utils.misc import (AverageMeter, ProgressMeter, concat_all_gather, trainMetricGPU, get_seg_image)
from utils.grasp_eval import (detect_grasps, calculate_iou, calculate_max_iou, calculate_jacquard_index, visualization)
import matplotlib.pyplot as plt
import colorsys
from datetime import datetime
from toolrgs.runtime import autocast, current_device, move_to_device

import os
import cv2
import numpy as np
import torch


def _scalar(value):
    if isinstance(value, torch.Tensor):
        return value.detach().mean().item()
    return float(value)


def _apply_affine_points(points_xy: np.ndarray, mat_2x3: np.ndarray) -> np.ndarray:
    """points_xy: (N,2) in (x,y); mat_2x3: 2x3; return (N,2)"""
    if points_xy is None or len(points_xy) == 0:
        return points_xy
    homo = np.concatenate([points_xy.astype(np.float32),
                           np.ones((points_xy.shape[0], 1), dtype=np.float32)], axis=1)  # (N,3)
    out = (mat_2x3 @ homo.T).T
    return out.astype(np.float32)

def _rect_center_xy(rect_xy4: np.ndarray) -> np.ndarray:
    """rect_xy4: (4,2) corners; return (2,) center"""
    return rect_xy4.mean(axis=0)

def _translate_rect_xy4(rect_xy4: np.ndarray, dx: float, dy: float) -> np.ndarray:
    out = rect_xy4.copy()
    out[:, 0] += dx
    out[:, 1] += dy
    return out

def _sample_off_nn(off_1x2_hw: torch.Tensor, x: float, y: float) -> tuple[float, float]:
    """off_1x2_hw: [1,2,H,W], values are normalized dx,dy; nearest-neighbor sample."""
    H, W = off_1x2_hw.shape[-2:]
    xi = int(np.clip(round(float(x)), 0, W-1))
    yi = int(np.clip(round(float(y)), 0, H-1))
    dx = float(off_1x2_hw[0, 0, yi, xi].item())
    dy = float(off_1x2_hw[0, 1, yi, xi].item())
    return dx, dy

def _refine_grasps_with_offset(
    grasps_list: list,
    off_map_1x2_hw: torch.Tensor,   # [1,2,H_in,W_in] normalized (dx,dy)
    inv_mat_2x3: np.ndarray,        # inverse affine: input->orig (already in your batch)
    r_norm: float,                  # same r used when training GT offsets
    assume_rect_is_xy4: bool = True # True if detect_grasps returns 4-point boxes; False if returns (cx,cy,w,h,theta)
):
    """
    Strategy: (orig center) -> forward to input coords -> add off*r_norm -> inverse back -> translate rectangle by Δ.
    """
    # forward matrix: orig->input
    fwd_mat = cv2.invertAffineTransform(inv_mat_2x3)
    refined = []
    for g in grasps_list:
        if assume_rect_is_xy4:
            rect_xy4 = np.asarray(g, dtype=np.float32)             # (4,2)
            c_orig = _rect_center_xy(rect_xy4)                      # (2,)
        else:
            # (cx,cy,w,h,theta) -> center only
            cx, cy = float(g[0]), float(g[1])
            rect_xy4 = None
            c_orig = np.array([cx, cy], dtype=np.float32)

        # center to input
        c_in = _apply_affine_points(c_orig[None, :], fwd_mat)[0]
        # sample off and de-normalize
        dx_n, dy_n = _sample_off_nn(off_map_1x2_hw, c_in[0], c_in[1])
        dx_pix_in = dx_n * float(r_norm)
        dy_pix_in = dy_n * float(r_norm)
        # move center in input, then map back to original
        c_in_ref = np.array([[c_in[0] + dx_pix_in, c_in[1] + dy_pix_in]], dtype=np.float32)
        c_orig_ref = _apply_affine_points(c_in_ref, inv_mat_2x3)[0]
        delta = c_orig_ref - c_orig  # (dx,dy) in original pixels

        if assume_rect_is_xy4:
            rect_ref = _translate_rect_xy4(rect_xy4, float(delta[0]), float(delta[1]))
            refined.append(rect_ref)
        else:
            # keep w,h,theta the same; only shift center
            cxr, cyr = float(c_orig_ref[0]), float(c_orig_ref[1])
            refined.append(np.array([cxr, cyr, g[2], g[3], g[4]], dtype=np.float32))
    return refined


def save_image_and_targets_grid(
    image,                   # (B,C,H,W) or (C,H,W) or (H,W,3)
    ins_mask=None,           # (B,1,H,W) or (H,W)
    qua=None,                # grasp_qua_mask
    sin=None,
    cos=None,
    wid=None,
    save_path="vis/compare.png",
    imagenet_mean=(0.485, 0.456, 0.406),
    imagenet_std=(0.229, 0.224, 0.225),
    try_denorm=True,         # 若怀疑 image 是 ImageNet 标准化，则反归一化
    target_size=None,        # 指定可视化尺寸 (W,H)。默认跟 image 尺寸一致
    colormap=cv2.COLORMAP_JET
):
    """
    生成 2x3 网格：
      Row1: Image | ins_mask | qua
      Row2: sin   | cos      | wid
    缺失的项会自动填充“空白”。

    返回: 保存路径 save_path
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    def to_numpy(x):
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        return np.asarray(x)

    def squeeze_to_2d(x):
        """从 (B,C,H,W)/(C,H,W)/(H,W)/(H,W,1) 中拿到 2D (H,W)"""
        if x is None:
            return None
        if x.ndim == 4:  # (B,C,H,W)
            x = x[0]
            if x.shape[0] in (1,3):  # (C,H,W)
                x = x[0] if x.shape[0] > 1 else x[0]
        if x.ndim == 3:
            # (C,H,W) or (H,W,C)
            if x.shape[0] in (1,3):   # (C,H,W)
                if x.shape[0] == 1:
                    x = x[0]
                else:
                    x = x.mean(0)
            elif x.shape[2] in (1,3): # (H,W,C)
                if x.shape[2] == 1:
                    x = x[...,0]
                else:
                    x = cv2.cvtColor(x.astype(np.float32), cv2.COLOR_RGB2GRAY)
            else:
                x = x[0]
        return x.astype(np.float32)

    def norm_to_uint8(x):
        if x is None:
            return None
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        xmin, xmax = float(x.min()), float(x.max())
        if xmax > xmin:
            x = (x - xmin) / (xmax - xmin)
        else:
            x = np.zeros_like(x, dtype=np.float32)
        return (x * 255.0).clip(0, 255).astype(np.uint8)

    def make_color_from_gray(gray, cm=colormap):
        if gray is None:
            return None
        return cv2.applyColorMap(gray, cm)  # BGR

    def put_title(img_bgr, title):
        """在图上方画标题条"""
        if img_bgr is None:
            return None
        h, w = img_bgr.shape[:2]
        bar_h = max(24, h // 20)
        overlay = img_bgr.copy()
        cv2.rectangle(overlay, (0,0), (w, bar_h), (0,0,0), -1)
        alpha = 0.5
        img_bgr = cv2.addWeighted(overlay, alpha, img_bgr, 1-alpha, 0)

        cv2.putText(img_bgr, title, (8, bar_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)
        return img_bgr

    img = to_numpy(image)

    if img.ndim == 4:  # (B,C,H,W)
        img = img[0]
    if img.ndim == 3 and img.shape[0] in (1,3):  # (C,H,W) -> (H,W,C)
        img = np.transpose(img, (1,2,0))

    img_vis = img.astype(np.float32)
    if try_denorm and img_vis.ndim == 3 and img_vis.shape[2] == 3:

        if (img_vis.min() < -0.5) or (img_vis.max() > 1.5):
            mean = np.array(imagenet_mean, dtype=np.float32).reshape(1,1,3)
            std  = np.array(imagenet_std, dtype=np.float32).reshape(1,1,3)
            img_vis = img_vis * std + mean

        img_vis = np.clip(img_vis, 0.0, 1.0)
        img_vis = (img_vis * 255.0).astype(np.uint8)
    else:
        if img_vis.max() <= 1.5:
            img_vis = (np.clip(img_vis, 0.0, 1.0) * 255.0).astype(np.uint8)
        else:
            img_vis = np.clip(img_vis, 0, 255).astype(np.uint8)

    if img_vis.ndim == 3 and img_vis.shape[2] == 3:
        img_bgr = cv2.cvtColor(img_vis, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = cv2.cvtColor(norm_to_uint8(squeeze_to_2d(img_vis)), cv2.COLOR_GRAY2BGR)

    H, W = img_bgr.shape[:2]
    if target_size is not None:
        Wt, Ht = target_size
        img_bgr = cv2.resize(img_bgr, (Wt, Ht), interpolation=cv2.INTER_LINEAR)
        H, W = Ht, Wt

    ins = make_color_from_gray(norm_to_uint8(squeeze_to_2d(to_numpy(ins_mask)))) if ins_mask is not None else None
    qua = make_color_from_gray(norm_to_uint8(squeeze_to_2d(to_numpy(qua)))) if qua is not None else None
    sinm = make_color_from_gray(norm_to_uint8(squeeze_to_2d(to_numpy(sin)))) if sin is not None else None
    cosm = make_color_from_gray(norm_to_uint8(squeeze_to_2d(to_numpy(cos)))) if cos is not None else None
    widm = make_color_from_gray(norm_to_uint8(squeeze_to_2d(to_numpy(wid)))) if wid is not None else None

    def resize_to(img, W, H):
        return cv2.resize(img, (W,H), interpolation=cv2.INTER_NEAREST)

    panels = [
        ("Image", img_bgr),
        ("InsMask", ins),
        ("Qua", qua),
        ("Sin", sinm),
        ("Cos", cosm),
        ("Wid", widm),
    ]

    for i, (name, pane) in enumerate(panels):
        if pane is None:
            pane = np.zeros_like(img_bgr)
        else:
            pane = resize_to(pane, W, H)
        panels[i] = (name, put_title(pane, name))

    row1 = np.hstack([panels[0][1], panels[1][1], panels[2][1]])
    row2 = np.hstack([panels[3][1], panels[4][1], panels[5][1]])
    grid = np.vstack([row1, row2])

    cv2.imwrite(save_path, grid)
    return save_path


def _to_numpy(x):
    if isinstance(x, torch.Tensor):
        x = x.detach().cpu()
        if x.ndim == 4:  # [B,1,H,W]
            x = x[0]
        if x.ndim == 3:  # [1,H,W]
            x = x.squeeze(0)
        x = x.numpy()
    return x

def _minmax(img, eps=1e-8):
    m, M = img.min(), img.max()
    if M - m < eps:
        return np.zeros_like(img)
    return (img - m) / (M - m)

def draw_grasp(mask, grasp, color, thickness=2):

    if len(mask.shape) == 2:
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    cx, cy, w, h, theta = grasp[:5]
    if abs(theta) <= np.pi:  # 如果是弧度
        theta = np.rad2deg(theta)

    box = cv2.boxPoints(((cx, cy), (w, h), theta))
    box = np.intp(box)
    cv2.drawContours(mask, [box], 0, color, thickness)
    return mask

def save_grasp_targets(
    ins_mask_target,
    grasp_qua_mask_target,
    grasp_sin_mask_target,
    grasp_cos_mask_target,
    grasp_wid_mask_target,
    save_dir="./debug_masks",
    filename=None
):
    os.makedirs(save_dir, exist_ok=True)

    ins  = _to_numpy(ins_mask_target)
    qua  = _to_numpy(grasp_qua_mask_target)
    s    = _to_numpy(grasp_sin_mask_target)
    c    = _to_numpy(grasp_cos_mask_target)
    wid  = _to_numpy(grasp_wid_mask_target)

    ins_viz = _minmax(ins)
    qua_viz = _minmax(qua)
    wid_viz = _minmax(wid)

    ang = np.arctan2(s, c)
    ang_deg = np.degrees(ang)
    ang_viz = _minmax(ang_deg)

    hsv = np.zeros((ang_viz.shape[0], ang_viz.shape[1], 3), dtype=np.float32)
    hsv[..., 0] = ang_viz
    hsv[..., 1] = 1.0
    hsv[..., 2] = qua_viz
    rgb = np.zeros_like(hsv)
    for i in range(hsv.shape[0]):
        for j in range(hsv.shape[1]):
            rgb[i, j] = colorsys.hsv_to_rgb(hsv[i, j, 0], hsv[i, j, 1], hsv[i, j, 2])

    fig, axs = plt.subplots(2, 3, figsize=(16, 7))
    fig.suptitle("Targets after inverse warp", fontsize=14)

    axs[0, 0].imshow(ins_viz, cmap='viridis'); axs[0, 0].set_title('ins_mask_target'); axs[0, 0].axis('off')
    axs[0, 1].imshow(qua_viz, cmap='viridis'); axs[0, 1].set_title('grasp_qua_mask_target'); axs[0, 1].axis('off')
    axs[0, 2].imshow(wid_viz, cmap='viridis'); axs[0, 2].set_title('grasp_wid_mask_target'); axs[0, 2].axis('off')
    axs[1, 0].imshow(s, cmap='viridis'); axs[1, 0].set_title('grasp_sin_mask_target'); axs[1, 0].axis('off')
    axs[1, 1].imshow(c, cmap='viridis'); axs[1, 1].set_title('grasp_cos_mask_target'); axs[1, 1].axis('off')
    axs[1, 2].imshow(rgb); axs[1, 2].set_title('Angle (qua as V)'); axs[1, 2].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])

    # 默认用时间戳命名，防止覆盖
    if filename is None:
        filename = f"grasp_targets_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"

    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, dpi=150)
    plt.close(fig)

    print(f"[Saved] {save_path}")


def save_prediction_visualization(ins_mask_pred, ins_mask_target, save_dir="./debug_vis"):


    os.makedirs(save_dir, exist_ok=True)

    pred_np = ins_mask_pred#.detach().cpu().numpy()
    target_np = ins_mask_target#.detach().cpu().numpy()

    if pred_np.ndim == 4:
        pred_np = pred_np[0][0]
        target_np = target_np[0][0]


    if pred_np.ndim == 3 and pred_np.shape[0] > 1:
        pred_np = pred_np[0]
        target_np = target_np[0]

    # 可视化保存
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].imshow(pred_np, cmap='viridis')
    axs[0].set_title("Predicted Mask")
    axs[0].axis('off')

    axs[1].imshow(target_np, cmap='viridis')
    axs[1].set_title("Ground Truth Mask")
    axs[1].axis('off')

    fig.suptitle(f"Step {10}")
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"validate_mask_step_{10}.png")
    plt.savefig(save_path)
    plt.close(fig)


def save_grasp_pred_all_gt(mask, pred, targets, save_path):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 转成 BGR 彩图
    if mask.ndim == 2:
        mask_vis = cv2.cvtColor(mask*255, cv2.COLOR_GRAY2BGR)
    else:
        mask_vis = mask.copy()

    def draw_one(g, color):
        cx, cy, w, h, theta = g[:5]
        if abs(theta) <= np.pi:  # 弧度转角度
            theta = np.degrees(theta)
        box = cv2.boxPoints(((float(cx), float(cy)), (float(w), float(h)), float(theta)))
        box = np.intp(box)
        cv2.drawContours(mask_vis, [box], 0, color, 2)

    # 画预测 grasp[0]（绿色）
    if len(pred) > 0:
        draw_one(pred[0], (0, 255, 0))

    # 画所有 GT（红色）
    for gt in targets:
        draw_one(gt, (0, 0, 255))

    cv2.imwrite(save_path, mask_vis)

def train_with_grasp(train_loader, model, optimizer, scheduler, scaler, epoch, args):
    device = current_device(int(getattr(args, "npu", getattr(args, "gpu", 0))))
    batch_time = AverageMeter('Batch', ':2.2f')
    data_time = AverageMeter('Data', ':2.2f')
    lr = AverageMeter('Lr', ':1.6f')
    loss_meter = AverageMeter('Loss', ':2.4f')
    qua_loss_metter = AverageMeter('Loss_qua', ':2.4f')
    sin_loss_metter = AverageMeter('Loss_sin', ':2.4f')
    cos_loss_metter = AverageMeter('Loss_cos', ':2.4f')
    wid_loss_metter = AverageMeter('Loss_wid', ':2.4f')
    off_loss_metter = AverageMeter('Loss_off', ':2.4f')
    iou_meter = AverageMeter('IoU', ':2.2f')
    pr_meter = AverageMeter('Prec@50', ':2.2f')
    progress = ProgressMeter(
        len(train_loader),
        [
            batch_time, data_time, lr, loss_meter,
            qua_loss_metter, sin_loss_metter, cos_loss_metter, wid_loss_metter, off_loss_metter,
            iou_meter, pr_meter
        ],
        prefix="Training: Epoch=[{}/{}] ".format(epoch, args.epochs))

    model.train()
    time.sleep(2)
    end = time.time()

    for i, data in enumerate(train_loader):
        # image, target, text = data
        # ins_mask, grasp_quality_mask, grasp_sin_mask, grasp_cos_mask, grasp_width_mask = target

        image = data["img"]
        text = data["word_vec"]
        ins_mask = data["mask"]
        grasp_qua_mask = data["grasp_masks"]["qua"]
        grasp_sin_mask = data["grasp_masks"]["sin"]
        grasp_cos_mask = data["grasp_masks"]["cos"]
        grasp_wid_mask = data["grasp_masks"]["wid"]
        grasp_off_mask = data["grasp_masks"].get("off")
        grasp_off_weight = data["grasp_masks"].get("off_w")


        data_time.update(time.time() - end)
        # data
        image = move_to_device(image, device)
        text = move_to_device(text, device)
        ins_mask = move_to_device(ins_mask, device).unsqueeze(1)
        grasp_qua_mask = move_to_device(grasp_qua_mask, device).unsqueeze(1)
        grasp_sin_mask = move_to_device(grasp_sin_mask, device).unsqueeze(1)
        grasp_cos_mask = move_to_device(grasp_cos_mask, device).unsqueeze(1)
        grasp_wid_mask = move_to_device(grasp_wid_mask, device).unsqueeze(1)
        if grasp_off_mask is not None:
            grasp_off_mask = move_to_device(grasp_off_mask, device)
        if grasp_off_weight is not None:
            grasp_off_weight = move_to_device(grasp_off_weight, device)

        # # multi-scale training
        # image = F.interpolate(image, size=(new_size, new_size), mode='bilinear')

        # forward
        with autocast(enabled=bool(getattr(args, "amp", True))):
            pred, target, loss, loss_dict = model(
                image, text, ins_mask, grasp_qua_mask, grasp_sin_mask,
                grasp_cos_mask, grasp_wid_mask, grasp_off_mask,
                grasp_off_weight,
            )

        ins_mask_pred = pred[0]
        ins_mask_target = target[0]

        # if (i % 10) == 0:
        #     save_path = f"debug_gt/epoch{epoch:03d}_iter{i:06d}.png"
        #     save_image_and_targets_grid(
        #         image,                              # 原始输入图 (B,C,H,W)
        #         ins_mask=ins_mask,                  # GT 实例掩码
        #         qua=grasp_qua_mask,                 # GT qua
        #         sin=grasp_sin_mask,                 # GT sin
        #         cos=grasp_cos_mask,                 # GT cos
        #         wid=grasp_wid_mask,                 # GT wid
        #         save_path=save_path,
        #         try_denorm=True,                    # 如你的图像是 ImageNet 标准化
        #         # target_size=(512, 512),           # 如需强制尺寸，解开这一行
        #     )
        #     print("saved:", save_path)

        # save_grasp_targets(target[0], target[1], target[2], target[3], target[4], filename= "target")
        # save_grasp_targets(pred[0], pred[1], pred[2], pred[3], pred[4], filename= "pred")

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if args.max_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        scaler.step(optimizer)
        scaler.update()

        # metric
        iou, pr5 = trainMetricGPU(ins_mask_pred, ins_mask_target, 0.35, 0.5)
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(loss.detach())
            dist.all_reduce(iou)
            dist.all_reduce(pr5)
            world_size = dist.get_world_size()
            loss = loss / world_size
            iou = iou / world_size
            pr5 = pr5 / world_size

        loss_meter.update(loss.item(), image.size(0))
        qua_loss_metter.update(_scalar(loss_dict["m_qua"]), image.size(0))
        sin_loss_metter.update(_scalar(loss_dict["m_sin"]), image.size(0))
        cos_loss_metter.update(_scalar(loss_dict["m_cos"]), image.size(0))
        wid_loss_metter.update(_scalar(loss_dict["m_wid"]), image.size(0))
        off_loss_metter.update(_scalar(loss_dict.get("m_off", 0.0)), image.size(0))
        iou_meter.update(iou.item(), image.size(0))
        pr_meter.update(pr5.item(), image.size(0))
        lr.update(scheduler.get_last_lr()[-1])
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0:
            progress.display(i + 1)


@torch.no_grad()
def _legacy_validate_with_grasp(val_loader, model, epoch, args):
    device = current_device(int(getattr(args, "npu", getattr(args, "gpu", 0))))



    import cv2, time
    import torch.nn.functional as F


    # ---------- helpers ----------
    def inverse(img, mat, w, h):
        return cv2.warpAffine(img, mat, (w, h), flags=cv2.INTER_NEAREST, borderValue=0.)

    def _apply_affine_points(points_xy: np.ndarray, mat_2x3: np.ndarray) -> np.ndarray:
        if points_xy is None or len(points_xy) == 0:
            return points_xy
        homo = np.concatenate([points_xy.astype(np.float32),
                               np.ones((points_xy.shape[0], 1), dtype=np.float32)], axis=1)
        out = (mat_2x3 @ homo.T).T
        return out.astype(np.float32)

    def _rect_center_xy(rect_xy4: np.ndarray) -> np.ndarray:
        return rect_xy4.mean(axis=0)  # (2,)

    def _translate_rect_xy4(rect_xy4: np.ndarray, dx: float, dy: float) -> np.ndarray:
        out = rect_xy4.copy()
        out[:, 0] += dx
        out[:, 1] += dy
        return out

    def _sample_off_nn(off_1x2_hw, x, y):
        """
        支持 off 为 (2,H,W) 或 (1,2,H,W)，返回最近邻采样的 (dx, dy)（网络输出的归一化单位）
        """
        # 转 numpy
        if isinstance(off_1x2_hw, torch.Tensor):
            off = off_1x2_hw.detach().to('cpu', dtype=torch.float32).numpy()
        else:
            off = np.asarray(off_1x2_hw, dtype=np.float32)

        # 统一成 (2,H,W)
        if off.ndim == 4:
            # (1,2,H,W) -> (2,H,W)
            assert off.shape[0] == 1 and off.shape[1] == 2, f"bad off shape {off.shape}"
            off2 = off[0]
        elif off.ndim == 3:
            # (2,H,W)
            assert off.shape[0] == 2, f"bad off shape {off.shape}"
            off2 = off
        else:
            raise ValueError(f"_sample_off_nn expects (2,H,W) or (1,2,H,W), got {off.shape}")

        C, H, W = off2.shape
        assert C == 2, f"channel must be 2, got {C}"

        # 最近邻取整 + 边界裁剪（注意 x 对应宽，y 对应高）
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        xi = max(0, min(W - 1, xi))
        yi = max(0, min(H - 1, yi))

        dx = float(off2[0, yi, xi])
        dy = float(off2[1, yi, xi])
        return dx, dy

    def _to_xy4_one(rect, assume_theta_degree=True, wh_is_full=True):
        """
        将抓取框统一为 (4,2) 顶点格式。
        rect: 5参 [cx,cy,w,h,theta] 或 8参 [x1,y1,...,x4,y4]
        assume_theta_degree: theta 是否为角度（是就会自动转弧度）
        wh_is_full: w/h 是否为“整宽整高”（若传的是半宽/半高，设为 False）
        """
        arr = np.asarray(rect, dtype=np.float32).reshape(-1)
        if arr.size == 8:
            return arr.reshape(4, 2)

        if arr.size != 5:
            raise ValueError(f"rect length {arr.size} not in (5, 8). rect={arr.tolist()}")

        cx, cy, w, h, th = arr.tolist()
        if assume_theta_degree:
            th = np.deg2rad(th)
        c, s = np.cos(th), np.sin(th)

        # 处理 w/h 的定义
        if wh_is_full:
            dx, dy = w / 2.0, h / 2.0
        else:
            dx, dy = w, h

        # 顺时针：左上 -> 右上 -> 右下 -> 左下
        corners = np.array([[-dx, -dy], [dx, -dy], [dx,  dy], [-dx,  dy]], dtype=np.float32)
        R = np.array([[c, -s], [s,  c]], dtype=np.float32)
        rot = corners @ R.T
        rot[:, 0] += cx; rot[:, 1] += cy
        return rot  # (4,2)


    def _xy4_to_five(rect_xy4, out_degrees=True):
        """
        (4,2) -> [cx, cy, w, h, theta] using cv2.minAreaRect
        返回 theta 默认是度数，范围约 (-90, 90]
        """
        pts = np.asarray(rect_xy4, dtype=np.float32).reshape(4, 2)
        (cx, cy), (w, h), ang = cv2.minAreaRect(pts)

        # 可选归一化：确保 w 表示较长边，角度落在 (-90, 90]
        if w < h:
            w, h = h, w
            ang = ang + 90.0
        if ang <= -90.0:
            ang += 180.0
        if ang > 90.0:
            ang -= 180.0

        theta = ang if out_degrees else np.deg2rad(ang)
        return np.array([cx, cy, w, h, theta], dtype=np.float32)

    def _to_five_any(rect, assume_degrees=True):
        """
        将任意常见格式统一为 5 参 [cx, cy, w, h, theta]（theta 默认度数）
        支持：
        - 5 参: [cx, cy, w, h, theta] -> 直通
        - 6 参: [cx, cy, w, h, theta, extra] -> 取前 5 个，忽略附加字段
        - 8 参: [x1,y1,x2,y2,x3,y3,x4,y4] -> 先转 (4,2) 再转 5 参
        - (4,2): 顶点 -> 5 参
        - 4 参: (xmin,ymin,xmax,ymax) -> 5 参（theta=0）
        """
        arr = np.asarray(rect, dtype=np.float32).reshape(-1)
        n = arr.size

        if n == 5:
            # [cx, cy, w, h, theta]
            return arr

        if n == 6:
            # 兼容带一个附加字段的 6 参，通常最后一个是 score/cls/jaw_width 等
            cand = arr[:5].copy()
            # 简单健壮性校验：若 cand[4] 不像角度（>180 且不是弧度范围），尝试另一种排列 (cx,cy,theta,w,h)
            th = float(cand[4])
            if not (abs(th) <= 180.0 or abs(th) <= 6.283185):  # 不是度也不是弧度
                # 常见变体： [cx,cy,theta,w,h,extra] -> 重新整理到 [cx,cy,w,h,theta]
                cx, cy, theta, w, h = arr[:5].tolist()
                cand = np.array([cx, cy, w, h, theta], dtype=np.float32)
            return cand

        if n == 8:
            # [x1,y1,...,x4,y4]
            return _xy4_to_five(arr.reshape(4, 2), out_degrees=assume_degrees)

        if n == 4:
            # 轴对齐框
            xmin, ymin, xmax, ymax = arr.tolist()
            cx = 0.5 * (xmin + xmax)
            cy = 0.5 * (ymin + ymax)
            w  = max(1e-6, xmax - xmin)
            h  = max(1e-6, ymax - ymin)
            theta = 0.0
            return np.array([cx, cy, w, h, theta], dtype=np.float32)

        if n == 2*4:
            return _xy4_to_five(arr.reshape(4, 2), out_degrees=assume_degrees)

        # 打印一条更友好的调试信息
        raise ValueError(
            f"Unexpected rect length {n}. Supported: 4/(4,2)/5/6/8. "
            f"Got rect={arr.tolist()}"
        )

    def _batch_to_five(rects, assume_degrees=True):
        return [ _to_five_any(r, assume_degrees) for r in rects ]

    def _to_six_any(rect, assume_degrees=True):
        """
        将任意常见格式统一为 6 参 [cx,cy,w,h,theta,extra]
        - 若已是 6 参 -> 直通
        - 若是 5 参/8 参/(4,2)/4 参 -> 先转 5 参，再在末尾补一个占位 0.0
        说明：评估里只用到前 5 个；第 6 个仅为解包占位。
        """
        arr = np.asarray(rect, dtype=np.float32).reshape(-1)
        if arr.size == 6:
            return arr
        five = _to_five_any(arr, assume_degrees=assume_degrees)  # 你之前已实现
        extra = np.array([0.0], dtype=np.float32)
        return np.concatenate([five, extra], axis=0)

    def _batch_to_six(rects, assume_degrees=True):
        return [_to_six_any(r, assume_degrees=assume_degrees) for r in rects]

    def _refine_grasps_with_offset(rects_xy4_list, off_map_1x2_hw, inv_mat_2x3, r_norm):
        """
        rects_xy4_list: list，元素可以是 5参或 8参；本函数会统一成 (4,2)
        off_map_1x2_hw: (2,H,W) 或 (1,2,H,W)
        inv_mat_2x3: 2x3 仿射矩阵（input->original）
        r_norm: 将网络归一化位移还原为像素的尺度
        """
        inv = np.asarray(inv_mat_2x3, dtype=np.float32)
        fwd = cv2.invertAffineTransform(inv)
        r_norm = float(r_norm)

        # 统一 off_map -> (2,H,W)
        if isinstance(off_map_1x2_hw, torch.Tensor):
            off = off_map_1x2_hw.detach().to('cpu', dtype=torch.float32).numpy()
        else:
            off = np.asarray(off_map_1x2_hw, dtype=np.float32)
        if off.ndim == 4:  # (1,2,H,W)
            off = off[0]
        assert off.ndim == 3 and off.shape[0] == 2, f"off_map shape should be (2,H,W) or (1,2,H,W), got {off.shape}"

        # —— 新增：把 5参/8参统一成 (4,2) —— #
        rects_xy4_norm = []
        for idx, r in enumerate(rects_xy4_list):
            try:
                rects_xy4_norm.append(_to_xy4_one(r, assume_theta_degree=True, wh_is_full=True))
            except Exception as e:
                print(f"[rect-norm] bad rect at idx={idx}: {r}")
                raise

        refined = []
        for rect_xy4 in rects_xy4_norm:
            rect_xy4 = np.asarray(rect_xy4, dtype=np.float32).reshape(4, 2)

            # --- 中心点 (1,2) in original ---
            c_orig = np.asarray(_rect_center_xy(rect_xy4), dtype=np.float32).reshape(1, 2)

            # original -> input
            c_in = _apply_affine_points(c_orig, fwd)   # (1,2)
            x_in, y_in = float(c_in[0, 0]), float(c_in[0, 1])

            # 采样 offset（网络输出为归一化单位）
            dx_n, dy_n = _sample_off_nn(off, x_in, y_in)   # 标量
            dx_in, dy_in = dx_n * r_norm, dy_n * r_norm    # 还原到像素（input）

            # 偏移后的点（input） -> original
            c_in_ref = np.array([[x_in + dx_in, y_in + dy_in]], dtype=np.float32)
            c_orig_ref = _apply_affine_points(c_in_ref, inv)

            # original 坐标中的平移量
            delta = (c_orig_ref - c_orig)[0]  # (2,)
            dx_o, dy_o = float(delta[0]), float(delta[1])

            # 平移矩形
            rect_ref = _translate_rect_xy4(rect_xy4, dx_o, dy_o)
            refined.append(rect_ref)

        return refined

    # ---------- init meters ----------
    iou_list = []
    model.eval()
    time.sleep(2)

    num_grasps = [1, 5]
    num_correct_grasps = [0, 0]
    num_total_grasps = [0, 0]

    pbar = tqdm(val_loader)

    # r_norm：与训练/GT 生成时一致；优先用 args.offset_r，否则回退为 min(H,W)/20
    def _get_rnorm(H_in, W_in):
        if hasattr(args, "offset_r") and args.offset_r and args.offset_r > 0:
            return float(args.offset_r)
        return max(1.0, min(H_in, W_in) / 20.0)

    for data in pbar:
        # ----- fetch batch -----
        image = data["img"]                      # [B,3,H,W]
        text  = data["word_vec"]
        ins_mask = data["mask"]
        grasp_qua_mask = data["grasp_masks"]["qua"]
        grasp_sin_mask = data["grasp_masks"]["sin"]
        grasp_cos_mask = data["grasp_masks"]["cos"]
        grasp_wid_mask = data["grasp_masks"]["wid"]
        inverse_matrix = data["inverse"]         # list[tensor/np] of 2x3
        ori_sizes = data["ori_size"]             # list[(H,W)]
        grasp_targets = data["grasps"]

        has_off = ("off" in data["grasp_masks"])

        image = move_to_device(image, device)
        text = move_to_device(text, device)
        ins_mask = move_to_device(ins_mask, device).unsqueeze(1)
        grasp_qua_mask = move_to_device(grasp_qua_mask, device).unsqueeze(1)
        grasp_sin_mask = move_to_device(grasp_sin_mask, device).unsqueeze(1)
        grasp_cos_mask = move_to_device(grasp_cos_mask, device).unsqueeze(1)
        grasp_wid_mask = move_to_device(grasp_wid_mask, device).unsqueeze(1)

        # ----- forward -----
        pred, target = model(image, text, ins_mask, grasp_qua_mask, grasp_sin_mask, grasp_cos_mask, grasp_wid_mask)

        ins_mask_preds       = pred[0]
        grasp_qua_mask_preds = pred[1]
        grasp_sin_mask_preds = pred[2]
        grasp_cos_mask_preds = pred[3]
        grasp_wid_mask_preds = pred[4]

        grasp_off_mask_preds = None
        if isinstance(pred, (list, tuple)) and len(pred) >= 6 and pred[5] is not None:
            grasp_off_mask_preds = pred[5]   # [B,2,h,w]

        ins_mask_targets       = target[0]
        grasp_qua_mask_targets = target[1]
        grasp_sin_mask_targets = target[2]
        grasp_cos_mask_targets = target[3]
        grasp_wid_mask_targets = target[4]

        ins_mask_preds       = torch.sigmoid(ins_mask_preds)
        grasp_qua_mask_preds = torch.sigmoid(grasp_qua_mask_preds)
        grasp_wid_mask_preds = torch.sigmoid(grasp_wid_mask_preds)

        if ins_mask_preds.shape[-2:] != image.shape[-2:]:
            tgt_hw = image.shape[-2:]
            ins_mask_preds       = F.interpolate(ins_mask_preds,       size=tgt_hw, mode='bicubic',  align_corners=True).squeeze(1)
            grasp_qua_mask_preds = F.interpolate(grasp_qua_mask_preds, size=tgt_hw, mode='bicubic',  align_corners=True).squeeze(1)
            grasp_sin_mask_preds = F.interpolate(grasp_sin_mask_preds, size=tgt_hw, mode='bicubic',  align_corners=True).squeeze(1)
            grasp_cos_mask_preds = F.interpolate(grasp_cos_mask_preds, size=tgt_hw, mode='bicubic',  align_corners=True).squeeze(1)
            grasp_wid_mask_preds = F.interpolate(grasp_wid_mask_preds, size=tgt_hw, mode='bicubic',  align_corners=True).squeeze(1)
            if grasp_off_mask_preds is not None:
                grasp_off_mask_preds = F.interpolate(grasp_off_mask_preds, size=tgt_hw, mode='bilinear', align_corners=False)  # [B,2,H,W]

        # ----- per-sample -----
        for idx in range(ins_mask_preds.shape[0]):
            inv_mat = inverse_matrix[idx]
            if isinstance(inv_mat, torch.Tensor):
                inv_mat = inv_mat.cpu().numpy()
            ori_h, ori_w = int(ori_sizes[idx][0]), int(ori_sizes[idx][1])

            # numpy maps on INPUT size
            ins_mask_pred       = ins_mask_preds[idx].cpu().numpy()
            grasp_qua_mask_pred = grasp_qua_mask_preds[idx].squeeze().cpu().numpy()
            grasp_sin_mask_pred = grasp_sin_mask_preds[idx].squeeze().cpu().numpy()
            grasp_cos_mask_pred = grasp_cos_mask_preds[idx].squeeze().cpu().numpy()
            grasp_wid_mask_pred = grasp_wid_mask_preds[idx].squeeze().cpu().numpy()

            # GT (for viz / metrics)
            ins_mask_target       = ins_mask_targets[idx].squeeze().cpu().numpy()
            grasp_target          = grasp_targets[idx]
            grasp_qua_mask_target = grasp_qua_mask_targets[idx].squeeze().cpu().numpy()
            grasp_sin_mask_target = grasp_sin_mask_targets[idx].squeeze().cpu().numpy()
            grasp_cos_mask_target = grasp_cos_mask_targets[idx].squeeze().cpu().numpy()
            grasp_wid_mask_target = grasp_wid_mask_targets[idx].squeeze().cpu().numpy()

            # ---- inverse warp to ORIGINAL size (for detect_grasps and IoU/J) ----
            ins_mask_pred = inverse(ins_mask_pred, inv_mat, ori_w, ori_h)
            ins_mask_pred = (ins_mask_pred > 0.35)
            grasp_qua_mask_pred = inverse(grasp_qua_mask_pred, inv_mat, ori_w, ori_h)
            grasp_sin_mask_pred = inverse(grasp_sin_mask_pred, inv_mat, ori_w, ori_h)
            grasp_cos_mask_pred = inverse(grasp_cos_mask_pred, inv_mat, ori_w, ori_h)
            grasp_wid_mask_pred = inverse(grasp_wid_mask_pred, inv_mat, ori_w, ori_h)

            ins_mask_target       = inverse(ins_mask_target,       inv_mat, ori_w, ori_h)
            grasp_qua_mask_target = inverse(grasp_qua_mask_target, inv_mat, ori_w, ori_h)
            grasp_sin_mask_target = inverse(grasp_sin_mask_target, inv_mat, ori_w, ori_h)
            grasp_cos_mask_target = inverse(grasp_cos_mask_target, inv_mat, ori_w, ori_h)
            grasp_wid_mask_target = inverse(grasp_wid_mask_target, inv_mat, ori_w, ori_h)

            # 可选调试图
            # save_prediction_visualization(grasp_qua_mask_pred, grasp_qua_mask_target, save_dir="./debug_all_validate")

            # ---- instance IoU ----
            inter = np.logical_and(ins_mask_pred, ins_mask_target)
            union = np.logical_or(ins_mask_pred, ins_mask_target)
            iou = np.sum(inter) / (np.sum(union) + 1e-6)
            iou_list.append(iou)

            # ---- grasp detection + (optional) offset refine ----
            for i, num_g in enumerate(num_grasps):
                grasp_preds, _ = detect_grasps(
                    grasp_qua_mask_pred, grasp_sin_mask_pred, grasp_cos_mask_pred, grasp_wid_mask_pred, num_g
                )

                # offset refine（若存在）
                if grasp_off_mask_preds is not None:
                    H_in, W_in = image.shape[-2], image.shape[-1]
                    r_norm_eval = _get_rnorm(H_in, W_in)
                    off_map = grasp_off_mask_preds[idx:idx+1]  # [1,2,H_in,W_in]
                    grasp_preds = _refine_grasps_with_offset(
                        rects_xy4_list=grasp_preds,
                        off_map_1x2_hw=off_map,
                        inv_mat_2x3=inv_mat,
                        r_norm=r_norm_eval
                    )

                # 评估
                grasp_preds_5  = _batch_to_five(grasp_preds, assume_degrees=True)
                grasp_target_6 = _batch_to_six(grasp_target, assume_degrees=True)
                j_index = calculate_jacquard_index(grasp_preds_5, grasp_target_6)
                num_correct_grasps[i] += j_index
                num_total_grasps[i]   += 1

            # 可选保存（使用最后一次 num_g 的预测）
            # save_grasp_pred_all_gt(grasp_qua_mask_pred, grasp_preds, grasp_target, save_path='./debug_infer/gt_grasp.png')

    # ---------- reduce & log ----------
    J_index = [0, 0]
    for i in range(len(num_grasps)):
        J_index[i] = num_correct_grasps[i] / max(1, num_total_grasps[i])

    iou_list = np.stack(iou_list)
    iou_list = torch.from_numpy(iou_list).to(image.device)
    if dist.is_available() and dist.is_initialized():
        iou_list = concat_all_gather(iou_list)
    prec_list = []
    for thres in torch.arange(0.5, 1.0, 0.1):
        tmp = (iou_list > thres).float().mean()
        prec_list.append(tmp)
    iou = iou_list.mean()
    prec = {}
    temp = '  '
    for i, thres in enumerate(range(5, 10)):
        key = 'Pr@{}'.format(thres * 10)
        value = prec_list[i].item()
        prec[key] = value
        temp += "{}: {:.2f}  ".format(key, 100. * value)

    head = 'Evaluation: Epoch=[{}/{}]  IoU={:.2f}  J_index@1: {:.2f}  J_index@5: {:.2f}'.format(
        epoch, args.epochs, 100. * iou.item(), 100. * J_index[0], 100. * J_index[1]
    )
    logger.info(head + temp)
    return iou.item(), prec, J_index


@torch.no_grad()
def validate_with_grasp(val_loader, model, epoch, args):
    """Compatibility entry point backed by the registered validation loop."""
    from toolrgs.engine import GraspValLoop

    return GraspValLoop(
        dataloader=val_loader,
        model=model,
        cfg=args,
        hooks=getattr(args, "val_hooks", None),
    ).run_epoch(epoch)


@torch.no_grad()
def inference_with_grasp(test_loader, model, args):
    device = current_device(int(getattr(args, "npu", getattr(args, "gpu", 0))))
    def inverse(img, mat, w, h):
        inv_img = cv2.warpAffine(img, mat, (w, h),
                                    flags=cv2.INTER_CUBIC,
                                    borderValue=0.)
        return inv_img

    def _apply_affine_points(points_xy: np.ndarray, mat_2x3: np.ndarray) -> np.ndarray:
        if points_xy is None or len(points_xy) == 0:
            return points_xy
        homo = np.concatenate([points_xy.astype(np.float32),
                               np.ones((points_xy.shape[0], 1), dtype=np.float32)], axis=1)
        out = (mat_2x3 @ homo.T).T
        return out.astype(np.float32)

    def _rect_center_xy(rect_xy4: np.ndarray) -> np.ndarray:
        return rect_xy4.mean(axis=0)  # (2,)

    def _translate_rect_xy4(rect_xy4: np.ndarray, dx: float, dy: float) -> np.ndarray:
        out = rect_xy4.copy()
        out[:, 0] += dx
        out[:, 1] += dy
        return out

    def _sample_off_nn(off_1x2_hw, x, y):
        """
        支持 off 为 (2,H,W) 或 (1,2,H,W)，返回最近邻采样的 (dx, dy)（网络输出的归一化单位）
        """
        # 转 numpy
        if isinstance(off_1x2_hw, torch.Tensor):
            off = off_1x2_hw.detach().to('cpu', dtype=torch.float32).numpy()
        else:
            off = np.asarray(off_1x2_hw, dtype=np.float32)

        # 统一成 (2,H,W)
        if off.ndim == 4:
            # (1,2,H,W) -> (2,H,W)
            assert off.shape[0] == 1 and off.shape[1] == 2, f"bad off shape {off.shape}"
            off2 = off[0]
        elif off.ndim == 3:
            # (2,H,W)
            assert off.shape[0] == 2, f"bad off shape {off.shape}"
            off2 = off
        else:
            raise ValueError(f"_sample_off_nn expects (2,H,W) or (1,2,H,W), got {off.shape}")

        C, H, W = off2.shape
        assert C == 2, f"channel must be 2, got {C}"

        # 最近邻取整 + 边界裁剪（注意 x 对应宽，y 对应高）
        xi = int(round(float(x)))
        yi = int(round(float(y)))
        xi = max(0, min(W - 1, xi))
        yi = max(0, min(H - 1, yi))

        dx = float(off2[0, yi, xi])
        dy = float(off2[1, yi, xi])
        return dx, dy

    def _to_xy4_one(rect, assume_theta_degree=True, wh_is_full=True):
        """
        将抓取框统一为 (4,2) 顶点格式。
        rect: 5参 [cx,cy,w,h,theta] 或 8参 [x1,y1,...,x4,y4]
        assume_theta_degree: theta 是否为角度（是就会自动转弧度）
        wh_is_full: w/h 是否为“整宽整高”（若传的是半宽/半高，设为 False）
        """
        arr = np.asarray(rect, dtype=np.float32).reshape(-1)
        if arr.size == 8:
            return arr.reshape(4, 2)

        if arr.size != 5:
            raise ValueError(f"rect length {arr.size} not in (5, 8). rect={arr.tolist()}")

        cx, cy, w, h, th = arr.tolist()
        if assume_theta_degree:
            th = np.deg2rad(th)
        c, s = np.cos(th), np.sin(th)

        # 处理 w/h 的定义
        if wh_is_full:
            dx, dy = w / 2.0, h / 2.0
        else:
            dx, dy = w, h

        # 顺时针：左上 -> 右上 -> 右下 -> 左下
        corners = np.array([[-dx, -dy], [dx, -dy], [dx,  dy], [-dx,  dy]], dtype=np.float32)
        R = np.array([[c, -s], [s,  c]], dtype=np.float32)
        rot = corners @ R.T
        rot[:, 0] += cx; rot[:, 1] += cy
        return rot  # (4,2)


    def _xy4_to_five(rect_xy4, out_degrees=True):
        """
        (4,2) -> [cx, cy, w, h, theta] using cv2.minAreaRect
        返回 theta 默认是度数，范围约 (-90, 90]
        """
        pts = np.asarray(rect_xy4, dtype=np.float32).reshape(4, 2)
        (cx, cy), (w, h), ang = cv2.minAreaRect(pts)

        # 可选归一化：确保 w 表示较长边，角度落在 (-90, 90]
        if w < h:
            w, h = h, w
            ang = ang + 90.0
        if ang <= -90.0:
            ang += 180.0
        if ang > 90.0:
            ang -= 180.0

        theta = ang if out_degrees else np.deg2rad(ang)
        return np.array([cx, cy, w, h, theta], dtype=np.float32)

    def _to_five_any(rect, assume_degrees=True):
        """
        将任意常见格式统一为 5 参 [cx, cy, w, h, theta]（theta 默认度数）
        支持：
        - 5 参: [cx, cy, w, h, theta] -> 直通
        - 6 参: [cx, cy, w, h, theta, extra] -> 取前 5 个，忽略附加字段
        - 8 参: [x1,y1,x2,y2,x3,y3,x4,y4] -> 先转 (4,2) 再转 5 参
        - (4,2): 顶点 -> 5 参
        - 4 参: (xmin,ymin,xmax,ymax) -> 5 参（theta=0）
        """
        arr = np.asarray(rect, dtype=np.float32).reshape(-1)
        n = arr.size

        if n == 5:
            # [cx, cy, w, h, theta]
            return arr

        if n == 6:

            cand = arr[:5].copy()
            # 简单健壮性校验：若 cand[4] 不像角度（>180 且不是弧度范围），尝试另一种排列 (cx,cy,theta,w,h)
            th = float(cand[4])
            if not (abs(th) <= 180.0 or abs(th) <= 6.283185):  # 不是度也不是弧度
                # 常见变体： [cx,cy,theta,w,h,extra] -> 重新整理到 [cx,cy,w,h,theta]
                cx, cy, theta, w, h = arr[:5].tolist()
                cand = np.array([cx, cy, w, h, theta], dtype=np.float32)
            return cand

        if n == 8:
            # [x1,y1,...,x4,y4]
            return _xy4_to_five(arr.reshape(4, 2), out_degrees=assume_degrees)

        if n == 4:
            # 轴对齐框
            xmin, ymin, xmax, ymax = arr.tolist()
            cx = 0.5 * (xmin + xmax)
            cy = 0.5 * (ymin + ymax)
            w  = max(1e-6, xmax - xmin)
            h  = max(1e-6, ymax - ymin)
            theta = 0.0
            return np.array([cx, cy, w, h, theta], dtype=np.float32)

        if n == 2*4:
            return _xy4_to_five(arr.reshape(4, 2), out_degrees=assume_degrees)

        # 打印一条更友好的调试信息
        raise ValueError(
            f"Unexpected rect length {n}. Supported: 4/(4,2)/5/6/8. "
            f"Got rect={arr.tolist()}"
        )

    def _batch_to_five(rects, assume_degrees=True):
        return [ _to_five_any(r, assume_degrees) for r in rects ]

    def _to_six_any(rect, assume_degrees=True):
        """
        """
        arr = np.asarray(rect, dtype=np.float32).reshape(-1)
        if arr.size == 6:
            return arr
        five = _to_five_any(arr, assume_degrees=assume_degrees)  # 你之前已实现
        extra = np.array([0.0], dtype=np.float32)
        return np.concatenate([five, extra], axis=0)

    def _batch_to_six(rects, assume_degrees=True):
        return [_to_six_any(r, assume_degrees=assume_degrees) for r in rects]

    def _refine_grasps_with_offset(rects_xy4_list, off_map_1x2_hw, inv_mat_2x3, r_norm):
        """
        rects_xy4_list: list，元素可以是 5参或 8参；本函数会统一成 (4,2)
        off_map_1x2_hw: (2,H,W) 或 (1,2,H,W)
        inv_mat_2x3: 2x3 仿射矩阵（input->original）
        r_norm: 将网络归一化位移还原为像素的尺度
        """
        inv = np.asarray(inv_mat_2x3, dtype=np.float32)
        fwd = cv2.invertAffineTransform(inv)
        r_norm = float(r_norm)

        # 统一 off_map -> (2,H,W)
        if isinstance(off_map_1x2_hw, torch.Tensor):
            off = off_map_1x2_hw.detach().to('cpu', dtype=torch.float32).numpy()
        else:
            off = np.asarray(off_map_1x2_hw, dtype=np.float32)
        if off.ndim == 4:  # (1,2,H,W)
            off = off[0]
        assert off.ndim == 3 and off.shape[0] == 2, f"off_map shape should be (2,H,W) or (1,2,H,W), got {off.shape}"

        # —— 新增：把 5参/8参统一成 (4,2) —— #
        rects_xy4_norm = []
        for idx, r in enumerate(rects_xy4_list):
            try:
                rects_xy4_norm.append(_to_xy4_one(r, assume_theta_degree=True, wh_is_full=True))
            except Exception as e:
                print(f"[rect-norm] bad rect at idx={idx}: {r}")
                raise

        refined = []
        for rect_xy4 in rects_xy4_norm:
            rect_xy4 = np.asarray(rect_xy4, dtype=np.float32).reshape(4, 2)

            # --- 中心点 (1,2) in original ---
            c_orig = np.asarray(_rect_center_xy(rect_xy4), dtype=np.float32).reshape(1, 2)

            # original -> input
            c_in = _apply_affine_points(c_orig, fwd)   # (1,2)
            x_in, y_in = float(c_in[0, 0]), float(c_in[0, 1])

            # 采样 offset（网络输出为归一化单位）
            dx_n, dy_n = _sample_off_nn(off, x_in, y_in)   # 标量
            dx_in, dy_in = dx_n * r_norm, dy_n * r_norm    # 还原到像素（input）

            # 偏移后的点（input） -> original
            c_in_ref = np.array([[x_in + dx_in, y_in + dy_in]], dtype=np.float32)
            c_orig_ref = _apply_affine_points(c_in_ref, inv)

            # original 坐标中的平移量
            delta = (c_orig_ref - c_orig)[0]  # (2,)
            dx_o, dy_o = float(delta[0]), float(delta[1])

            # 平移矩形
            rect_ref = _translate_rect_xy4(rect_xy4, dx_o, dy_o)
            refined.append(rect_ref)

        return refined

    def _get_rnorm(H_in, W_in):
        if hasattr(args, "offset_r") and args.offset_r and args.offset_r > 0:
            return float(args.offset_r)
        return max(1.0, min(H_in, W_in) / 20.0)

    iou_list = []
    num_correct_grasps = 0
    num_total_grasps = 0
    model.eval()
    time.sleep(2)

    num_grasps = [1,5]
    num_correct_grasps = [0, 0]
    num_total_grasps = [0, 0]

    tbar = tqdm(test_loader, desc='Inference:', ncols=100)
    for cnt, data in enumerate(tbar):

        # data
        image = data["img"]
        text = data["word_vec"]
        ins_mask = data["mask"]
        grasp_qua_mask = data["grasp_masks"]["qua"]
        grasp_sin_mask = data["grasp_masks"]["sin"]
        grasp_cos_mask = data["grasp_masks"]["cos"]
        grasp_wid_mask = data["grasp_masks"]["wid"]
        inverse_matrix = data["inverse"]
        ori_sizes = data["ori_size"]
        grasp_targets = data["grasps"]
        sentences = data["sentence"]
        img_paths = data["img_path"]

        image = move_to_device(image, device)
        text = move_to_device(text, device)
        ins_mask = move_to_device(ins_mask, device).unsqueeze(1)
        grasp_qua_mask = move_to_device(grasp_qua_mask, device).unsqueeze(1)
        grasp_sin_mask = move_to_device(grasp_sin_mask, device).unsqueeze(1)
        grasp_cos_mask = move_to_device(grasp_cos_mask, device).unsqueeze(1)
        grasp_wid_mask = move_to_device(grasp_wid_mask, device).unsqueeze(1)

        # inference & get predictions from model
        pred, target = model(image, text, ins_mask, grasp_qua_mask, grasp_sin_mask, grasp_cos_mask, grasp_wid_mask)

        # predictions
        ins_mask_preds = pred[0]
        grasp_qua_mask_preds = pred[1]
        grasp_sin_mask_preds = pred[2]
        grasp_cos_mask_preds = pred[3]
        grasp_wid_mask_preds = pred[4]
        if isinstance(pred, (list, tuple)) and len(pred) >= 6 and pred[5] is not None:
            grasp_off_mask_preds = pred[5]

        # targets
        ins_mask_targets = target[0]
        grasp_qua_mask_targets = target[1]
        grasp_sin_mask_targets = target[2]
        grasp_cos_mask_targets = target[3]
        grasp_wid_mask_targets = target[4]

        # Interpolate the predicted ins mask to the same size of input image
        ins_mask_preds = torch.sigmoid(ins_mask_preds)
        grasp_qua_mask_preds = torch.sigmoid(grasp_qua_mask_preds)
        grasp_wid_mask_preds = torch.sigmoid(grasp_wid_mask_preds)

        if ins_mask_preds.shape[-2:] != image.shape[-2:]:
            tgt_hw = image.shape[-2:]
            ins_mask_preds = F.interpolate(ins_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)

            grasp_qua_mask_preds = F.interpolate(grasp_qua_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)

            grasp_sin_mask_preds = F.interpolate(grasp_sin_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)

            grasp_cos_mask_preds = F.interpolate(grasp_cos_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)

            grasp_wid_mask_preds = F.interpolate(grasp_wid_mask_preds,
                                  size=image.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)
            if grasp_off_mask_preds is not None:
                grasp_off_mask_preds = F.interpolate(grasp_off_mask_preds,
                                                    size=tgt_hw,
                                                    mode='bilinear',
                                                    align_corners=False)  # [B,2,H,W]


        # iterate over the whole batch
        for idx in range(ins_mask_preds.shape[0]):
            inv_mat = inverse_matrix[idx]
            ori_size = ori_sizes[idx]
            h, w = ori_size
            sent = sentences[idx]
            img_path = img_paths[idx]

            ins_mask_pred = ins_mask_preds[idx].cpu().numpy()
            grasp_qua_mask_pred = grasp_qua_mask_preds[idx].squeeze().cpu().numpy()
            grasp_sin_mask_pred = grasp_sin_mask_preds[idx].squeeze().cpu().numpy()
            grasp_cos_mask_pred = grasp_cos_mask_preds[idx].squeeze().cpu().numpy()
            grasp_wid_mask_pred = grasp_wid_mask_preds[idx].squeeze().cpu().numpy()

            ins_mask_target = ins_mask_targets[idx].squeeze().cpu().numpy()
            grasp_target = grasp_targets[idx]
            grasp_qua_mask_target = grasp_qua_mask_targets[idx].squeeze().cpu().numpy()
            grasp_sin_mask_target = grasp_sin_mask_targets[idx].squeeze().cpu().numpy()
            grasp_cos_mask_target = grasp_cos_mask_targets[idx].squeeze().cpu().numpy()
            grasp_wid_mask_target = grasp_wid_mask_targets[idx].squeeze().cpu().numpy()

            # Inverse to original size
            ins_mask_pred = inverse(ins_mask_pred, inv_mat, w, h)
            ins_mask_pred = (ins_mask_pred > 0.35)
            grasp_qua_mask_pred = inverse(grasp_qua_mask_pred, inv_mat, w, h)
            grasp_sin_mask_pred = inverse(grasp_sin_mask_pred, inv_mat, w, h)
            grasp_cos_mask_pred = inverse(grasp_cos_mask_pred, inv_mat, w, h)
            grasp_wid_mask_pred = inverse(grasp_wid_mask_pred, inv_mat, w, h)

            ins_mask_target = inverse(ins_mask_target, inv_mat, w, h)
            grasp_qua_mask_target = inverse(grasp_qua_mask_target, inv_mat, w, h)
            grasp_sin_mask_target = inverse(grasp_sin_mask_target, inv_mat, w, h)
            grasp_cos_mask_target = inverse(grasp_cos_mask_target, inv_mat, w, h)
            grasp_wid_mask_target = inverse(grasp_wid_mask_target, inv_mat, w, h)

            # Calculate IoU between predicted instance mask and gt
            inter = np.logical_and(ins_mask_pred, ins_mask_target)
            union = np.logical_or(ins_mask_pred, ins_mask_target)

            iou = np.sum(inter) / (np.sum(union) + 1e-6)
            iou_list.append(iou)

            # Calculate grasp configurations
            for i in range(len(num_grasps)):
                num_g = num_grasps[i]
                grasp_preds, grasp_ang_mask_pred = detect_grasps(grasp_qua_mask_pred, grasp_sin_mask_pred, grasp_cos_mask_pred, grasp_wid_mask_pred, num_g)

                if grasp_off_mask_preds is not None:
                    H_in, W_in = image.shape[-2], image.shape[-1]
                    r_norm_eval = _get_rnorm(H_in, W_in)
                    off_map = grasp_off_mask_preds[idx:idx+1]  # [1,2,H_in,W_in]
                    grasp_preds = _refine_grasps_with_offset(
                        rects_xy4_list=grasp_preds,
                        off_map_1x2_hw=off_map,
                        inv_mat_2x3=inv_mat,
                        r_norm=r_norm_eval
                    )

                grasp_preds_5  = _batch_to_five(grasp_preds, assume_degrees=True)
                grasp_target_6 = _batch_to_six(grasp_target, assume_degrees=True)
                j_index = calculate_jacquard_index(grasp_preds_5, grasp_target_6)
                num_correct_grasps[i] += j_index
                num_total_grasps[i]   += 1

                # Visualization
                if args.visualize:
                    img_bgr = cv2.imread(img_path)
                    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    visualization(img, ins_mask_pred, (grasp_qua_mask_pred, grasp_ang_mask_pred, grasp_wid_mask_pred), grasp_preds, sent, save_path=os.path.join("./results", args.exp_name, f"results_{cnt}_{num_g}_grasps.png"))
                # save_grasp_pred_all_gt(grasp_qua_mask_pred, grasp_preds, grasp_target, save_path='./debug_grasp/validate_all_gt_grasp.png')
    J_index = [0, 0]
    for i in range(len(num_grasps)):
        J_index[i] = num_correct_grasps[i]/num_total_grasps[i]

    iou_list = np.stack(iou_list)
    iou_list = torch.from_numpy(iou_list).to(image.device)
    # print(iou_list)
    # iou_list = concat_all_gather(iou_list)
    prec_list = []
    for thres in torch.arange(0.5, 1.0, 0.1):
        tmp = (iou_list > thres).float().mean()
        prec_list.append(tmp)
    iou = iou_list.mean()
    prec = {}
    for i, thres in enumerate(range(5, 10)):
        key = 'Pr@{}'.format(thres*10)
        value = prec_list[i].item()
        prec[key] = value
    logger.info('IoU={:.2f}'.format(100.*iou.item()))
    for k, v in prec.items():
        logger.info('{}: {:.2f}.'.format(k, 100.*v))
    logger.info("J@1: {:.2f}, J@5: {:.2f}".format(100. * J_index[0], 100. * J_index[1]))

    return iou.item(), prec, J_index
