"""
General Single Image Inference Script for Pedestrian Attribute Recognition (PAR).
Generates high-quality side-by-side visual reports with colored attribute bars.

Usage:
    python inference/predict_image.py --image "0007_002.jpg"
    python inference/predict_image.py --image "i-LID/0007_002.jpg"
    python inference/predict_image.py --image "CAVIAR4REID/0007_002.jpg"
"""
import os
import sys
import argparse
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms as T

HAS_MATPLOTLIB = False
MATPLOTLIB_ERR = ""

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except Exception as e:
    HAS_MATPLOTLIB = False
    MATPLOTLIB_ERR = str(e)

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.hydraplus.par_model import UnifiedPARModel

UPAR_ATTRIBUTES = [
    "Age-Young", "Age-Adult", "Age-Old", "Gender-Female",
    "Hair-Length-Short", "Hair-Length-Long", "Hair-Length-Bald",
    "UpperBody-Length-Short", "UpperBody-Color-Black", "UpperBody-Color-Blue",
    "UpperBody-Color-Brown", "UpperBody-Color-Green", "UpperBody-Color-Grey",
    "UpperBody-Color-Orange", "UpperBody-Color-Pink", "UpperBody-Color-Purple",
    "UpperBody-Color-Red", "UpperBody-Color-White", "UpperBody-Color-Yellow",
    "UpperBody-Color-Other", "LowerBody-Length-Short", "LowerBody-Color-Black",
    "LowerBody-Color-Blue", "LowerBody-Color-Brown", "LowerBody-Color-Green",
    "LowerBody-Color-Grey", "LowerBody-Color-Orange", "LowerBody-Color-Pink",
    "LowerBody-Color-Purple", "LowerBody-Color-Red", "LowerBody-Color-White",
    "LowerBody-Color-Yellow", "LowerBody-Color-Other", "LowerBody-Type-Trousers&Shorts",
    "LowerBody-Type-Skirt&Dress", "Accessory-Backpack", "Accessory-Bag",
    "Accessory-Glasses-Normal", "Accessory-Glasses-Sun", "Accessory-Hat"
]


def resolve_image_path(query_path: str) -> str:
    """Smart path resolution with subfolder specificity."""
    query_clean = query_path.strip('"').strip("'").replace('/', os.sep).replace('\\', os.sep)
    
    if os.path.exists(query_clean):
        return os.path.abspath(query_clean)
        
    search_roots = [
        r"D:\AI DATASET\3 Datasets",
        r"D:\AI DATASET",
        r"D:\AI_Project"
    ]
    
    matches = []
    filename = os.path.basename(query_clean)
    query_parts = [p.lower() for p in query_clean.split(os.sep) if p]
    
    for root in search_roots:
        if os.path.exists(root):
            for r, d, files in os.walk(root):
                for f in files:
                    if f.lower() == filename.lower():
                        full_p = os.path.join(r, f)
                        full_p_lower = full_p.lower()
                        if len(query_parts) > 1:
                            sub_query = os.sep.join(query_parts)
                            if sub_query in full_p_lower or all(part in full_p_lower for part in query_parts):
                                return full_p
                        if full_p not in matches:
                            matches.append(full_p)
                        
    if not matches:
        return None
        
    if len(matches) > 1:
        print("\n" + "!" * 70)
        print(f"CANH BAO: PETA co {len(matches)} anh trung ten '{filename}' tai cac thu muc khac nhau:")
        for idx, m in enumerate(matches, 1):
            sub_p = os.path.relpath(m, r"D:\AI DATASET")
            print(f"  [{idx}] {sub_p}")
        first_rel = os.path.relpath(matches[0], r"D:\AI DATASET")
        print(f"-> Dang su dung anh dau tien: {first_rel}")
        last_subfolder = os.path.basename(os.path.dirname(os.path.dirname(matches[-1])))
        print(">>> MEO: Ban co the truyen thu muc con de chon chinh xac. Vi du:")
        print(f'    python inference/predict_image.py --image "{last_subfolder}\\{filename}"')
        print("!" * 70 + "\n")
        
    return matches[0]


def get_default_font(size=14):
    """Load clean TrueType font or fallback."""
    font_candidates = ["arial.ttf", "calibri.ttf", "dejavusans.ttf", "segoeui.ttf"]
    for font_name in font_candidates:
        try:
            return ImageFont.truetype(font_name, size)
        except Exception:
            continue
    return ImageFont.load_default()


def visualize_with_pil(raw_img: Image.Image, img_name: str, probs: np.ndarray, attr_names: list, threshold: float = 0.5, save_path: str = "result_prediction.png"):
    """
    High-quality PIL Bar Chart Renderer.
    """
    sorted_indices = np.argsort(probs)[::-1]
    top_indices = [i for i in sorted_indices if probs[i] >= threshold]
    if len(top_indices) < 8:
        top_indices = list(sorted_indices[:10])
        
    plot_items = [(attr_names[i], probs[i]) for i in top_indices]

    img_w, img_h = raw_img.size
    target_img_h = max(480, len(plot_items) * 45 + 100)
    aspect_ratio = img_w / img_h
    target_img_w = int(target_img_h * aspect_ratio)

    resized_img = raw_img.resize((target_img_w, target_img_h), Image.Resampling.LANCZOS)

    chart_w = 680
    canvas_w = target_img_w + chart_w + 50
    canvas_h = target_img_h + 40

    canvas = Image.new('RGB', (canvas_w, canvas_h), (255, 255, 255))
    canvas.paste(resized_img, (20, 20))

    draw = ImageDraw.Draw(canvas)
    
    font_title = get_default_font(18)
    font_sub = get_default_font(13)
    font_label = get_default_font(13)
    font_val = get_default_font(12)

    x_chart = target_img_w + 40
    y_chart = 30
    draw.text((x_chart, y_chart), f"Pedestrian: {img_name}", fill=(30, 30, 30), font=font_title)
    y_chart += 30
    draw.text((x_chart, y_chart), f"Predicted Attributes (Threshold = {int(threshold*100)}%)", fill=(100, 100, 100), font=font_sub)
    y_chart += 40

    bar_max_w = 320
    bar_start_x = x_chart + 240
    thresh_x = bar_start_x + int(bar_max_w * threshold)

    for name, prob in plot_items:
        is_pos = prob >= threshold
        color = (46, 204, 113) if is_pos else (231, 76, 60)

        draw.text((x_chart, y_chart + 2), name, fill=(40, 40, 40), font=font_label)
        draw.rectangle([bar_start_x, y_chart, bar_start_x + bar_max_w, y_chart + 20], fill=(238, 238, 238))

        fill_w = int(bar_max_w * prob)
        if fill_w > 0:
            draw.rectangle([bar_start_x, y_chart, bar_start_x + fill_w, y_chart + 20], fill=color)

        percent_str = f"{prob * 100:.1f}%"
        draw.text((bar_start_x + bar_max_w + 12, y_chart + 2), percent_str, fill=(30, 30, 30), font=font_val)
        y_chart += 38

    for y_line in range(70, y_chart, 6):
        draw.line([(thresh_x, y_line), (thresh_x, y_line + 3)], fill=(211, 84, 0), width=2)
    draw.text((thresh_x - 30, y_chart + 10), f"Threshold {int(threshold*100)}%", fill=(211, 84, 0), font=font_val)

    canvas.save(save_path)
    print(f"\n[Visual Report Saved (PIL-Pro)]: {os.path.abspath(save_path)}")


def visualize_results(raw_img: Image.Image, img_name: str, probs: np.ndarray, attr_names: list, threshold: float = 0.5, save_path: str = "result_prediction.png"):
    """Generate side-by-side plot using Matplotlib if available, otherwise PIL-Pro."""
    if not HAS_MATPLOTLIB:
        if MATPLOTLIB_ERR:
            print(f"(Thong bao: Matplotlib import note: {MATPLOTLIB_ERR})")
        visualize_with_pil(raw_img, img_name, probs, attr_names, threshold=threshold, save_path=save_path)
        return

    sorted_indices = np.argsort(probs)[::-1]
    top_indices = [i for i in sorted_indices if probs[i] >= threshold]
    if len(top_indices) < 8:
        top_indices = list(sorted_indices[:12])
        
    top_indices = top_indices[::-1]
    plot_names = [attr_names[i] for i in top_indices]
    plot_probs = [probs[i] * 100 for i in top_indices]
    colors = ['#2ecc71' if probs[i] >= threshold else '#e74c3c' for i in top_indices]
    
    fig, (ax_img, ax_bar) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={'width_ratios': [1, 1.8]})
    
    ax_img.imshow(raw_img)
    ax_img.set_title(f" Pedestrian Image: {img_name}", fontsize=12, fontweight='bold', pad=10)
    ax_img.axis('off')
    
    bars = ax_bar.barh(plot_names, plot_probs, color=colors, height=0.6)
    ax_bar.axvline(x=threshold * 100, color='#d35400', linestyle='--', linewidth=1.5, label=f'Threshold ({int(threshold*100)}%)')
    
    ax_bar.set_xlim(0, 105)
    ax_bar.set_xlabel('Probability (%)', fontsize=11, fontweight='bold')
    ax_bar.set_title(f'Predicted Attributes (Threshold = {threshold})', fontsize=12, fontweight='bold', pad=10)
    ax_bar.grid(axis='x', linestyle=':', alpha=0.6)
    ax_bar.legend(loc='lower right')
    
    for bar, prob in zip(bars, plot_probs):
        ax_bar.text(bar.get_width() + 1.5, bar.get_y() + bar.get_height()/2, f'{prob:.1f}%', 
                    va='center', ha='left', fontsize=9, fontweight='bold', color='#2c3e50')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    print(f"\n[Visual Report Saved (Matplotlib)]: {os.path.abspath(save_path)}")
    plt.close(fig)


def main():
    img_arg = None
    for idx, arg in enumerate(sys.argv):
        if arg.startswith('--image='):
            img_arg = arg.split('=', 1)[1]
        elif arg in ['--image', '-i'] and idx + 1 < len(sys.argv):
            img_arg = sys.argv[idx + 1]

    parser = argparse.ArgumentParser(description="Predict Pedestrian Attributes for Any Image")
    parser.add_argument('--image', '-i', type=str, default=img_arg, help="Path or filename of any pedestrian image")
    parser.add_argument('--checkpoint', '-c', type=str, default=r"D:\AI DATASET\UPAR_UNIFIED\checkpoints\hydraplus_upar_best.pth")
    parser.add_argument('--threshold', '-t', type=float, default=0.5, help="Classification probability threshold (default 0.5)")
    args = parser.parse_args()

    if not args.image:
        print("LOI: Vui long cung cap duong dan hoac ten file anh voi arg --image!")
        sys.exit(1)

    img_path = resolve_image_path(args.image)
    if img_path is None or not os.path.exists(img_path):
        print(f"LOI: Khong tim thay file anh '{args.image}' tren dia!")
        sys.exit(1)

    print("=" * 70)
    print("HYDRAPLUS-NET PAR SINGLE IMAGE INFERENCE")
    print("=" * 70)
    print(f"File anh: {img_path}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Threshold: {args.threshold}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if not os.path.exists(args.checkpoint):
        print(f"LOI: Khong tim thay checkpoint tai '{args.checkpoint}'!")
        sys.exit(1)

    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    attr_names = checkpoint.get("attr_names", UPAR_ATTRIBUTES)
    num_attributes = checkpoint.get("num_attributes", len(attr_names))
    backbone_name = checkpoint.get("backbone", "resnet50")

    model = UnifiedPARModel(num_attributes=num_attributes, backbone_name=backbone_name, pretrained=True).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    transform = T.Compose([
        T.Resize((256, 128)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        raw_img = Image.open(img_path).convert('RGB')
    except Exception as e:
        print(f"LOI: Khong the doc file anh. Chi tiet: {e}")
        sys.exit(1)

    input_tensor = transform(raw_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor)
        probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()

    preds = (probs >= args.threshold).astype(int)

    print("\n" + "=" * 70)
    print(f"BANG KET QUA 40 UPAR ATTRIBUTES (Threshold = {args.threshold})")
    print("=" * 70)
    print(f"| {'ID':<2} | {'Attribute Name':<32} | {'Probability':<12} | {'Prediction':<10} |")
    print("|----|----------------------------------|--------------|------------|")

    detected_attrs = []
    for idx, name in enumerate(attr_names):
        prob = probs[idx]
        is_pos = preds[idx] == 1
        status_str = "[CO] YES" if is_pos else "     NO"
        if is_pos:
            detected_attrs.append((name, prob))
        print(f"| {idx:02d} | {name:<32} | {prob * 100:>9.2f}%   | {status_str:<10} |")

    print("\n" + "=" * 70)
    print(f"CAC THUOC TINH NHAN DIEN CHINH (Co {len(detected_attrs)} attributes [CO]):")
    print("=" * 70)
    for name, prob in detected_attrs:
        print(f"  + {name:<32} : {prob * 100:.2f}%")

    sub_name = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
    file_base = os.path.splitext(os.path.basename(img_path))[0].lstrip('-')
    save_filename = f"result_{sub_name}_{file_base}.png"
    save_filepath = os.path.join(r"D:\AI DATASET", save_filename)
    visualize_results(raw_img, f"{sub_name}/{os.path.basename(img_path)}", probs, attr_names, threshold=args.threshold, save_path=save_filepath)


if __name__ == '__main__':
    main()
