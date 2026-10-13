"""Render the implemented NAFNet25D architecture as a vector poster and PNG."""
from pathlib import Path
import hashlib
import importlib.util
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle


ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT/'docs/figures'
W, H = 2400, 3600
COLORS = dict(ink='#172b42', muted='#54687b', line='#50657a', blue='#2864a0',
              blue_bg='#eef5fc', teal='#087f85', teal_bg='#e9f7f5',
              gold='#ac7311', gold_bg='#fff6df', coral='#b65437', coral_bg='#fff0e9',
              violet='#6953a0', violet_bg='#f3effa', border='#d6e1ec')


def source_model():
    path = ROOT/'denoise/nafnet25d.py'
    spec = importlib.util.spec_from_file_location('architecture_model', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = module.NAFNet25D()
    counts = {name: sum(p.numel() for p in child.parameters())
              for name, child in model.named_children()}
    import torch
    with torch.device('meta'):
        shape_model = module.NAFNet25D()
    shapes = {}
    handles = []
    for name, child in shape_model.named_children():
        def capture(layer, inputs, output, name=name):
            shapes[name] = list(output.shape)
        handles.append(child.register_forward_hook(capture))
    with torch.inference_mode():
        default_output = shape_model(torch.empty(4, 5, 256, 256, device='meta'))
    expected_shapes = dict(stem=[20, 16, 256, 256], neighbor_gate=[4, 16, 256, 256],
                           fuse=[4, 32, 256, 256], detail=[4, 32, 256, 256],
                           enc0=[4, 32, 256, 256], down0=[4, 64, 128, 128],
                           enc1=[4, 64, 128, 128], down1=[4, 128, 64, 64],
                           middle=[4, 128, 64, 64], up1=[4, 64, 128, 128],
                           dec1=[4, 64, 128, 128], up0=[4, 32, 256, 256],
                           dec0=[4, 32, 256, 256], head=[4, 1, 256, 256])
    assert shapes == expected_shapes
    assert tuple(default_output.shape) == (4, 1, 256, 256)
    for handle in handles:
        handle.remove()
    with torch.inference_mode():
        output = shape_model(torch.empty(2, 5, 255, 257, device='meta'))
    assert tuple(output.shape) == (2, 1, 255, 257)
    assert sum(counts.values()) == 989745
    assert len(model.detail) == 4 and len(model.middle) == 6
    return dict(source='denoise/nafnet25d.py', source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                parameters=sum(counts.values()), module_parameters=counts,
                default_options=dict(n_slices=5, width=32, stem_width=16, detail_blocks=4,
                                     encoder_blocks=[2, 2], middle_blocks=6, decoder_blocks=[2, 2]),
                default_module_output_shapes=shapes,
                shape_check=dict(input=[2, 5, 255, 257], output=list(output.shape), device='meta'),
                verification='Parameter inspection and shape propagation only; no model training.')


def render():
    metadata = source_model()
    DESTINATION.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'path',
                         'svg.hashsalt': 'krg-nafnet25d-architecture',
                         'savefig.facecolor': 'white'})
    fig = plt.figure(figsize=(24, 36), dpi=100)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, W), ylim=(H, 0))
    ax.axis('off')
    text_boxes = []

    def text(x, y, value, size=23, color='ink', weight='normal', ha='left', va='top'):
        return ax.text(x, y, value, fontsize=size*.72, color=COLORS.get(color, color),
                       weight=weight, ha=ha, va=va, linespacing=1.43)

    def box(x, y, w, h, title, lines=(), theme='blue', size=23, title_size=27):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0,rounding_size=13',
                                   facecolor=COLORS[theme+'_bg'], edgecolor=COLORS[theme], linewidth=1.35))
        artists = [text(x+21, y+18, title, size=title_size, color=theme, weight='bold')]
        if lines:
            artists.append(text(x+21, y+57, '\n'.join(lines), size=size))
        text_boxes.append((artists[0], (x+18, y+12, x+w-18, y+52)))
        if lines:
            text_boxes.append((artists[1], (x+18, y+54, x+w-18, y+h-12)))

    def panel(y, h):
        ax.add_patch(FancyBboxPatch((55, y), W-110, h, boxstyle='round,pad=0,rounding_size=18',
                                   facecolor='#fbfcfe', edgecolor=COLORS['border'], linewidth=1.2))

    def line(points, color='line', dashed=False, width=2, arrow=True):
        color = COLORS.get(color, color)
        xs, ys = zip(*points)
        ax.plot(xs, ys, color=color, linewidth=width, linestyle=(0, (5, 4)) if dashed else '-',
                solid_capstyle='round', zorder=1)
        if arrow:
            a, b = points[-2:]
            ax.add_patch(FancyArrowPatch(a, b, arrowstyle='-|>', mutation_scale=15,
                                         linewidth=width, color=color, zorder=2))

    def operator(x, y, symbol='+', color='blue', radius=29):
        ax.add_patch(Circle((x, y), radius, facecolor='white', edgecolor=COLORS[color], linewidth=2))
        text(x, y, symbol, size=34, color=color, weight='bold', ha='center', va='center')

    def section(y, number, title, description):
        text(75, y, number, size=28, color='blue', weight='bold')
        text(140, y, title, size=32, weight='bold')
        text(140, y+45, description, size=22, color='muted')

    text(75, 55, 'NAFNet25D', size=65, weight='bold')
    text(75, 133, 'Micro-CT denoising with learned axial fusion and a full-resolution detail path', size=31)
    text(75, 182, 'KRG-alsDenoise  /  implemented architecture  /  repository defaults, not a specific trained checkpoint', size=22, color='muted')
    for x, title, body, theme in (
        (75, '5 input slices → 1 output', 'K is configurable: odd values ≥ 3', 'coral'),
        (655, '989,745 parameters', '18 NAFBlocks · widths 32 / 64 / 128', 'blue'),
        (1235, '256 × 256 training patches', 'B = 4 per GPU in the default preset', 'teal'),
        (1815, 'Two downsampling stages', 'Context reaches 64 × 64; detail stays 256 × 256', 'violet')):
        box(x, 234, 510, 100, title, [body], theme=theme, size=18, title_size=24)

    section(370, '01', 'End-to-end data flow', 'Shapes use B × channels × height × width. The example has K = 5 and H = W = 256.')
    box(75, 465, 330, 157, 'Normalized axial stack', ['X = [X₋₂, X₋₁, X₀, X₊₁, X₊₂]', '[B, 5, 256, 256]', 'Xc = X₀ is the center image'], theme='gold', size=21, title_size=24)
    box(445, 465, 280, 157, 'Replicate padding', ['Pad right / bottom edges', 'P = 4⌈H/4⌉, Q = 4⌈W/4⌉', 'No padding at 256 × 256'], size=19, title_size=24)
    box(765, 465, 370, 157, 'Shared slice stem', ['Reshape to [B·K, 1, P, Q]', '3 × 3 Conv: 1 → 16, stride 1', 'Features F: [B, K, 16, P, Q]'], size=21)
    box(1175, 465, 520, 157, 'Gated neighbor corrections', ['For each neighbor: Δi = Fi − Fc', 'Learn spatial / channel gate gi (panel 02)', 'R = Σᵢ≠c (gi ⊙ Δi) / (K − 1)'], theme='coral', size=22)
    box(1735, 465, 430, 157, 'Fuse center + correction', ['Concatenate [Fc, R]: 16 + 16 channels', '1 × 1 Conv: 32 → 32', 'Fused features: [B, 32, P, Q]'], theme='coral', size=21, title_size=24)
    for start, end in ((405,445), (725,765), (1135,1175), (1695,1735)):
        line([(start, 544), (end, 544)])

    line([(1950,622),(1950,670),(245,670),(245,790)], color='coral')
    ax.add_patch(Circle((245,790), 7, color=COLORS['coral']))
    text(270, 703, 'The same fused features feed both branches', size=23, color='muted')
    box(350, 740, 610, 125, 'Detail branch · 4 × NAFBlock (C = 32)', ['[B, 32, 256, 256] → [B, 32, 256, 256]', 'Full-resolution processing without downsampling'], theme='teal', size=22, title_size=25)
    line([(245,790),(350,790)], color='teal')
    text(1655, 727, 'Context branch · encoder–decoder', size=27, color='blue', weight='bold')

    enc_y, dec_y = 945, 1275
    box(350, enc_y, 330, 137, 'Encoder 0 · 2 blocks', ['C = 32', '[B, 32, 256, 256]'], size=22, title_size=25)
    box(835, enc_y, 220, 137, 'Downsample', ['2 × 2 Conv · stride 2', '32 → 64 channels'], size=18, title_size=23)
    box(1170, enc_y, 330, 137, 'Encoder 1 · 2 blocks', ['C = 64', '[B, 64, 128, 128]'], size=22, title_size=25)
    box(1655, enc_y, 220, 137, 'Downsample', ['2 × 2 Conv · stride 2', '64 → 128 channels'], size=18, title_size=23)
    box(1980, 1010, 310, 164, 'Middle · 6 blocks', ['C = 128 (bottleneck)', '[B, 128, 64, 64]', '713,472 parameters'], theme='violet', size=22, title_size=24)
    line([(245,790),(245,1013),(350,1013)], color='blue')
    for a, b in ((680,835),(1055,1170),(1500,1655)):
        line([(a,1013),(b,1013)], color='blue')
    line([(1875,1013),(1920,1013),(1920,1090),(1980,1090)], color='blue')

    box(1655, dec_y, 220, 137, 'Upsample × 2', ['1 × 1: 128 → 256', 'PixelShuffle × 2', 'Output C = 64'], size=19, title_size=23)
    box(1170, dec_y, 330, 137, 'Decoder 1 · 2 blocks', ['C = 64', '[B, 64, 128, 128]'], size=22, title_size=25)
    box(835, dec_y, 220, 137, 'Upsample × 2', ['1 × 1: 64 → 128', 'PixelShuffle × 2', 'Output C = 32'], size=19, title_size=23)
    box(350, dec_y, 330, 137, 'Decoder 0 · 2 blocks', ['C = 32', '[B, 32, 256, 256]'], size=22, title_size=25)
    line([(2290,1090),(2320,1090),(2320,1343),(1875,1343)], color='blue')
    operator(1335,1190)
    line([(1655,1343),(1585,1343),(1585,1190),(1364,1190)], color='blue')
    line([(1335,1082),(1335,1161)], color='blue', dashed=True)
    line([(1335,1219),(1335,1275)], color='blue')
    text(1380, 1130, 'add encoder skip', size=21, color='blue')
    line([(1170,1343),(1055,1343)], color='blue')
    operator(515,1190)
    line([(835,1343),(760,1343),(760,1190),(544,1190)], color='blue')
    line([(515,1082),(515,1161)], color='blue', dashed=True)
    line([(515,1219),(515,1275)], color='blue')
    text(560, 1130, 'add encoder skip', size=21, color='blue')

    box(750, 1470, 370, 125, 'Concatenate branches', ['Detail (32) + context (32)', '[B, 64, P, Q]'], theme='teal', size=22, title_size=25)
    box(1220, 1470, 620, 125, 'Residual head + crop', ['1 × 1 Conv: 64 → 32; 3 × 3 Conv: 32 → 1', 'Crop to H × W, then cast correction to FP32'], size=23)
    box(1980, 1470, 345, 125, 'Denoised center slice', ['Ŷc = Xc + residual', '[B, 1, H, W] · FP32'], theme='gold', size=22, title_size=24)
    line([(960,803),(2340,803),(2340,1440),(980,1440),(980,1470)], color='teal')
    line([(350,1343),(300,1343),(300,1425),(830,1425),(830,1470)], color='blue')
    line([(1120,1533),(1220,1533)], color='teal')
    operator(1900,1533,color='gold')
    line([(1840,1533),(1871,1533)], color='gold')
    line([(1929,1533),(1980,1533)], color='gold')
    line([(190,622),(95,622),(95,1665),(1900,1665),(1900,1562)], color='gold', width=2.7)
    text(390, 1625, 'Unmodified center-image shortcut  ·  Xc [B, 1, H, W]  ·  final addition in FP32', size=24, color='gold')
    text(1980, 1625, 'Final 3 × 3 head starts at zero.', size=20, color='muted')

    panel(1740, 440)
    section(1770, '02', 'Learned axial fusion', 'One shared gate is reused for all K − 1 neighboring slices. All spatial operations below preserve P × Q.')
    box(80, 1870, 350, 169, 'Center Fc + neighbor Fi', ['Both: [B, 16, P, Q]', 'From the same shared 3 × 3 stem', 'i ≠ c; K = 5 gives four neighbors'], theme='coral', size=21, title_size=24)
    box(480, 1870, 265, 169, 'Feature difference', ['Δi = Fi − Fc', '[B, 16, P, Q]'], theme='coral', size=22, title_size=24)
    box(795, 1870, 330, 169, 'Gate input', ['Concatenate [Fc, Fi, Δi]', '16 + 16 + 16 = 48 channels', '[B, 48, P, Q]'], theme='coral', size=21)
    box(1175, 1870, 570, 169, 'Spatial / channel gate gi', ['Depthwise 3 × 3: 48 → 48, groups = 48', 'Pointwise 1 × 1: 48 → 16, then sigmoid', 'gi: [B, 16, P, Q], values between 0 and 1'], theme='coral', size=22, title_size=25)
    box(1795, 1870, 525, 169, 'Average gated corrections', ['Multiply gi ⊙ Δi for each neighbor', 'R = Σᵢ≠c (gi ⊙ Δi) / (K − 1)', 'Concatenate [Fc, R]; 1 × 1 Conv → 32'], theme='coral', size=22, title_size=25)
    for a,b in ((430,480),(745,795),(1125,1175),(1745,1795)):
        line([(a,1954),(b,1954)],color='coral')
    text(90, 2075, 'Gate pointwise bias starts at −2 to initially limit neighbor influence; the exact gate also depends on features and weights.', size=23)
    text(90, 2120, 'No axial-distance embedding is implemented. Neighbor order is interchangeable; only the center slice has a distinct role.', size=23, color='muted')

    panel(2220, 785)
    section(2250, '03', 'Inside every NAFBlock', 'C = 32, 64 or 128; spatial size h × w is unchanged. Each block has two independently scaled residual updates.')
    text(80, 2355, 'A', size=24, weight='bold', color='violet')
    box(80, 2410, 150, 115, 'Input x', ['[B,C,h,w]'], theme='violet', size=19, title_size=23)
    box(275, 2410, 220, 115, 'LayerNorm', ['C channels / pixel'], theme='violet', size=19, title_size=23)
    box(540, 2410, 230, 115, '1 × 1 Conv', ['Expand C → 2C'], theme='violet', size=21, title_size=23)
    box(815, 2410, 230, 115, 'Depthwise 3 × 3', ['2C groups; 2C → 2C'], theme='violet', size=19, title_size=22)
    box(1090, 2410, 200, 115, 'SimpleGate', ['Split 2C into a,b', 'T = a ⊙ b: C'], theme='violet', size=18, title_size=23)
    operator(1370,2468,'×','violet')
    box(1440, 2410, 260, 115, '1 × 1 Conv', ['Project C → C'], theme='violet', size=21, title_size=23)
    box(1750, 2410, 170, 115, 'Scale β', ['Per channel', 'Initially zero'], theme='violet', size=18, title_size=23)
    operator(1980,2468,color='violet')
    box(2060, 2410, 260, 115, 'Intermediate u', ['u = x + β ⊙ A(x)'], theme='violet', size=19, title_size=23)
    for a,b in ((230,275),(495,540),(770,815),(1045,1090),(1290,1341),(1399,1440),(1700,1750),(1920,1951),(2009,2060)):
        line([(a,2468),(b,2468)],color='violet')
    line([(155,2410),(155,2368),(1980,2368),(1980,2439)], color='violet', dashed=True)
    text(850, 2339, 'Identity shortcut', size=20, color='violet')
    box(1080, 2590, 450, 120, 'Simplified channel attention', ['Global mean of T → [B, C, 1, 1]', '1 × 1 Conv: C → C; no sigmoid'], theme='violet', size=21, title_size=23)
    line([(1190,2525),(1190,2555),(1040,2555),(1040,2650),(1080,2650)], color='violet')
    line([(1530,2650),(1570,2650),(1570,2555),(1370,2555),(1370,2497)], color='violet')
    text(1620, 2600, 'Broadcast channel weights over h × w.', size=22, color='muted')
    text(1620, 2640, 'Attention = T ⊙ Conv₁×₁(mean over h,w(T)).', size=22)

    text(80, 2740, 'B', size=24, weight='bold', color='violet')
    box(420, 2785, 250, 105, 'LayerNorm', ['C channels / pixel'], theme='violet', size=20, title_size=24)
    box(730, 2785, 250, 105, '1 × 1 Conv', ['Expand C → 2C'], theme='violet', size=21, title_size=24)
    box(1040, 2785, 220, 105, 'SimpleGate', ['a ⊙ b: 2C → C'], theme='violet', size=20, title_size=24)
    box(1320, 2785, 250, 105, '1 × 1 Conv', ['Project C → C'], theme='violet', size=21, title_size=24)
    box(1630, 2785, 170, 115, 'Scale γ', ['Per channel', 'Starts at 0'], theme='violet', size=18, title_size=24)
    operator(1900,2838,color='violet')
    box(1990, 2785, 330, 105, 'Block output v', ['v = u + γ ⊙ B(u)'], theme='violet', size=21, title_size=24)
    line([(2190,2525),(2190,2545),(2330,2545),(2330,2740),(180,2740),(180,2838),(420,2838)], color='violet')
    for a,b in ((670,730),(980,1040),(1260,1320),(1570,1630),(1800,1871),(1929,1990)):
        line([(a,2838),(b,2838)],color='violet')
    line([(180,2838),(180,2920),(1900,2920),(1900,2867)],color='violet',dashed=True)
    text(90, 2960, 'LayerNorm uses FP32 with ε = 10⁻⁶; β and γ have shape [1,C,1,1]. NAFBlocks use multiplication; axial gates use sigmoid.', size=22, color='muted')

    section(3050, '04', 'Training context and implementation details', 'Both directions use the same network weights. The loss is computed against the opposite reconstruction split’s center slice.')
    box(80, 3150, 1120, 294, 'Symmetric paired training', [
        'Even-angle stack A → fθ(A) → compare with odd center Bc',
        'Odd-angle stack B → fθ(B) → compare with even center Ac',
        'L = ½(L1(A→Bc) + L1(B→Ac)) + ½λ(LCL(fθ(A)) + LCL(fθ(B)))',
        'λ = 0 during warmup; λ = 0.01 after warmup (default 2,000 updates).',
        'LCL = mean flat-region |Laplacian| / (mean edge-region |Laplacian| + ε).',
        'The detached 80th-percentile mask separates flat / edge responses.'
    ], theme='gold', size=23, title_size=27)
    box(1250, 3150, 1070, 294, 'Parameters, precision and initialization', [
        'Axial stem + gate + fusion: 2,480  ·  detail branch: 32,896',
        'Context encoder–decoder: 952,000  ·  residual head: 2,369',
        'Each NAFBlock: 7C² + 33C parameters; 18 blocks in total.',
        'β = γ = 0 and the final head is zero: initial output equals Xc.',
        'BF16 autocast on supported GPUs; LayerNorm and final addition use FP32.',
        'Adam: lr = 0.001; gradient norm clipped to 1.0; no LR scheduler.'
    ], theme='blue', size=23, title_size=27)
    text(90, 3472, 'Stopping: trailing 10-epoch evaluation total-loss mean after warmup; min_delta = 0.001; patience = 100 in notebook defaults.', size=23)
    text(90, 3510, 'L2 / SSIM are reporting metrics; raw L1 / LCL / EDGE checkpoints remain separate. Evaluation resamples training data; metrics use the final GPU rank.', size=21, color='muted')
    text(90, 3553, 'Source: denoise/nafnet25d.py, denoise/train.py, denoise/loss.py, workflow/src/krg_workflow/presets.py  ·  NAFBlock adapted from MEGVII NAFNet (MIT).', size=20, color='muted')

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    for artist, boundary in text_boxes:
        bounds = artist.get_window_extent(renderer).transformed(ax.transData.inverted())
        x0, y0, x1, y1 = boundary
        width = x1 - artist.get_position()[0]
        height = y1 - artist.get_position()[1]
        scale = min(1, width/(bounds.xmax-bounds.xmin), height/(bounds.ymax-bounds.ymin))
        if scale < 1:
            artist.set_fontsize(artist.get_fontsize()*scale*.97)
        if artist.get_fontsize() < 11:
            raise ValueError('Diagram label needs more space: '+artist.get_text())
    fig.canvas.draw()
    overflow = []
    for artist, boundary in text_boxes:
        bounds = artist.get_window_extent(renderer).transformed(ax.transData.inverted())
        x0, y0, x1, y1 = boundary
        if bounds.xmin < x0 or bounds.xmax > x1 or bounds.ymin < y0 or bounds.ymax > y1:
            overflow.append((artist.get_text(), boundary, (bounds.xmin, bounds.ymin, bounds.xmax, bounds.ymax)))
    if overflow:
        raise ValueError('Text exceeds diagram nodes: '+repr(overflow))
    for suffix in ('svg', 'png'):
        path = DESTINATION/('nafnet25d-architecture.'+suffix)
        export_metadata = {'Date': None, 'Creator': 'KRG-alsDenoise architecture renderer'} if suffix == 'svg' else None
        fig.savefig(path, dpi=160, metadata=export_metadata)
        if suffix == 'svg':
            path.write_text('\n'.join(line.rstrip() for line in path.read_text(encoding='utf-8').splitlines())+'\n', encoding='utf-8')
        print(path)
    plt.close(fig)
    (DESTINATION/'nafnet25d-architecture.json').write_text(json.dumps(metadata, indent=2)+'\n', encoding='utf-8')
    print('Verified:', metadata['parameters'], 'parameters; meta shape propagation passed; no training.')


if __name__ == '__main__':
    render()
