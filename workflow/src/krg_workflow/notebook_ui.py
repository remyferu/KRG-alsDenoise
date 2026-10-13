"""Notebook ui helpers extracted from the v3.3 CT workflow.

No notebook cell execution or training occurs on import.
"""
try:
    from IPython.display import display
except ImportError:
    def display(value, **kwargs):
        print(value)
from PIL import Image
from pathlib import Path
import base64
import html
import io
import matplotlib.pyplot as plt
import numpy as np
import tifffile
try:
    import ipywidgets as widgets
except ImportError:
    widgets = None


def image_zoom_bounds(shape, zoom=1.0, center=None):
    """Return exclusive pixel bounds for a zoomed viewport, without resampling."""
    if len(shape) != 2 or min(shape) < 1:
        raise ValueError('The viewer needs a nonempty two-dimensional image.')
    height, width = map(int, shape)
    zoom = float(zoom)
    if not np.isfinite(zoom) or zoom < 1:
        raise ValueError('Zoom must be finite and at least 1.')
    center = ((width - 1) / 2, (height - 1) / 2) if center is None else tuple(center)
    if len(center) != 2 or not np.isfinite(center).all():
        raise ValueError('Center must be finite (x, y) pixel coordinates.')
    crop_width = max(1, int(round(width / zoom)))
    crop_height = max(1, int(round(height / zoom)))
    x0 = min(width - crop_width, max(0, int(round(center[0] - (crop_width - 1) / 2))))
    y0 = min(height - crop_height, max(0, int(round(center[1] - (crop_height - 1) / 2))))
    return (x0, x0 + crop_width, y0, y0 + crop_height)


def image_display_limits(image):
    """Choose a robust full-image window once, rather than renormalizing each crop."""
    finite = np.asarray(image)[np.isfinite(image)]
    if not finite.size:
        raise ValueError('The image contains no finite pixels.')
    low, high = map(float, np.percentile(finite, [1, 99]))
    if high <= low:
        high = low + max(1.0, abs(low) * 1e-06)
    return (low, high)


def image_view_png(image, bounds, limits, cmap='gray', overlay=None):
    """Encode only the selected native pixels; keep NaN borders transparent."""
    x0, x1, y0, y1 = bounds
    low, high = map(float, limits)
    if not np.isfinite([low, high]).all() or high <= low:
        raise ValueError('White level must be greater than black level; both must be finite.')
    crop = np.asarray(image)[y0:y1, x0:x1]
    rgba = plt.get_cmap(cmap)(np.ma.masked_invalid((crop - low) / (high - low)), bytes=True)
    if overlay is not None:
        over_image, over_cmap, opacity = overlay
        over_crop = np.asarray(over_image)[y0:y1, x0:x1]
        over_rgba = plt.get_cmap(over_cmap)(np.ma.masked_invalid((over_crop - low) / (high - low)), bytes=True)
        alpha = np.where(np.isfinite(over_crop), float(opacity), 0.0)[..., None]
        rgba[..., :3] = np.round(rgba[..., :3] * (1 - alpha) + over_rgba[..., :3] * alpha).astype(np.uint8)
    stream = io.BytesIO()
    Image.fromarray(rgba).save(stream, format='PNG')
    return stream.getvalue()


def zoom_image_viewer(images, titles=None, limits=None, cmaps=None, overlays=None, captured=None, title='Image inspection', zoom=None, center=None, load_images=None, count=None, index=0, index_label='Slice', *, context=None):
    """Linked crop/contrast controls for arrays or a lazily loaded TIFF slice.

    Zoom and center coordinates affect display only. No image is binned or modified.
    The viewer retains only the current slice, and callbacks verify their run context.
    """
    if captured is not None and context is None:
        raise ValueError('A captured viewer requires its RunContext.')
    defaults = {}
    zoom = float(defaults.get('zoom', 1.0) if zoom is None else zoom)
    center = defaults.get('center') if center is None else center
    arrays = [np.asarray(image) for image in images]
    if not arrays or any((a.ndim != 2 or a.shape != arrays[0].shape or (not np.isrealobj(a)) for a in arrays)):
        raise ValueError('Viewer images must be real, matching two-dimensional arrays.')
    for array in arrays:
        if not np.isfinite(array).any():
            raise ValueError('Viewer images must contain finite pixels.')
    shape = arrays[0].shape
    bounds = image_zoom_bounds(shape, zoom, center)
    number = len(arrays)
    titles = list(titles) if titles is not None else [f'Image {i + 1}' for i in range(number)]
    cmaps = list(cmaps) if cmaps is not None else ['gray'] * number
    limits = [tuple(map(float, pair)) for pair in limits] if limits is not None else [image_display_limits(arrays[0])] * number
    overlays = list(overlays) if overlays is not None else [None] * number
    if any((len(items) != number for items in (titles, cmaps, limits, overlays))):
        raise ValueError('Titles, windows, color maps, and overlays must match the image count.')
    for pair in limits:
        if len(pair) != 2 or not np.isfinite(pair).all() or pair[1] <= pair[0]:
            raise ValueError('Each contrast window needs finite black < white levels.')
    for overlay in overlays:
        if overlay is not None and (len(overlay) != 3 or np.shape(overlay[0]) != shape or (not np.isfinite(overlay[2])) or (not 0 <= overlay[2] <= 1)):
            raise ValueError('Each overlay needs a matching image, color map and opacity in [0,1].')
    if count is not None and (type(count) is not int or count < 1 or (not callable(load_images)) or (not 0 <= index < count)):
        raise ValueError('A slice browser needs a loader, a positive count and a valid initial index.')
    if captured is not None:
        context.assert_current(captured)
    groups, group_keys, panel_groups = ([], [], [])
    for i, (pair, cmap) in enumerate(zip(limits, cmaps)):
        key = (pair, cmap)
        if key not in group_keys:
            group_keys.append(key)
            groups.append(dict(window=pair, initial=pair, titles=[]))
        group = group_keys.index(key)
        groups[group]['titles'].append(titles[i])
        panel_groups.append(group)
    state = dict(images=arrays, index=index, bounds=bounds, groups=groups, error=None)
    if not widgets:
        fig, axes = plt.subplots(1, number, figsize=(7 * number, 7), squeeze=False)
        x0, x1, y0, y1 = bounds
        extent = (x0 - 0.5, x1 - 0.5, y1 - 0.5, y0 - 0.5)
        try:
            for i, ax in enumerate(axes[0]):
                ax.imshow(arrays[i][y0:y1, x0:x1], cmap=cmaps[i], vmin=limits[i][0], vmax=limits[i][1], interpolation='nearest', origin='upper', extent=extent)
                if overlays[i] is not None:
                    over, over_cmap, opacity = overlays[i]
                    ax.imshow(np.asarray(over)[y0:y1, x0:x1], cmap=over_cmap, alpha=opacity, vmin=limits[i][0], vmax=limits[i][1], interpolation='nearest', origin='upper', extent=extent)
                ax.set_title(titles[i])
                ax.set_xlabel('x (pixel)')
                ax.set_ylabel('y (pixel)')
            fig.suptitle(title)
            fig.subplots_adjust(left=0.07, right=0.98, bottom=0.12, top=0.85, wspace=0.25)
            plt.show()
        finally:
            plt.close(fig)
        print('Static pixel view. Enable ipywidgets for zoom controls, or pass zoom=8, center=(x, y) to the viewer; example:', "{'zoom': 8, 'center': (x, y)} and rerun the display cell.")
        return dict(state=state, widget=None, controls={})
    height, width = shape
    start_center = ((width - 1) // 2, (height - 1) // 2) if center is None else center
    layout = widgets.Layout
    zoom_control = widgets.FloatSlider(value=min(zoom, 64.0), min=1.0, max=64.0, step=1.0, description='Zoom:', readout_format='.0f', continuous_update=False, layout=layout(width='330px'))
    x_control = widgets.IntSlider(value=int(np.clip(round(start_center[0]), 0, width - 1)), min=0, max=width - 1, description='Center x:', continuous_update=False, layout=layout(width='330px'))
    y_control = widgets.IntSlider(value=int(np.clip(round(start_center[1]), 0, height - 1)), min=0, max=height - 1, description='Center y:', continuous_update=False, layout=layout(width='330px'))
    reset = widgets.Button(description='Reset view', icon='expand')
    directions = [widgets.Button(description=label, layout=layout(width='65px')) for label in ('←', '→', '↑', '↓')]
    group_control = widgets.Dropdown(options=[(' / '.join(g['titles']), i) for i, g in enumerate(groups)], value=0, description='Contrast:', layout=layout(width='350px'))
    low_control = widgets.FloatText(value=groups[0]['window'][0], description='Black:', layout=layout(width='235px'))
    high_control = widgets.FloatText(value=groups[0]['window'][1], description='White:', layout=layout(width='235px'))
    reset_contrast = widgets.Button(description='Reset contrast')
    panels = [widgets.HTML(layout=layout(flex='1 1 300px', min_width='280px')) for _ in arrays]
    status = widgets.HTML()
    index_control = None if count is None else widgets.IntSlider(value=index, min=0, max=count - 1, description=index_label + ':', continuous_update=False, layout=layout(width='550px'))
    controls = dict(zoom=zoom_control, x=x_control, y=y_control, reset=reset, contrast_group=group_control, black=low_control, white=high_control, reset_contrast=reset_contrast, index=index_control, pan=directions)
    muted = [False]

    def render(change=None):
        if muted[0]:
            return
        try:
            if captured is not None:
                context.assert_current(captured)
            selected = index if index_control is None else index_control.value
            new_arrays = state['images']
            if selected != state['index']:
                new_arrays = [np.asarray(a) for a in load_images(selected)]
                if len(new_arrays) != number or any((a.shape != shape or not np.isrealobj(a) or (not np.isfinite(a).any()) for a in new_arrays)):
                    raise ValueError('Selected slice has invalid or mismatched image pixels.')
            window = (float(low_control.value), float(high_control.value))
            if not np.isfinite(window).all() or window[1] <= window[0]:
                raise ValueError('Set finite contrast levels with White greater than Black.')
            new_bounds = image_zoom_bounds(shape, zoom_control.value, (x_control.value, y_control.value))
            group = group_control.value
            windows = [window if i == group else g['window'] for i, g in enumerate(groups)]
            payloads = []
            for i, array in enumerate(new_arrays):
                png = image_view_png(array, new_bounds, windows[panel_groups[i]], cmaps[i], overlays[i])
                encoded = base64.b64encode(png).decode('ascii')
                label = html.escape(str(titles[i]))
                low, high = windows[panel_groups[i]]
                payloads.append(f'<div style="border:1px solid #dbe2ea;border-radius:8px;padding:8px;background:#f8fafc"><b>{label}</b><div style="font-size:11px;color:#58677a;margin:4px 0">Black {low:.5g} · White {high:.5g}</div><img alt="{label}" src="data:image/png;base64,{encoded}" style="display:block;width:100%;height:auto;max-height:620px;object-fit:contain;image-rendering:crisp-edges;image-rendering:pixelated;background:#17212f"></div>')
            state.update(images=new_arrays, index=selected, bounds=new_bounds, error=None)
            groups[group]['window'] = window
            for panel, payload in zip(panels, payloads):
                panel.value = payload
            x0, x1, y0, y1 = new_bounds
            prefix = '' if index_control is None else f'{html.escape(index_label)} {selected} · '
            status.value = f'<div style="color:#42536a;padding:6px 0">{prefix}{zoom_control.value:g}× zoom · x {x0}–{x1 - 1}, y {y0}–{y1 - 1} · {x1 - x0} × {y1 - y0} source pixels. Zoom and center controls are linked across panels.</div>'
        except Exception as error:
            state['error'] = str(error)
            status.value = '<b style="color:#b42318">Viewer: ' + html.escape(str(error)) + '</b>'

    def choose_group(change):
        muted[0] = True
        try:
            low_control.value, high_control.value = groups[group_control.value]['window']
        finally:
            muted[0] = False
        render()

    def reset_view(button):
        muted[0] = True
        try:
            zoom_control.value = 1.0
            x_control.value, y_control.value = ((width - 1) // 2, (height - 1) // 2)
        finally:
            muted[0] = False
        render()

    def reset_windows(button):
        for group in groups:
            group['window'] = group['initial']
        choose_group(None)

    def pan(dx, dy):
        x0, x1, y0, y1 = state['bounds']
        muted[0] = True
        try:
            x_control.value = int(np.clip(x_control.value + dx * max(1, (x1 - x0) // 4), 0, width - 1))
            y_control.value = int(np.clip(y_control.value + dy * max(1, (y1 - y0) // 4), 0, height - 1))
        finally:
            muted[0] = False
        render()
    for control in (zoom_control, x_control, y_control, low_control, high_control, index_control):
        if control is not None:
            control.observe(render, names='value')
    group_control.observe(choose_group, names='value')
    reset.on_click(reset_view)
    reset_contrast.on_click(reset_windows)
    for button, (dx, dy) in zip(directions, ((-1, 0), (1, 0), (0, -1), (0, 1))):
        button.on_click(lambda button, dx=dx, dy=dy: pan(dx, dy))
    wrap = layout(display='flex', flex_flow='row wrap', align_items='center', width='100%')
    rows = [widgets.HTML(value='<b>' + html.escape(title) + '</b>')]
    if index_control is not None:
        rows.append(index_control)
    rows.extend([widgets.Box([zoom_control, x_control, y_control], layout=wrap), widgets.Box([reset] + directions, layout=wrap), widgets.Box([group_control, low_control, high_control, reset_contrast], layout=wrap), status, widgets.Box(panels, layout=layout(display='flex', flex_flow='row wrap', width='100%'))])
    viewer = widgets.VBox(rows, layout=layout(width='100%'))
    render()
    display(viewer)
    return dict(state=state, widget=viewer, controls=controls, panels=panels, status=status)


def browse_tiffs(paths, title='Slice browser', *, context, zoom=None, center=None):
    """Read one TIFF at a time and retain the same zoom/window as slices change."""
    captured = context.snapshot()
    paths = tuple((Path(p) for p in paths))
    if not paths:
        raise ValueError('No TIFF files found.')
    index = len(paths) // 2
    sample = tifffile.imread(paths[index])

    def load(index, captured=captured, paths=paths):
        context.assert_current(captured)
        return [tifffile.imread(paths[index])]
    return zoom_image_viewer([sample], titles=[title], limits=[image_display_limits(sample)], captured=captured, title=title, load_images=load, count=len(paths), index=index, context=context,zoom=zoom,center=center)


def live_log_box():
    """Return a progress callback; keep raw ANSI output in the complete log file."""
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            handle = display({'text/plain':'Ready'},raw=True,display_id=True)
            if handle is not None:
                return lambda text: handle.update({'text/plain':text},raw=True)
    except ImportError:
        pass
    return lambda text: print(text,flush=True)


def manual_review(run, rows=None, adjustment=0., apply=False):
    """Review one selected COR; movement never saves a plan or applies a choice."""
    from .cor import manual_start, cor_comparison, apply_manual
    from .files import exclusive_lock
    captured = run.snapshot()
    rows = [run.shape[1]//2] if rows is None else list(rows)
    base, reason = manual_start(captured,run.automatic_cor,run.alignment_center,context=run)
    print(f'Review starts at {base:+.3f} px: {reason}')

    def preview(selected_rows, offset):
        run.assert_current(captured,check_source=True)
        with exclusive_lock(run.run_dir/'.run.lock'):
            cor_comparison(captured,[base+offset],selected_rows,
                           run.run_dir/'manual_cor_previews',context=run)

    def save(selected_rows, offset, method):
        run.manual_cor = apply_manual(captured,base+offset,selected_rows,method,base,context=run)
        print('Manual COR applied; run Save reviewed COR to freeze it.')

    if not widgets:
        preview(rows,adjustment)
        if apply:
            save(rows,adjustment,'manual_numeric')
        return None
    slider = widgets.FloatSlider(value=adjustment,min=min(-1.,adjustment),max=max(1.,adjustment),
                                step=.05,description='Adjust (px):',continuous_update=False)
    row_input = widgets.Text(value=','.join(map(str,rows)),description='Rows:')
    label = widgets.HTML(value=f'Midpoint COR: {base+adjustment:+.3f} px')
    preview_button = widgets.Button(description='Preview selected COR')
    apply_button = widgets.Button(description='Apply selected COR')
    output = widgets.Output()
    def update(change):
        with output:
            run.assert_current(captured)
            label.value = f'Midpoint COR: {base+slider.value:+.3f} px'
    def act(function, method=None):
        def clicked(button):
            with output:
                run.assert_current(captured)
                selected = sorted(set(int(value.strip()) for value in row_input.value.split(',')))
                if method is None:
                    function(selected,slider.value)
                else:
                    function(selected,slider.value,method)
        return clicked
    slider.observe(update,names='value')
    preview_button.on_click(act(preview))
    apply_button.on_click(act(save,'manual_widget'))
    viewer = widgets.VBox([slider,label,row_input,widgets.HBox([preview_button,apply_button]),output])
    display(viewer)
    return viewer
