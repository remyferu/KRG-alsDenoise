"""Scans helpers extracted from the v3.3 CT workflow.

No notebook cell execution or training occurs on import.
"""
from pathlib import Path
import h5py
import hashlib
import numpy as np
import stat
import uuid
from .files import digest


def fingerprint(path):
    """Identify an HDF5 source by path, file stat, data shape, type, and angles."""
    path = Path(path).resolve()
    stat = path.stat()
    with h5py.File(path, 'r') as f:
        data = f['/exchange/data']
        theta = np.asarray(f['/exchange/theta'][:])
        return dict(path=str(path), size=stat.st_size, mtime_ns=stat.st_mtime_ns, shape=list(data.shape), dtype=str(data.dtype), theta_sha256=hashlib.sha256(theta.tobytes()).hexdigest())


def validate_als_metadata(path):
    """Validate the metadata read by the modern ALS backend before expensive work."""
    with h5py.File(path, 'r') as f:
        nproj, nrows, ncols = f['/exchange/data'].shape
        expected = {'/measurement/instrument/detector/dimension_x': ncols, '/measurement/instrument/detector/dimension_y': nrows, '/process/acquisition/rotation/num_angles': nproj}
        required = list(expected) + ['/process/acquisition/rotation/range', '/measurement/instrument/detector/pixel_size', '/measurement/instrument/monochromator/energy']
        for key in required:
            if key not in f:
                raise ValueError(f'ALS backend requires metadata: {key}')
            values = np.asarray(f[key][()])
            if values.ndim != 1 or not values.size or (not np.isfinite(values).all()):
                raise ValueError(f'ALS requires finite 1D metadata: {key}')
            if key in expected and values[0] != expected[key]:
                raise ValueError(f'ALS metadata disagrees with image dimensions: {key}')
        if f['/measurement/instrument/detector/pixel_size'][0] <= 0:
            raise ValueError('ALS attenuation scaling requires positive pixel size in mm.')
        theta = np.asarray(f['/exchange/theta'][:], dtype=np.float64)
        if theta.shape != (nproj,) or nproj < 2 or (not np.isfinite(theta).all()) or (not np.all(np.diff(theta) > 0)):
            raise ValueError('Expected one finite, increasing angle per projection.')
        span = float(theta[-1] - theta[0])
        stored = float(f['/process/acquisition/rotation/range'][0])
        if abs(stored - span) > max(1.0, 2 * float(np.median(np.diff(theta)))):
            raise ValueError('ALS angular range metadata disagrees with measured angles.')


def optional_camera_distance(path):
    """Report only the upstream second entry; never reinterpret a singleton distance."""
    key = '/measurement/instrument/camera_motor_stack/setup/camera_distance'
    with h5py.File(path, 'r') as f:
        if key not in f:
            return None
        values = np.asarray(f[key][()])
        if values.ndim != 1 or values.size < 2 or values.dtype.kind not in 'iuf':
            return None
        value = float(values[1])
        return value if np.isfinite(value) else None


def install_reconstruction_metadata_reader(functions):
    """Adapt unused distance metadata for this notebook's no-phase FBP/gridrec path.

    Changes only the privately imported module, never the checkout or raw HDF5.
    Missing distance remains None; no measurement or units are guessed.
    """
    original = functions.read_metadata

    def read_metadata(path, print_flag=True):
        if optional_camera_distance(path) is not None:
            return original(path, print_flag=print_flag)
        validate_als_metadata(path)
        with h5py.File(path, 'r') as f:

            def first(key):
                return f[key][0]
            metadata = dict(numslices=int(first('/measurement/instrument/detector/dimension_y')), numrays=int(first('/measurement/instrument/detector/dimension_x')), pxsize=float(first('/measurement/instrument/detector/pixel_size')) / 10.0, numangles=int(first('/process/acquisition/rotation/num_angles')), propagation_dist=None, kev=float(first('/measurement/instrument/monochromator/energy')) / 1000.0, angularrange=float(first('/process/acquisition/rotation/range')))
        if print_flag:
            print(Path(path).name, metadata)
            print('Camera distance unavailable; unused in this no-phase FBP/gridrec workflow.')
        return metadata
    functions.read_metadata = read_metadata


def inspect_scan(path):
    """Check the ALS-style 180-degree HDF5 layout and return readable metadata."""
    with h5py.File(path, 'r') as f:
        required = ['data', 'data_white', 'data_dark', 'theta']
        for name in required:
            assert '/exchange/' + name in f, f'Missing /exchange/{name}'
        shape = f['/exchange/data'].shape
        if len(shape) != 3 or not all((n > 0 for n in shape)) or shape[0] < 4:
            raise ValueError('Expected [projection, row, column] with at least four projections for even/odd views.')
        for name in ('data_white', 'data_dark'):
            calibration = f['/exchange/' + name]
            if calibration.ndim != 3 or calibration.shape[0] == 0 or calibration.shape[1:] != shape[1:]:
                raise ValueError(f'{name} must contain nonempty [frame, row, column] images matching the scan.')
        validate_als_metadata(path)
        theta = np.asarray(f['/exchange/theta'][:], dtype=np.float64)
        assert theta.shape == (shape[0],) and np.isfinite(theta).all()
        assert np.all(np.diff(theta) > 0), 'This workflow requires increasing projection angles.'
        step = float(np.median(np.diff(theta)))
        assert abs(float(theta[-1] - theta[0]) - 180) <= max(1, 2 * step), 'Expected an ALS 180-degree scan. Check angle units/geometry before proceeding.'

        def scalar(key):
            if key not in f:
                return None
            values = np.asarray(f[key][()])
            if values.ndim != 1 or not values.size:
                raise ValueError(f'Expected nonempty 1D metadata: {key}')
            val = float(values[0])
            return val if np.isfinite(val) else None
        pixel_mm = scalar('/measurement/instrument/detector/pixel_size')
        energy_ev = scalar('/measurement/instrument/monochromator/energy')
        info = dict(shape=list(shape), dtype=str(f['/exchange/data'].dtype), flats=int(f['/exchange/data_white'].shape[0]), darks=int(f['/exchange/data_dark'].shape[0]), angle_start_deg=float(theta[0]), angle_end_deg=float(theta[-1]), angle_step_deg=step, pixel_um=pixel_mm * 1000 if pixel_mm and pixel_mm > 0 else None, distance_mm=optional_camera_distance(path), stored_energy_ev=energy_ev, energy_kev=energy_ev / 1000 if energy_ev and energy_ev > 0 else None)
        if info['distance_mm'] is None:
            print('Camera distance has no usable second entry; recorded as unknown. This workflow does not use phase retrieval or camera distance.')
        return info


def projection_view(source_path, destination, parity, expected_source):
    """Create and verify an even or odd HDF5 virtual projection view.

    The view stores angles and ALS metadata for the selected projections;
    image pixels and flat/dark frames remain in the unchanged source file.
    `parity` is 0 for even indices and 1 for odd indices.
    """
    source_path, destination = (Path(source_path).resolve(), Path(destination))
    assert fingerprint(source_path) == expected_source, 'Source file moved or changed.'
    if parity not in (0, 1):
        raise ValueError('Projection parity must be 0 (even) or 1 (odd).')
    validate_als_metadata(source_path)
    nproj, rows, columns = expected_source['shape']
    indices = np.arange(parity, nproj, 2, dtype=np.int64)
    assert len(indices) >= 2
    signature = digest(dict(view_schema=4, source=expected_source, parity=parity))
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        tmp = destination.with_name(destination.name + '.' + uuid.uuid4().hex + '.tmp')
        with h5py.File(source_path, 'r') as src, h5py.File(tmp, 'w', libver='latest') as out:
            original = src['/exchange/data']
            layout = h5py.VirtualLayout(shape=(len(indices), rows, columns), dtype=original.dtype)
            virtual = h5py.VirtualSource(str(source_path), '/exchange/data', shape=original.shape)
            for out_index, original_index in enumerate(indices):
                layout[out_index] = virtual[int(original_index)]
            ex = out.create_group('exchange')
            ex.create_virtual_dataset('data', layout)
            ex['data_white'] = h5py.ExternalLink(str(source_path), '/exchange/data_white')
            ex['data_dark'] = h5py.ExternalLink(str(source_path), '/exchange/data_dark')
            ex.create_dataset('theta', data=src['/exchange/theta'][indices])
            values = {'/measurement/instrument/detector/dimension_x': columns, '/measurement/instrument/detector/dimension_y': rows, '/process/acquisition/rotation/num_angles': len(indices), '/process/acquisition/rotation/range': float(src['/exchange/theta'][int(indices[-1])] - src['/exchange/theta'][int(indices[0])]), '/process/acquisition/flat_fields/num_flat_fields': src['/exchange/data_white'].shape[0], '/process/acquisition/dark_fields/num_dark_fields': src['/exchange/data_dark'].shape[0], '/process/acquisition/flat_fields/i0cycle': 0}
            for key, value in values.items():
                out.create_dataset(key, data=[value])
            for key in ('/measurement/instrument/detector/pixel_size', '/measurement/instrument/camera_motor_stack/setup/camera_distance', '/measurement/instrument/monochromator/energy'):
                if key in src:
                    out.create_dataset(key, data=src[key][()])
            out.create_dataset('source_projection_indices', data=indices)
            out.attrs['signature'] = signature
        tmp.replace(destination)
    with h5py.File(source_path, 'r') as src, h5py.File(destination, 'r') as view:
        assert view.attrs.get('signature') == signature, 'Stale view; choose a new run directory.'
        assert view['/exchange/data'].is_virtual
        assert view['/exchange/data'].shape == (len(indices), rows, columns)
        np.testing.assert_array_equal(view['source_projection_indices'][:], indices)
        np.testing.assert_array_equal(view['/exchange/theta'][:], src['/exchange/theta'][indices])
        for key in ('data_white', 'data_dark'):
            np.testing.assert_array_equal(view['/exchange/' + key][0], src['/exchange/' + key][0])
        for i in sorted(set([0, len(indices) // 2, len(indices) - 1])):
            np.testing.assert_array_equal(view['/exchange/data'][i], src['/exchange/data'][int(indices[i])])
    return destination
