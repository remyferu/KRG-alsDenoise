# Data importers
+
+Convert volumetric data into the TIFF directories required by Noise2Inverse360.
+Each reconstruction stack contains 2D files named `00000.tiff`, `00001.tiff`,
+and so on; `denoise.tiffs.glob()` loads them in natural filename order.
+
+`3dtiff_converter.py` reads `beta_even.tiff`, `beta_odd.tiff` and `beta_all.tiff`
+as `[slice, row, column]` volumes and writes `beta_even/`, `beta_odd/` and
+`beta_all/` directories:
+
+```bash
+python data_importers/3dtiff_converter.py /path/to/input /path/to/output
+```
+
+The output directory defaults to the input directory when omitted. Existing
+slice files with matching names are overwritten. The three input filenames
+are fixed in the script.
+
+New converters should use descriptive filenames and write `%05d.tiff` slices.
+Document their input format and output layout here.
+