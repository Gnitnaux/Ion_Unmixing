import os
import re
import numpy as np


def _parse_concentration_to_nm(value_str, unit_str):
    # Convert a concentration value+unit to nM.
    value = int(value_str)
    if unit_str == 'μM':
        return value * 1000
    return value


def read_data(data_dir):
    """
    Read and preprocess spectral data from txt files in data_dir.

    Parameters
    ----------
    data_dir : str
        Path to the directory containing the spectral txt files.

    Returns
    -------
    group_numbers : list of int
        Group numbers parsed from filenames, one entry per spectrum
        (repeated for each spectrum within the same file).
    concentrations : list of list of int
        Each entry is [Cu_nM, Fe_nM, Zn_nM] — concentrations in nM
        for the corresponding spectrum.
    Raman_Shifts : np.ndarray
        1D array of Raman Shift values (shared across all spectra).
    intensities : list of np.ndarray
        Each entry is a 1D array of intensity values for one spectrum.
    """
    print(f"Reading data from {data_dir}...")


    _FILE_RE = re.compile(
        r'^(\d+)\s*\(\s*'
        r'(\d+)\s*(μM|nM)\s+'
        r'(\d+)\s*(μM|nM)\s+'
        r'(\d+)\s*(μM|nM)\s*\)\.txt$'
    )

    # Collect and sort txt files by group number
    txt_files = [
        f for f in os.listdir(data_dir)
        if f.endswith('.txt') and _FILE_RE.match(f)
    ]
    txt_files.sort(key=lambda x: int(_FILE_RE.match(x).group(1)))

    if not txt_files:
        raise FileNotFoundError(
            f"No matching txt files found in {data_dir}"
        )

    group_numbers = []
    concentrations = []
    Raman_Shifts = None
    intensities = []

    for filename in txt_files:
        match = _FILE_RE.match(filename)
        group_num = int(match.group(1))

        cu_nm = _parse_concentration_to_nm(match.group(2), match.group(3))
        fe_nm = _parse_concentration_to_nm(match.group(4), match.group(5))
        zn_nm = _parse_concentration_to_nm(match.group(6), match.group(7))

        filepath = os.path.join(data_dir, filename)

        with open(filepath, 'r') as f:
            lines = f.readlines()

        # --- Parse Raman Shifts from the first line ---
        # The first line starts with a leading tab (empty first column for
        # alignment); strip() removes it, so all remaining tokens are values.
        raman_tokens = lines[0].strip().split('\t')
        current_raman = np.array([float(x) for x in raman_tokens])

        if Raman_Shifts is None:
            Raman_Shifts = current_raman
        else:
            if not np.allclose(Raman_Shifts, current_raman):
                raise ValueError(
                    f"Raman Shift mismatch in file: {filename}. "
                    f"Expected all files to share the same Raman Shift axis."
                )

        # --- Parse intensity rows ---
        # Each data line: metadata_col1, metadata_col2, intensity_1, intensity_2, ...
        for line in lines[1:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split('\t')
            # First two columns are metadata (e.g. -100, -50), skip them
            intensity = np.array([float(x) for x in parts[2:]])
            intensities.append(intensity)
            group_numbers.append(group_num)
            concentrations.append([cu_nm, fe_nm, zn_nm])

    print(
        f"Loaded {len(intensities)} spectra from {len(txt_files)} files. "
        f"Raman Shift range: {Raman_Shifts[0]:.1f} – {Raman_Shifts[-1]:.1f} "
        f"({len(Raman_Shifts)} points)."
    )

    return group_numbers, concentrations, Raman_Shifts, intensities
