import os
import re
import numpy as np
import matplotlib.pyplot as plt
from scipy.sparse import diags, eye, csc_matrix
from scipy.sparse.linalg import spsolve


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


def WhittakerSmooth(x, w, lambda_, differences=1):
    X = np.matrix(x)
    m = X.size
    E = eye(m, format='csc')
    for i in range(differences):
        E = E[1:] - E[:-1]
    W = diags(w, 0, shape=(m, m))
    A = csc_matrix(W + (lambda_ * E.T * E))
    B = csc_matrix(W * X.T)
    background = spsolve(A, B)
    return np.array(background)

def airPLS(x, lambda_=1e7, porder=3, itermax=15):
    m = x.shape[0]
    w = np.ones(m)
    for i in range(1, itermax + 1):
        z = WhittakerSmooth(x, w, lambda_, porder)
        d = x - z
        dssn = np.abs(d[d < 0].sum())
        if (dssn < 0.001 * (abs(x)).sum() or i == itermax):
            if (i == itermax): print('WARING max iteration reached!')
            break
        w[d >= 0] = 0
        w[d < 0] = np.exp(i * np.abs(d[d < 0]) / dssn)
        w[0] = np.exp(i * (d[d < 0]).max() / dssn)
        w[-1] = w[0]
    return z




def preprocess_data(Raman_Shifts, intensities, cut_range=(0, 2000),
                    lamb=1e7, polyorder=3, max_iters=150, plot=False,
                    minmax_normalize=True):
    """Preprocess spectral data: cut to a Raman Shift range and remove
    baseline via airPLS.

    Parameters
    ----------
    Raman_Shifts : np.ndarray
        1D array of Raman Shift values.
    intensities : list of np.ndarray
        Each entry is a 1D array of intensity values for one spectrum.
    cut_range : tuple of (float, float), optional
        The (min, max) Raman Shift range to retain (default: (0, 2000)).
    lamb : float
        Smoothness parameter passed to ``airPLS`` (default: 1e7).
    polyorder : int
        Order of the difference penalty for ``airPLS`` (default: 3).
    max_iters : int
        Maximum iterations for ``airPLS`` (default: 150).
    plot : bool, optional
        Whether to plot the original and baseline-corrected spectra for
        the first few spectra (default: False).
    minmax_normalize : bool, optional
        Whether to apply min-max normalization to [0, 1] (default: True).
        Set to False for PLSR quantification where concentration-dependent
        intensity differences must be preserved.

    Returns
    -------
    cut_Raman_Shifts : np.ndarray
        Raman Shift axis restricted to ``cut_range``.
    processed_intensities : list of np.ndarray
        Baseline-corrected intensity arrays (cut to ``cut_range``).
    """
    min_shift, max_shift = cut_range

    cut_indices = np.where((Raman_Shifts >= min_shift) & (Raman_Shifts <= max_shift))[0]

    if len(cut_indices) == 0:
        raise ValueError(
            f"No Raman Shift points found in the specified cut range: "
            f"{min_shift} – {max_shift} cm⁻¹."
        )

    # Cut the Raman Shifts and intensities to the specified range
    cut_Raman_Shifts = Raman_Shifts[cut_indices]
    processed_intensities = [
        intensity[cut_indices].copy() for intensity in intensities
    ]

    print(f"Applying airPLS baseline correction "
          f"(λ={lamb:.0e}, polyorder={polyorder})...")

    for i in range(len(processed_intensities)):
        baseline = airPLS(processed_intensities[i], lamb, polyorder, max_iters)
        processed_intensities[i] = processed_intensities[i] - baseline

    # min-max normalization to [0, 1]
    if minmax_normalize:
        for i in range(len(processed_intensities)):
            min_val = processed_intensities[i].min()
            max_val = processed_intensities[i].max()
            if max_val > min_val:  # Avoid division by zero
                processed_intensities[i] = (
                    processed_intensities[i] - min_val) / (max_val - min_val)
            else:
                processed_intensities[i] = np.zeros_like(processed_intensities[i])

    # ---- Optional diagnostic plot ----
    if plot:
        plt.figure(figsize=(10, 6))
        for intensity in processed_intensities:
            plt.plot(cut_Raman_Shifts, intensity)
        plt.xlabel('Raman Shift (cm⁻¹)')
        plt.ylabel('Intensity')
        plt.title('Preprocessed Spectra')
        plt.savefig('visualization/preprocessed_spectra.png', dpi=600)
        plt.show(block = False)
        # wait for 5s then close the plot
        plt.pause(5)
        plt.close()


    return cut_Raman_Shifts, processed_intensities





    
