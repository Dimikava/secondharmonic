"""
This module contains the Fourier analysis widgets for the secondharmonic plugin.
"""


from typing import TYPE_CHECKING, Literal

from magicgui import magic_factory, magicgui
from magicgui.widgets import CheckBox, Container, create_widget
from qtpy.QtWidgets import QHBoxLayout, QPushButton, QWidget
from skimage.util import img_as_float
import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import rfft
import dask.array as da

import napari
from napari.types import ImageData
from napari.viewer import Viewer

from .fourier_analysis import theoretical, theoretical2, analyze_pixel_signal


def _ensure_layer(viewer: Viewer, name: str, data: np.ndarray, cmap: str, overwrite: bool):
    
    if overwrite and name in viewer.layers:
        viewer.layers.pop(name)
    viewer.add_image(data, name=name, colormap=cmap)


def fourier_phase_analysis_widget():
    """Return a widget that performs FFT phase analysis on the whole image.

    The computation is fully vectorized: all pixel time series are transformed
    with a single ``rfft`` call and metrics are derived using NumPy
    broadcasting.
    """
    @magicgui(
        call_button="Run Fourier Phase Analysis",
        frame_start={"label": "Frame start", "min": 0, "max": 10_000, "step": 1},
        frame_stop={"label": "Frame stop (exclusive)", "min": 1, "max": 10_000, "step": 1},
        use_frame_slice={"label": "Use frame slice"},
        fft_axis={"label": "FFT axis", "choices": [0, 1, 2]},
        detrend_mean={"label": "Detrend (remove mean)"},
        # ---- Time base ----
        time_span={"label": "Time span (radians)", "min": 0.0, "max": 1000.0, "step": 0.01},
        custom_time={"label": "Custom time array (comma/space sep)", "widget_type": "LineEdit"},
        # ---- Harmonics / model ----
        model_choice={"label": "Model for R²", "choices": ["d3h"]},
        # ---- Thresholds ----
        r2_thresh={"label": "R² threshold", "min": 0.0, "max": 1.0, "step": 0.005},
        # Output / UX
        layer_prefix={"label": "Layer name prefix"},
        overwrite_layers={"label": "Overwrite existing layers"},
    )
    def run(
        image: ImageData,
        viewer: Viewer,
        # Data / slicing
        frame_start: int = 0,
        frame_stop: int = 360,
        use_frame_slice: bool = True,
        fft_axis: int = 0,
        detrend_mean: bool = False,
        # Time base
        time_span: float = 2 * np.pi,
        custom_time: str = "",
        # Harmonics / model
        model_choice: Literal["d3h"] = "d3h",
        # Thresholds
        r2_thresh: float = 0.8,
        # Output / UX
        layer_prefix: str = "",
        overwrite_layers: bool = True,    
    ):
        if image.ndim != 3:
            raise ValueError("Expected 3D image: (time, y, x)")
        
        T, H, W = image.shape
        
        if use_frame_slice:
            s0 = max(0, min(frame_start, T - 1))
            s1 = max(s0 + 1, min(frame_stop, T))
            image = image[s0:s1]
            T = image.shape[0]
       

        def r_squared_vec(obs: np.ndarray, pred: np.ndarray) -> np.ndarray:
            ss_res = np.sum((obs - pred) ** 2, axis=0)
            ss_tot = np.sum((obs - np.mean(obs, axis=0)) ** 2, axis=0)
            return 1 - ss_res / ss_tot
       
        if detrend_mean:
            image = image - image.mean(axis=0, keepdims=True)

        # ---------- time base ----------
        if custom_time.strip():
            parts = custom_time.replace(",", " ").split()
            t = np.array([float(p) for p in parts], dtype=float)
            if len(t) != T:
                raise ValueError(f"Custom time length ({len(t)}) must match number of frames ({T}).")
        else:
            t = np.linspace(0.0, time_span, T, endpoint=False)
        
        t_arr = t[:, None, None]
        
        fft_result = rfft(image, axis=0)
        fft_phases = np.angle(fft_result)
        if model_choice == "d3h":
            a_0 = np.abs(fft_result[0]) / len(t)
            a_4 = 2 * np.abs(fft_result[4]) / len(t)
            phase = np.mod(fft_phases[4], 2 * np.pi) / 6


            fit1 = theoretical(
                a_0[None, ...], a_4[None, ...], phase[None, ...], t_arr
            )
            r1 = r_squared_vec(image, fit1)
            fit2 = theoretical2(
                a_0[None, ...], a_4[None, ...], phase[None, ...], t_arr
            )
            r2 = r_squared_vec(image, fit2)

            alt_phase = np.abs(np.mod(fft_phases[4], 2 * np.pi) - 2 * np.pi) / 6
            fit_alt = theoretical(
                a_0[None, ...], a_4[None, ...], alt_phase[None, ...], t_arr
            )
            r_alt = r_squared_vec(image, fit_alt)

            use_alt = r2 > r1
            r2_map = np.where(use_alt, r_alt, r1)
            phase_final = np.where(use_alt, alt_phase, phase)
        else:
            raise ValueError(f"Unknown model choice: {model_choice}")

        phase_map = np.where(
            r2_map > r2_thresh, np.degrees(phase_final), np.nan
        )
        prefix = (layer_prefix + " ").strip() if layer_prefix else ""
        _ensure_layer(viewer, f"{prefix}Armchair angle Map", phase_map.reshape(H, W), "magma", overwrite_layers)
        _ensure_layer(viewer, f"{prefix}R² Map", r2_map.reshape(H, W), "magma", overwrite_layers)
        return None

    return run

def add_fourier_phase_tools_to_viewer(viewer: Viewer):
    fpw = fourier_phase_analysis_widget()
    viewer.window.add_dock_widget(fpw, area="right", name="Fourier Phase Analysis")
    viewer.window.add_dock_widget(export_with_matplotlib, area="right", name="Export with Matplotlib")
    return fpw

import matplotlib.pyplot as plt 
from .fourier_analysis import analyze_pixel_signal


def single_pixel_analysis_widget():
    @magicgui(
        call_button="Analyze Pixel FFT",
        x={"widget_type": "SpinBox", "min": 0, "max": 512},
        y={"widget_type": "SpinBox", "min": 0, "max": 512},
        #max_time={"widget_type": "SpinBox", "min": 0, "max": 360, "value": 180},
    )
    def widget(
        image: ImageData, x: int = 0, y: int = 0, viewer: Viewer = None
    ):
        import dask.array as da

        def theoretical(a_0, a_4, phi, alpha):
            return a_0 + a_4 * np.cos(2 * (3 * phi - 2 * alpha))

        def theoretical2(a_0, a_4, phi, alpha):
            return a_0 + a_4 * np.cos(2 * (3 * phi + 2 * alpha))

        def r_squared(signal, predicted):
            ss_res = np.sum((signal - predicted)**2)
            ss_tot = np.sum((signal - np.mean(signal))**2)
            r_squared = 1 - (ss_res / ss_tot)
            return r_squared

        if isinstance(image, da.Array):
            image = image.compute()

        if image.ndim != 3:
            raise ValueError("Expected 3D image: (time, y, x)")
        signal = image[:180, y, x]
        t = np.linspace(0, 2 * np.pi, signal.shape[0])
        

        # Debug info
        print(f"Selected pixel: (x={x}, y={y})")
        print(f"Signal shape: {signal.shape}, Time shape: {t.shape}")

        result = analyze_pixel_signal(signal, t)
        print("Result:", result)

        # Plot signal and fitted model
        a_0 = np.mean(signal)
        a_4 = 2 * np.abs(np.fft.rfft(signal)[4]) / len(t)

        fft_phase = np.mod(np.angle(np.fft.rfft(signal)[4]), 2 * np.pi) / 6
        r_1 = r_squared(signal,theoretical(a_0, a_4,fft_phase, t))
        r_2 = r_squared(signal,theoretical2(a_0, a_4,fft_phase, t))
        if r_2>r_1:
            fft_phase=np.abs(np.mod(fft_phase,2*np.pi)-2*np.pi)/6
        
        
        if fft_phase is not None:
            fitted = theoretical(a_0, a_4, fft_phase, t) 
        else:
            fitted = np.zeros_like(signal)

        plt.figure()
        plt.plot(t, signal, "ko", label="Raw Signal")
        plt.plot(t, fitted, "r-", label="Fitted Model")
        plt.title(
            f"Pixel ({x},{y}) - Phase: {fft_phase:.2f}°"
            if fft_phase
            else "No Fit"
        )
        plt.legend()
        plt.tight_layout()
        plt.show()

    return widget
